from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import argparse
import sys
import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split

import flwr as fl
from flwr.client import Client, ClientApp, NumPyClient
from flwr.simulation import run_simulation, start_simulation
from flwr_datasets import FederatedDataset
from flwr.common import ndarrays_to_parameters, NDArrays, Scalar, Context

import absl
import tensorflow_model_analysis as tfma
from tfx.components import CsvExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import Pusher
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.dsl.components.common import resolver
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing
from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx_bsl.tfxio import dataset_options
import tensorflow_transform as tft

import tensorflow as tf

def create_pipeline(client_id: int, module_file: str, output_dir: str) -> pipeline.Pipeline:
    data_root = f"../tfx-flower/data/simple/client_{client_id}.csv"
    pipeline_root = f"{output_dir}/pipeline_client_{client_id}"
    metadata_path = f"{output_dir}/metadata_client_{client_id}.db"
    serving_model_dir = os.path.join(output_dir, f"serving_model_client_{client_id}")

    # Brings data into the pipeline or otherwise joins/converts training data.
    example_gen = CsvExampleGen(input_base=os.path.dirname(data_root))
    
    # Computes statistics over data for visualization and example validation.
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

    # Generates schema based on statistics files.
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs['statistics'],
        infer_feature_shape=True)
    
    # Performs anomaly detection based on statistics and data schema.
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema'])
        
    # Performs transformations and feature engineering in training and serving.
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=module_file)
    
    return pipeline.Pipeline(
        pipeline_name=f"client_{client_id}_pipeline",
        pipeline_root=pipeline_root,
        components=[example_gen, statistics_gen, schema_gen, example_validator, transform],
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(metadata_path),
    )

def run_pipeline(client_id: int, module_file: str, output_dir: str):
    print(f"Running TFX pipeline for client {client_id}...")
    BeamDagRunner().run(create_pipeline(client_id=client_id, module_file=module_file, output_dir=output_dir))
    print(f"Pipeline completed for client {client_id}")

# Categorical features are assumed to each have a maximum value in the dataset.
_MAX_CATEGORICAL_FEATURE_VALUES = [24, 31, 13, 2000, 2000, 80, 80]

_CATEGORICAL_FEATURE_KEYS = [
    'trip_start_hour', 'trip_start_day', 'trip_start_month',
    'pickup_census_tract', 'dropoff_census_tract', 'pickup_community_area',
    'dropoff_community_area'
]

_DENSE_FLOAT_FEATURE_KEYS = ['trip_miles', 'fare', 'trip_seconds']

# Number of buckets used by tf.transform for encoding each feature.
_FEATURE_BUCKET_COUNT = 10

_BUCKET_FEATURE_KEYS = [
    'pickup_latitude', 'pickup_longitude', 'dropoff_latitude',
    'dropoff_longitude'
]

# Number of vocabulary terms used for encoding VOCAB_FEATURES by tf.transform
_VOCAB_SIZE = 1000

# Count of out-of-vocab buckets in which unrecognized VOCAB_FEATURES are hashed.
_OOV_SIZE = 10

_VOCAB_FEATURE_KEYS = [
    'payment_type',
    'company',
]

# Keys
_LABEL_KEY = 'tips'
_FARE_KEY = 'fare'

def _transformed_name(key):
  return key + '_xf'

def _transformed_names(keys):
  return [_transformed_name(key) for key in keys]

def _input_fn(file_pattern: List[str],
              data_accessor: DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 200) -> tf.data.Dataset:
  """Generates features and label for tuning/training.

  Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    data_accessor: DataAccessor for converting input to RecordBatch.
    tf_transform_output: A TFTransformOutput.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  return data_accessor.tf_dataset_factory(
      file_pattern,
      dataset_options.TensorFlowDatasetOptions(
          batch_size=batch_size, label_key=_transformed_name(_LABEL_KEY)),
      tf_transform_output.transformed_metadata.schema).repeat()

def create_model(hidden_units: List[int]=None) -> tf.keras.Model:
    """Creates a DNN Keras model for classifying taxi data.
    Mostly identical to the _build_keras_model in the taxi_utils_native_keras.py

    Args:
        hidden_units: [int], the layer sizes of the DNN (input layer first).

    Returns:
        A Wide and Deep keras Model.
    """
    # Keras needs the feature definitions at compile time.
    deep_input = {
        colname: tf.keras.layers.Input(name=colname, shape=(1,), dtype=tf.float32)
        for colname in _transformed_names(_DENSE_FLOAT_FEATURE_KEYS)
    }

    wide_vocab_input = {
        colname: tf.keras.layers.Input(name=colname, shape=(1,), dtype='int32')
        for colname in _transformed_names(_VOCAB_FEATURE_KEYS)
    }

    wide_bucket_input = {
        colname: tf.keras.layers.Input(name=colname, shape=(1,), dtype='int32')
        for colname in _transformed_names(_BUCKET_FEATURE_KEYS)
    }

    wide_categorical_inpt = {
        colname: tf.keras.layers.Input(name=colname, shape=(1,), dtype='int32')
        for colname in _transformed_names(_CATEGORICAL_FEATURE_KEYS)
    }

    input_layers = {
        **deep_input,
        **wide_vocab_input,
        **wide_bucket_input,
        **wide_categorical_inpt,
    }

    # Build deep branch
    deep = tf.keras.layers.concatenate(
        [tf.keras.layers.Normalization()(layer) for layer in deep_input.values()]
    )

    for numnodes in (hidden_units or [100, 70, 50, 25]):
        deep = tf.keras.layers.Dense(numnodes)(deep)

    # Build wide branch
    wide_layers = []

    for key in _transformed_names(_VOCAB_FEATURE_KEYS):
        wide_layers.append(
            tf.keras.layers.CategoryEncoding(num_tokens=_VOCAB_SIZE + _OOV_SIZE)(
                input_layers[key]
            )
        )

    for key in _transformed_names(_BUCKET_FEATURE_KEYS):
        wide_layers.append(
            tf.keras.layers.CategoryEncoding(num_tokens=_FEATURE_BUCKET_COUNT)(
                input_layers[key]
            )
        )
    
    for key, num_tokens in zip(
        _transformed_names(_CATEGORICAL_FEATURE_KEYS),
        _MAX_CATEGORICAL_FEATURE_VALUES,
    ):
        wide_layers.append(
            tf.keras.layers.CategoryEncoding(num_tokens=num_tokens + 1)(
                input_layers[key]
            )
        )

    wide = tf.keras.layers.concatenate(wide_layers)

    # Combine and create output
    combined = tf.keras.layers.concatenate([deep, wide])
    output = tf.keras.layers.Dense(1, activation='sigmoid')(combined)
    output = tf.keras.layers.Reshape((1,))(output)

    model = tf.keras.Model(inputs=input_layers, outputs=output)

    # Compile
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )

    return model

def load_transformed_data(client_id: int, split: str='train', batch_size:int=32):
    """
    Load preprocessed data from TFX Transform component output

    Args:
        client_id:
        split: data split ('train' or 'eval')
        batch_size: batch size for dataset

    Returns:
        tf.data.Dataset with preprocessed features and labels
    """
    # Paths to transformed data
    transform_output_path = glob.glob(
    f"{OUTPUT_DIR}/pipeline_client_{client_id}/Transform/transform_graph/*"
    )[-1] 

    print(f"{transform_output_path}")

    
    data_path = os.path.join(
        OUTPUT_DIR,
        f"pipeline_client_{client_id}",
        "Transform",
        "transformed_examples",
        "*",                   # this matches span directory (e.g., 5/)
        f"Split-{split}",
        "*.gz"
    )

    try:
        # Load the transformed output
        tf_transform_output = tft.TFTransformOutput(transform_output_path)

        # Find tfrecord files
        tfrecord_files = glob.glob(data_path)
        if not tfrecord_files:
            raise FileNotFoundError(f"No tfrecord files found at {data_path}")
        
        print(f"Loading data for client {client_id}, split: {split}")
        print(f"Found {len(tfrecord_files)} tfrecord files")

        # Create dataset from tfrecord files
        dataset = tf.data.TFRecordDataset(tfrecord_files, compression_type='GZIP')

        # Parse the examples using the transformed schema
        feature_spec = tf_transform_output.transformed_feature_spec()

        def parse_fn(example_proto):
            parsed = tf.io.parse_single_example(example_proto, feature_spec)

            label_key = _transformed_name(_LABEL_KEY)
            label = parsed.pop(label_key)

            return parsed, label
        
        dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        # check dataset
        for batch in dataset.take(1):
            features, labels = batch
            print(f"Feature keys: {list(features.keys())}")
            print(f"Batch shape - Features: {len(features)}, Labels: {labels.shape}")
            break

        return dataset 
    
    except Exception as e:
        print(f"Error loading data for client {client_id}: {e}")


### Flower Client Implementation
class TaxiFlowerClient(NumPyClient):

    def __init__(self, client_id: int):
        self.client_id = client_id
        self.model = create_model()

        # Load client's data
        self.train_dataset = load_transformed_data(client_id, 'train', batch_size=32)
        self.eval_dataset = load_transformed_data(client_id, 'eval', batch_size=32)

        self.train_size = sum(1 for _ in self.train_dataset.unbatch())
        self.eval_size = sum(1 for _ in self.eval_dataset.unbatch())

        print(f"Client {client_id} initialized with {self.train_size} train examples and {self.eval_size} eval examples")

    def get_parameters(self, config):
        return self.model.get_weights()
    
    def set_parameters(self, parameters) -> None:
        return self.model.set_weights(parameters)
    
    def fit(self, parameters, config):
        """Train themodel on the local dataset"""
        self.set_parameters(parameters)  # set parameters received from the server

        epochs = int(config.get("epochs", 1))

        history = self.model.fit(
            self.train_dataset,
            epochs=epochs,
            validation_data=self.eval_dataset,
            verbose=1
        )

        return (
            self.get_parameters({}),
            self.train_size,
            {
                "train_loss": float(history.history["loss"][-1]),
                "train_accuracy": float(history.history["binary_accuracy"][-1])
            }
        )
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        loss, accuracy = self.model.evaluate(self.eval_dataset, verbose=0)

        return float(loss), self.eval_size, {"accuracy": float(accuracy)}
    

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="FL Client for Taxi Tip Classification")
    parser.add_argument(
        "--client-id",
        type=int,
        required=True,
        help="Client ID (0-4)"
    )

    parser.add_argument(
        "--server-address",
        type=str,
        default="localhost:8080",
        help="Server address (default: localhost:8080)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./tfx_output",
        help="Directory containing TFX pipeline outputs (default: ./tfx_output)"
    )

    return parser.parse_args()


def client_fn(cid: str) ->  TaxiFlowerClient:
    client_id = int(cid)
    return TaxiFlowerClient(client_id).to_client()

# Create the ClientApp (for simulation)
client = ClientApp(client_fn=client_fn)


if __name__=='__main__':
    MODULE_FILE = "./taxi_utils_native_keras.py"  # Your preprocessing module file
    OUTPUT_DIR = "./tfx_output"

    NUM_CLIENTS = 5

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    args = parse_arguments()

    print(f"Starting Flower client {args.client_id}")
    print(f"Server addresS: {args.server_address}")
    print(f"TFX output directory: {args.output_dir}")

    if not (0 <= args.client_id <= 4):
        print("Error: Client ID must be between 0 and 4")
        sys.exit(1)

    try:
        run_pipeline(args.client_id, MODULE_FILE, OUTPUT_DIR)
    except Exception as e:
        print(f"Error running pipeline for client {args.client_id}: {e}")

    try:
        client = TaxiFlowerClient(args.client_id)

        fl.client.start_client(
            server_address=args.server_address,
            client=client
        )
    except Exception as e:
        print(f"Error starting client: {e}")
        sys.exit(1)

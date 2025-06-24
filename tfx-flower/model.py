# Categorical features are assumed to each have a maximum value in the dataset.
from typing import List
import tensorflow as tf
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
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
from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx_bsl.tfxio import dataset_options
import tensorflow_transform as tft

import tensorflow as tf

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
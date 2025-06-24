from collections import OrderedDict
import pickle
from typing import Dict, List, Optional, Tuple

import os 

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import flwr as fl
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg, FedAdagrad
from flwr.simulation import run_simulation, start_simulation
from flwr_datasets import FederatedDataset
from flwr.common import Context, parameters_to_ndarrays, ndarrays_to_parameters

def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    if not metrics:
        return {}
    
    accuracies = [num_examples * m.get("accuracy", 0.0) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    total_examples = sum(examples)
    if total_examples == 0:
        return {"accuracy": 0.0}
    
    weighted_accuracy = sum(accuracies) / total_examples
    return {"accuracy": weighted_accuracy}

class FedAvgWithSaving(FedAvg):
    """Custom strategy that saves the final model"""
    def __init__(self, model_save_path: str = "saved_models", model_create_fn=None, **kwargs):
        super().__init__(**kwargs)
        self.model_save_path = model_save_path 
        self.model_create_fn = model_create_fn
        self.final_parameters = None 

        os.makedirs(model_save_path, exist_ok=True)

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate fit results and save final model"""
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            self.final_parameters = aggregated_parameters
            if server_round == self.num_rounds:
                self.save_model(server_round)

        return aggregated_parameters, aggregated_metrics
    
    # TODO: can also extend to store model as flwr, numpy or pkl
    def save_model(self, round_num: int):
        if self.final_parameters is None:
            print("No parameters to save")
            return
        
        arrays = parameters_to_ndarrays(self.final_parameters)
        
        if self.model_create_fn is not None:
            self.save_as_keras_model(arrays, round_num)
        else:
            print("Model creation function not provided.")


def load_saved_model(model_path: str, format_type: str = "flwr"):
    if format_type == "flwr":
        with open(model_path, 'rb') as f:
            parameters = pickle.load(f)
        return parameters 
    else:
        raise ValueError(f"Invalid format type: {format_type}")
    

def save_as_keras_model(self, arrays, round_num: int):
    """Save as complete Keras model."""
    try:
        model = self.model_create_fn()

        # Set the weights
        model.set_weights(arrays)

        # Save the complete Keras model
        keras_path = os.path.join(f"parameters_round_{round_num}.keras")
        model.save(keras_path)
        print(f"Keras model saved to: {keras_path}")

        # Also save weights (lighter file)
        weights_path = os.path.join(
            self.model_save_path,
            f"parameters_weights_round_{round_num}.h5"
        )
        model.save_weights(weights_path)
        print(f"Keras weights saved to: {weights_path}")

    except Exception as e:
        print(f"Error saving Keras model: {e}")


def load_saved_keras_model(model_path: str, format_type: str = "keras", model_create_fn=None):
    """
    Load a previously saved Keras model.
    
    Args:
        model_path: Path to the saved model
        format_type: "keras", "weights", "flwr", or "numpy"
        model_create_fn: Function to create model (needed for weights/flwr/numpy formats)
    """
    if format_type == "keras":
        # Load complete Keras model
        return tf.keras.models.load_model(model_path)
    elif format_type == "weights":
        # Load weights into new model
        if model_create_fn is None:
            raise ValueError("model_create_fn required for weights format")
        model = model_create_fn()
        model.load_weights(model_path)
        return model
    
    else:
        raise ValueError("Invalid format_type")


NUM_CLIENTS = 5
NUM_ROUNDS = 5



def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour.

    You can use the settings in `context.run_config` to parameterize the
    construction of all elements (e.g the strategy or the number of rounds)
    wrapped in the returned ServerAppComponents object.
    """

    # Configure the server for 3 rounds of training
    config = ServerConfig(num_rounds=NUM_ROUNDS)

    # Use the custom strategy with saving capability
    strategy = FedAvgWithSaving(
        model_save_path="saved_models",
        # model_create_fn=create_model,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
    )
    # Store num_rounds in strategy for final round detection
    strategy.num_rounds = NUM_ROUNDS

    return ServerAppComponents(strategy=strategy, config=config)

# Create the ServerApp
server = ServerApp(server_fn=server_fn)

# Specify the resources each of your clients need
# By default, each client will be allocated 1x CPU and 0x GPUs
backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}

# When running on GPU, assign an entire GPU for each client
DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU

if DEVICE == "cuda":
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}
    # Refer to our Flower framework documentation for more details about Flower simulations
    # and how to set up the `backend_config`

# Global variable to store history for analysis
# fl_history = None

if __name__ == "__main__":

    from model import create_model

    NUM_ROUNDS = 5
    NUM_CLIENTS = 5


    # strategy = FedAvg(
    # fraction_fit=1.0,
    # fraction_evaluate=1.0,
    # min_fit_clients=NUM_CLIENTS,
    # min_evaluate_clients=NUM_CLIENTS,
    # min_available_clients=NUM_CLIENTS,
    # evaluate_metrics_aggregation_fn=weighted_average,
    # fit_metrics_aggregation_fn=weighted_average,
    # )

    # Use the custom strategy with model saving
    strategy = FedAvgWithSaving(
        model_save_path="saved_models",
        model_create_fn=create_model,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
    )
    strategy.num_rounds = NUM_ROUNDS

    # Start server
    print(f"Starting FL server for {NUM_ROUNDS} rounds...")
    print(f"Minimum clients required: {NUM_CLIENTS}")
    print(f"Models will be saved to: saved_models/")
    
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    print("Training completed! Check the 'saved_models' directory for the final global model.")


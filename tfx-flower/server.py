from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

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
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg, FedAdagrad
from flwr.simulation import run_simulation, start_simulation
from flwr_datasets import FederatedDataset
from flwr.common import ndarrays_to_parameters, NDArrays, Scalar, Context

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
fl_history = None

if __name__ == "__main__":
    NUM_ROUNDS = 5
    NUM_CLIENTS = 5

    strategy = FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=NUM_CLIENTS,
    min_evaluate_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    evaluate_metrics_aggregation_fn=weighted_average,
    fit_metrics_aggregation_fn=weighted_average,
    )

    # Start server
    print(f"Starting FL server for {NUM_ROUNDS} rounds...")
    print(f"Minimum clients required: {NUM_CLIENTS}")
    print("Results will be saved to fl_results.json")
    
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )


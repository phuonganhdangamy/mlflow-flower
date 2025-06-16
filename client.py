import argparse
import mlflow
import warnings
from collections import OrderedDict
import numpy as np
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import os
import sys
sys.path.append("cords-semantics-lib-main")
import cords_semantics.tags as cords_tags

# Import the prepare_dataset function
from dataset import prepare_dataset
# Import model and training functions
from model import ElectricityModel, train, evaluate

# Suppress warnings
warnings.filterwarnings("ignore")

# Device configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define MLflow experiment name
experiment_name = "Federated-Learning-Energy-Prediction"
run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
logdir = os.path.join("logs", experiment_name, run_name)


# Get client id
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--client-id",
    choices=[0, 1],
    default=0,
    type=int,
    help="Partition of the dataset divided into 2 iid partitions created artificially.",
)
client_id = parser.parse_known_args()[0].client_id

partition_id = np.random.choice(5)

train_loaders, test_loaders = prepare_dataset(client_id, partition_id)


# Define Flower client
class FlowerElectricityClient(NumPyClient):
    def __init__(self, current_round):
        super().__init__()
        self.model = ElectricityModel(input_dim=5).to(DEVICE)
        self.criterion = nn.MSELoss()  # Mean Squared Error loss for regression
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.003)
        self.current_round = current_round

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        
        with mlflow.start_run(run_name=f"client_{client_id}_{run_name}") as mlflow_run:
            # Train model
            train(self.model, train_loaders, epochs=5, criterion=self.criterion, 
                  optimizer=self.optimizer, device=DEVICE)
            
            # Log metrics
            loss = evaluate(self.model, test_loaders, self.criterion, DEVICE)
            
            # Log MLflow tags and metrics
            mlflow.set_tag(cords_tags.CORDS_RUN, mlflow_run.info.run_id)
            mlflow.set_tag(cords_tags.CORDS_RUN_EXECUTES, "ANN")
            mlflow.set_tag(cords_tags.CORDS_IMPLEMENTATION, "python")
            mlflow.set_tag(cords_tags.CORDS_SOFTWARE, "pytorch")
            mlflow.set_tag("client_id", client_id)
            mlflow.set_tag("round", self.current_round)
            
            mlflow.log_metric("loss", loss)
            mlflow.log_param("learning_rate", 0.003)
            mlflow.log_param("epochs", 5)
            
            # Log client model
            mlflow.pytorch.log_model(
                self.model, 
                f"Round_{self.current_round}_Client_{client_id}"
            )

        self.current_round += 1
        return self.get_parameters(config={}), len(train_loaders.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss = evaluate(self.model, test_loaders, self.criterion, DEVICE)
        return loss, len(test_loaders.dataset), {"loss": loss}


def client_fn(context: Context):
    """Create and return an instance of Flower `Client`."""
    current_round = 0
    return FlowerElectricityClient(current_round).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)

# Legacy mode
if __name__ == "__main__":
    from flwr.client import start_client
    current_round = 0 
    client = FlowerElectricityClient(current_round=current_round).to_client()
    start_client(
        server_address="127.0.0.1:8080",
        client=client,
    )
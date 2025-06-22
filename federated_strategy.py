from typing import List, Tuple
import mlflow
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Metrics, FitRes
import torch
import numpy as np
from collections import OrderedDict
import sys
sys.path.append("cords-semantics-lib-main")
import cords_semantics.tags as cords_tags

from model import ElectricityModel


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Define metric aggregation function."""
    # Multiply loss of each client by number of examples used
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    # Aggregate and return custom metric (weighted average)
    total_examples = sum([num_examples for num_examples, _ in metrics])
    return {"loss": sum(losses) / total_examples}


class CustomFedAvg(FedAvg):
    """Custom FedAvg strategy that logs the global model after each round."""
    
    def __init__(self, input_dim=5, num_rounds=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_model = ElectricityModel(input_dim=input_dim)
        self.mlflow_run_id = None
        self.num_rounds = num_rounds
        
    def set_mlflow_run_id(self, run_id: str):
        """Set the MLflow run ID for logging."""
        self.mlflow_run_id = run_id
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.client.Client, FitRes]],
        failures: List[BaseException],
    ):
        """Aggregate fit results and log the global model."""
        # Call parent aggregate_fit to get aggregated parameters
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        if aggregated_parameters is not None:
            # Convert aggregated parameters back to model state dict
            params_dict = zip(self.global_model.state_dict().keys(), aggregated_parameters)
            state_dict = OrderedDict({k: torch.tensor(np.array(v)) for k, v in params_dict})
            self.global_model.load_state_dict(state_dict, strict=True)
            
            # Log the global model to MLflow
            if self.mlflow_run_id:
                # Check if there's already an active run
                active_run = mlflow.active_run()
                if active_run and active_run.info.run_id == self.mlflow_run_id:
                    # Use the existing active run
                    mlflow.pytorch.log_model(
                        self.global_model, 
                        f"global_model_round_{server_round}",
                        registered_model_name=f"FederatedElectricityModel_Round_{server_round}"
                    )
                    mlflow.log_metric("server_round", server_round)
                    
                    # If it's the final round, also log as the final model
                    if server_round == self.num_rounds:
                        mlflow.pytorch.log_model(
                            self.global_model, 
                            "final_global_model",
                            registered_model_name="FederatedElectricityModel_Final"
                        )
                        mlflow.set_tag("model_type", "final_global_model")
                        mlflow.set_tag(cords_tags.CORDS_RUN_EXECUTES, "ANN")
                        mlflow.set_tag(cords_tags.CORDS_IMPLEMENTATION, "python")
                        mlflow.set_tag(cords_tags.CORDS_SOFTWARE, "pytorch")
                        print(f"Final global model logged to MLflow (Round {server_round})")
                else:
                    # Start a new run if no active run or different run ID
                    with mlflow.start_run(run_id=self.mlflow_run_id):
                        mlflow.pytorch.log_model(
                            self.global_model, 
                            f"global_model_round_{server_round}",
                            registered_model_name=f"FederatedElectricityModel_Round_{server_round}"
                        )
                        mlflow.log_metric("server_round", server_round)
                        
                        # If it's the final round, also log as the final model
                        if server_round == self.num_rounds:
                            mlflow.pytorch.log_model(
                                self.global_model, 
                                "final_global_model",
                                registered_model_name="FederatedElectricityModel_Final"
                            )
                            mlflow.set_tag("model_type", "final_global_model")
                            mlflow.set_tag(cords_tags.CORDS_RUN_EXECUTES, "ANN")
                            mlflow.set_tag(cords_tags.CORDS_IMPLEMENTATION, "python")
                            mlflow.set_tag(cords_tags.CORDS_SOFTWARE, "pytorch")
                            print(f"Final global model logged to MLflow (Round {server_round})")
        
        
        return aggregated_parameters, aggregated_metrics
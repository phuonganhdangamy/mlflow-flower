import mlflow
import flwr as fl
from flwr.server import ServerApp, ServerConfig
from datetime import datetime
import os

# Import custom strategy
from federated_strategy import CustomFedAvg, weighted_average

# Configuration
experiment_name = "Federated-Learning-Energy-Prediction"
run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
logdir = os.path.join("logs", experiment_name, run_name)

# Server configuration
NUM_ROUNDS = 10
INPUT_DIM = 5

# Define strategy with custom aggregation
strategy = CustomFedAvg(
    input_dim=INPUT_DIM,
    num_rounds=NUM_ROUNDS,
    evaluate_metrics_aggregation_fn=weighted_average
)

# Define config
config = ServerConfig(num_rounds=NUM_ROUNDS)

# Flower ServerApp
app = ServerApp(
    config=config,
    strategy=strategy,
)

# Legacy mode
if __name__ == "__main__":
    from flwr.server import start_server
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"server_{run_name}") as mlflow_run:
        # Set the MLflow run ID in the strategy
        strategy.set_mlflow_run_id(mlflow_run.info.run_id)
        
        # Log server parameters
        mlflow.log_param("num_rounds", config.num_rounds)
        mlflow.log_param("strategy", "CustomFedAvg")
        mlflow.log_param("server_address", "0.0.0.0:8080")
        mlflow.log_param("input_dim", INPUT_DIM)
        
        print(f"Starting federated learning server...")
        print(f"MLflow run ID: {mlflow_run.info.run_id}")
        print(f"Number of rounds: {NUM_ROUNDS}")
        
        start_server(
            server_address="0.0.0.0:8080",
            config=config,
            strategy=strategy,
        )
        
        print("Federated learning completed!")
        print(f"Final global model should be logged to MLflow run: {mlflow_run.info.run_id}")
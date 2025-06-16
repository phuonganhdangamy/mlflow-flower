import mlflow
import torch
import numpy as np
from mlflow.tracking import MlflowClient
import torch.nn as nn

# Import the model class
from model import ElectricityModel


class GlobalModelTester:
    """Class to handle loading and testing of the global federated model."""
    
    def __init__(self, experiment_name="Federated-Learning-Energy-Prediction", 
                 mlflow_uri="http://127.0.0.1:5000"):
        self.experiment_name = experiment_name
        self.mlflow_uri = mlflow_uri
        mlflow.set_tracking_uri(self.mlflow_uri)
        self.client = MlflowClient()
        
    def load_final_global_model(self):
        """Load the final global model from MLflow."""
        
        # Get the experiment
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{self.experiment_name}' not found")
        
        # Search for runs with the final global model tag
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.model_type = 'final_global_model'",
            order_by=["start_time DESC"]
        )
        
        if not runs:
            print("No final global model found. Trying to load from registered models...")
            # Alternative: Load from registered models
            try:
                model_uri = "models:/FederatedElectricityModel_Final/latest"
                model = mlflow.pytorch.load_model(model_uri)
                print("Loaded model from model registry")
                return model
            except Exception as e:
                print(f"Could not load from registered models: {e}")
                return self._load_latest_global_model()
        
        # Get the most recent run with final global model
        latest_run = runs[0]
        run_id = latest_run.info.run_id
        
        # Load the model
        model_uri = f"runs:/{run_id}/final_global_model"
        model = mlflow.pytorch.load_model(model_uri)
        
        print(f"Loaded final global model from run: {run_id}")
        return model
    
    def _load_latest_global_model(self):
        """Fallback method to load the latest global model from any round."""
        print("Trying to load the latest global model from any round...")
        
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        if not experiment:
            raise ValueError(f"Experiment '{self.experiment_name}' not found")
        
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )
        
        for run in runs:
            artifacts = self.client.list_artifacts(run.info.run_id)
            global_model_artifacts = [a for a in artifacts if 'global_model' in a.path]
            
            if global_model_artifacts:
                # Load the most recent global model found
                latest_artifact = sorted(global_model_artifacts, key=lambda x: x.path)[-1]
                model_uri = f"runs:/{run.info.run_id}/{latest_artifact.path}"
                model = mlflow.pytorch.load_model(model_uri)
                print(f"Loaded global model from run: {run.info.run_id}, artifact: {latest_artifact.path}")
                return model
        
        raise ValueError("No global model found in any run")
    
    def predict_single_record(self, model, data_point):
        """Make prediction on a single data record."""
        model.eval()
        
        # Convert to tensor if it's not already
        if not isinstance(data_point, torch.Tensor):
            data_point = torch.tensor(data_point, dtype=torch.float32)
        
        # Ensure it has the right shape (1, num_features)
        if data_point.dim() == 1:
            data_point = data_point.unsqueeze(0)
        
        with torch.no_grad():
            prediction = model(data_point)
        
        return prediction.item()
    
    def test_on_batch(self, model, test_data, test_targets):
        """Test the model on a batch of data."""
        model.eval()
        
        if not isinstance(test_data, torch.Tensor):
            test_data = torch.tensor(test_data, dtype=torch.float32)
        if not isinstance(test_targets, torch.Tensor):
            test_targets = torch.tensor(test_targets, dtype=torch.float32)
        
        with torch.no_grad():
            predictions = model(test_data)
            
            # Calculate metrics
            mse = nn.MSELoss()(predictions.squeeze(), test_targets)
            mae = nn.L1Loss()(predictions.squeeze(), test_targets)
        
        return {
            'predictions': predictions.numpy(),
            'mse': mse.item(),
            'mae': mae.item()
        }
    
    def run_tests(self, test_data=None, test_targets=None):
        """Run comprehensive tests on the global model."""
        try:
            # Load the final global model
            global_model = self.load_final_global_model()
            
            if global_model is not None:
                print("Successfully loaded the final global model!")
                
                # Test on a single record
                single_test_record = np.array([0.5, 0.3, 0.8, 0.2, 0.9])  # 5 features
                prediction = self.predict_single_record(global_model, single_test_record)
                print(f"Prediction for single record {single_test_record}: {prediction:.4f}")
                
                # Test on batch if data provided
                if test_data is not None and test_targets is not None:
                    results = self.test_on_batch(global_model, test_data, test_targets)
                    print(f"Batch test results:")
                    print(f"MSE: {results['mse']:.4f}")
                    print(f"MAE: {results['mae']:.4f}")
                    print(f"Predictions shape: {results['predictions'].shape}")
                else:
                    # Use dummy data for demonstration
                    print("No test data provided, using dummy data...")
                    dummy_data = np.random.rand(10, 5)  # 10 samples, 5 features
                    dummy_targets = np.random.rand(10)  # 10 target values
                    
                    results = self.test_on_batch(global_model, dummy_data, dummy_targets)
                    print(f"Dummy batch test results:")
                    print(f"MSE: {results['mse']:.4f}")
                    print(f"MAE: {results['mae']:.4f}")
                
                return global_model
            else:
                print("Could not load the final global model")
                return None
                
        except Exception as e:
            print(f"Error during testing: {e}")
            return None


# Example usage
if __name__ == "__main__":
    # Initialize the tester
    tester = GlobalModelTester()
    
    # Run tests
    model = tester.run_tests()
    
    if model:
        print("\nModel loaded successfully! You can now use it for predictions.")
        
        # Example of how to use for new predictions
        new_data_point = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        prediction = tester.predict_single_record(model, new_data_point)
        print(f"New prediction: {prediction:.4f}")
    else:
        print("Failed to load model. Please check if federated learning has completed and models are logged.")
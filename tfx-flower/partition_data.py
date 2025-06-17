import os
import pandas as pd
from pathlib import Path

# Config
NUM_CLIENTS = 5
TOTAL_PARTITIONS = NUM_CLIENTS + 1  # 5 clients + 1 global test
DATA_PATH = os.path.join("..", "tfx-flower", "data", "simple", "data.csv")
OUTPUT_DIR = os.path.dirname(DATA_PATH)

def partition_data():
    # Load the original data
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Calculate partition sizes
    total_rows = len(df)
    chunk_size = total_rows // TOTAL_PARTITIONS

    for i in range(NUM_CLIENTS):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        client_df = df.iloc[start_idx:end_idx]
        client_file = Path(OUTPUT_DIR) / f"client_{i+1}.csv"
        client_df.to_csv(client_file, index=False)
        print(f"Client {i+1}: Saved {len(client_df)} rows to {client_file}")

    # Final chunk for global test set
    test_df = df.iloc[NUM_CLIENTS * chunk_size:]
    test_file = Path(OUTPUT_DIR) / "global_test.csv"
    test_df.to_csv(test_file, index=False)
    print(f"Global Test Set: Saved {len(test_df)} rows to {test_file}")

if __name__ == "__main__":
    partition_data()

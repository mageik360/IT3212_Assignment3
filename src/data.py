from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

# Load dataset from path specified in .env file
def load_dataset():
    dataset_path = os.getenv("PATH_DATASET")
    if not dataset_path:
        raise ValueError("PATH_DATASET is not set in the .env file.")
    
    try:
        dataset = pd.read_csv(dataset_path)
        return dataset
    except Exception as e:
        raise ValueError(f"Failed to load dataset from {dataset_path}: {e}")
      
if __name__ == "__main__":
    dataset = load_dataset()
    print("Dataset loaded successfully.")
    print(dataset.head())
import os
import argparse
from huggingface_hub import snapshot_download, hf_hub_download

def download_resources(dataset_repo_id):
    # 1. Download UMT5 Model
    print("Downloading UMT5-Base model...")
    model_path = "models/umt5"
    os.makedirs(model_path, exist_ok=True)
    
    # We need specific files for the model to load locally
    files_to_download = ["config.json", "pytorch_model.bin", "spiece.model", "tokenizer.json", "special_tokens_map.json", "tokenizer_config.json"]
    
    try:
        snapshot_download(repo_id="google/umt5-base", local_dir=model_path, local_dir_use_symlinks=False)
        print(f"UMT5 model downloaded to {model_path}")
    except Exception as e:
        print(f"Error downloading UMT5: {e}")
        print("You might need to install huggingface_hub: pip install huggingface_hub")

    # 2. Download Dataset
    print(f"Downloading Dataset from {dataset_repo_id}...")
    data_path = "data"
    os.makedirs(data_path, exist_ok=True)
    
    try:
        snapshot_download(repo_id=dataset_repo_id, local_dir=data_path, repo_type="dataset", local_dir_use_symlinks=False)
        print(f"Dataset downloaded to {data_path}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download resources for Sign Language Recognition")
    parser.add_argument("--dataset_repo_id", type=str, required=True, help="Hugging Face Dataset Repo ID (e.g., 'username/dataset_name')")
    
    args = parser.parse_args()
    
    download_resources(args.dataset_repo_id)

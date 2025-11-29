import os
import argparse
from huggingface_hub import HfApi

def upload_folder(local_dir, repo_id, token):
    if not os.path.exists(local_dir):
        print(f"Directory {local_dir} does not exist. Skipping upload.")
        return

    print(f"Uploading {local_dir} to {repo_id}...")
    try:
        api = HfApi(token=token)
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            repo_type="model"
        )
        print("Upload complete!")
    except Exception as e:
        print(f"Error uploading to Hugging Face: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload folder to Hugging Face Hub")
    parser.add_argument("--local_dir", type=str, required=True, help="Local directory to upload")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face Repo ID (e.g., username/model-name)")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face Write Token")
    
    args = parser.parse_args()
    
    upload_folder(args.local_dir, args.repo_id, args.token)

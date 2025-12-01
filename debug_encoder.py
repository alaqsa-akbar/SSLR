import torch
from modules.pose_encoder import PoseEncoder, PoseEncoderConfig
from modules.dataset import SignLanguageDataset, collate_fn
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import numpy as np
import os

def check_pose_encoder_outputs():
    # Load Config and Model
    config_path = "configs/pose_encoder_config.json"
    config = PoseEncoderConfig.from_json_file(config_path)
    
    # Try to load trained weights if available, otherwise random
    model_dir = "output/pose_encoder_contrastive"
    if os.path.exists(os.path.join(model_dir, "pytorch_model.bin")):
        print(f"Loading trained weights from {model_dir}")
        pose_encoder = PoseEncoder.from_pretrained(model_dir)
    else:
        print("Using random weights (untrained)")
        pose_encoder = PoseEncoder(config)
    
    pose_encoder.eval()
    
    # Load a few samples
    tokenizer = AutoTokenizer.from_pretrained("google/umt5-base")
    dataset = SignLanguageDataset("data/dev.csv", "data/data.pkl", tokenizer, max_frames=100)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    
    batch = next(iter(loader))
    poses = batch['pose']
    masks = batch['pose_attention_mask']
    
    print(f"Input Pose Shape: {poses.shape}")
    print(f"Input Mask Shape: {masks.shape}")
    
    # Check if inputs are all zeros or identical
    print(f"Pose Std: {poses.std(dim=(1,2,3))}") # Should be > 0
    
    with torch.no_grad():
        outputs = pose_encoder(poses, attention_mask=masks)
        
    print(f"Output Shape: {outputs.shape}")
    
    # Check if outputs are identical across batch
    # Calculate cosine similarity between different batch elements
    from torch.nn.functional import cosine_similarity
    
    sim_0_1 = cosine_similarity(outputs[0].mean(0), outputs[1].mean(0), dim=0)
    sim_0_2 = cosine_similarity(outputs[0].mean(0), outputs[2].mean(0), dim=0)
    
    print(f"Cosine Sim (0 vs 1): {sim_0_1.item():.4f}")
    print(f"Cosine Sim (0 vs 2): {sim_0_2.item():.4f}")
    
    if sim_0_1 > 0.99:
        print("WARNING: Outputs are almost identical! Model has collapsed.")
    else:
        print("Outputs seem distinct.")

if __name__ == "__main__":
    check_pose_encoder_outputs()

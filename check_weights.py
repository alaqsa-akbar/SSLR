import torch
import os
from modules.pose_encoder import PoseEncoder, PoseEncoderConfig
from safetensors.torch import load_file

def check_weights():
    model_dir = "output/pose_encoder_contrastive"
    print(f"Checking weights in {model_dir}")
    
    if os.path.exists(os.path.join(model_dir, "model.safetensors")):
        state_dict = load_file(os.path.join(model_dir, "model.safetensors"))
    elif os.path.exists(os.path.join(model_dir, "pytorch_model.bin")):
        state_dict = torch.load(os.path.join(model_dir, "pytorch_model.bin"))
    else:
        print("No weights found!")
        return

    print(f"Total parameters: {len(state_dict)}")
    
    has_nan = False
    has_inf = False
    all_zeros = True
    
    for name, param in state_dict.items():
        if torch.isnan(param).any():
            print(f"NANs found in {name}")
            has_nan = True
        if torch.isinf(param).any():
            print(f"INFs found in {name}")
            has_inf = True
        if param.abs().sum() > 1e-9:
            all_zeros = False
            
        # Print stats for a few layers
        if "gcn" in name or "layers.0" in name:
            print(f"{name}: Mean={param.mean().item():.4f}, Std={param.std().item():.4f}, Min={param.min().item():.4f}, Max={param.max().item():.4f}")
            
    if has_nan:
        print("CRITICAL: Model contains NaNs!")
    if has_inf:
        print("CRITICAL: Model contains Infs!")
    if all_zeros:
        print("CRITICAL: Model is all zeros!")
    
    if not has_nan and not has_inf and not all_zeros:
        print("Weights look numerically valid (no NaNs/Infs/Zeros).")

if __name__ == "__main__":
    check_weights()

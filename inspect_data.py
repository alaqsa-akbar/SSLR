import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from modules.dataset import SignLanguageDataset, GlossTokenizer, collate_fn_hybrid

# CONFIG
DATA_DIR = "data" # Change this to your path
VOCAB_FILE = "vocab.json"
PKL_FILE = f"{DATA_DIR}/data.pkl"
CSV_FILE = f"{DATA_DIR}/train.csv"

def inspect():
    print("--- Inspecting Dataset Stability ---")
    tokenizer = GlossTokenizer(VOCAB_FILE)
    
    # Initialize dataset with augmentation OFF to check raw stability
    dataset = SignLanguageDataset(
        CSV_FILE, PKL_FILE, tokenizer, 
        max_frames=None, downsample_factor=1, mode='hybrid', augment=False
    )
    
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn_hybrid)
    
    print(f"Total Samples: {len(dataset)}")
    
    max_val = -float('inf')
    min_val = float('inf')
    nan_count = 0
    short_samples = 0
    
    for i, batch in enumerate(tqdm(loader)):
        pose = batch['pose'] # (B, T, 86, 4)
        labels = batch['labels']
        input_len = batch['input_lengths'][0].item()
        target_len = batch['target_lengths'][0].item()
        
        # 1. Check for NaNs
        if torch.isnan(pose).any():
            print(f"❌ NaN found in Sample ID: {batch['ids'][0]}")
            nan_count += 1
            
        # 2. Check Value Range
        curr_max = pose.max().item()
        curr_min = pose.min().item()
        max_val = max(max_val, curr_max)
        min_val = min(min_val, curr_min)
        
        # 3. Check CTC Constraint (Input Frames >= Target Labels)
        if input_len < target_len:
            print(f"⚠️ CTC Violation: ID {batch['ids'][0]} | Frames: {input_len} < Tokens: {target_len}")
            print(f"   Gloss: {batch['raw_text'][0]}")
            short_samples += 1

        # 4. Check for Massive Values (Instability)
        if abs(curr_max) > 100 or abs(curr_min) > 100:
            print(f"⚠️ Massive Values: ID {batch['ids'][0]} | Range: [{curr_min:.2f}, {curr_max:.2f}]")

    print("\n--- Summary ---")
    print(f"NaN Samples: {nan_count}")
    print(f"CTC Violations (Frame < Label): {short_samples}")
    print(f"Global Value Range: [{min_val:.4f}, {max_val:.4f}]")
    print("----------------")
    
    if max_val > 10.0 or min_val < -10.0:
        print("🚨 CRITICAL: Input values are too large! Normalization is dividing by near-zero.")
        print("   Fix: Increase the epsilon in dataset.py normalization.")
    elif short_samples > 0:
        print("🚨 CRITICAL: Some samples are too short for their labels.")
        print("   Fix: Reduce downsample_factor or clean the dataset.")
    else:
        print("✅ Data looks healthy. If NaNs persist, it is Learning Rate or Gradients.")

if __name__ == "__main__":
    inspect()
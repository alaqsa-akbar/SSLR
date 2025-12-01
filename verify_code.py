import torch
import os
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, UMT5EncoderModel, UMT5Config, UMT5ForConditionalGeneration
from modules.pose_encoder import PoseEncoder, PoseEncoderConfig
from modules.dataset import SignLanguageDataset, collate_fn
import torch.nn as nn
import sys

# Add current dir to path to allow imports
sys.path.append(os.getcwd())

def verify():
    print("=== Verifying Setup ===")
    data_dir = "data"
    model_dir = "models/umt5"
    pose_config_path = "configs/pose_encoder_config.json"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Data Loading
    print("[1/5] Verifying Data Loading...")
    try:
        if os.path.exists(os.path.join(model_dir, "tokenizer.json")):
             print("    Loading local tokenizer...")
             tokenizer = AutoTokenizer.from_pretrained(model_dir)
        else:
             print("    Loading hub tokenizer...")
             tokenizer = AutoTokenizer.from_pretrained("google/umt5-base")
        
        train_csv = os.path.join(data_dir, "train.csv")
        pkl_file = os.path.join(data_dir, "data.pkl")
        
        # Load tiny subset
        dataset = SignLanguageDataset(train_csv, pkl_file, tokenizer, max_frames=100)
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        batch = next(iter(loader))
        print("    Data loaded successfully. Batch keys:", batch.keys())
    except Exception as e:
        print(f"    FAILED: {e}")
        return

    # 2. Model Initialization (Pose Encoder)
    print("[2/5] Verifying Pose Encoder Init...")
    try:
        pose_config = PoseEncoderConfig.from_json_file(pose_config_path)
        pose_config.hidden_dim = 768 # Force match
        pose_encoder = PoseEncoder(pose_config)
        pose_encoder.to(device)
        print("    Pose Encoder initialized.")
    except Exception as e:
        print(f"    FAILED: {e}")
        return

    # 3. Model Initialization (UMT5)
    print("[3/5] Verifying UMT5 Init (Robust Loading)...")
    try:
        # Mimic the robust loading logic
        try:
            config = UMT5Config.from_pretrained(model_dir)
            if config.d_model != 768:
                print("    Config mismatch detected (simulated), using manual config")
                raise ValueError("Wrong dim")
        except:
            config = UMT5Config(vocab_size=256384, d_model=768, d_kv=64, d_ff=2048, num_layers=12, num_heads=12, is_encoder_decoder=True)
            
        umt5_encoder = UMT5EncoderModel(config)
        umt5_gen = UMT5ForConditionalGeneration(config)
        umt5_encoder.to(device)
        umt5_gen.to(device)
        print("    UMT5 initialized.")
    except Exception as e:
        print(f"    FAILED: {e}")
        return

    # 4. Pre-training Step (MSE)
    print("[4/5] Verifying Pre-training Step (MSE)...")
    try:
        pose = batch['pose'].to(device)
        pose_mask = batch['pose_attention_mask'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Forward Pose
        pose_emb = pose_encoder(pose, attention_mask=pose_mask)
        # Pooling (simplified for check)
        pose_mask_expanded = pose_mask.unsqueeze(-1).expand(pose_emb.size()).float()
        pose_sum = torch.sum(pose_emb * pose_mask_expanded, dim=1)
        pose_counts = torch.clamp(pose_mask_expanded.sum(1), min=1e-9)
        pose_rep = pose_sum / pose_counts
        
        # Forward Text
        text_out = umt5_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = text_out.last_hidden_state
        text_mask_expanded = attention_mask.unsqueeze(-1).expand(text_emb.size()).float()
        text_sum = torch.sum(text_emb * text_mask_expanded, dim=1)
        text_counts = torch.clamp(text_mask_expanded.sum(1), min=1e-9)
        text_rep = text_sum / text_counts
        
        # Loss
        loss = nn.MSELoss()(pose_rep, text_rep)
        loss.backward()
        print(f"    Pre-training backward pass successful. Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"    FAILED: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Final Training Step
    print("[5/5] Verifying Final Training Step...")
    try:
        # Import the model class to ensure it matches
        from train_final import SignLanguageModel
        model = SignLanguageModel(pose_encoder, umt5_gen)
        model.to(device)
        
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        
        outputs = model(pose, pose_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        print(f"    Final training backward pass successful. Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"    FAILED: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\nALL CHECKS PASSED! The code is ready for training.")

if __name__ == "__main__":
    verify()

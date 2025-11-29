import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, UMT5ForConditionalGeneration, UMT5Config
from modules.pose_encoder import PoseEncoder, PoseEncoderConfig
from modules.dataset import SignLanguageDataset, collate_fn
from tqdm import tqdm
import os
import argparse
from torch.utils.tensorboard import SummaryWriter

class SignLanguageModel(nn.Module):
    def __init__(self, pose_encoder, umt5_model):
        super().__init__()
        self.pose_encoder = pose_encoder
        self.umt5_model = umt5_model
        
    def forward(self, pose, pose_attention_mask, labels=None, decoder_attention_mask=None):
        # Pose Encoder
        pose_embeddings = self.pose_encoder(pose, attention_mask=pose_attention_mask) # (B, T, H)
        
        # UMT5 Decoder
        outputs = self.umt5_model(
            encoder_outputs=(pose_embeddings,),
            attention_mask=pose_attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        )
        
        return outputs

def parse_args():
    parser = argparse.ArgumentParser(description="Train Sign Language Model (Full Fine-tuning)")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--model_dir", type=str, default="models/umt5", help="Path to directory containing tokenizer and UMT5 model")
    parser.add_argument("--contrastive_model_dir", type=str, default="output/pose_encoder_contrastive", help="Path to contrastive checkpoints")
    parser.add_argument("--output_dir", type=str, default="output/sign_language_model", help="Path to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="runs/sign_language_model", help="Path to save tensorboard logs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames for dataset (default: None, calculated from data)")
    parser.add_argument("--pose_config", type=str, default="configs/pose_encoder_config.json", help="Path to Pose Encoder config file")
    
    return parser.parse_args()

def train():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Tensorboard Writer
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Paths
    train_csv = os.path.join(args.data_dir, "train.csv")
    dev_csv = os.path.join(args.data_dir, "dev.csv")
    pkl_file = os.path.join(args.data_dir, "data.pkl")
    
    # Load Tokenizer
    tokenizer_path = os.path.join(args.model_dir, "tokenizer.json")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except:
        print(f"Could not load tokenizer from {tokenizer_path}, trying 'google/umt5-base'")
        tokenizer = AutoTokenizer.from_pretrained("google/umt5-base")
        
    # Load UMT5
    print("Loading UMT5...")
    umt5_weights_path = os.path.join(args.model_dir, "umt5-base.bin")
    
    try:
        if os.path.exists(umt5_weights_path):
            print(f"Loading UMT5 weights from: {umt5_weights_path}")
            config_path = os.path.join(args.model_dir, "config.json")
            if os.path.exists(config_path):
                config = UMT5Config.from_pretrained(args.model_dir)
            else:
                print("Local config not found, using 'google/umt5-base' config")
                config = UMT5Config.from_pretrained("google/umt5-base")
            
            umt5_model = UMT5ForConditionalGeneration(config)
            state_dict = torch.load(umt5_weights_path, map_location="cpu")
            umt5_model.load_state_dict(state_dict, strict=False)
        else:
            print(f"Local model file {umt5_weights_path} not found, loading 'google/umt5-base'")
            umt5_model = UMT5ForConditionalGeneration.from_pretrained("google/umt5-base")
    except Exception as e:
        print(f"Error loading UMT5: {e}")
        config = UMT5Config.from_pretrained("google/umt5-base") if os.path.exists("google/umt5-base") else UMT5Config()
        umt5_model = UMT5ForConditionalGeneration(config)
        
    # Datasets
    print("Loading Datasets...")
    train_dataset = SignLanguageDataset(train_csv, pkl_file, tokenizer, max_frames=args.max_frames)
    dev_dataset = SignLanguageDataset(dev_csv, pkl_file, tokenizer, max_frames=args.max_frames)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Load Pose Encoder
    print(f"Loading Pose Encoder from {args.pose_config}...")
    # Ensure RoPE cache is at least as large as the max frames in dataset
    max_pos = max(512, train_dataset.max_frames) # Default 512 if dataset is small, else max_frames
    print(f"Setting RoPE cache size (max_position_embeddings) to {max_pos}")

    pose_config = PoseEncoderConfig.from_json_file(args.pose_config)
    pose_config.max_position_embeddings = max_pos # Update dynamic cache size
    
    # Override hidden_dim to match UMT5
    if pose_config.hidden_dim != umt5_model.config.d_model:
        print(f"Overriding Pose Encoder hidden_dim ({pose_config.hidden_dim}) to match UMT5 ({umt5_model.config.d_model})")
        pose_config.hidden_dim = umt5_model.config.d_model
        
    pose_encoder = PoseEncoder(pose_config)
    
    # Try to find latest checkpoint
    checkpoint_path = None
    if os.path.exists(args.contrastive_model_dir):
        checkpoints = [d for d in os.listdir(args.contrastive_model_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            # Sort by epoch number
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            checkpoint_path = os.path.join(args.contrastive_model_dir, checkpoints[-1])
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading pretrained Pose Encoder from {checkpoint_path}")
        try:
            pose_encoder = PoseEncoder.from_pretrained(checkpoint_path)
        except Exception as e:
            print(f"Failed to load pretrained pose encoder: {e}")
    else:
        print("Warning: No contrastive checkpoint found. Training Pose Encoder from scratch.")

    # Combined Model
    model = SignLanguageModel(pose_encoder, umt5_model)
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    print("Starting Training...")
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            # Move to device
            pose = batch['pose'].to(device)
            pose_mask = batch['pose_attention_mask'].to(device)
            input_ids = batch['input_ids'].to(device)
            
            # Labels for T5 (replace pad with -100)
            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            
            outputs = model(pose, pose_mask, labels=labels)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            writer.add_scalar("Loss/train", loss.item(), global_step)
            global_step += 1
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            print(f"Saving checkpoint at epoch {epoch+1}")
            save_dir = os.path.join(args.output_dir, f"checkpoint-{epoch+1}")
            os.makedirs(save_dir, exist_ok=True)
            model.pose_encoder.save_pretrained(os.path.join(save_dir, "pose_encoder"))
            model.umt5_model.save_pretrained(os.path.join(save_dir, "umt5"))
        
    writer.close()

if __name__ == "__main__":
    train()

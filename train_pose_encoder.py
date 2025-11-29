import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, UMT5EncoderModel, UMT5Config
from modules.pose_encoder import PoseEncoder, PoseEncoderConfig
from modules.dataset import SignLanguageDataset, collate_fn
from tqdm import tqdm
import os
import argparse
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser(description="Train Pose Encoder with Contrastive Learning")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--model_dir", type=str, default="models/umt5", help="Path to directory containing tokenizer and UMT5 model")
    parser.add_argument("--output_dir", type=str, default="output/pose_encoder_contrastive", help="Path to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="runs/pose_encoder_contrastive", help="Path to save tensorboard logs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
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

    # Load UMT5 Encoder
    print("Loading UMT5 Encoder...")
    umt5_weights_path = os.path.join(args.model_dir, "umt5-base.bin")
    
    try:
        if os.path.exists(umt5_weights_path):
            print(f"Loading UMT5 weights from: {umt5_weights_path}")
            # Load config first
            config_path = os.path.join(args.model_dir, "config.json")
            if os.path.exists(config_path):
                config = UMT5Config.from_pretrained(args.model_dir)
            else:
                print("Local config not found, using 'google/umt5-base' config")
                config = UMT5Config.from_pretrained("google/umt5-base")
            
            text_encoder = UMT5EncoderModel(config)
            state_dict = torch.load(umt5_weights_path, map_location="cpu")
            # Handle potential prefix mismatch if saving/loading differently
            # Usually transformers models have specific keys. 
            # If the bin file is just the model state dict, this should work.
            text_encoder.load_state_dict(state_dict, strict=False) # strict=False to allow missing keys if any (e.g. decoder keys)
        else:
            print(f"Local model file {umt5_weights_path} not found, loading 'google/umt5-base'")
            text_encoder = UMT5EncoderModel.from_pretrained("google/umt5-base")
    except Exception as e:
        print(f"Error loading UMT5: {e}")
        print("Initializing random UMT5 (WARNING: Contrastive learning will fail to align to language if text encoder is random)")
        config = UMT5Config.from_pretrained("google/umt5-base") if os.path.exists("google/umt5-base") else UMT5Config()
        text_encoder = UMT5EncoderModel(config)

    text_encoder.to(device)
    text_encoder.eval() # Freeze
    for param in text_encoder.parameters():
        param.requires_grad = False
        
    # Datasets
    print("Loading Datasets...")
    train_dataset = SignLanguageDataset(train_csv, pkl_file, tokenizer, max_frames=args.max_frames)
    dev_dataset = SignLanguageDataset(dev_csv, pkl_file, tokenizer, max_frames=args.max_frames)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize Pose Encoder
    print(f"Initializing Pose Encoder from {args.pose_config}...")
    pose_config = PoseEncoderConfig.from_json_file(args.pose_config)
    
    # Override hidden_dim to match UMT5
    if pose_config.hidden_dim != text_encoder.config.d_model:
        print(f"Overriding Pose Encoder hidden_dim ({pose_config.hidden_dim}) to match UMT5 ({text_encoder.config.d_model})")
        pose_config.hidden_dim = text_encoder.config.d_model
        
    pose_encoder = PoseEncoder(pose_config)
    pose_encoder.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(pose_encoder.parameters(), lr=args.lr)
    
    # Loss Function (Contrastive)
    cross_entropy_loss = nn.CrossEntropyLoss()
    
    print("Starting Training...")
    global_step = 0
    for epoch in range(args.epochs):
        pose_encoder.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            # Move to device
            pose = batch['pose'].to(device) # (B, T_pose, 84, 2)
            pose_mask = batch['pose_attention_mask'].to(device) # (B, T_pose)
            input_ids = batch['input_ids'].to(device) # (B, T_text)
            attention_mask = batch['attention_mask'].to(device) # (B, T_text)
            
            # Forward Pose Encoder
            pose_embeddings = pose_encoder(pose, attention_mask=pose_mask) # (B, T_pose, H)
            
            # Forward Text Encoder
            with torch.no_grad():
                text_outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                text_embeddings = text_outputs.last_hidden_state # (B, T_text, H)
                
            # Pooling
            # Pose Pooling
            pose_mask_expanded = pose_mask.unsqueeze(-1).expand(pose_embeddings.size()).float()
            pose_sum = torch.sum(pose_embeddings * pose_mask_expanded, dim=1)
            pose_counts = torch.clamp(pose_mask_expanded.sum(1), min=1e-9)
            pose_rep = pose_sum / pose_counts # (B, H)
            
            # Text Pooling
            text_mask_expanded = attention_mask.unsqueeze(-1).expand(text_embeddings.size()).float()
            text_sum = torch.sum(text_embeddings * text_mask_expanded, dim=1)
            text_counts = torch.clamp(text_mask_expanded.sum(1), min=1e-9)
            text_rep = text_sum / text_counts # (B, H)
            
            # Normalize
            pose_rep = nn.functional.normalize(pose_rep, p=2, dim=1)
            text_rep = nn.functional.normalize(text_rep, p=2, dim=1)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, UMT5EncoderModel, UMT5Config
from modules.pose_encoder import PoseEncoder, PoseEncoderConfig
from modules.dataset import SignLanguageDataset, collate_fn
from tqdm import tqdm
import os
import argparse
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser(description="Train Pose Encoder with Contrastive Learning")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--model_dir", type=str, default="models/umt5", help="Path to directory containing tokenizer and UMT5 model")
    parser.add_argument("--output_dir", type=str, default="output/pose_encoder_contrastive", help="Path to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="runs/pose_encoder_contrastive", help="Path to save tensorboard logs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames for dataset (default: None, calculated from data)")
    parser.add_argument("--pose_config", type=str, default="configs/pose_encoder_config.json", help="Path to Pose Encoder config file")
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    
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

    # Load UMT5 Encoder
    print("Loading UMT5 Encoder...")
    umt5_weights_path = os.path.join(args.model_dir, "umt5-base.bin")
    
    try:
        if os.path.exists(umt5_weights_path):
            print(f"Loading UMT5 weights from: {umt5_weights_path}")
            # Load config first
            config_path = os.path.join(args.model_dir, "config.json")
            if os.path.exists(config_path):
                config = UMT5Config.from_pretrained(args.model_dir)
            else:
                print("Local config not found, using 'google/umt5-base' config")
                config = UMT5Config.from_pretrained("google/umt5-base")
            
            text_encoder = UMT5EncoderModel(config)
            state_dict = torch.load(umt5_weights_path, map_location="cpu")
            # Handle potential prefix mismatch if saving/loading differently
            # Usually transformers models have specific keys. 
            # If the bin file is just the model state dict, this should work.
            text_encoder.load_state_dict(state_dict, strict=False) # strict=False to allow missing keys if any (e.g. decoder keys)
        else:
            print(f"Local model file {umt5_weights_path} not found, loading 'google/umt5-base'")
            text_encoder = UMT5EncoderModel.from_pretrained("google/umt5-base")
    except Exception as e:
        print(f"Error loading UMT5: {e}")
        print("Initializing random UMT5 (WARNING: Contrastive learning will fail to align to language if text encoder is random)")
        config = UMT5Config.from_pretrained("google/umt5-base") if os.path.exists("google/umt5-base") else UMT5Config()
        text_encoder = UMT5EncoderModel(config)

    text_encoder.to(device)
    text_encoder.eval() # Freeze
    for param in text_encoder.parameters():
        param.requires_grad = False
        
    # Datasets
    print("Loading Datasets...")
    train_dataset = SignLanguageDataset(train_csv, pkl_file, tokenizer, max_frames=args.max_frames)
    dev_dataset = SignLanguageDataset(dev_csv, pkl_file, tokenizer, max_frames=args.max_frames)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize Pose Encoder
    print(f"Initializing Pose Encoder from {args.pose_config}...")
    pose_config = PoseEncoderConfig.from_json_file(args.pose_config)
    
    # Override hidden_dim to match UMT5
    if pose_config.hidden_dim != text_encoder.config.d_model:
        print(f"Overriding Pose Encoder hidden_dim ({pose_config.hidden_dim}) to match UMT5 ({text_encoder.config.d_model})")
        pose_config.hidden_dim = text_encoder.config.d_model
        
    pose_encoder = PoseEncoder(pose_config)
    pose_encoder.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(pose_encoder.parameters(), lr=args.lr)
    
    # Loss Function (Contrastive)
    cross_entropy_loss = nn.CrossEntropyLoss()
    
    print("Starting Training...")
    global_step = 0
    for epoch in range(args.epochs):
        pose_encoder.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in progress_bar:
            # Move to device
            pose = batch['pose'].to(device) # (B, T_pose, 84, 2)
            pose_mask = batch['pose_attention_mask'].to(device) # (B, T_pose)
            input_ids = batch['input_ids'].to(device) # (B, T_text)
            attention_mask = batch['attention_mask'].to(device) # (B, T_text)
            
            # Forward Pose Encoder
            pose_embeddings = pose_encoder(pose, attention_mask=pose_mask) # (B, T_pose, H)
            
            # Forward Text Encoder
            with torch.no_grad():
                text_outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                text_embeddings = text_outputs.last_hidden_state # (B, T_text, H)
                
            # Pooling
            # Pose Pooling
            pose_mask_expanded = pose_mask.unsqueeze(-1).expand(pose_embeddings.size()).float()
            pose_sum = torch.sum(pose_embeddings * pose_mask_expanded, dim=1)
            pose_counts = torch.clamp(pose_mask_expanded.sum(1), min=1e-9)
            pose_rep = pose_sum / pose_counts # (B, H)
            
            # Text Pooling
            text_mask_expanded = attention_mask.unsqueeze(-1).expand(text_embeddings.size()).float()
            text_sum = torch.sum(text_embeddings * text_mask_expanded, dim=1)
            text_counts = torch.clamp(text_mask_expanded.sum(1), min=1e-9)
            text_rep = text_sum / text_counts # (B, H)
            
            # Normalize
            pose_rep = nn.functional.normalize(pose_rep, p=2, dim=1)
            text_rep = nn.functional.normalize(text_rep, p=2, dim=1)
            
            # Contrastive Loss (InfoNCE / CLIP)
            temperature = 0.07
            logits = torch.matmul(pose_rep, text_rep.T) / temperature # (B, B)
            
            labels = torch.arange(logits.size(0), device=device)
            loss_i = cross_entropy_loss(logits, labels)
            loss_t = cross_entropy_loss(logits.T, labels)
            loss = (loss_i + loss_t) / 2
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            writer.add_scalar("Loss/train", loss.item(), global_step)
            global_step += 1

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            pose_encoder.save_pretrained(os.path.join(args.output_dir, f"checkpoint-{epoch+1}"))
        
    writer.close()

if __name__ == "__main__":
    train()

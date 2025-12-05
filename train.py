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
import wandb

class SignLanguageModel(nn.Module):
    def __init__(self, pose_encoder, umt5_model):
        super().__init__()
        self.pose_encoder = pose_encoder
        self.umt5_model = umt5_model
        
    def forward(self, pose, pose_attention_mask, labels=None, decoder_attention_mask=None):
        # Pose Encoder
        pose_embeddings = self.pose_encoder(pose, attention_mask=pose_attention_mask) # (B, T, H)
        
        # UMT5 Decoder
        from transformers.modeling_outputs import BaseModelOutput
        encoder_outputs = BaseModelOutput(last_hidden_state=pose_embeddings)
        
        outputs = self.umt5_model(
            encoder_outputs=encoder_outputs,
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
    parser.add_argument("--stage1_epochs", type=int, default=50, help="Number of epochs for Stage 1 (Pose Encoder only)")
    parser.add_argument("--stage2_epochs", type=int, default=50, help="Number of epochs for Stage 2 (Full Fine-tuning)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients before update")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames for dataset (default: None, calculated from data)")
    parser.add_argument("--pose_config", type=str, default="configs/pose_encoder_config.json", help="Path to Pose Encoder config file")
    
    return parser.parse_args()

def train():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Tensorboard Writer
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Initialize WandB
    wandb.init(project="sign-language-translation", name="final-model-training", config=args)
    
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
            
            if config.d_model != 768:
                print(f"Warning: Loaded config has d_model={config.d_model}, forcing 768 and vocab 256384")
                config.d_model = 768
                config.vocab_size = 256384
            
            umt5_model = UMT5ForConditionalGeneration(config)
            state_dict = torch.load(umt5_weights_path, map_location="cpu")
            umt5_model.load_state_dict(state_dict, strict=False)
        else:
            print(f"Local model file {umt5_weights_path} not found, loading 'google/umt5-base'")
            umt5_model = UMT5ForConditionalGeneration.from_pretrained("google/umt5-base")
    except Exception as e:
        print(f"Error loading UMT5: {e}")
        print("Attempting to create manual UMT5-Base config...")
        config = UMT5Config(
            vocab_size=256384,
            d_model=768,
            d_kv=64,
            d_ff=2048,
            num_layers=12,
            num_heads=12,
            relative_attention_num_buckets=32,
            dropout_rate=0.1,
            layer_norm_epsilon=1e-6,
            initializer_factor=1.0,
            feed_forward_proj="gated-gelu",
            is_encoder_decoder=True
        )
        umt5_model = UMT5ForConditionalGeneration(config)
        
    # Enable Gradient Checkpointing
    print("Enabling Gradient Checkpointing for UMT5...")
    umt5_model.gradient_checkpointing_enable()
        
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
    
    # Initialize Pose Encoder from scratch (Random Weights)
    print("Initializing Pose Encoder from scratch...")
    # (Already initialized above)

    # Combined Model
    model = SignLanguageModel(pose_encoder, umt5_model)
    model.to(device)
    
    # --- STAGE 1: Train Pose Encoder Only ---
    print("\n" + "="*50)
    print(f"STAGE 1: Training Pose Encoder Only for {args.stage1_epochs} epochs")
    print("="*50 + "\n")
    
    # Freeze UMT5
    print("Freezing UMT5 parameters...")
    for param in model.umt5_model.parameters():
        param.requires_grad = False
        
    # Optimizer (Pose Encoder Only)
    optimizer = torch.optim.AdamW(model.pose_encoder.parameters(), lr=args.lr)
    
    # Scheduler
    from transformers import get_linear_schedule_with_warmup
    num_training_steps = len(train_loader) * args.stage1_epochs // args.gradient_accumulation_steps
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    
    best_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(args.stage1_epochs):
        # Train
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Stage 1 Epoch {epoch+1}/{args.stage1_epochs} [Train]"):
            pose = batch['pose'].to(device)
            pose_mask = batch['pose_attention_mask'].to(device)
            input_ids = batch['input_ids'].to(device)
            
            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            
            optimizer.zero_grad()
            outputs = model(pose, pose_mask, labels=labels)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            
            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.pose_encoder.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * args.gradient_accumulation_steps
            writer.add_scalar("Loss/train_stage1", loss.item() * args.gradient_accumulation_steps, global_step)
            wandb.log({"train_loss_stage1": loss.item() * args.gradient_accumulation_steps, "epoch": epoch, "global_step": global_step})
            global_step += 1
            
        avg_train_loss = total_loss / len(train_loader)
        print(f"Stage 1 Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")
        
        # Val
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc=f"Stage 1 Epoch {epoch+1}/{args.stage1_epochs} [Val]"):
                pose = batch['pose'].to(device)
                pose_mask = batch['pose_attention_mask'].to(device)
                input_ids = batch['input_ids'].to(device)
                labels = input_ids.clone()
                labels[labels == tokenizer.pad_token_id] = -100
                outputs = model(pose, pose_mask, labels=labels)
                total_val_loss += outputs.loss.item()
        
        avg_val_loss = total_val_loss / len(dev_loader)
        print(f"Stage 1 Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")
        writer.add_scalar("Loss/val_stage1", avg_val_loss, epoch)
        wandb.log({"val_loss_stage1": avg_val_loss, "epoch": epoch})
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"New best Stage 1 loss! Saving to {args.output_dir}/stage1")
            os.makedirs(os.path.join(args.output_dir, "stage1"), exist_ok=True)
            model.pose_encoder.save_pretrained(os.path.join(args.output_dir, "stage1", "pose_encoder"))
            # Don't need to save UMT5 as it's frozen
            
    # --- STAGE 2: Full Fine-tuning ---
    print("\n" + "="*50)
    print(f"STAGE 2: Full Fine-tuning for {args.stage2_epochs} epochs")
    print("="*50 + "\n")
    
    # Unfreeze UMT5
    print("Unfreezing UMT5 parameters...")
    for param in model.umt5_model.parameters():
        param.requires_grad = True
        
    # Optimizer (All Parameters)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Scheduler
    num_training_steps = len(train_loader) * args.stage2_epochs // args.gradient_accumulation_steps
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    
    best_val_loss = float('inf') # Reset for Stage 2
    
    for epoch in range(args.stage2_epochs):
        # Train
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Stage 2 Epoch {epoch+1}/{args.stage2_epochs} [Train]"):
            pose = batch['pose'].to(device)
            pose_mask = batch['pose_attention_mask'].to(device)
            input_ids = batch['input_ids'].to(device)
            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            
            optimizer.zero_grad()
            outputs = model(pose, pose_mask, labels=labels)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            
            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * args.gradient_accumulation_steps
            writer.add_scalar("Loss/train_stage2", loss.item() * args.gradient_accumulation_steps, global_step)
            wandb.log({"train_loss_stage2": loss.item() * args.gradient_accumulation_steps, "epoch": epoch, "global_step": global_step})
            global_step += 1
            
        avg_train_loss = total_loss / len(train_loader)
        print(f"Stage 2 Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")
        
        # Val
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc=f"Stage 2 Epoch {epoch+1}/{args.stage2_epochs} [Val]"):
                pose = batch['pose'].to(device)
                pose_mask = batch['pose_attention_mask'].to(device)
                input_ids = batch['input_ids'].to(device)
                labels = input_ids.clone()
                labels[labels == tokenizer.pad_token_id] = -100
                outputs = model(pose, pose_mask, labels=labels)
                total_val_loss += outputs.loss.item()
        
        avg_val_loss = total_val_loss / len(dev_loader)
        print(f"Stage 2 Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")
        writer.add_scalar("Loss/val_stage2", avg_val_loss, epoch)
        wandb.log({"val_loss_stage2": avg_val_loss, "epoch": epoch})
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"New best Stage 2 loss! Saving to {args.output_dir}/final")
            os.makedirs(os.path.join(args.output_dir, "final"), exist_ok=True)
            model.pose_encoder.save_pretrained(os.path.join(args.output_dir, "final", "pose_encoder"))
            model.umt5_model.save_pretrained(os.path.join(args.output_dir, "final", "umt5"))
            
    writer.close()

if __name__ == "__main__":
    train()

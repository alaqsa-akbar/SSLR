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
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description="Train Pose Encoder with Contrastive Learning")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--model_dir", type=str, default="models/umt5", help="Path to directory containing tokenizer and UMT5 model")
    parser.add_argument("--output_dir", type=str, default="output/pose_encoder_contrastive", help="Path to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="runs/pose_encoder_contrastive", help="Path to save tensorboard logs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients before update")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames for dataset (default: None, calculated from data)")
    parser.add_argument("--pose_config", type=str, default="configs/pose_encoder_config.json", help="Path to Pose Encoder config file")
    
    return parser.parse_args()

def train():
    args = parse_args()
    
    # Initialize Accelerator
    from accelerate import Accelerator
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, log_with="wandb")
    device = accelerator.device
    
    # Initialize WandB (only on main process)
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="sign-language-translation", 
            config=vars(args),
            init_kwargs={"wandb": {"name": "pose-encoder-pretrain"}}
        )
    
    # Paths
    train_csv = os.path.join(args.data_dir, "train.csv")
    dev_csv = os.path.join(args.data_dir, "dev.csv")
    pkl_file = os.path.join(args.data_dir, "data.pkl")
    
    # Load Tokenizer
    tokenizer_path = os.path.join(args.model_dir, "tokenizer.json")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except:
        if accelerator.is_main_process:
            print(f"Could not load tokenizer from {tokenizer_path}, trying 'google/umt5-base'")
        tokenizer = AutoTokenizer.from_pretrained("google/umt5-base")

    # Load UMT5 Encoder
    if accelerator.is_main_process:
        print("Loading UMT5 Encoder...")
    umt5_weights_path = os.path.join(args.model_dir, "umt5-base.bin")
    
    try:
        if os.path.exists(umt5_weights_path):
            if accelerator.is_main_process:
                print(f"Loading UMT5 weights from: {umt5_weights_path}")
            # Load config first
            config_path = os.path.join(args.model_dir, "config.json")
            if os.path.exists(config_path):
                config = UMT5Config.from_pretrained(args.model_dir)
            else:
                if accelerator.is_main_process:
                    print("Local config not found, using 'google/umt5-base' config")
                config = UMT5Config.from_pretrained("google/umt5-base")
            
            if config.d_model != 768:
                if accelerator.is_main_process:
                    print(f"Warning: Loaded config has d_model={config.d_model}, forcing 768 and vocab 256384")
                config.d_model = 768
                config.vocab_size = 256384
            
            text_encoder = UMT5EncoderModel(config)
            state_dict = torch.load(umt5_weights_path, map_location="cpu")
            text_encoder.load_state_dict(state_dict, strict=False)
        else:
            if accelerator.is_main_process:
                print(f"Local model file {umt5_weights_path} not found, loading 'google/umt5-base'")
            text_encoder = UMT5EncoderModel.from_pretrained("google/umt5-base")
    except Exception as e:
        if accelerator.is_main_process:
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
        text_encoder = UMT5EncoderModel(config)

    text_encoder.to(device)
    text_encoder.eval() # Freeze
    for param in text_encoder.parameters():
        param.requires_grad = False
        
    # Datasets
    if accelerator.is_main_process:
        print("Loading Datasets...")
    train_dataset = SignLanguageDataset(train_csv, pkl_file, tokenizer, max_frames=args.max_frames)
    dev_dataset = SignLanguageDataset(dev_csv, pkl_file, tokenizer, max_frames=args.max_frames)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize Pose Encoder
    if accelerator.is_main_process:
        print(f"Initializing Pose Encoder from {args.pose_config}...")
    pose_config = PoseEncoderConfig.from_json_file(args.pose_config)
    
    # Override hidden_dim to match UMT5
    if pose_config.hidden_dim != text_encoder.config.d_model:
        if accelerator.is_main_process:
            print(f"Overriding Pose Encoder hidden_dim ({pose_config.hidden_dim}) to match UMT5 ({text_encoder.config.d_model})")
        pose_config.hidden_dim = text_encoder.config.d_model
        
    pose_encoder = PoseEncoder(pose_config)
    # pose_encoder.to(device) # Handled by prepare
    
    # Optimizer
    optimizer = torch.optim.AdamW(pose_encoder.parameters(), lr=args.lr)
    
    # Scheduler
    from transformers import get_linear_schedule_with_warmup
    num_training_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    num_warmup_steps = int(0.1 * num_training_steps) # 10% warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    
    # Prepare with Accelerator
    pose_encoder, optimizer, train_loader, dev_loader, scheduler = accelerator.prepare(
        pose_encoder, optimizer, train_loader, dev_loader, scheduler
    )
    
    # Loss Function (Contrastive)
    cross_entropy_loss = nn.CrossEntropyLoss()
    
    if accelerator.is_main_process:
        print("Starting Training...")
    global_step = 0
    # Define temperature for contrastive loss (example value, adjust as needed)
    temperature = 0.07 
    
    for epoch in range(args.epochs):
        pose_encoder.train()
        total_loss = 0
        
        # Only show progress bar on main process
        if accelerator.is_main_process:
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        else:
            progress_bar = train_loader
            
        for batch in progress_bar:
            with accelerator.accumulate(pose_encoder):
                # Move to device (Handled by accelerate if using prepared loader, but explicit doesn't hurt)
                # pose = batch['pose'].to(device) 
                # pose_mask = batch['pose_attention_mask'].to(device)
                # input_ids = batch['input_ids'].to(device)
                # attention_mask = batch['attention_mask'].to(device)
                
                pose = batch['pose']
                pose_mask = batch['pose_attention_mask']
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                
                optimizer.zero_grad()
                
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
                
                # Contrastive Loss (InfoNCE)
                # Normalize embeddings
                pose_rep = torch.nn.functional.normalize(pose_rep, p=2, dim=1)
                text_rep = torch.nn.functional.normalize(text_rep, p=2, dim=1)
                
                # Cosine similarity matrix
                logits = torch.matmul(pose_rep, text_rep.T) / temperature # (B, B)
                
                # Labels: diagonal elements are positives
                labels = torch.arange(logits.size(0)).to(device)
                
                # Symmetric loss (Pose->Text and Text->Pose)
                loss_i = nn.CrossEntropyLoss()(logits, labels)
                loss_t = nn.CrossEntropyLoss()(logits.T, labels)
                loss = (loss_i + loss_t) / 2
                
                # Backward
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(pose_encoder.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Logging
                total_loss += loss.item()
                if accelerator.is_main_process:
                    accelerator.log({"train_loss": loss.item(), "epoch": epoch, "global_step": global_step}, step=global_step)
                    writer.add_scalar("Loss/train", loss.item(), global_step)
                global_step += 1
            
        avg_loss = total_loss / len(train_loader)
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
            
    # Save at the end
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f"Saving final model to {args.output_dir}")
        unwrapped_model = accelerator.unwrap_model(pose_encoder)
        unwrapped_model.save_pretrained(args.output_dir)
        writer.close()
        accelerator.end_training()

if __name__ == "__main__":
    train()

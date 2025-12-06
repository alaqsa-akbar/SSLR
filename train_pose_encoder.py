"""
Contrastive Pre-training for Sign Language Pose Encoder (Unique Caching)

Key improvements:
1. Caches RAW UMT5 embeddings for UNIQUE GLOSSES only (~687 items).
2. Solves "Duplicate Conflict": Identical glosses now map to the same target index.
3. Optimizes both Pose Encoder AND Text Projection Head.
4. Uses Global Contrastive Loss (Batch vs. Unique Vocab).

Usage:
    accelerate launch --num_processes 2 train_pose_encoder.py \
        --epochs 75 \
        --batch_size 64 \
        --lr 5e-5 \
        --projection_dim 256 \
        --output_dir output/pose_encoder_cached
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, UMT5EncoderModel, UMT5Config
from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import set_seed

from modules.pose_encoder import (
    PoseEncoderForContrastive,
    PoseEncoderConfig,
    TextProjectionHead
)
from modules.dataset import SignLanguageDataset, collate_fn_contrastive

from tqdm import tqdm
import os
import argparse
import json
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Train Pose Encoder with Cached Contrastive Learning")

    # Data
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model_dir", type=str, default="models/umt5")
    parser.add_argument("--vocab_file", type=str, default="vocab.json", help="Path to vocab file for unique caching")

    # Output
    parser.add_argument("--output_dir", type=str, default="output/pose_encoder_cached")
    parser.add_argument("--log_dir", type=str, default="runs/pose_encoder_cached")
    parser.add_argument("--wandb_project", type=str, default="slr")

    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    # Model
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--pose_config", type=str, default="configs/pose_encoder_config.json")
    parser.add_argument("--projection_dim", type=int, default=256)

    # Contrastive
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--learnable_temperature", action="store_true")

    # Checkpointing
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--eval_every", type=int, default=1)

    return parser.parse_args()


def load_text_encoder(model_dir: str, device: torch.device, accelerator: Accelerator):
    """Load and freeze the UMT5 text encoder."""
    umt5_weights_path = os.path.join(model_dir, "umt5-base.bin")

    try:
        if os.path.exists(umt5_weights_path):
            if accelerator.is_main_process:
                print(f"Loading UMT5 weights from: {umt5_weights_path}")
            config = UMT5Config.from_pretrained(model_dir)
            if config.d_model != 768:
                config.d_model = 768
                config.vocab_size = 256384
            text_encoder = UMT5EncoderModel(config)
            state_dict = torch.load(umt5_weights_path, map_location="cpu")
            text_encoder.load_state_dict(state_dict, strict=False)
        else:
            if accelerator.is_main_process:
                print("Loading UMT5 from HuggingFace...")
            text_encoder = UMT5EncoderModel.from_pretrained("google/umt5-base")

    except Exception as e:
        if accelerator.is_main_process:
            print(f"Error loading UMT5: {e}, creating from config...")
        config = UMT5Config(
            vocab_size=256384, d_model=768, d_kv=64, d_ff=2048,
            num_layers=12, num_heads=12, relative_attention_num_buckets=32,
            dropout_rate=0.1, layer_norm_epsilon=1e-6, feed_forward_proj="gated-gelu"
        )
        text_encoder = UMT5EncoderModel(config)

    text_encoder.eval()
    for param in text_encoder.parameters():
        param.requires_grad = False

    return text_encoder


def pool_text_embeddings(embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pool text embeddings with attention mask."""
    mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def cache_unique_raw_embeddings(
    text_encoder: nn.Module,
    tokenizer,
    vocab_file: str,
    device: torch.device,
    accelerator: Accelerator
) -> tuple:
    """
    Caches RAW (768-dim) embeddings for UNIQUE glosses from the vocab file.
    
    Returns:
        cached_embeddings: (Vocab_Size, 768) tensor
        gloss_to_idx: Dictionary mapping "GLOSS" -> Index
    """
    if accelerator.is_main_process:
        print("Caching UNIQUE RAW text embeddings from vocab...")
    
    # Load Unique Glosses
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    # Sort keys to ensure deterministic order across runs
    # We use the keys (gloss strings) from the vocab dict
    unique_glosses = sorted(list(vocab_data.keys()))
    
    text_encoder.eval()
    embeddings = []
    batch_size = 32
    
    # Process in batches
    # Only main process needs to show progress bar, but all must compute if not distributed caching
    # For simplicity, we run on all processes (redundant but safe) or just main and broadcast.
    # Given the small size (<1000), running on all is instant.
    
    iterator = range(0, len(unique_glosses), batch_size)
    if accelerator.is_main_process:
        iterator = tqdm(iterator, desc="Caching Unique")
        
    for i in iterator:
        batch_text = unique_glosses[i : i + batch_size]
        
        encoded = tokenizer(
            batch_text, 
            padding='max_length', 
            truncation=True, 
            max_length=128, 
            return_tensors='pt'
        )
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)
        
        with torch.no_grad():
            text_outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_pooled = pool_text_embeddings(text_outputs.last_hidden_state, attention_mask)
            embeddings.append(text_pooled.cpu())
            
    cached_embeddings = torch.cat(embeddings, dim=0)
    gloss_to_idx = {gloss: i for i, gloss in enumerate(unique_glosses)}
    
    if accelerator.is_main_process:
        print(f"Cached {len(cached_embeddings)} unique embeddings, shape: {cached_embeddings.shape}")
    
    return cached_embeddings, gloss_to_idx


def train():
    args = parse_args()

    # Initialize Accelerator
    accelerator = Accelerator(
        log_with=["wandb"],
        project_dir=args.log_dir
    )

    set_seed(args.seed)
    device = accelerator.device

    # Initialize logging
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)

        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=vars(args),
            init_kwargs={"wandb": {"name": f"unique-contrastive-{args.batch_size}bs"}}
        )

        with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    # Load Tokenizer
    if accelerator.is_main_process:
        print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    except:
        tokenizer = AutoTokenizer.from_pretrained("google/umt5-base")

    # Load Text Encoder (Teacher)
    if accelerator.is_main_process:
        print("Loading UMT5 text encoder...")
    text_encoder = load_text_encoder(args.model_dir, device, accelerator)
    text_encoder.to(device)

    # ============================================
    # Cache Step (Unique)
    # ============================================
    cached_unique_raw, gloss_map = cache_unique_raw_embeddings(
        text_encoder, tokenizer, args.vocab_file, device, accelerator
    )
    
    # Move to GPU
    cached_unique_raw = cached_unique_raw.to(device)
    
    # Free memory
    del text_encoder
    torch.cuda.empty_cache()

    # ============================================
    # Datasets & Target Mapping
    # ============================================
    if accelerator.is_main_process:
        print("Loading datasets...")
        
    # We use 'contrastive' mode to get standard tensor output, but we need raw glosses for mapping.
    # Dataset stores 'gloss' in self.data dataframe.
    train_dataset = SignLanguageDataset(os.path.join(args.data_dir, "train.csv"), 
                                      os.path.join(args.data_dir, "data.pkl"), 
                                      tokenizer, max_frames=args.max_frames, mode='contrastive')
                                      
    dev_dataset = SignLanguageDataset(os.path.join(args.data_dir, "dev.csv"), 
                                    os.path.join(args.data_dir, "data.pkl"), 
                                    tokenizer, max_frames=args.max_frames, mode='contrastive')

    # Pre-compute ID -> Target Index map
    # This avoids slow string lookups inside the training loop
    if accelerator.is_main_process: print("Mapping dataset IDs to Unique Indices...")
    
    def build_id_map(ds):
        mapping = {}
        for idx in range(len(ds)):
            row = ds.data.iloc[idx]
            sid = row['id']
            # Important: Strip whitespace to match vocab keys
            gloss = str(row['gloss']).strip() 
            # Default to 0 if not found (should not happen with valid vocab)
            mapping[sid] = gloss_map.get(gloss, 0)
        return mapping

    # Build maps (every process needs this for its batch)
    train_id_map = build_id_map(train_dataset)
    dev_id_map = build_id_map(dev_dataset)

    # ============================================
    # Model Initialization
    # ============================================
    if accelerator.is_main_process:
        print(f"Initializing Pose Encoder from {args.pose_config}...")
        
    pose_config = PoseEncoderConfig.from_json_file(args.pose_config)
    pose_config.channels = 4
    pose_config.hidden_dim = 768
    pose_config.projection_dim = args.projection_dim
    
    pose_model = PoseEncoderForContrastive(pose_config)

    # Text Projection Head (Trainable)
    text_projection = TextProjectionHead(
        input_dim=768,
        hidden_dim=768,
        output_dim=args.projection_dim
    )
    text_projection.to(device)

    # Loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        collate_fn=collate_fn_contrastive, num_workers=4, pin_memory=True, drop_last=True
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False, 
        collate_fn=collate_fn_contrastive, num_workers=4, pin_memory=True, drop_last=True
    )

    # Temperature
    if args.learnable_temperature:
        log_temperature = nn.Parameter(torch.log(torch.tensor(args.temperature)))
    else:
        log_temperature = torch.log(torch.tensor(args.temperature)).to(device)

    # Optimizer
    trainable_params = list(pose_model.parameters()) + list(text_projection.parameters())
    if args.learnable_temperature:
        trainable_params.append(log_temperature)

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    # Prepare
    pose_model, text_projection, optimizer, train_loader, dev_loader, scheduler = accelerator.prepare(
        pose_model, text_projection, optimizer, train_loader, dev_loader, scheduler
    )

    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"Training Configuration:")
        print(f"  - Batch size per GPU: {args.batch_size}")
        print(f"  - Unique Classes (Vocab): {len(gloss_map)}")
        print(f"  - Projection dim: {args.projection_dim}")
        print(f"  - Temperature: {args.temperature}")
        print(f"  - Training steps: {num_training_steps}")
        print(f"{'='*60}\n")

    # ============================================
    # Training Loop
    # ============================================
    global_step = 0
    best_val_acc = 0.0 # Track Accuracy for Classification

    for epoch in range(args.epochs):
        pose_model.train()
        text_projection.train()

        total_loss = 0
        total_acc = 0
        num_batches = 0

        if accelerator.is_main_process:
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        else:
            progress_bar = train_loader

        for batch in progress_bar:
            optimizer.zero_grad()

            # 1. Visual Side
            pose_embeds = pose_model(batch['pose'], attention_mask=batch['pose_attention_mask'], return_projection=True)
            pose_embeds = F.normalize(pose_embeds, p=2, dim=1)

            # 2. Text Side (Project Unique Cache)
            # Input: (Vocab_Size, 768) -> Output: (Vocab_Size, 256)
            all_text_embeds = text_projection(cached_unique_raw)
            all_text_embeds = F.normalize(all_text_embeds, p=2, dim=1)

            # 3. Targets (Lookup via ID)
            # We map the Sample ID to the Unique Gloss Index
            target_indices = torch.tensor([train_id_map[sid] for sid in batch['id']], device=device)

            # 4. Global Contrastive Loss
            temperature = torch.exp(log_temperature)
            logits = torch.matmul(pose_embeds, all_text_embeds.T) / temperature
            loss = F.cross_entropy(logits, target_indices)

            # Backward
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(pose_model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            # Metrics
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                acc = (preds == target_indices).float().mean().item()

            total_loss += loss.item()
            total_acc += acc
            num_batches += 1

            if accelerator.is_main_process:
                current_temp = temperature.item() if isinstance(temperature, torch.Tensor) else temperature
                accelerator.log({
                    "train/loss": loss.item(),
                    "train/acc": acc,
                    "train/temperature": current_temp,
                    "train/lr": scheduler.get_last_lr()[0],
                }, step=global_step)
                
                if global_step % 10 == 0:
                    progress_bar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{acc:.3f}"})

            global_step += 1

        avg_train_loss = total_loss / num_batches
        avg_train_acc = total_acc / num_batches

        if accelerator.is_main_process:
            print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.3f}")

        # ============================================
        # Validation Loop
        # ============================================
        if (epoch + 1) % args.eval_every == 0:
            pose_model.eval()
            text_projection.eval()

            total_val_loss = 0
            total_val_acc = 0
            num_val_batches = 0

            with torch.no_grad():
                # Project Unique Cache once
                all_dev_embeds = text_projection(cached_unique_raw)
                all_dev_embeds = F.normalize(all_dev_embeds, p=2, dim=1)
                temperature = torch.exp(log_temperature)

                if accelerator.is_main_process:
                    val_pbar = tqdm(dev_loader, desc=f"Epoch {epoch+1} [Dev]")
                else:
                    val_pbar = dev_loader

                for batch in val_pbar:
                    pose_embeds = pose_model(batch['pose'], attention_mask=batch['pose_attention_mask'], return_projection=True)
                    pose_embeds = F.normalize(pose_embeds, p=2, dim=1)

                    target_indices = torch.tensor([dev_id_map[sid] for sid in batch['id']], device=device)

                    logits = torch.matmul(pose_embeds, all_dev_embeds.T) / temperature
                    loss = F.cross_entropy(logits, target_indices)

                    preds = logits.argmax(dim=1)
                    acc = (preds == target_indices).float().mean().item()

                    total_val_loss += loss.item()
                    total_val_acc += acc
                    num_val_batches += 1

            avg_val_loss = total_val_loss / num_val_batches
            avg_val_acc = total_val_acc / num_val_batches

            if accelerator.is_main_process:
                print(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f}, Acc: {avg_val_acc:.3f}")

                accelerator.log({
                    "val/loss": avg_val_loss,
                    "val/acc": avg_val_acc,
                    "epoch": epoch + 1
                }, step=global_step)

                # Save best model based on Accuracy (since loss can be tricky with temp)
                if avg_val_acc > best_val_acc:
                    best_val_acc = avg_val_acc
                    save_path = os.path.join(args.output_dir, "best_model")
                    os.makedirs(save_path, exist_ok=True)

                    unwrapped_model = accelerator.unwrap_model(pose_model)
                    unwrapped_model.encoder.save_pretrained(os.path.join(save_path, "pose_encoder"))
                    unwrapped_model.save_pretrained(os.path.join(save_path, "pose_encoder_full"))
                    
                    print(f"  -> Saved best model (Acc: {avg_val_acc:.4f})")

        # ============================================
        # Checkpointing
        # ============================================
        if (epoch + 1) % args.save_every == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                checkpoint_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
                os.makedirs(checkpoint_path, exist_ok=True)
                
                unwrapped_model = accelerator.unwrap_model(pose_model)
                unwrapped_model.encoder.save_pretrained(os.path.join(checkpoint_path, "pose_encoder"))
                print(f"  -> Saved checkpoint at epoch {epoch+1}")

    accelerator.end_training()


if __name__ == "__main__":
    train()
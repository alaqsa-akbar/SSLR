"""
Training Script for Hybrid CTC/Attention Sign Language Recognition Model

Uses improved architecture:
- Grouped GCN (separate for hands, face, body)
- Dual-stream (pose + velocity)
- Transformer temporal encoder
- CTC + Attention decoder

Usage:
    python train_final.py \
        --data_dir data \
        --output_dir output/sign_language_model \
        --epochs 100 \
        --batch_size 16 \
        --lr 1e-4 \
        --lambda_ctc 0.3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import get_linear_schedule_with_warmup
import argparse
import os
import json
from tqdm import tqdm

# Import modules
from modules.dataset import SignLanguageDataset, GlossTokenizer, collate_fn_hybrid
from modules.pose_encoder import PoseEncoder, TinyAdvancedDecoder, PoseEncoderConfig


class HybridModel(nn.Module):
    """
    Hybrid SLR Model with CTC and Attention branches.
    """
    def __init__(self, pose_encoder: PoseEncoder, vocab_size: int, hidden_dim: int = 768, num_decoder_layers: int = 2):
        super().__init__()
        self.encoder = pose_encoder
        self.vocab_size = vocab_size
        
        # CTC Head: Projects to Vocab + Blank
        self.ctc_head = nn.Linear(hidden_dim, vocab_size + 1)
        
        # Decoder
        dec_config = pose_encoder.config
        dec_config.num_decoder_layers = num_decoder_layers
        self.decoder = TinyAdvancedDecoder(vocab_size, dec_config)
        
    def forward(self, pose, input_lengths, decoder_input_ids=None):
        """
        Args:
            pose: (B, T, 86, 4) with [x, y, vx, vy]
            input_lengths: (B,) valid frame counts
            decoder_input_ids: (B, S) shifted targets for teacher forcing
        Returns:
            ctc_logits: (B, T, vocab+1)
            dec_logits: (B, S, vocab) or None
        """
        max_len = pose.shape[1]
        device = pose.device
        
        # Create encoder mask
        enc_mask = torch.arange(max_len, device=device)[None, :] < input_lengths[:, None]
        
        # Encode
        enc_out = self.encoder(pose, attention_mask=enc_mask.float())
        
        # CTC branch
        ctc_logits = self.ctc_head(enc_out)
        
        # Decoder branch
        dec_logits = None
        if decoder_input_ids is not None:
            dec_logits = self.decoder(decoder_input_ids, enc_out, enc_mask=enc_mask.float())
            
        return ctc_logits, dec_logits


def parse_args():
    parser = argparse.ArgumentParser(description="Train Hybrid SLR Model")
    
    # Data & Paths
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--vocab_file", type=str, default="vocab.json")
    parser.add_argument("--pose_config", type=str, default="configs/pose_encoder_config.json")
    
    # Pre-trained encoder (optional)
    parser.add_argument("--pretrained_encoder", type=str, default=None,
                        help="Path to pre-trained pose encoder")
    parser.add_argument("--fresh_start", action="store_true",
                        help="Initialize random encoder (ignore pretrained)")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="output/sign_language_model_final")
    parser.add_argument("--log_dir", type=str, default="runs/sign_language_model_final")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="Weights & Biases project (None to disable)")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    
    # Loss weights
    parser.add_argument("--lambda_ctc", type=float, default=0.3,
                        help="Weight for CTC loss (decoder weight = 1 - lambda)")
    
    # Checkpointing
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=15)

    return parser.parse_args()


def train():
    args = parse_args()
    
    # Initialize Accelerator
    log_with = ["wandb"] if args.wandb_project else None
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=log_with,
        project_dir=args.log_dir
    )
    
    set_seed(args.seed)
    device = accelerator.device
    
    # Setup logging
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        
        if args.wandb_project:
            accelerator.init_trackers(
                project_name=args.wandb_project,
                config=vars(args),
                init_kwargs={"wandb": {"name": f"hybrid-{args.batch_size}bs-{args.lambda_ctc}ctc"}}
            )
        
        with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
            
        print("=" * 60)
        print("TRAINING CONFIGURATION")
        print("=" * 60)
        for k, v in vars(args).items():
            print(f"  {k}: {v}")
        print("=" * 60)

    # =========================================================================
    # 1. Load Tokenizer
    # =========================================================================
    if not os.path.exists(args.vocab_file):
        raise FileNotFoundError(f"Vocab file not found: {args.vocab_file}")
    tokenizer = GlossTokenizer(args.vocab_file)
    if accelerator.is_main_process:
        print(f"Vocabulary size: {tokenizer.vocab_size}")

    # =========================================================================
    # 2. Load Datasets
    # =========================================================================
    if accelerator.is_main_process:
        print("Loading datasets...")
    
    train_dataset = SignLanguageDataset(
        f"{args.data_dir}/train.csv",
        f"{args.data_dir}/data.pkl",
        tokenizer,
        mode='hybrid',
        use_augmentation=True
    )
    
    dev_dataset = SignLanguageDataset(
        f"{args.data_dir}/dev.csv",
        f"{args.data_dir}/data.pkl",
        tokenizer,
        mode='hybrid',
        use_augmentation=False
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn_hybrid, num_workers=4, pin_memory=True
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn_hybrid, num_workers=4
    )
    
    if accelerator.is_main_process:
        print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
        print(f"Dev: {len(dev_dataset)} samples")

    # =========================================================================
    # 3. Initialize Model
    # =========================================================================
    if accelerator.is_main_process:
        print("Initializing model...")
    
    config = PoseEncoderConfig.from_json_file(args.pose_config)
    
    # Load pretrained or initialize fresh
    if args.fresh_start or args.pretrained_encoder is None:
        if accelerator.is_main_process:
            print("Initializing random encoder")
        pose_encoder = PoseEncoder(config)
    else:
        if accelerator.is_main_process:
            print(f"Loading pretrained encoder from {args.pretrained_encoder}")
        try:
            pose_encoder = PoseEncoder.from_pretrained(args.pretrained_encoder)
        except Exception as e:
            if accelerator.is_main_process:
                print(f"Warning: Could not load pretrained ({e}), initializing random")
            pose_encoder = PoseEncoder(config)
    
    model = HybridModel(
        pose_encoder, tokenizer.vocab_size,
        hidden_dim=config.hidden_dim,
        num_decoder_layers=config.num_decoder_layers
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if accelerator.is_main_process:
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    # =========================================================================
    # 4. Loss Functions
    # =========================================================================
    ctc_criterion = nn.CTCLoss(blank=tokenizer.vocab_size, zero_infinity=True)
    ce_criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # =========================================================================
    # 5. Optimizer & Scheduler
    # =========================================================================
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    
    num_training_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * num_training_steps),
        num_training_steps=num_training_steps
    )
    
    # Prepare with accelerator
    model, optimizer, train_loader, dev_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, dev_loader, scheduler
    )

    # =========================================================================
    # 6. Training Loop
    # =========================================================================
    global_step = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_ctc = 0
        total_ce = 0
        num_batches = 0
        
        if accelerator.is_main_process:
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        else:
            progress_bar = train_loader
            
        for batch in progress_bar:
            with accelerator.accumulate(model):
                # Get batch data
                pose = batch['pose']
                input_lens = batch['input_lengths']
                
                # Handle both old and new dataset formats
                if 'ctc_labels' in batch:
                    # New format with separate CTC and decoder labels
                    ctc_labels = batch['ctc_labels']
                    ctc_lengths = batch['ctc_lengths']
                    dec_labels = batch['dec_labels']
                else:
                    # Old format - labels include EOS (not ideal but backward compatible)
                    ctc_labels = batch['labels']
                    ctc_lengths = batch['target_lengths']
                    dec_labels = batch['labels']
                
                # Prepare decoder input: [SOS, tok1, tok2, ...] (shifted right)
                dec_input = dec_labels.clone()
                dec_input[dec_input == -100] = tokenizer.pad_token_id
                sos_col = torch.full((dec_input.shape[0], 1), tokenizer.sos_token_id, 
                                     dtype=torch.long, device=device)
                dec_input = torch.cat([sos_col, dec_input[:, :-1]], dim=1)
                
                # Forward pass
                ctc_logits, dec_logits = model(pose, input_lens, dec_input)
                
                # CTC Loss
                ctc_log_probs = ctc_logits.permute(1, 0, 2).log_softmax(2)  # (T, B, V+1)
                
                # Flatten CTC targets
                ctc_targets_flat = []
                for i in range(ctc_labels.shape[0]):
                    valid_len = ctc_lengths[i].item()
                    ctc_targets_flat.append(ctc_labels[i, :valid_len])
                ctc_targets_flat = torch.cat(ctc_targets_flat)
                
                loss_ctc = ctc_criterion(ctc_log_probs, ctc_targets_flat, input_lens, ctc_lengths)
                
                # Decoder Loss
                loss_ce = ce_criterion(
                    dec_logits.reshape(-1, tokenizer.vocab_size),
                    dec_labels.reshape(-1)
                )
                
                # Combined loss
                loss = (args.lambda_ctc * loss_ctc) + ((1 - args.lambda_ctc) * loss_ce)
                
                # Backward
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Logging
                total_loss += loss.item()
                total_ctc += loss_ctc.item()
                total_ce += loss_ce.item()
                num_batches += 1
                global_step += 1
                
                if accelerator.is_main_process:
                    if args.wandb_project and global_step % 10 == 0:
                        accelerator.log({
                            "train/loss": loss.item(),
                            "train/ctc_loss": loss_ctc.item(),
                            "train/ce_loss": loss_ce.item(),
                            "train/lr": scheduler.get_last_lr()[0]
                        }, step=global_step)
                    
                    if global_step % 10 == 0:
                        progress_bar.set_postfix({
                            'loss': f"{loss.item():.4f}",
                            'ctc': f"{loss_ctc.item():.3f}",
                            'ce': f"{loss_ce.item():.3f}"
                        })

        avg_train_loss = total_loss / num_batches
        avg_ctc = total_ctc / num_batches
        avg_ce = total_ce / num_batches

        # =====================================================================
        # Validation
        # =====================================================================
        if (epoch + 1) % args.eval_every == 0:
            model.eval()
            val_loss = 0
            val_ctc = 0
            val_ce = 0
            val_batches = 0
            
            with torch.no_grad():
                for i, batch in enumerate(dev_loader):
                    pose = batch['pose']
                    input_lens = batch['input_lengths']
                    
                    if 'ctc_labels' in batch:
                        ctc_labels = batch['ctc_labels']
                        ctc_lengths = batch['ctc_lengths']
                        dec_labels = batch['dec_labels']
                    else:
                        ctc_labels = batch['labels']
                        ctc_lengths = batch['target_lengths']
                        dec_labels = batch['labels']
                    
                    # Decoder input
                    dec_input = dec_labels.clone()
                    dec_input[dec_input == -100] = tokenizer.pad_token_id
                    sos_col = torch.full((dec_input.shape[0], 1), tokenizer.sos_token_id,
                                         dtype=torch.long, device=device)
                    dec_input = torch.cat([sos_col, dec_input[:, :-1]], dim=1)
                    
                    # Forward
                    ctc_logits, dec_logits = model(pose, input_lens, dec_input)
                    
                    # Losses
                    ctc_log_probs = ctc_logits.permute(1, 0, 2).log_softmax(2)
                    ctc_targets_flat = []
                    for j in range(ctc_labels.shape[0]):
                        valid_len = ctc_lengths[j].item()
                        ctc_targets_flat.append(ctc_labels[j, :valid_len])
                    ctc_targets_flat = torch.cat(ctc_targets_flat)
                    
                    loss_ctc = ctc_criterion(ctc_log_probs, ctc_targets_flat, input_lens, ctc_lengths)
                    loss_ce = ce_criterion(dec_logits.reshape(-1, tokenizer.vocab_size), dec_labels.reshape(-1))
                    loss = (args.lambda_ctc * loss_ctc) + ((1 - args.lambda_ctc) * loss_ce)
                    
                    val_loss += loss.item()
                    val_ctc += loss_ctc.item()
                    val_ce += loss_ce.item()
                    val_batches += 1
                    
                    # Show sample prediction (first batch only)
                    if i == 0 and accelerator.is_main_process:
                        # CTC decode
                        pred_ids = torch.argmax(ctc_logits[0], dim=-1)
                        tokens = []
                        prev = -1
                        for t in pred_ids[:input_lens[0]]:
                            t_val = t.item()
                            if t_val != prev and t_val != tokenizer.vocab_size:
                                tokens.append(t_val)
                            prev = t_val
                        ctc_pred = tokenizer.decode(tokens)
                        
                        # Attention decode (greedy)
                        generated = torch.tensor([[tokenizer.sos_token_id]], device=device)
                        enc_mask = torch.arange(pose.shape[1], device=device)[None, :] < input_lens[0:1, None]
                        enc_out = accelerator.unwrap_model(model).encoder(pose[0:1], attention_mask=enc_mask.float())
                        
                        for _ in range(50):
                            logits = accelerator.unwrap_model(model).decoder(generated, enc_out, enc_mask=enc_mask.float())
                            next_token = logits[0, -1, :].argmax().item()
                            if next_token == tokenizer.eos_token_id:
                                break
                            generated = torch.cat([generated, torch.tensor([[next_token]], device=device)], dim=1)
                        
                        attn_pred = tokenizer.decode(generated[0, 1:].tolist())
                        ref_text = batch['raw_text'][0]
                        
                        print(f"\n[Sample] Ref:  {ref_text}")
                        print(f"[Sample] CTC:  {ctc_pred}")
                        print(f"[Sample] Attn: {attn_pred}")

            avg_val_loss = val_loss / val_batches
            avg_val_ctc = val_ctc / val_batches
            avg_val_ce = val_ce / val_batches
            
            accelerator.wait_for_everyone()
            
            if accelerator.is_main_process:
                print(f"\nEpoch {epoch+1}")
                print(f"  Train - Loss: {avg_train_loss:.4f} (CTC: {avg_ctc:.4f}, CE: {avg_ce:.4f})")
                print(f"  Val   - Loss: {avg_val_loss:.4f} (CTC: {avg_val_ctc:.4f}, CE: {avg_val_ce:.4f})")
                
                if args.wandb_project:
                    accelerator.log({
                        "val/loss": avg_val_loss,
                        "val/ctc_loss": avg_val_ctc,
                        "val/ce_loss": avg_val_ce
                    }, step=global_step)
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    
                    save_path = os.path.join(args.output_dir, "best_model")
                    os.makedirs(save_path, exist_ok=True)
                    
                    torch.save(
                        accelerator.unwrap_model(model).state_dict(),
                        os.path.join(save_path, "pytorch_model.bin")
                    )
                    accelerator.unwrap_model(model).encoder.save_pretrained(
                        os.path.join(save_path, "pose_encoder")
                    )
                    
                    with open(os.path.join(save_path, "config.json"), "w") as f:
                        json.dump({
                            "vocab_size": tokenizer.vocab_size,
                            "hidden_dim": config.hidden_dim,
                            "best_val_loss": best_val_loss
                        }, f, indent=2)
                    
                    print(f"  ✓ Saved best model (loss: {best_val_loss:.4f})")
                else:
                    patience_counter += 1
                    print(f"  No improvement. Patience: {patience_counter}/{args.patience}")
                    if patience_counter >= args.patience:
                        print("Early stopping!")
                        break

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                ckpt_path = os.path.join(args.output_dir, f"checkpoint-ep{epoch+1}")
                os.makedirs(ckpt_path, exist_ok=True)
                torch.save(
                    accelerator.unwrap_model(model).state_dict(),
                    os.path.join(ckpt_path, "model.pt")
                )

    # Finish
    if args.wandb_project:
        accelerator.end_training()
    
    if accelerator.is_main_process:
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Model saved to: {args.output_dir}/best_model")
        print("=" * 60)


if __name__ == "__main__":
    train()
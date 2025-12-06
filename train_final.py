"""
Final Training: Hybrid CTC/Attention Model for Sign Language Recognition

Architecture:
1. Pose Encoder (Pre-trained contrastively, frozen or fine-tuned)
2. CTC Head (For temporal alignment, loss λ)
3. Tiny Decoder (For sequence generation/grammar, loss 1-λ)

Usage:
    accelerate launch --num_processes 2 train_final.py \
        --contrastive_model_dir output/pose_encoder_cached/best_model/pose_encoder \
        --output_dir output/sign_language_model_final \
        --epochs 100 \
        --batch_size 16 \
        --lr 1e-4 \
        --lambda_ctc 0.5
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
    Hybrid SLR Model:
    - Encoder: Processes video frames (Spatial GCN + Temporal Transformer)
    - Branch 1 (CTC): Predicts gloss sequence alignment
    - Branch 2 (Decoder): Autoregressively generates gloss sequence
    """
    def __init__(self, pose_encoder: PoseEncoder, vocab_size: int, hidden_dim: int = 768, num_decoder_layers: int = 2):
        super().__init__()
        self.encoder = pose_encoder
        self.vocab_size = vocab_size
        
        # CTC Head: Projects to Vocab + Blank (Blank is usually index = vocab_size)
        self.ctc_head = nn.Linear(hidden_dim, vocab_size + 1)
        
        # Advanced Tiny Decoder (uses RoPE, RMSNorm from encoder config)
        dec_config = pose_encoder.config
        dec_config.num_decoder_layers = num_decoder_layers
        self.decoder = TinyAdvancedDecoder(vocab_size, dec_config)
        
    def forward(self, pose, input_lengths, decoder_input_ids=None):
        """
        Args:
            pose: (B, T, 86, 4)
            input_lengths: (B,) Valid frames for each sample
            decoder_input_ids: (B, S) Shifted targets for teacher forcing (optional)
        """
        # Create mask for encoder (1 for valid frames, 0 for padding)
        max_len = pose.shape[1]
        enc_mask = torch.arange(max_len, device=pose.device)[None, :] < input_lengths[:, None]
        
        # 1. Encoder Forward
        enc_out = self.encoder(pose, attention_mask=enc_mask.float())
        
        # 2. CTC Branch
        ctc_logits = self.ctc_head(enc_out)
        
        # 3. Decoder Branch
        dec_logits = None
        if decoder_input_ids is not None:
            # Encoder mask is passed to decoder to ignore padding keys in cross-attention
            dec_logits = self.decoder(decoder_input_ids, enc_out, enc_mask=enc_mask.float())
            
        return ctc_logits, dec_logits

def parse_args():
    parser = argparse.ArgumentParser(description="Train Hybrid SLR Model")
    
    # Data & Paths
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--contrastive_model_dir", type=str, required=True, 
                        help="Path to pre-trained pose encoder from Stage 1")
    parser.add_argument("--vocab_file", type=str, default="vocab.json", 
                        help="Path to vocabulary JSON (from build_vocab.py)")
    parser.add_argument("--pose_config", type=str, default="configs/pose_encoder_config.json",
                        help="Path to pose encoder config JSON")
    parser.add_argument("--fresh_start", action="store_true", 
                        help="If set, do not load weights from pre-trained encoder.")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="output/sign_language_model_final")
    parser.add_argument("--log_dir", type=str, default="runs/sign_language_model_final")
    parser.add_argument("--wandb_project", type=str, default="slr")
    
    # Training Hyperparams
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    
    # Hybrid Loss Weights
    parser.add_argument("--lambda_ctc", type=float, default=0.5, 
                        help="Weight for CTC loss (0.0 to 1.0). Decoder weight is 1 - lambda.")
    
    # Checkpointing
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")

    return parser.parse_args()

def train():
    args = parse_args()
    
    # Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=["wandb"],
        project_dir=args.log_dir
    )
    
    set_seed(args.seed)
    device = accelerator.device
    
    # Logging Setup
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=vars(args),
            init_kwargs={"wandb": {"name": f"hybrid-{args.batch_size}bs"}}
        )
        # Save args
        with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    # 1. Load Vocab & Tokenizer
    if not os.path.exists(args.vocab_file):
        raise FileNotFoundError(f"Vocab file {args.vocab_file} not found. Run build_vocab.py first.")
    tokenizer = GlossTokenizer(args.vocab_file)
    
    # 2. Load Datasets
    if accelerator.is_main_process: print("Loading datasets...")
    train_dataset = SignLanguageDataset(f"{args.data_dir}/train.csv", f"{args.data_dir}/data.pkl", tokenizer, mode='hybrid')
    dev_dataset = SignLanguageDataset(f"{args.data_dir}/dev.csv", f"{args.data_dir}/data.pkl", tokenizer, mode='hybrid')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_hybrid, num_workers=4, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=collate_fn_hybrid, num_workers=4)
    
    # 3. Model Initialization
    if accelerator.is_main_process:
        if args.fresh_start:
            print("Initializing random Pose Encoder (fresh start).")
        else:
            print(f"Loading Pre-trained Pose Encoder from {args.contrastive_model_dir}...")
    
    config = PoseEncoderConfig.from_json_file(args.pose_config)
    if args.fresh_start:
        pose_encoder = PoseEncoder(config)
    else:
        try:
            pose_encoder = PoseEncoder.from_pretrained(args.contrastive_model_dir)
        except Exception as e:
            if accelerator.is_main_process:
                print(f"Warning: Could not load weights ({e}). initializing random encoder.")
        pose_encoder = PoseEncoder(config)

    model = HybridModel(pose_encoder, tokenizer.vocab_size, hidden_dim=config.hidden_dim, num_decoder_layers=config.num_decoder_layers)
    
    # 4. Losses
    # CTC: Blank token is usually the last index (vocab_size)
    ctc_criterion = nn.CTCLoss(blank=tokenizer.vocab_size, zero_infinity=True)
    ce_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # 5. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    num_training_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(args.warmup_ratio * num_training_steps), 
        num_training_steps=num_training_steps
    )
    
    model, optimizer, train_loader, dev_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, dev_loader, scheduler
    )
    
    # ============================================
    # Training Loop
    # ============================================
    global_step = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_ctc = 0
        total_ce = 0
        
        if accelerator.is_main_process:
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        else:
            progress_bar = train_loader
            
        for batch in progress_bar:
            with accelerator.accumulate(model):
                # Prepare Inputs
                pose = batch['pose']
                input_lens = batch['input_lengths']
                labels = batch['labels'] # (B, S) padded with -100
                
                # Decoder Inputs: Replace -100 with PAD, Shift Right (Prepend SOS)
                dec_input = labels.clone()
                dec_input[dec_input == -100] = tokenizer.pad_token_id
                sos_col = torch.full((dec_input.shape[0], 1), tokenizer.sos_token_id, device=device)
                dec_input = torch.cat([sos_col, dec_input[:, :-1]], dim=1)
                
                # Forward
                ctc_logits, dec_logits = model(pose, input_lens, dec_input)
                
                # --- Loss 1: CTC ---
                # LogSoftmax: (T, B, V)
                ctc_log_probs = ctc_logits.permute(1, 0, 2).log_softmax(2)
                # Targets: Flattened, remove padding
                ctc_targets = labels[labels != -100]
                loss_ctc = ctc_criterion(ctc_log_probs, ctc_targets, input_lens, batch['target_lengths'])
                
                # --- Loss 2: Cross Entropy ---
                # Flatten: (B*S, V)
                loss_ce = ce_criterion(dec_logits.reshape(-1, tokenizer.vocab_size), labels.reshape(-1))
                
                # Weighted Sum
                loss = (args.lambda_ctc * loss_ctc) + ((1 - args.lambda_ctc) * loss_ce)
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Metrics
                total_loss += loss.item()
                total_ctc += loss_ctc.item()
                total_ce += loss_ce.item()
                global_step += 1
                
                if accelerator.is_main_process:
                    accelerator.log({
                        "train/loss": loss.item(),
                        "train/ctc_loss": loss_ctc.item(),
                        "train/ce_loss": loss_ce.item(),
                        "train/lr": scheduler.get_last_lr()[0]
                    }, step=global_step)
                    
                    if global_step % 10 == 0:
                        progress_bar.set_postfix({'loss': f"{loss.item():.4f}", 'ctc': f"{loss_ctc.item():.2f}"})

        # ============================================
        # Validation
        # ============================================
        if (epoch + 1) % args.eval_every == 0:
            model.eval()
            val_loss = 0
            
            # For validation, we compute Loss (teacher forcing) 
            # AND perform one Greedy Decode just to see progress
            with torch.no_grad():
                if accelerator.is_main_process:
                    val_pbar = tqdm(dev_loader, desc="Validating")
                else:
                    val_pbar = dev_loader
                    
                for i, batch in enumerate(val_pbar):
                    pose = batch['pose']
                    input_lens = batch['input_lengths']
                    labels = batch['labels']
                    
                    # Prepare Decoder Input
                    dec_input = labels.clone()
                    dec_input[dec_input == -100] = tokenizer.pad_token_id
                    sos_col = torch.full((dec_input.shape[0], 1), tokenizer.sos_token_id, device=device)
                    dec_input = torch.cat([sos_col, dec_input[:, :-1]], dim=1)
                    
                    # Forward
                    ctc_logits, dec_logits = model(pose, input_lens, dec_input)
                    
                    # Calculate Validation Loss (Same metric as train)
                    ctc_log_probs = ctc_logits.permute(1, 0, 2).log_softmax(2)
                    ctc_targets = labels[labels != -100]
                    loss_ctc = ctc_criterion(ctc_log_probs, ctc_targets, input_lens, batch['target_lengths'])
                    loss_ce = ce_criterion(dec_logits.reshape(-1, tokenizer.vocab_size), labels.reshape(-1))
                    
                    loss = (args.lambda_ctc * loss_ctc) + ((1 - args.lambda_ctc) * loss_ce)
                    val_loss += loss.item()
                    
                    # Qualitative Check (Greedy CTC) - Only on first batch of first GPU
                    if i == 0 and accelerator.is_main_process:
                        pred_ids = torch.argmax(ctc_logits[0], dim=-1)
                        # Decode
                        tokens = []
                        prev = -1
                        for t in pred_ids[:input_lens[0]]:
                            t = t.item()
                            if t != prev and t != tokenizer.vocab_size:
                                tokens.append(t)
                            prev = t
                        
                        pred_str = tokenizer.decode(tokens)
                        ref_str = batch['raw_text'][0]
                        print(f"\n[Val Sample] Ref: {ref_str} || Pred: {pred_str}")

            avg_val_loss = val_loss / len(dev_loader)
            accelerator.wait_for_everyone()
            
            if accelerator.is_main_process:
                # print train and val losses
                print(f"Epoch {epoch+1} Train Loss: {total_loss/len(train_loader):.4f} (CTC: {total_ctc/len(train_loader):.4f}, CE: {total_ce/len(train_loader):.4f})")
                print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")
                accelerator.log({"val/loss": avg_val_loss}, step=global_step)
                
                # Checkpointing & Early Stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    
                    save_path = os.path.join(args.output_dir, "best_model")
                    accelerator.unwrap_model(model).encoder.save_pretrained(os.path.join(save_path, "pose_encoder"))
                    torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(save_path, "pytorch_model.bin"))
                    print(f"--> Saved Best Model (Loss: {best_val_loss:.4f})")
                else:
                    patience_counter += 1
                    print(f"No improvement. Patience: {patience_counter}/{args.patience}")
                    if patience_counter >= args.patience:
                        print("Early stopping triggered!")
                        break

        # Periodic Save
        if (epoch + 1) % args.save_every == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                save_path = os.path.join(args.output_dir, f"checkpoint-ep{epoch+1}")
                os.makedirs(save_path, exist_ok=True)
                torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(save_path, "model.pt"))

    accelerator.end_training()

if __name__ == "__main__":
    train()
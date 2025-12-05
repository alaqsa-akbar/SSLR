"""
Final Sign Language Translation Model Training

Architecture:
- Pose Encoder (pre-trained from contrastive learning)
- UMT5 Decoder (only decoder, not full encoder-decoder)

This saves ~40% memory by not loading the unused UMT5 encoder.

Usage:
    accelerate launch --num_processes 2 train_final.py \
        --contrastive_model_dir output/pose_encoder_contrastive/best_model/pose_encoder \
        --output_dir output/sign_language_model_final
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, UMT5Config, UMT5ForConditionalGeneration
from transformers import get_linear_schedule_with_warmup
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from accelerate import Accelerator
from accelerate.utils import set_seed

from modules.pose_encoder import PoseEncoder, PoseEncoderConfig
from modules.dataset import SignLanguageDataset, collate_fn

from tqdm import tqdm
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
import json


class SignLanguageModel(nn.Module):
    """
    Sign Language Translation Model
    
    Combines a pre-trained Pose Encoder with UMT5 Decoder.
    The pose encoder produces embeddings that the decoder translates to text.
    """
    
    def __init__(self, pose_encoder: PoseEncoder, decoder: nn.Module, decoder_config: UMT5Config):
        super().__init__()
        self.pose_encoder = pose_encoder
        self.decoder = decoder
        self.config = decoder_config
        
        # LM head for token prediction
        self.lm_head = nn.Linear(decoder_config.d_model, decoder_config.vocab_size, bias=False)
        
    def forward(
        self,
        pose: torch.Tensor,
        pose_attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
        decoder_input_ids: torch.Tensor = None,
        decoder_attention_mask: torch.Tensor = None,
    ):
        """
        Forward pass for training.
        
        Args:
            pose: (B, T, 86, 2) pose keypoints
            pose_attention_mask: (B, T) mask for pose frames
            labels: (B, S) target token ids (shifted internally)
            decoder_input_ids: (B, S) decoder inputs (optional, derived from labels if not provided)
            decoder_attention_mask: (B, S) decoder attention mask
        """
        # Encode pose
        encoder_hidden_states = self.pose_encoder(pose, attention_mask=pose_attention_mask)
        
        # Prepare decoder inputs from labels (shift right)
        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self._shift_right(labels)
        
        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=pose_attention_mask,
        )
        
        # Project to vocabulary
        sequence_output = decoder_outputs.last_hidden_state
        logits = self.lm_head(sequence_output)
        
        # Compute loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
        )
    
    def _shift_right(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Shift input ids right for decoder input (teacher forcing)."""
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id
        
        if decoder_start_token_id is None:
            decoder_start_token_id = pad_token_id
        
        shifted = input_ids.new_zeros(input_ids.shape)
        shifted[:, 1:] = input_ids[:, :-1].clone()
        shifted[:, 0] = decoder_start_token_id
        
        # Replace -100 (ignore index) with pad token
        shifted.masked_fill_(shifted == -100, pad_token_id)
        
        return shifted
    
    @torch.no_grad()
    def generate(
        self,
        pose: torch.Tensor,
        pose_attention_mask: torch.Tensor,
        max_length: int = 128,
        num_beams: int = 4,
        early_stopping: bool = True,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        decoder_start_token_id: int = None,
    ):
        """
        Generate text from pose input using beam search.
        
        Args:
            pose: (B, T, 86, 2) pose keypoints
            pose_attention_mask: (B, T) mask for pose frames
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            
        Returns:
            Generated token ids: (B, S)
        """
        batch_size = pose.size(0)
        device = pose.device
        
        if decoder_start_token_id is None:
            decoder_start_token_id = self.config.decoder_start_token_id or pad_token_id
        
        # Encode pose once
        encoder_hidden_states = self.pose_encoder(pose, attention_mask=pose_attention_mask)
        
        # Initialize decoder input
        decoder_input_ids = torch.full(
            (batch_size, 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=device
        )
        
        # Simple greedy decoding (for beam search, use HuggingFace's generate)
        for _ in range(max_length - 1):
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=pose_attention_mask,
            )
            
            logits = self.lm_head(decoder_outputs.last_hidden_state[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)
            
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
            
            # Check if all sequences have generated EOS
            if (next_token == eos_token_id).all():
                break
        
        return decoder_input_ids
    
    def save_pretrained(self, save_directory: str):
        """Save model components separately."""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save pose encoder
        pose_encoder_path = os.path.join(save_directory, "pose_encoder")
        self.pose_encoder.save_pretrained(pose_encoder_path)
        
        # Save decoder
        decoder_path = os.path.join(save_directory, "decoder")
        os.makedirs(decoder_path, exist_ok=True)
        torch.save(self.decoder.state_dict(), os.path.join(decoder_path, "pytorch_model.bin"))
        self.config.save_pretrained(decoder_path)
        
        # Save LM head
        torch.save(self.lm_head.state_dict(), os.path.join(save_directory, "lm_head.pt"))
        
        # Save full config
        config_dict = {
            "pose_encoder_path": "pose_encoder",
            "decoder_path": "decoder",
            "model_type": "sign_language_model"
        }
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, load_directory: str, device: torch.device = None):
        """Load model from saved directory."""
        # Load pose encoder
        pose_encoder_path = os.path.join(load_directory, "pose_encoder")
        pose_encoder = PoseEncoder.from_pretrained(pose_encoder_path)
        
        # Load decoder config and weights
        decoder_path = os.path.join(load_directory, "decoder")
        decoder_config = UMT5Config.from_pretrained(decoder_path)
        
        # Create decoder from config
        full_model = UMT5ForConditionalGeneration(decoder_config)
        decoder = full_model.decoder
        
        # Load decoder weights
        decoder_weights = torch.load(
            os.path.join(decoder_path, "pytorch_model.bin"),
            map_location=device or "cpu"
        )
        decoder.load_state_dict(decoder_weights)
        
        # Create model
        model = cls(pose_encoder, decoder, decoder_config)
        
        # Load LM head
        lm_head_path = os.path.join(load_directory, "lm_head.pt")
        if os.path.exists(lm_head_path):
            model.lm_head.load_state_dict(torch.load(lm_head_path, map_location=device or "cpu"))
        
        return model


def load_umt5_decoder(model_dir: str, accelerator: Accelerator):
    """
    Load only the decoder from UMT5.
    
    Returns:
        decoder: The UMT5 decoder module
        config: The UMT5 config
    """
    umt5_weights_path = os.path.join(model_dir, "umt5-base.bin")
    
    try:
        if os.path.exists(umt5_weights_path):
            if accelerator.is_main_process:
                print(f"Loading UMT5 from: {umt5_weights_path}")
            
            config_path = os.path.join(model_dir, "config.json")
            if os.path.exists(config_path):
                config = UMT5Config.from_pretrained(model_dir)
            else:
                config = UMT5Config.from_pretrained("google/umt5-base")
            
            if config.d_model != 768:
                config.d_model = 768
                config.vocab_size = 256384
            
            # Load full model temporarily to extract decoder
            full_model = UMT5ForConditionalGeneration(config)
            state_dict = torch.load(umt5_weights_path, map_location="cpu")
            full_model.load_state_dict(state_dict, strict=False)
            
        else:
            if accelerator.is_main_process:
                print("Loading UMT5 from HuggingFace...")
            full_model = UMT5ForConditionalGeneration.from_pretrained("google/umt5-base")
            config = full_model.config
            
    except Exception as e:
        if accelerator.is_main_process:
            print(f"Error loading UMT5: {e}, creating from config...")
        config = UMT5Config(
            vocab_size=256384, d_model=768, d_kv=64, d_ff=2048,
            num_layers=12, num_decoder_layers=12, num_heads=12,
            relative_attention_num_buckets=32, dropout_rate=0.1,
            layer_norm_epsilon=1e-6, feed_forward_proj="gated-gelu",
            is_encoder_decoder=True
        )
        full_model = UMT5ForConditionalGeneration(config)
    
    # Extract only decoder
    decoder = full_model.decoder
    
    # Delete the full model to free memory
    del full_model.encoder
    del full_model
    torch.cuda.empty_cache()
    
    if accelerator.is_main_process:
        print(f"Loaded UMT5 decoder with {sum(p.numel() for p in decoder.parameters())/1e6:.1f}M parameters")
    
    return decoder, config


def load_pose_encoder(contrastive_model_dir: str, pose_config_path: str, hidden_dim: int, accelerator: Accelerator):
    """
    Load pose encoder from contrastive pre-training or initialize from scratch.
    
    Args:
        contrastive_model_dir: Path to contrastive model checkpoint
        pose_config_path: Path to pose encoder config JSON
        hidden_dim: Hidden dimension (must match UMT5)
        
    Returns:
        PoseEncoder model
    """
    # First, try to load from contrastive checkpoint
    if contrastive_model_dir and os.path.exists(contrastive_model_dir):
        # Check various possible paths
        possible_paths = [
            contrastive_model_dir,  # Direct path
            os.path.join(contrastive_model_dir, "pose_encoder"),  # Subdirectory
            os.path.join(contrastive_model_dir, "best_model", "pose_encoder"),  # Best model
            os.path.join(contrastive_model_dir, "final_model", "pose_encoder"),  # Final model
        ]
        
        for path in possible_paths:
            config_file = os.path.join(path, "config.json")
            if os.path.exists(config_file):
                if accelerator.is_main_process:
                    print(f"Loading pre-trained Pose Encoder from: {path}")
                try:
                    pose_encoder = PoseEncoder.from_pretrained(path)
                    
                    # Verify hidden_dim matches
                    if pose_encoder.config.hidden_dim != hidden_dim:
                        if accelerator.is_main_process:
                            print(f"Warning: Pose encoder hidden_dim ({pose_encoder.config.hidden_dim}) "
                                  f"doesn't match UMT5 ({hidden_dim}). This may cause issues.")
                    
                    if accelerator.is_main_process:
                        print(f"Successfully loaded pose encoder with {sum(p.numel() for p in pose_encoder.parameters())/1e6:.1f}M parameters")
                    return pose_encoder
                    
                except Exception as e:
                    if accelerator.is_main_process:
                        print(f"Failed to load from {path}: {e}")
                    continue
        
        if accelerator.is_main_process:
            print(f"Could not find valid checkpoint in {contrastive_model_dir}")
    
    # Fallback: Initialize from config
    if accelerator.is_main_process:
        print(f"Initializing Pose Encoder from config: {pose_config_path}")
    
    pose_config = PoseEncoderConfig.from_json_file(pose_config_path)
    pose_config.hidden_dim = hidden_dim  # Ensure it matches UMT5
    
    pose_encoder = PoseEncoder(pose_config)
    
    if accelerator.is_main_process:
        print(f"Initialized fresh pose encoder with {sum(p.numel() for p in pose_encoder.parameters())/1e6:.1f}M parameters")
        print("WARNING: Training without contrastive pre-training may result in worse performance!")
    
    return pose_encoder


def parse_args():
    parser = argparse.ArgumentParser(description="Train Sign Language Translation Model")
    
    # Data
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--model_dir", type=str, default="models/umt5",
                        help="Path to UMT5 model directory")
    parser.add_argument("--contrastive_model_dir", type=str, 
                        default="output/pose_encoder_contrastive/best_model/pose_encoder",
                        help="Path to pre-trained pose encoder from contrastive learning")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="output/sign_language_model_final",
                        help="Path to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="runs/sign_language_model_final",
                        help="Path to tensorboard logs")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Model
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames for poses")
    parser.add_argument("--pose_config", type=str, default="configs/pose_encoder_config.json",
                        help="Path to pose encoder config")
    
    # Early stopping
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (epochs without improvement)")
    
    # Checkpointing
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--eval_every", type=int, default=1, help="Evaluate every N epochs")
    
    # Generation (for validation)
    parser.add_argument("--num_beams", type=int, default=4, help="Beam size for generation")
    parser.add_argument("--max_gen_length", type=int, default=128, help="Max generation length")
    
    return parser.parse_args()


def train():
    args = parse_args()
    
    # Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=["tensorboard"],
        project_dir=args.log_dir
    )
    
    set_seed(args.seed)
    device = accelerator.device
    
    # Initialize logging
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        
        accelerator.init_trackers(
            project_name="sign-language-translation",
            config=vars(args)
        )
        
        with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    
    # Paths
    train_csv = os.path.join(args.data_dir, "train.csv")
    dev_csv = os.path.join(args.data_dir, "dev.csv")
    pkl_file = os.path.join(args.data_dir, "data.pkl")
    
    # Load tokenizer
    if accelerator.is_main_process:
        print("Loading tokenizer...")
    tokenizer_path = os.path.join(args.model_dir, "tokenizer.json")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except:
        tokenizer = AutoTokenizer.from_pretrained("google/umt5-base")
    
    # Load UMT5 decoder only
    if accelerator.is_main_process:
        print("Loading UMT5 decoder...")
    decoder, umt5_config = load_umt5_decoder(args.model_dir, accelerator)
    
    # Load pose encoder
    if accelerator.is_main_process:
        print("Loading Pose Encoder...")
    pose_encoder = load_pose_encoder(
        args.contrastive_model_dir,
        args.pose_config,
        umt5_config.d_model,
        accelerator
    )
    
    # Create combined model
    model = SignLanguageModel(pose_encoder, decoder, umt5_config)
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(decoder, 'gradient_checkpointing_enable'):
        decoder.gradient_checkpointing_enable()
        if accelerator.is_main_process:
            print("Enabled gradient checkpointing for decoder")
    
    # Load datasets
    if accelerator.is_main_process:
        print("Loading datasets...")
    train_dataset = SignLanguageDataset(train_csv, pkl_file, tokenizer, max_frames=args.max_frames)
    dev_dataset = SignLanguageDataset(dev_csv, pkl_file, tokenizer, max_frames=args.max_frames)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Scheduler
    num_training_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    
    # Prepare with accelerator
    model, optimizer, train_loader, dev_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, dev_loader, scheduler
    )
    
    # Print training info
    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{'='*60}")
        print(f"Training Configuration:")
        print(f"  - Total parameters: {total_params/1e6:.1f}M")
        print(f"  - Trainable parameters: {trainable_params/1e6:.1f}M")
        print(f"  - Batch size per GPU: {args.batch_size}")
        print(f"  - Number of GPUs: {accelerator.num_processes}")
        print(f"  - Gradient accumulation: {args.gradient_accumulation_steps}")
        print(f"  - Effective batch size: {args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")
        print(f"  - Training steps: {num_training_steps}")
        print(f"  - Warmup steps: {num_warmup_steps}")
        print(f"{'='*60}\n")
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # ============ Training ============
        model.train()
        total_loss = 0
        num_batches = 0
        
        if accelerator.is_main_process:
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        else:
            progress_bar = train_loader
        
        for batch in progress_bar:
            with accelerator.accumulate(model):
                pose = batch['pose']
                pose_mask = batch['pose_attention_mask']
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                
                # Prepare labels (replace pad with -100)
                labels = input_ids.clone()
                labels[labels == tokenizer.pad_token_id] = -100
                
                # Forward pass
                outputs = model(
                    pose=pose,
                    pose_attention_mask=pose_mask,
                    labels=labels,
                    decoder_attention_mask=attention_mask
                )
                
                loss = outputs.loss
                
                # Backward pass
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                num_batches += 1
                
                if accelerator.is_main_process:
                    accelerator.log({
                        "train/loss": loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                    }, step=global_step)
                    
                    progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
                
                global_step += 1
        
        avg_train_loss = total_loss / num_batches
        
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}")
        
        # ============ Validation ============
        if (epoch + 1) % args.eval_every == 0:
            model.eval()
            total_val_loss = 0
            num_val_batches = 0
            
            if accelerator.is_main_process:
                val_progress_bar = tqdm(dev_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            else:
                val_progress_bar = dev_loader
            
            with torch.no_grad():
                for batch in val_progress_bar:
                    pose = batch['pose']
                    pose_mask = batch['pose_attention_mask']
                    input_ids = batch['input_ids']
                    attention_mask = batch['attention_mask']
                    
                    labels = input_ids.clone()
                    labels[labels == tokenizer.pad_token_id] = -100
                    
                    outputs = model(
                        pose=pose,
                        pose_attention_mask=pose_mask,
                        labels=labels,
                        decoder_attention_mask=attention_mask
                    )
                    
                    loss = outputs.loss
                    
                    # Gather loss
                    gathered_loss = accelerator.gather(loss.unsqueeze(0))
                    total_val_loss += gathered_loss.mean().item()
                    num_val_batches += 1
            
            avg_val_loss = total_val_loss / num_val_batches
            
            if accelerator.is_main_process:
                print(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f}")
                
                accelerator.log({
                    "val/loss": avg_val_loss,
                    "epoch": epoch + 1
                }, step=global_step)
                
                # Check for improvement
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    
                    # Save best model
                    save_path = os.path.join(args.output_dir, "best_model")
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    
                    print(f"  → Saved best model (val_loss: {avg_val_loss:.4f})")
                else:
                    patience_counter += 1
                    print(f"  → No improvement. Patience: {patience_counter}/{args.patience}")
                    
                    if patience_counter >= args.patience:
                        print(f"\nEarly stopping triggered after {epoch+1} epochs!")
                        break
        
        # ============ Periodic Checkpoint ============
        if (epoch + 1) % args.save_every == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                checkpoint_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(checkpoint_path)
                
                # Save training state
                torch.save({
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'best_val_loss': best_val_loss,
                    'patience_counter': patience_counter,
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                }, os.path.join(checkpoint_path, "training_state.pt"))
                
                print(f"  → Saved checkpoint at epoch {epoch+1}")
    
    # ============ Final Save ============
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f"\nTraining complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        
        # Save final model
        final_path = os.path.join(args.output_dir, "final_model")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        
        accelerator.end_training()
        print("Done!")


if __name__ == "__main__":
    train()
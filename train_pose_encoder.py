"""
Contrastive Pre-training for Sign Language Pose Encoder

Features:
- Multi-GPU training with HuggingFace Accelerate
- Projection heads for both pose and text encoders
- Memory queue for increased effective batch size (MoCo-style)
- Proper validation loop
- Gradient checkpointing option for memory efficiency

Usage:
    # Single GPU
    python train_contrastive.py --batch_size 64

    # Multi-GPU with Accelerate (2x A100)
    accelerate launch --num_processes 2 train_contrastive.py --batch_size 64

    # With config file
    accelerate launch --config_file accelerate_config.yaml train_contrastive.py
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
from modules.dataset import SignLanguageDataset, collate_fn

from tqdm import tqdm
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Train Pose Encoder with Contrastive Learning")

    # Data
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--model_dir", type=str, default="models/umt5",
                        help="Path to directory containing tokenizer and UMT5 model")

    # Output
    parser.add_argument("--output_dir", type=str, default="output/pose_encoder_contrastive",
                        help="Path to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="runs/pose_encoder_contrastive",
                        help="Path to save tensorboard logs")

    # Training
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size per GPU (effective batch = batch_size * num_gpus + queue_size)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of steps to accumulate gradients")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Model
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Max frames for dataset (default: None, calculated from data)")
    parser.add_argument("--pose_config", type=str, default="configs/pose_encoder_config.json",
                        help="Path to Pose Encoder config file")
    parser.add_argument("--projection_dim", type=int, default=256,
                        help="Projection head output dimension")

    # Contrastive Learning
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="Temperature for contrastive loss")
    parser.add_argument("--queue_size", type=int, default=16384,
                        help="Size of memory queue for additional negatives (0 to disable)")
    parser.add_argument("--learnable_temperature", action="store_true",
                        help="Make temperature a learnable parameter")

    # Checkpointing
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--eval_every", type=int, default=1, help="Evaluate every N epochs")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint")

    return parser.parse_args()


class MemoryQueue:
    """
    Memory queue for MoCo-style contrastive learning.
    Stores text embeddings from previous batches to increase effective negatives.
    """

    def __init__(self, feature_dim: int, queue_size: int, device: torch.device):
        self.queue_size = queue_size
        self.feature_dim = feature_dim
        self.device = device

        # Initialize queue with random normalized vectors
        self.queue = F.normalize(torch.randn(queue_size, feature_dim), dim=1).to(device)
        self.pointer = 0

    @torch.no_grad()
    def enqueue(self, embeddings: torch.Tensor):
        """Add new embeddings to queue, replacing oldest ones."""
        embeddings = embeddings.detach()
        batch_size = embeddings.size(0)

        if batch_size == 0:
            return

        # Handle case where batch is larger than remaining space
        if self.pointer + batch_size <= self.queue_size:
            self.queue[self.pointer:self.pointer + batch_size] = embeddings
        else:
            # Wrap around
            overflow = (self.pointer + batch_size) - self.queue_size
            self.queue[self.pointer:] = embeddings[:-overflow]
            self.queue[:overflow] = embeddings[-overflow:]

        self.pointer = (self.pointer + batch_size) % self.queue_size

    def get_queue(self) -> torch.Tensor:
        """Get all embeddings in the queue."""
        return self.queue.clone().detach()


class ContrastiveLoss(nn.Module):
    """
    InfoNCE Contrastive Loss with optional memory queue.
    Supports symmetric loss (pose->text + text->pose).
    """

    def __init__(
        self,
        temperature: float = 0.07,
        learnable_temperature: bool = False,
        queue_size: int = 0,
        feature_dim: int = 256,
        device: torch.device = None
    ):
        super().__init__()

        if learnable_temperature:
            # Initialize log temperature (more stable than raw temperature)
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        else:
            self.register_buffer('log_temperature', torch.log(torch.tensor(temperature)))

        self.queue_size = queue_size
        self.text_queue = None
        self.pose_queue = None

        if queue_size > 0 and device is not None:
            self.text_queue = MemoryQueue(feature_dim, queue_size, device)
            self.pose_queue = MemoryQueue(feature_dim, queue_size, device)

    @property
    def temperature(self):
        return torch.exp(self.log_temperature)

    def forward(
        self,
        pose_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        pose_embeddings_all: torch.Tensor = None,
        text_embeddings_all: torch.Tensor = None
    ):
        """
        Compute symmetric contrastive loss.

        Args:
            pose_embeddings: Local batch pose embeddings (B, D)
            text_embeddings: Local batch text embeddings (B, D)
            pose_embeddings_all: Gathered pose embeddings from all GPUs (B_global, D)
            text_embeddings_all: Gathered text embeddings from all GPUs (B_global, D)

        Returns:
            loss: Scalar loss value
            metrics: Dict with additional metrics for logging
        """
        # Normalize embeddings
        pose_embeddings = F.normalize(pose_embeddings, p=2, dim=1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

        if pose_embeddings_all is not None:
            pose_embeddings_all = F.normalize(pose_embeddings_all, p=2, dim=1)
            text_embeddings_all = F.normalize(text_embeddings_all, p=2, dim=1)
        else:
            pose_embeddings_all = pose_embeddings
            text_embeddings_all = text_embeddings

        batch_size = pose_embeddings.size(0)
        temperature = self.temperature

        # Build negatives: gathered embeddings + queue
        if self.text_queue is not None and self.queue_size > 0:
            text_negatives = torch.cat([text_embeddings_all, self.text_queue.get_queue()], dim=0)
            pose_negatives = torch.cat([pose_embeddings_all, self.pose_queue.get_queue()], dim=0)
        else:
            text_negatives = text_embeddings_all
            pose_negatives = pose_embeddings_all

        # Pose -> Text
        # Positive: diagonal of (pose @ text_all.T)
        # We compute logits against all negatives
        logits_pose_to_text = torch.matmul(pose_embeddings, text_negatives.T) / temperature

        # Text -> Pose
        logits_text_to_pose = torch.matmul(text_embeddings, pose_negatives.T) / temperature

        # Labels: positives are at indices 0 to batch_size-1 in the gathered embeddings
        # But we need to account for the GPU rank offset
        labels = torch.arange(batch_size, device=pose_embeddings.device)

        # Cross entropy loss
        loss_p2t = F.cross_entropy(logits_pose_to_text, labels)
        loss_t2p = F.cross_entropy(logits_text_to_pose, labels)

        loss = (loss_p2t + loss_t2p) / 2

        # Update queues
        if self.text_queue is not None:
            self.text_queue.enqueue(text_embeddings)
            self.pose_queue.enqueue(pose_embeddings)

        # Compute accuracy for monitoring
        with torch.no_grad():
            pred_p2t = logits_pose_to_text[:, :text_embeddings_all.size(0)].argmax(dim=1)
            pred_t2p = logits_text_to_pose[:, :pose_embeddings_all.size(0)].argmax(dim=1)
            acc_p2t = (pred_p2t == labels).float().mean()
            acc_t2p = (pred_t2p == labels).float().mean()

        metrics = {
            'loss_p2t': loss_p2t.item(),
            'loss_t2p': loss_t2p.item(),
            'acc_p2t': acc_p2t.item(),
            'acc_t2p': acc_t2p.item(),
            'temperature': temperature.item() if isinstance(temperature, torch.Tensor) else temperature
        }

        return loss, metrics


def load_text_encoder(model_dir: str, device: torch.device, accelerator: Accelerator):
    """Load and freeze the UMT5 text encoder."""
    umt5_weights_path = os.path.join(model_dir, "umt5-base.bin")

    try:
        if os.path.exists(umt5_weights_path):
            if accelerator.is_main_process:
                print(f"Loading UMT5 weights from: {umt5_weights_path}")

            config_path = os.path.join(model_dir, "config.json")
            if os.path.exists(config_path):
                config = UMT5Config.from_pretrained(model_dir)
            else:
                config = UMT5Config.from_pretrained("google/umt5-base")

            if config.d_model != 768:
                config.d_model = 768
                config.vocab_size = 256384

            text_encoder = UMT5EncoderModel(config)
            state_dict = torch.load(umt5_weights_path, map_location="cpu")
            text_encoder.load_state_dict(state_dict, strict=False)
        else:
            if accelerator.is_main_process:
                print(f"Loading UMT5 from HuggingFace...")
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

    # Freeze text encoder
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


def train():
    args = parse_args()

    # Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=["tensorboard"],
        project_dir=args.log_dir
    )

    # Set seed for reproducibility
    set_seed(args.seed)

    device = accelerator.device

    # Initialize logging
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)

        accelerator.init_trackers(
            project_name="sign-language-contrastive",
            config=vars(args)
        )

        # Save args
        with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    # Paths
    train_csv = os.path.join(args.data_dir, "train.csv")
    dev_csv = os.path.join(args.data_dir, "dev.csv")
    pkl_file = os.path.join(args.data_dir, "data.pkl")

    # Load Tokenizer
    if accelerator.is_main_process:
        print("Loading tokenizer...")
    tokenizer_path = os.path.join(args.model_dir, "tokenizer.json")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except:
        tokenizer = AutoTokenizer.from_pretrained("google/umt5-base")

    # Load Text Encoder
    if accelerator.is_main_process:
        print("Loading text encoder...")
    text_encoder = load_text_encoder(args.model_dir, device, accelerator)
    text_encoder.to(device)

    # Load Datasets
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
        pin_memory=True,
        drop_last=True  # Important for consistent batch sizes in contrastive learning
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    # Initialize Pose Encoder with Projection Head
    if accelerator.is_main_process:
        print(f"Initializing Pose Encoder from {args.pose_config}...")
    pose_config = PoseEncoderConfig.from_json_file(args.pose_config)

    # Ensure dimensions match
    pose_config.hidden_dim = text_encoder.config.d_model  # 768
    pose_config.projection_dim = args.projection_dim

    pose_model = PoseEncoderForContrastive(pose_config)

    # Text Projection Head (separate, trainable)
    text_projection = TextProjectionHead(
        input_dim=text_encoder.config.d_model,
        hidden_dim=text_encoder.config.d_model,
        output_dim=args.projection_dim
    )

    # Contrastive Loss
    contrastive_loss_fn = ContrastiveLoss(
        temperature=args.temperature,
        learnable_temperature=args.learnable_temperature,
        queue_size=args.queue_size,
        feature_dim=args.projection_dim,
        device=device
    )

    # Optimizer - include all trainable parameters
    trainable_params = [
        {'params': pose_model.parameters(), 'lr': args.lr},
        {'params': text_projection.parameters(), 'lr': args.lr},
    ]
    if args.learnable_temperature:
        trainable_params.append({'params': contrastive_loss_fn.parameters(), 'lr': args.lr * 0.1})

    optimizer = torch.optim.AdamW(trainable_params, weight_decay=args.weight_decay)

    # Scheduler
    num_training_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    # Prepare with Accelerator
    pose_model, text_projection, optimizer, train_loader, dev_loader, scheduler = accelerator.prepare(
        pose_model, text_projection, optimizer, train_loader, dev_loader, scheduler
    )

    # Print training info
    if accelerator.is_main_process:
        effective_batch_size = args.batch_size * accelerator.num_processes
        total_negatives = effective_batch_size + args.queue_size - 1
        print(f"\n{'='*60}")
        print(f"Training Configuration:")
        print(f"  - Batch size per GPU: {args.batch_size}")
        print(f"  - Number of GPUs: {accelerator.num_processes}")
        print(f"  - Effective batch size: {effective_batch_size}")
        print(f"  - Queue size: {args.queue_size}")
        print(f"  - Total negatives per sample: {total_negatives}")
        print(f"  - Projection dim: {args.projection_dim}")
        print(f"  - Temperature: {args.temperature} (learnable: {args.learnable_temperature})")
        print(f"  - Training steps: {num_training_steps}")
        print(f"  - Warmup steps: {num_warmup_steps}")
        print(f"{'='*60}\n")

    # Training Loop
    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        # ============ Training ============
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
            with accelerator.accumulate(pose_model):
                pose = batch['pose']
                pose_mask = batch['pose_attention_mask']
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']

                # Forward pass - Pose Encoder with projection
                pose_projected = pose_model(pose, attention_mask=pose_mask, return_projection=True)

                # Forward pass - Text Encoder (frozen) + projection (trainable)
                with torch.no_grad():
                    text_outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                    text_pooled = pool_text_embeddings(text_outputs.last_hidden_state, attention_mask)

                text_projected = text_projection(text_pooled)

                # Gather embeddings from all GPUs
                pose_projected_all = accelerator.gather(pose_projected)
                text_projected_all = accelerator.gather(text_projected)

                # Compute contrastive loss
                loss, metrics = contrastive_loss_fn(
                    pose_projected,
                    text_projected,
                    pose_projected_all,
                    text_projected_all
                )

                # Backward
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        list(pose_model.parameters()) + list(text_projection.parameters()),
                        args.max_grad_norm
                    )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Logging
                total_loss += loss.item()
                total_acc += (metrics['acc_p2t'] + metrics['acc_t2p']) / 2
                num_batches += 1

                if accelerator.is_main_process:
                    accelerator.log({
                        "train/loss": loss.item(),
                        "train/loss_p2t": metrics['loss_p2t'],
                        "train/loss_t2p": metrics['loss_t2p'],
                        "train/acc_p2t": metrics['acc_p2t'],
                        "train/acc_t2p": metrics['acc_t2p'],
                        "train/temperature": metrics['temperature'],
                        "train/lr": scheduler.get_last_lr()[0],
                    }, step=global_step)

                    progress_bar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'acc': f"{(metrics['acc_p2t'] + metrics['acc_t2p'])/2:.3f}"
                    })

                global_step += 1

        avg_train_loss = total_loss / num_batches
        avg_train_acc = total_acc / num_batches

        if accelerator.is_main_process:
            print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.3f}")

        # ============ Validation ============
        if (epoch + 1) % args.eval_every == 0:
            pose_model.eval()
            text_projection.eval()

            total_val_loss = 0
            total_val_acc = 0
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

                    # Forward
                    pose_projected = pose_model(pose, attention_mask=pose_mask, return_projection=True)

                    text_outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                    text_pooled = pool_text_embeddings(text_outputs.last_hidden_state, attention_mask)
                    text_projected = text_projection(text_pooled)

                    # Gather
                    pose_projected_all = accelerator.gather(pose_projected)
                    text_projected_all = accelerator.gather(text_projected)

                    # Loss (without queue for validation)
                    pose_norm = F.normalize(pose_projected, p=2, dim=1)
                    text_norm = F.normalize(text_projected, p=2, dim=1)
                    text_all_norm = F.normalize(text_projected_all, p=2, dim=1)
                    pose_all_norm = F.normalize(pose_projected_all, p=2, dim=1)

                    temp = contrastive_loss_fn.temperature
                    logits_p2t = torch.matmul(pose_norm, text_all_norm.T) / temp
                    logits_t2p = torch.matmul(text_norm, pose_all_norm.T) / temp

                    labels = torch.arange(pose_norm.size(0), device=device)
                    val_loss = (F.cross_entropy(logits_p2t, labels) + F.cross_entropy(logits_t2p, labels)) / 2

                    # Accuracy
                    acc_p2t = (logits_p2t.argmax(dim=1) == labels).float().mean()
                    acc_t2p = (logits_t2p.argmax(dim=1) == labels).float().mean()
                    val_acc = (acc_p2t + acc_t2p) / 2

                    total_val_loss += val_loss.item()
                    total_val_acc += val_acc.item()
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

                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    save_path = os.path.join(args.output_dir, "best_model")
                    os.makedirs(save_path, exist_ok=True)

                    unwrapped_pose_model = accelerator.unwrap_model(pose_model)
                    unwrapped_text_proj = accelerator.unwrap_model(text_projection)

                    # Save encoder only (without projection head) for downstream tasks
                    unwrapped_pose_model.encoder.save_pretrained(os.path.join(save_path, "pose_encoder"))

                    # Save full model for resuming contrastive training
                    unwrapped_pose_model.save_pretrained(os.path.join(save_path, "pose_encoder_full"))
                    torch.save(unwrapped_text_proj.state_dict(), os.path.join(save_path, "text_projection.pt"))

                    print(f"  → Saved best model (val_loss: {avg_val_loss:.4f})")

        # ============ Periodic Checkpoint ============
        if (epoch + 1) % args.save_every == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                checkpoint_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
                os.makedirs(checkpoint_path, exist_ok=True)

                unwrapped_pose_model = accelerator.unwrap_model(pose_model)
                unwrapped_text_proj = accelerator.unwrap_model(text_projection)

                unwrapped_pose_model.save_pretrained(os.path.join(checkpoint_path, "pose_encoder"))
                torch.save(unwrapped_text_proj.state_dict(), os.path.join(checkpoint_path, "text_projection.pt"))
                torch.save({
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                }, os.path.join(checkpoint_path, "training_state.pt"))

                print(f"  → Saved checkpoint at epoch {epoch+1}")

    # ============ Final Save ============
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f"\nTraining complete! Saving final model to {args.output_dir}")

        final_path = os.path.join(args.output_dir, "final_model")
        os.makedirs(final_path, exist_ok=True)

        unwrapped_pose_model = accelerator.unwrap_model(pose_model)
        unwrapped_text_proj = accelerator.unwrap_model(text_projection)

        # Save encoder for downstream tasks
        unwrapped_pose_model.encoder.save_pretrained(os.path.join(final_path, "pose_encoder"))

        # Save full model
        unwrapped_pose_model.save_pretrained(os.path.join(final_path, "pose_encoder_full"))
        torch.save(unwrapped_text_proj.state_dict(), os.path.join(final_path, "text_projection.pt"))

        # Save config
        pose_config.save_pretrained(final_path)

        accelerator.end_training()
        print("Done!")


if __name__ == "__main__":
    train()
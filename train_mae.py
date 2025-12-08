"""
Masked Pose Reconstruction Pre-training for Sign Language Pose Encoder

Similar to BERT/MAE: mask random frames and reconstruct them.
This preserves temporal structure better than contrastive learning.

Key features:
1. Masks 15% of frames randomly
2. Encoder must reconstruct masked frame features
3. Learns local temporal patterns (useful for CTC/Attention)
4. No text encoder needed - self-supervised on pose only

Masking strategies:
- FRAME: Mask entire frames (all keypoints at timestep t)
- KEYPOINT: Mask specific keypoints across time
- SPAN: Mask contiguous spans of frames
- MIXED: Combination of above

Usage:
    python train_mae.py \
        --epochs 100 \
        --batch_size 64 \
        --lr 1e-4 \
        --mask_ratio 0.15 \
        --output_dir output/pretrained_mae

    # Multi-GPU:
    accelerate launch --num_processes 2 train_mae.py \
        --epochs 100 \
        --batch_size 64 \
        --output_dir output/pretrained_mae
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import set_seed

import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm
import os
import argparse
import json

# Import your model
from modules.pose_encoder import PoseEncoder, PoseEncoderConfig


# =============================================================================
# RECONSTRUCTION HEAD
# =============================================================================

class ReconstructionHead(nn.Module):
    """
    Reconstructs masked pose features from encoder output.
    """
    def __init__(self, hidden_dim, num_keypoints=86, num_channels=4):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.num_channels = num_channels
        output_dim = num_keypoints * num_channels  # 86 * 4 = 344
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        # x: (B, T, H) -> (B, T, 86*4)
        out = self.head(x)
        # Reshape to (B, T, 86, 4)
        B, T, _ = out.shape
        return out.view(B, T, self.num_keypoints, self.num_channels)


# =============================================================================
# MASKING STRATEGIES
# =============================================================================

class PoseMasker:
    """
    Different masking strategies for pose sequences.
    """
    def __init__(
        self,
        mask_ratio=0.15,
        strategy='frame',  # 'frame', 'keypoint', 'span', 'mixed'
        span_length=3,
        mask_value=0.0
    ):
        self.mask_ratio = mask_ratio
        self.strategy = strategy
        self.span_length = span_length
        self.mask_value = mask_value
    
    def __call__(self, pose, pose_mask):
        """
        Apply masking to pose sequence.
        
        Args:
            pose: (B, T, N, C) tensor
            pose_mask: (B, T) tensor indicating valid frames
            
        Returns:
            masked_pose: (B, T, N, C) with masked values
            mask: (B, T) boolean tensor indicating masked positions
        """
        if self.strategy == 'frame':
            return self._mask_frames(pose, pose_mask)
        elif self.strategy == 'keypoint':
            return self._mask_keypoints(pose, pose_mask)
        elif self.strategy == 'span':
            return self._mask_spans(pose, pose_mask)
        elif self.strategy == 'mixed':
            return self._mask_mixed(pose, pose_mask)
        else:
            raise ValueError(f"Unknown masking strategy: {self.strategy}")
    
    def _mask_frames(self, pose, pose_mask):
        """Mask entire frames randomly."""
        B, T, N, C = pose.shape
        device = pose.device
        
        # Create frame-level mask
        mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        
        for b in range(B):
            # Get valid frame indices
            valid_indices = torch.where(pose_mask[b] > 0)[0]
            num_valid = len(valid_indices)
            
            if num_valid > 2:  # Need at least some frames
                # Number of frames to mask (don't mask first and last)
                num_mask = max(1, int(num_valid * self.mask_ratio))
                
                # Avoid masking first and last frames
                maskable_indices = valid_indices[1:-1] if num_valid > 2 else valid_indices
                
                if len(maskable_indices) > 0:
                    # Random selection
                    perm = torch.randperm(len(maskable_indices), device=device)[:num_mask]
                    mask_indices = maskable_indices[perm]
                    mask[b, mask_indices] = True
        
        # Apply mask
        masked_pose = pose.clone()
        mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).expand_as(pose)
        masked_pose[mask_expanded] = self.mask_value
        
        return masked_pose, mask
    
    def _mask_keypoints(self, pose, pose_mask):
        """Mask specific keypoints across all time steps."""
        B, T, N, C = pose.shape
        device = pose.device
        
        # For this strategy, we mask specific keypoints
        # Return frame mask as "any keypoint masked at this frame"
        mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        masked_pose = pose.clone()
        
        # Keypoint groups
        RIGHT_HAND = list(range(0, 21))
        LEFT_HAND = list(range(21, 42))
        FACE = list(range(42, 70))
        BODY = list(range(70, 86))
        
        for b in range(B):
            valid_frames = pose_mask[b] > 0
            
            # Randomly select keypoints to mask
            num_keypoints_to_mask = max(1, int(N * self.mask_ratio))
            keypoint_indices = torch.randperm(N, device=device)[:num_keypoints_to_mask]
            
            # Mask these keypoints across all valid frames
            for kp in keypoint_indices:
                masked_pose[b, valid_frames, kp, :] = self.mask_value
            
            # Mark all valid frames as "masked" since we need to reconstruct
            mask[b, valid_frames] = True
        
        return masked_pose, mask
    
    def _mask_spans(self, pose, pose_mask):
        """Mask contiguous spans of frames."""
        B, T, N, C = pose.shape
        device = pose.device
        
        mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        
        for b in range(B):
            valid_indices = torch.where(pose_mask[b] > 0)[0]
            num_valid = len(valid_indices)
            
            if num_valid > self.span_length + 2:
                # Calculate number of spans to mask
                total_to_mask = int(num_valid * self.mask_ratio)
                num_spans = max(1, total_to_mask // self.span_length)
                
                # Avoid first and last frames
                max_start = num_valid - self.span_length - 1
                
                for _ in range(num_spans):
                    if max_start > 1:
                        start_idx = torch.randint(1, max_start, (1,), device=device).item()
                        span_indices = valid_indices[start_idx:start_idx + self.span_length]
                        mask[b, span_indices] = True
        
        # Apply mask
        masked_pose = pose.clone()
        mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).expand_as(pose)
        masked_pose[mask_expanded] = self.mask_value
        
        return masked_pose, mask
    
    def _mask_mixed(self, pose, pose_mask):
        """Mix of frame and span masking."""
        B, T, N, C = pose.shape
        device = pose.device
        
        # 50% frame masking, 50% span masking
        if torch.rand(1).item() < 0.5:
            return self._mask_frames(pose, pose_mask)
        else:
            return self._mask_spans(pose, pose_mask)


# =============================================================================
# DATASET
# =============================================================================

class MAEDataset(Dataset):
    """
    Dataset for Masked Autoencoder pre-training.
    Returns pose sequences without text labels.
    """
    def __init__(
        self,
        csv_file,
        pkl_file,
        max_frames=None,
        use_augmentation=True,
        augmentation_config=None
    ):
        self.data = pd.read_csv(csv_file)
        with open(pkl_file, 'rb') as f:
            self.keypoints = pickle.load(f)
        
        self.use_augmentation = use_augmentation
        self.training = True
        
        # Augmentation config (lighter than contrastive - we want clean targets)
        self.aug_config = {
            'noise_sigma': 0.01,  # Less noise for cleaner reconstruction targets
            'scale_range': (0.9, 1.1),
            'shift_range': 0.05,
            'rotation_prob': 0.3,
            'rotation_max_angle': 8,
            'speed_prob': 0.3,
            'speed_range': (0.9, 1.1),
        }
        if augmentation_config:
            self.aug_config.update(augmentation_config)
        
        # Calculate max_frames
        if max_frames is None:
            self.max_frames = 0
            for sample_id in self.data['id']:
                if sample_id in self.keypoints:
                    raw_len = self.keypoints[sample_id]['keypoints'].shape[0]
                    max_possible = int(raw_len * 1.2)
                    if max_possible > self.max_frames:
                        self.max_frames = max_possible
            print(f"Auto-detected max_frames: {self.max_frames}")
        else:
            self.max_frames = max_frames
        
        # Filter samples without keypoints
        valid_ids = [sid for sid in self.data['id'] if sid in self.keypoints]
        if len(valid_ids) < len(self.data):
            print(f"Warning: {len(self.data) - len(valid_ids)} samples missing keypoints")
            self.data = self.data[self.data['id'].isin(valid_ids)].reset_index(drop=True)
    
    def __len__(self):
        return len(self.data)
    
    def set_training(self, training=True):
        self.training = training
    
    def _normalize(self, keypoints, invalid_value=-1.0):
        """Normalize keypoints."""
        arr = keypoints.copy().astype(np.float32)
        T, N, C = arr.shape
        
        valid_mask = ~np.isclose(arr, invalid_value, atol=1e-6).any(axis=-1)
        num_valid = valid_mask.sum()
        
        if num_valid == 0:
            return np.zeros_like(arr), valid_mask
        
        valid_points = arr[valid_mask]
        center = valid_points.mean(axis=0)
        std = max(1e-6, valid_points.std())
        
        arr[valid_mask] = (arr[valid_mask] - center) / std
        arr[~valid_mask] = 0.0
        
        return arr, valid_mask
    
    def _compute_velocity(self, pose, valid_mask):
        """Compute velocity."""
        T, N, C = pose.shape
        velocity = np.zeros_like(pose)
        velocity[1:] = pose[1:] - pose[:-1]
        
        valid_velocity_mask = np.zeros((T, N), dtype=bool)
        valid_velocity_mask[1:] = valid_mask[1:] & valid_mask[:-1]
        velocity[~valid_velocity_mask] = 0.0
        velocity[0] = 0.0
        
        return velocity
    
    def _apply_augmentation(self, pose, valid_mask):
        """Apply light data augmentation."""
        from scipy import interpolate
        
        # Speed perturbation (light)
        if np.random.random() < self.aug_config['speed_prob']:
            T, N, C = pose.shape
            if T >= 4:
                speed = np.random.uniform(*self.aug_config['speed_range'])
                new_T = max(4, min(int(T / speed), int(T * 1.3)))
                if new_T != T:
                    t_orig = np.linspace(0, 1, T)
                    t_new = np.linspace(0, 1, new_T)
                    
                    resampled_pose = np.zeros((new_T, N, C), dtype=np.float32)
                    for n in range(N):
                        for c in range(C):
                            f = interpolate.interp1d(t_orig, pose[:, n, c], kind='linear', fill_value='extrapolate')
                            resampled_pose[:, n, c] = f(t_new)
                    
                    resampled_mask = np.zeros((new_T, N), dtype=bool)
                    mask_float = valid_mask.astype(float)
                    for n in range(N):
                        f = interpolate.interp1d(t_orig, mask_float[:, n], kind='nearest', fill_value='extrapolate')
                        resampled_mask[:, n] = f(t_new) > 0.5
                    
                    pose, valid_mask = resampled_pose, resampled_mask
        
        T, N, C = pose.shape
        
        # Rotation (light)
        if np.random.random() < self.aug_config['rotation_prob'] and valid_mask.sum() > 0:
            angle_deg = np.random.uniform(-self.aug_config['rotation_max_angle'], self.aug_config['rotation_max_angle'])
            angle_rad = np.radians(angle_deg)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
            
            valid_points = pose[valid_mask]
            center = valid_points.mean(axis=0)
            
            rotated = pose.copy()
            centered = pose[valid_mask] - center
            rotated[valid_mask] = centered @ rotation_matrix.T + center
            pose = rotated
        
        # Scale and shift (light)
        scale = np.random.uniform(*self.aug_config['scale_range'])
        pose[valid_mask] *= scale
        
        shift = np.random.uniform(-self.aug_config['shift_range'], self.aug_config['shift_range'], size=(1, 1, C)).astype(np.float32)
        pose[valid_mask] += shift.squeeze()
        
        return pose, valid_mask
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sample_id = row['id']
        
        # Get raw keypoints
        raw_pose = self.keypoints.get(sample_id, {}).get('keypoints', np.zeros((10, 86, 2), dtype=np.float32))
        
        # Normalize
        pose, valid_mask = self._normalize(raw_pose)
        
        # Augment (if training)
        if self.use_augmentation and self.training:
            pose, valid_mask = self._apply_augmentation(pose, valid_mask)
        
        # Compute velocity
        velocity = self._compute_velocity(pose, valid_mask)
        pose_combined = np.concatenate([pose, velocity], axis=-1)  # (T, 86, 4)
        
        # Truncate
        if pose_combined.shape[0] > self.max_frames:
            pose_combined = pose_combined[:self.max_frames]
        
        return {
            'id': sample_id,
            'pose': torch.tensor(pose_combined, dtype=torch.float32),
            'num_frames': pose_combined.shape[0]
        }


def collate_fn_mae(batch):
    """Collate function for MAE."""
    poses = [x['pose'] for x in batch]
    ids = [x['id'] for x in batch]
    
    max_frames = max(p.shape[0] for p in poses)
    num_keypoints = poses[0].shape[1]
    feature_dim = poses[0].shape[2]
    
    padded_poses = torch.zeros(len(poses), max_frames, num_keypoints, feature_dim)
    pose_mask = torch.zeros(len(poses), max_frames)
    
    for i, p in enumerate(poses):
        seq_len = p.shape[0]
        padded_poses[i, :seq_len] = p
        pose_mask[i, :seq_len] = 1
    
    return {
        'id': ids,
        'pose': padded_poses,
        'pose_mask': pose_mask,
    }


# =============================================================================
# MAIN TRAINING
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Masked Pose Reconstruction Pre-training")
    
    # Data
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--pose_config", type=str, default="configs/pose_encoder_config.json")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="output/pretrained_mae")
    parser.add_argument("--wandb_project", type=str, default=None)
    
    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    
    # Masking
    parser.add_argument("--mask_ratio", type=float, default=0.15)
    parser.add_argument("--mask_strategy", type=str, default="frame",
                        choices=["frame", "keypoint", "span", "mixed"])
    parser.add_argument("--span_length", type=int, default=3)
    
    # Loss
    parser.add_argument("--loss_type", type=str, default="mse",
                        choices=["mse", "l1", "smooth_l1"])
    parser.add_argument("--loss_on_all", action="store_true",
                        help="Compute loss on all frames, not just masked")
    
    # Checkpointing
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=1)
    
    return parser.parse_args()


def train():
    args = parse_args()
    
    # Initialize
    accelerator = Accelerator(
        log_with=["wandb"] if args.wandb_project else None,
    )
    set_seed(args.seed)
    device = accelerator.device
    
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        
        if args.wandb_project:
            accelerator.init_trackers(
                project_name=args.wandb_project,
                config=vars(args),
                init_kwargs={"wandb": {"name": f"mae-pretrain-{args.mask_strategy}"}}
            )
        
        print("=" * 70)
        print("MASKED POSE RECONSTRUCTION PRE-TRAINING")
        print("=" * 70)
        print(f"Pose Config: {args.pose_config}")
        print(f"Mask Ratio: {args.mask_ratio}")
        print(f"Mask Strategy: {args.mask_strategy}")
        print(f"Loss Type: {args.loss_type}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Learning Rate: {args.lr}")
        print("=" * 70)
    
    # Load datasets
    if accelerator.is_main_process:
        print("\nLoading datasets...")
    
    train_csv = os.path.join(args.data_dir, "train.csv")
    dev_csv = os.path.join(args.data_dir, "dev.csv")
    pkl_file = os.path.join(args.data_dir, "data.pkl")
    
    train_dataset = MAEDataset(train_csv, pkl_file, use_augmentation=True)
    dev_dataset = MAEDataset(dev_csv, pkl_file, use_augmentation=False)
    dev_dataset.set_training(False)
    
    if accelerator.is_main_process:
        print(f"Train samples: {len(train_dataset)}")
        print(f"Dev samples: {len(dev_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_mae,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_mae,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    # Initialize models
    if accelerator.is_main_process:
        print("\nInitializing models...")
    
    pose_config = PoseEncoderConfig.from_json_file(args.pose_config)
    pose_encoder = PoseEncoder(pose_config)
    
    reconstruction_head = ReconstructionHead(
        hidden_dim=pose_config.hidden_dim,
        num_keypoints=86,
        num_channels=4
    )
    
    # Masker
    masker = PoseMasker(
        mask_ratio=args.mask_ratio,
        strategy=args.mask_strategy,
        span_length=args.span_length,
        mask_value=0.0
    )
    
    # Count parameters
    num_encoder_params = sum(p.numel() for p in pose_encoder.parameters())
    num_head_params = sum(p.numel() for p in reconstruction_head.parameters())
    
    if accelerator.is_main_process:
        print(f"Encoder parameters: {num_encoder_params:,}")
        print(f"Reconstruction head parameters: {num_head_params:,}")
        print(f"Total: {num_encoder_params + num_head_params:,}")
    
    # Loss function
    if args.loss_type == "mse":
        criterion = nn.MSELoss(reduction='none')
    elif args.loss_type == "l1":
        criterion = nn.L1Loss(reduction='none')
    else:
        criterion = nn.SmoothL1Loss(reduction='none')
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        list(pose_encoder.parameters()) + list(reconstruction_head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Scheduler
    num_training_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * num_training_steps),
        num_training_steps=num_training_steps
    )
    
    # Prepare for distributed training
    pose_encoder, reconstruction_head, optimizer, train_loader, dev_loader, scheduler = accelerator.prepare(
        pose_encoder, reconstruction_head, optimizer, train_loader, dev_loader, scheduler
    )
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        pose_encoder.train()
        reconstruction_head.train()
        
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}") if accelerator.is_main_process else train_loader
        
        for batch in pbar:
            optimizer.zero_grad()
            
            pose = batch['pose']  # (B, T, 86, 4)
            pose_mask = batch['pose_mask']  # (B, T)
            
            # Store original for reconstruction target
            target = pose.clone()
            
            # Apply masking
            masked_pose, frame_mask = masker(pose, pose_mask)
            
            # Forward through encoder
            # Encoder expects (B, T, N, C), outputs (B, T, H)
            encoder_output = pose_encoder(masked_pose, attention_mask=pose_mask)
            
            # Reconstruct
            reconstructed = reconstruction_head(encoder_output)  # (B, T, 86, 4)
            
            # Compute loss
            loss_per_element = criterion(reconstructed, target)  # (B, T, 86, 4)
            
            # Average over keypoints and channels
            loss_per_frame = loss_per_element.mean(dim=(-1, -2))  # (B, T)
            
            if args.loss_on_all:
                # Loss on all valid frames
                valid_mask = pose_mask > 0
                loss = (loss_per_frame * valid_mask).sum() / valid_mask.sum().clamp(min=1)
            else:
                # Loss only on masked frames
                combined_mask = frame_mask & (pose_mask > 0)
                if combined_mask.sum() > 0:
                    loss = (loss_per_frame * combined_mask).sum() / combined_mask.sum()
                else:
                    loss = loss_per_frame.mean()
            
            # Backward
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(
                list(pose_encoder.parameters()) + list(reconstruction_head.parameters()),
                args.max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            if accelerator.is_main_process and global_step % 20 == 0:
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                accelerator.log({
                    'train/loss': loss.item(),
                    'train/lr': scheduler.get_last_lr()[0]
                }, step=global_step)
        
        avg_train_loss = total_loss / num_batches
        
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch+1}")
            print(f"  Train - Loss: {avg_train_loss:.4f}")
        
        # Validation
        if (epoch + 1) % args.eval_every == 0:
            pose_encoder.eval()
            reconstruction_head.eval()
            
            total_val_loss = 0
            num_val_batches = 0
            
            with torch.no_grad():
                for batch in dev_loader:
                    pose = batch['pose']
                    pose_mask = batch['pose_mask']
                    target = pose.clone()
                    
                    masked_pose, frame_mask = masker(pose, pose_mask)
                    encoder_output = pose_encoder(masked_pose, attention_mask=pose_mask)
                    reconstructed = reconstruction_head(encoder_output)
                    
                    loss_per_element = criterion(reconstructed, target)
                    loss_per_frame = loss_per_element.mean(dim=(-1, -2))
                    
                    if args.loss_on_all:
                        valid_mask = pose_mask > 0
                        loss = (loss_per_frame * valid_mask).sum() / valid_mask.sum().clamp(min=1)
                    else:
                        combined_mask = frame_mask & (pose_mask > 0)
                        if combined_mask.sum() > 0:
                            loss = (loss_per_frame * combined_mask).sum() / combined_mask.sum()
                        else:
                            loss = loss_per_frame.mean()
                    
                    total_val_loss += loss.item()
                    num_val_batches += 1
            
            avg_val_loss = total_val_loss / num_val_batches
            
            if accelerator.is_main_process:
                print(f"  Val   - Loss: {avg_val_loss:.4f}")
                accelerator.log({
                    'val/loss': avg_val_loss
                }, step=global_step)
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    
                    save_path = os.path.join(args.output_dir, "best_model")
                    os.makedirs(save_path, exist_ok=True)
                    
                    # Save encoder
                    unwrapped_encoder = accelerator.unwrap_model(pose_encoder)
                    unwrapped_encoder.save_pretrained(save_path)
                    
                    # Save reconstruction head (for visualization/evaluation)
                    unwrapped_head = accelerator.unwrap_model(reconstruction_head)
                    torch.save(unwrapped_head.state_dict(), 
                              os.path.join(save_path, "reconstruction_head.pt"))
                    
                    # Save config
                    pose_config.save_pretrained(save_path)
                    
                    # Save training info
                    with open(os.path.join(save_path, "training_info.json"), 'w') as f:
                        json.dump({
                            'epoch': epoch + 1,
                            'val_loss': avg_val_loss,
                            'mask_ratio': args.mask_ratio,
                            'mask_strategy': args.mask_strategy,
                            'loss_type': args.loss_type,
                        }, f, indent=2)
                    
                    print(f"  ✓ Saved best model (Loss: {avg_val_loss:.4f})")
        
        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            if accelerator.is_main_process:
                checkpoint_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
                os.makedirs(checkpoint_path, exist_ok=True)
                
                unwrapped_encoder = accelerator.unwrap_model(pose_encoder)
                unwrapped_encoder.save_pretrained(checkpoint_path)
                
                print(f"  ✓ Saved checkpoint at epoch {epoch+1}")
    
    # Final save
    if accelerator.is_main_process:
        final_path = os.path.join(args.output_dir, "final_model")
        os.makedirs(final_path, exist_ok=True)
        
        unwrapped_encoder = accelerator.unwrap_model(pose_encoder)
        unwrapped_encoder.save_pretrained(final_path)
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Models saved to: {args.output_dir}")
        print("=" * 70)
        print("\nTo use pretrained encoder for hybrid training:")
        print(f"  python train_final.py --pretrained_encoder {os.path.join(args.output_dir, 'best_model')}")
    
    if args.wandb_project:
        accelerator.end_training()


if __name__ == "__main__":
    train()
"""
Visualize and Evaluate MAE Pre-training

This script provides:
1. Reconstruction quality visualization (original vs masked vs reconstructed)
2. Quantitative metrics (MSE, per-keypoint error, per-body-part error)
3. Attention/feature analysis
4. Animation of reconstruction

Usage:
    python visualize_mae.py \
        --model_dir output/pretrained_mae/best_model \
        --data_dir data \
        --output_dir output/mae_visualizations \
        --num_samples 10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import seaborn as sns
import pandas as pd
import pickle
import json
import os
import argparse
from tqdm import tqdm
from pathlib import Path

# Import your modules
from modules.pose_encoder import PoseEncoder, PoseEncoderConfig


# =============================================================================
# RECONSTRUCTION HEAD (same as training)
# =============================================================================

class ReconstructionHead(nn.Module):
    """Reconstructs masked pose features from encoder output."""
    def __init__(self, hidden_dim, num_keypoints=86, num_channels=4):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.num_channels = num_channels
        output_dim = num_keypoints * num_channels
        
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
        out = self.head(x)
        B, T, _ = out.shape
        return out.view(B, T, self.num_keypoints, self.num_channels)


# =============================================================================
# DATASET (simplified for evaluation)
# =============================================================================

class MAEEvalDataset:
    """Simple dataset for MAE evaluation."""
    def __init__(self, csv_file, pkl_file):
        self.data = pd.read_csv(csv_file)
        with open(pkl_file, 'rb') as f:
            self.keypoints = pickle.load(f)
        
        # Filter valid samples
        valid_ids = [sid for sid in self.data['id'] if sid in self.keypoints]
        self.data = self.data[self.data['id'].isin(valid_ids)].reset_index(drop=True)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sample_id = row['id']
        gloss = row.get('gloss', '')
        
        raw_pose = self.keypoints[sample_id]['keypoints']
        pose, valid_mask = self._normalize(raw_pose)
        velocity = self._compute_velocity(pose, valid_mask)
        pose_combined = np.concatenate([pose, velocity], axis=-1)
        
        return {
            'id': sample_id,
            'gloss': gloss,
            'pose': torch.tensor(pose_combined, dtype=torch.float32),
            'pose_xy': torch.tensor(pose, dtype=torch.float32),  # Just x,y for visualization
            'valid_mask': torch.tensor(valid_mask.any(axis=1), dtype=torch.bool),
        }
    
    def _normalize(self, keypoints, invalid_value=-1.0):
        arr = keypoints.copy().astype(np.float32)
        T, N, C = arr.shape
        valid_mask = ~np.isclose(arr, invalid_value, atol=1e-6).any(axis=-1)
        
        if valid_mask.sum() == 0:
            return np.zeros_like(arr), valid_mask
        
        valid_points = arr[valid_mask]
        center = valid_points.mean(axis=0)
        std = max(1e-6, valid_points.std())
        
        arr[valid_mask] = (arr[valid_mask] - center) / std
        arr[~valid_mask] = 0.0
        
        return arr, valid_mask
    
    def _compute_velocity(self, pose, valid_mask):
        T, N, C = pose.shape
        velocity = np.zeros_like(pose)
        velocity[1:] = pose[1:] - pose[:-1]
        
        valid_velocity_mask = np.zeros((T, N), dtype=bool)
        valid_velocity_mask[1:] = valid_mask[1:] & valid_mask[:-1]
        velocity[~valid_velocity_mask] = 0.0
        
        return velocity


# =============================================================================
# KEYPOINT DEFINITIONS
# =============================================================================

# Body part indices
BODY_PARTS = {
    'right_hand': list(range(0, 21)),
    'left_hand': list(range(21, 42)),
    'face': list(range(42, 70)),
    'body': list(range(70, 86)),
}

# Skeleton connections for visualization
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17),  # Palm
]

BODY_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Arms
    (0, 5), (0, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # Torso/legs
]

BODY_PART_COLORS = {
    'right_hand': '#e74c3c',  # Red
    'left_hand': '#3498db',   # Blue
    'face': '#2ecc71',        # Green
    'body': '#9b59b6',        # Purple
}


# =============================================================================
# MASKING (same as training)
# =============================================================================

def mask_frames(pose, pose_mask, mask_ratio=0.15):
    """Mask random frames."""
    B, T, N, C = pose.shape
    device = pose.device
    
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    
    for b in range(B):
        valid_indices = torch.where(pose_mask[b] > 0)[0]
        num_valid = len(valid_indices)
        
        if num_valid > 2:
            num_mask = max(1, int(num_valid * mask_ratio))
            maskable_indices = valid_indices[1:-1] if num_valid > 2 else valid_indices
            
            if len(maskable_indices) > 0:
                perm = torch.randperm(len(maskable_indices), device=device)[:num_mask]
                mask_indices = maskable_indices[perm]
                mask[b, mask_indices] = True
    
    masked_pose = pose.clone()
    mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).expand_as(pose)
    masked_pose[mask_expanded] = 0.0
    
    return masked_pose, mask


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_skeleton(ax, pose_frame, title="", alpha=1.0, color_by_part=True):
    """Plot a single pose frame with skeleton connections."""
    ax.clear()
    
    # Plot each body part
    for part_name, indices in BODY_PARTS.items():
        color = BODY_PART_COLORS[part_name] if color_by_part else 'blue'
        
        # Get coordinates for this part
        x = pose_frame[indices, 0]
        y = -pose_frame[indices, 1]  # Flip y for display
        
        # Plot keypoints
        valid = ~np.isclose(pose_frame[indices], 0.0).all(axis=1)
        ax.scatter(x[valid], y[valid], c=color, s=20, alpha=alpha, zorder=2)
        
        # Plot connections
        if 'hand' in part_name:
            for (i, j) in HAND_CONNECTIONS:
                if i < len(indices) and j < len(indices):
                    if valid[i] and valid[j]:
                        ax.plot([x[i], x[j]], [y[i], y[j]], c=color, alpha=alpha*0.7, linewidth=1, zorder=1)
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.axis('off')


def visualize_reconstruction(original, masked, reconstructed, mask, frame_idx, save_path=None):
    """Visualize original, masked, and reconstructed poses side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    plot_skeleton(axes[0], original[frame_idx, :, :2].numpy(), "Original")
    
    # Masked (show which frame is masked)
    if mask[frame_idx]:
        axes[1].text(0.5, 0.5, "MASKED", ha='center', va='center', fontsize=20, 
                     transform=axes[1].transAxes, color='red')
        axes[1].set_xlim(-3, 3)
        axes[1].set_ylim(-3, 3)
        axes[1].axis('off')
        axes[1].set_title("Masked Frame")
    else:
        plot_skeleton(axes[1], masked[frame_idx, :, :2].numpy(), "Input (Not Masked)")
    
    # Reconstructed
    plot_skeleton(axes[2], reconstructed[frame_idx, :, :2].numpy(), "Reconstructed")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_sequence_comparison(original, reconstructed, mask, sample_id, save_path=None):
    """Visualize reconstruction quality across entire sequence."""
    T = original.shape[0]
    num_frames_to_show = min(10, T)
    frame_indices = np.linspace(0, T-1, num_frames_to_show, dtype=int)
    
    fig, axes = plt.subplots(3, num_frames_to_show, figsize=(2*num_frames_to_show, 6))
    
    for i, frame_idx in enumerate(frame_indices):
        # Original
        plot_skeleton(axes[0, i], original[frame_idx, :, :2].numpy(), 
                     f"t={frame_idx}" if i == 0 else f"{frame_idx}")
        
        # Masked indicator
        if mask[frame_idx]:
            axes[1, i].add_patch(plt.Rectangle((0, 0), 1, 1, color='red', alpha=0.3, 
                                                transform=axes[1, i].transAxes))
            axes[1, i].text(0.5, 0.5, "M", ha='center', va='center', fontsize=14,
                           transform=axes[1, i].transAxes, color='red', fontweight='bold')
        plot_skeleton(axes[1, i], original[frame_idx, :, :2].numpy() if not mask[frame_idx] 
                     else np.zeros_like(original[frame_idx, :, :2].numpy()), "")
        
        # Reconstructed
        plot_skeleton(axes[2, i], reconstructed[frame_idx, :, :2].numpy(), "")
    
    axes[0, 0].set_ylabel("Original", fontsize=12)
    axes[1, 0].set_ylabel("Masked", fontsize=12)
    axes[2, 0].set_ylabel("Reconstructed", fontsize=12)
    
    plt.suptitle(f"Sample: {sample_id}", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_error_heatmap(errors_by_part, save_path=None):
    """Plot heatmap of reconstruction errors by body part."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    parts = list(errors_by_part.keys())
    samples = list(range(len(errors_by_part[parts[0]])))
    
    data = np.array([errors_by_part[part] for part in parts])
    
    sns.heatmap(data, xticklabels=[f"S{i}" for i in samples], yticklabels=parts,
                cmap='YlOrRd', ax=ax, annot=True, fmt='.3f')
    ax.set_xlabel("Sample")
    ax.set_ylabel("Body Part")
    ax.set_title("Reconstruction Error by Body Part")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_error_distribution(all_errors, save_path=None):
    """Plot distribution of reconstruction errors."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Overall distribution
    axes[0].hist(all_errors['overall'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].axvline(np.mean(all_errors['overall']), color='red', linestyle='--', 
                    label=f"Mean: {np.mean(all_errors['overall']):.4f}")
    axes[0].set_xlabel("MSE")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Overall Reconstruction Error Distribution")
    axes[0].legend()
    
    # By body part (box plot)
    part_data = [all_errors[part] for part in BODY_PARTS.keys()]
    bp = axes[1].boxplot(part_data, labels=list(BODY_PARTS.keys()), patch_artist=True)
    
    colors = [BODY_PART_COLORS[part] for part in BODY_PARTS.keys()]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[1].set_xlabel("Body Part")
    axes[1].set_ylabel("MSE")
    axes[1].set_title("Reconstruction Error by Body Part")
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_temporal_error(temporal_errors, save_path=None):
    """Plot reconstruction error over time (relative position in sequence)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Bin errors by relative position
    num_bins = 20
    bins = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    binned_errors = [[] for _ in range(num_bins)]
    for rel_pos, error in temporal_errors:
        bin_idx = min(int(rel_pos * num_bins), num_bins - 1)
        binned_errors[bin_idx].append(error)
    
    means = [np.mean(b) if b else 0 for b in binned_errors]
    stds = [np.std(b) if b else 0 for b in binned_errors]
    
    ax.plot(bin_centers, means, 'b-', linewidth=2, label='Mean Error')
    ax.fill_between(bin_centers, 
                    np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds),
                    alpha=0.3, color='blue', label='±1 Std')
    
    ax.set_xlabel("Relative Position in Sequence")
    ax.set_ylabel("MSE")
    ax.set_title("Reconstruction Error Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_reconstruction_animation(original, masked, reconstructed, mask, save_path):
    """Create animation comparing original and reconstructed sequences."""
    T = original.shape[0]
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    def update(frame):
        for ax in axes:
            ax.clear()
        
        plot_skeleton(axes[0], original[frame, :, :2].numpy(), "Original")
        
        if mask[frame]:
            axes[1].text(0.5, 0.5, "MASKED", ha='center', va='center', fontsize=16,
                        transform=axes[1].transAxes, color='red', fontweight='bold')
            axes[1].set_xlim(-3, 3)
            axes[1].set_ylim(-3, 3)
            axes[1].axis('off')
            axes[1].set_title(f"Input (Frame {frame})")
        else:
            plot_skeleton(axes[1], original[frame, :, :2].numpy(), f"Input (Frame {frame})")
        
        plot_skeleton(axes[2], reconstructed[frame, :, :2].numpy(), "Reconstructed")
        
        return axes
    
    anim = FuncAnimation(fig, update, frames=T, interval=100, blit=False)
    anim.save(save_path, writer='pillow', fps=10)
    plt.close()


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def compute_reconstruction_metrics(original, reconstructed, mask):
    """Compute detailed reconstruction metrics."""
    metrics = {}
    
    # Overall MSE (only on masked frames)
    if mask.sum() > 0:
        masked_original = original[mask]
        masked_reconstructed = reconstructed[mask]
        metrics['overall_mse'] = F.mse_loss(masked_reconstructed, masked_original).item()
        metrics['overall_mae'] = F.l1_loss(masked_reconstructed, masked_original).item()
    else:
        metrics['overall_mse'] = 0
        metrics['overall_mae'] = 0
    
    # Per-body-part MSE
    for part_name, indices in BODY_PARTS.items():
        if mask.sum() > 0:
            part_original = original[mask][:, indices, :]
            part_reconstructed = reconstructed[mask][:, indices, :]
            metrics[f'{part_name}_mse'] = F.mse_loss(part_reconstructed, part_original).item()
        else:
            metrics[f'{part_name}_mse'] = 0
    
    # Position vs Velocity error (if 4 channels)
    if original.shape[-1] == 4:
        if mask.sum() > 0:
            pos_original = original[mask][:, :, :2]
            pos_reconstructed = reconstructed[mask][:, :, :2]
            vel_original = original[mask][:, :, 2:]
            vel_reconstructed = reconstructed[mask][:, :, 2:]
            
            metrics['position_mse'] = F.mse_loss(pos_reconstructed, pos_original).item()
            metrics['velocity_mse'] = F.mse_loss(vel_reconstructed, vel_original).item()
    
    return metrics


def evaluate_mae_model(model, reconstruction_head, dataset, device, num_samples=None, mask_ratio=0.15):
    """Evaluate MAE model on dataset."""
    model.eval()
    reconstruction_head.eval()
    
    all_metrics = []
    all_errors = {'overall': [], **{part: [] for part in BODY_PARTS.keys()}}
    temporal_errors = []
    
    num_samples = num_samples or len(dataset)
    
    with torch.no_grad():
        for idx in tqdm(range(min(num_samples, len(dataset))), desc="Evaluating"):
            sample = dataset[idx]
            
            pose = sample['pose'].unsqueeze(0).to(device)  # (1, T, 86, 4)
            T = pose.shape[1]
            pose_mask = torch.ones(1, T, device=device)
            
            # Mask
            masked_pose, frame_mask = mask_frames(pose, pose_mask, mask_ratio)
            
            # Forward
            encoder_output = model(masked_pose, attention_mask=pose_mask)
            reconstructed = reconstruction_head(encoder_output)
            
            # Compute metrics
            metrics = compute_reconstruction_metrics(
                pose.squeeze(0), 
                reconstructed.squeeze(0), 
                frame_mask.squeeze(0)
            )
            metrics['sample_id'] = sample['id']
            metrics['gloss'] = sample['gloss']
            metrics['num_masked_frames'] = frame_mask.sum().item()
            all_metrics.append(metrics)
            
            # Collect errors for distribution
            all_errors['overall'].append(metrics['overall_mse'])
            for part in BODY_PARTS.keys():
                all_errors[part].append(metrics[f'{part}_mse'])
            
            # Temporal errors
            masked_indices = torch.where(frame_mask.squeeze(0))[0]
            for t in masked_indices:
                rel_pos = t.item() / T
                frame_error = F.mse_loss(
                    reconstructed[0, t], pose[0, t]
                ).item()
                temporal_errors.append((rel_pos, frame_error))
    
    return all_metrics, all_errors, temporal_errors


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize MAE Pre-training Results")
    
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to pretrained MAE model")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--pose_config", type=str, default="configs/pose_encoder_config.json")
    parser.add_argument("--output_dir", type=str, default="output/mae_visualizations")
    
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to evaluate")
    parser.add_argument("--num_visualize", type=int, default=5,
                        help="Number of samples to visualize in detail")
    parser.add_argument("--mask_ratio", type=float, default=0.15)
    
    parser.add_argument("--create_animations", action="store_true",
                        help="Create reconstruction animations (slow)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("MAE Pre-training Visualization & Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    pose_config = PoseEncoderConfig.from_json_file(args.pose_config)
    model = PoseEncoder(pose_config)
    
    # Load weights
    model_path = os.path.join(args.model_dir, "model.safetensors")
    if not os.path.exists(model_path):
        model_path = os.path.join(args.model_dir, "pytorch_model.bin")
    
    if os.path.exists(model_path):
        if model_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(model_path)
        else:
            state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {model_path}")
    else:
        print(f"Warning: No weights found at {model_path}")
    
    model.to(device)
    model.eval()
    
    # Create reconstruction head
    reconstruction_head = ReconstructionHead(
        hidden_dim=pose_config.hidden_dim,
        num_keypoints=86,
        num_channels=4
    ).to(device)
    
    # Load reconstruction head weights if available
    head_path = os.path.join(args.model_dir, "reconstruction_head.pt")
    if os.path.exists(head_path):
        reconstruction_head.load_state_dict(torch.load(head_path, map_location=device))
        print(f"Loaded reconstruction head from {head_path}")
    else:
        print("Warning: No reconstruction head weights found")
        print("Results may not be accurate - reconstruction head is randomly initialized")
        print("Make sure to save reconstruction_head.pt during MAE training")
    
    # Load dataset
    print("\nLoading dataset...")
    dev_csv = os.path.join(args.data_dir, "dev.csv")
    pkl_file = os.path.join(args.data_dir, "data.pkl")
    dataset = MAEEvalDataset(dev_csv, pkl_file)
    print(f"Loaded {len(dataset)} samples")
    
    # Evaluate
    print("\nEvaluating reconstruction quality...")
    all_metrics, all_errors, temporal_errors = evaluate_mae_model(
        model, reconstruction_head, dataset, device,
        num_samples=args.num_samples,
        mask_ratio=args.mask_ratio
    )
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("RECONSTRUCTION METRICS")
    print("=" * 60)
    
    avg_mse = np.mean([m['overall_mse'] for m in all_metrics])
    avg_mae = np.mean([m['overall_mae'] for m in all_metrics])
    
    print(f"Overall MSE: {avg_mse:.6f}")
    print(f"Overall MAE: {avg_mae:.6f}")
    
    print("\nPer-Body-Part MSE:")
    for part in BODY_PARTS.keys():
        part_mse = np.mean([m[f'{part}_mse'] for m in all_metrics])
        print(f"  {part}: {part_mse:.6f}")
    
    if 'position_mse' in all_metrics[0]:
        pos_mse = np.mean([m['position_mse'] for m in all_metrics])
        vel_mse = np.mean([m['velocity_mse'] for m in all_metrics])
        print(f"\nPosition MSE: {pos_mse:.6f}")
        print(f"Velocity MSE: {vel_mse:.6f}")
    
    # Save metrics to JSON
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump({
            'overall_mse': avg_mse,
            'overall_mae': avg_mae,
            'per_part_mse': {part: np.mean([m[f'{part}_mse'] for m in all_metrics]) 
                           for part in BODY_PARTS.keys()},
            'num_samples': len(all_metrics),
            'mask_ratio': args.mask_ratio,
        }, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Error distribution
    plot_error_distribution(all_errors, 
                           os.path.join(args.output_dir, "error_distribution.png"))
    print("  - error_distribution.png")
    
    # Temporal error
    plot_temporal_error(temporal_errors,
                       os.path.join(args.output_dir, "temporal_error.png"))
    print("  - temporal_error.png")
    
    # Per-sample visualizations
    print(f"\nVisualizing {args.num_visualize} samples...")
    for idx in range(min(args.num_visualize, len(dataset))):
        sample = dataset[idx]
        sample_id = sample['id']
        
        pose = sample['pose'].unsqueeze(0).to(device)
        T = pose.shape[1]
        pose_mask = torch.ones(1, T, device=device)
        
        masked_pose, frame_mask = mask_frames(pose, pose_mask, args.mask_ratio)
        
        with torch.no_grad():
            encoder_output = model(masked_pose, attention_mask=pose_mask)
            reconstructed = reconstruction_head(encoder_output)
        
        # Sequence comparison
        visualize_sequence_comparison(
            pose.squeeze(0).cpu(),
            reconstructed.squeeze(0).cpu(),
            frame_mask.squeeze(0).cpu(),
            sample_id,
            os.path.join(args.output_dir, f"sequence_{sample_id}.png")
        )
        print(f"  - sequence_{sample_id}.png")
        
        # Animation (if requested)
        if args.create_animations:
            create_reconstruction_animation(
                pose.squeeze(0).cpu(),
                masked_pose.squeeze(0).cpu(),
                reconstructed.squeeze(0).cpu(),
                frame_mask.squeeze(0).cpu(),
                os.path.join(args.output_dir, f"animation_{sample_id}.gif")
            )
            print(f"  - animation_{sample_id}.gif")
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
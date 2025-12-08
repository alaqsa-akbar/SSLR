"""
Simple Single-Frame MAE Visualization

Visualizes one frame comparing:
- Original pose (ground truth)
- Reconstructed pose
- Overlay of both

Usage:
    python visualize_mae_frame.py \
        --model_dir output/pretrained_mae/best_model \
        --data_dir data \
        --sample_idx 0 \
        --frame_idx 50
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import argparse

from modules.pose_encoder import PoseEncoder, PoseEncoderConfig


# =============================================================================
# RECONSTRUCTION HEAD
# =============================================================================

class ReconstructionHead(nn.Module):
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
# BODY PART DEFINITIONS
# =============================================================================

BODY_PARTS = {
    'right_hand': (list(range(0, 21)), '#e74c3c'),      # Red
    'left_hand': (list(range(21, 42)), '#3498db'),      # Blue
    'face': (list(range(42, 70)), '#2ecc71'),           # Green
    'body': (list(range(70, 86)), '#9b59b6'),           # Purple
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),           # Palm
]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_sample(data_dir, sample_idx):
    """Load a single sample."""
    dev_csv = os.path.join(data_dir, "dev.csv")
    pkl_file = os.path.join(data_dir, "data.pkl")
    
    data = pd.read_csv(dev_csv)
    with open(pkl_file, 'rb') as f:
        keypoints = pickle.load(f)
    
    # Filter valid
    valid_ids = [sid for sid in data['id'] if sid in keypoints]
    data = data[data['id'].isin(valid_ids)].reset_index(drop=True)
    
    row = data.iloc[sample_idx]
    sample_id = row['id']
    gloss = row.get('gloss', '')
    
    raw_pose = keypoints[sample_id]['keypoints']
    
    # Normalize
    arr = raw_pose.copy().astype(np.float32)
    valid_mask = ~np.isclose(arr, -1.0, atol=1e-6).any(axis=-1)
    
    if valid_mask.sum() > 0:
        valid_points = arr[valid_mask]
        center = valid_points.mean(axis=0)
        std = max(1e-6, valid_points.std())
        arr[valid_mask] = (arr[valid_mask] - center) / std
        arr[~valid_mask] = 0.0
    
    # Compute velocity
    velocity = np.zeros_like(arr)
    velocity[1:] = arr[1:] - arr[:-1]
    
    pose_combined = np.concatenate([arr, velocity], axis=-1)
    
    return {
        'id': sample_id,
        'gloss': gloss,
        'pose': torch.tensor(pose_combined, dtype=torch.float32),
        'num_frames': pose_combined.shape[0]
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_skeleton(ax, pose_frame, title="", alpha=1.0, marker='o', markersize=15, linewidth=1.5):
    """Plot skeleton for one frame."""
    for part_name, (indices, color) in BODY_PARTS.items():
        x = pose_frame[indices, 0]
        y = -pose_frame[indices, 1]  # Flip y
        
        # Valid points (non-zero)
        valid = ~np.isclose(pose_frame[indices], 0.0).all(axis=1)
        
        # Plot keypoints
        ax.scatter(x[valid], y[valid], c=color, s=markersize, alpha=alpha, 
                   marker=marker, zorder=3, edgecolors='white', linewidths=0.5)
        
        # Plot hand connections
        if 'hand' in part_name:
            for (i, j) in HAND_CONNECTIONS:
                if i < len(indices) and j < len(indices) and valid[i] and valid[j]:
                    ax.plot([x[i], x[j]], [y[i], y[j]], c=color, 
                           alpha=alpha*0.7, linewidth=linewidth, zorder=2)
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')


def visualize_single_frame(original, reconstructed, frame_idx, sample_id, gloss, 
                           is_masked=True, save_path=None):
    """
    Visualize single frame: Original vs Reconstructed vs Overlay
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Sample: {sample_id} | Frame: {frame_idx} | Gloss: {gloss}", 
                 fontsize=16, fontweight='bold')
    
    orig_frame = original[frame_idx, :, :2]  # Just x, y
    recon_frame = reconstructed[frame_idx, :, :2]
    
    # 1. Original (Ground Truth)
    plot_skeleton(axes[0], orig_frame, "Ground Truth", alpha=1.0)
    
    # 2. Masked Input (what model sees)
    if is_masked:
        axes[1].text(0.5, 0.5, "MASKED\n(zeros)", ha='center', va='center', 
                    fontsize=20, transform=axes[1].transAxes, color='red',
                    fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat'))
        axes[1].set_xlim(-3, 3)
        axes[1].set_ylim(-3, 3)
        axes[1].set_aspect('equal')
        axes[1].set_title("Model Input", fontsize=14, fontweight='bold')
        axes[1].axis('off')
    else:
        plot_skeleton(axes[1], orig_frame, "Model Input (Not Masked)", alpha=1.0)
    
    # 3. Reconstructed
    plot_skeleton(axes[2], recon_frame, "Reconstructed", alpha=1.0)
    
    # 4. Overlay (Ground Truth + Reconstructed)
    axes[3].set_title("Overlay (GT=solid, Recon=X)", fontsize=14, fontweight='bold')
    
    for part_name, (indices, color) in BODY_PARTS.items():
        # Ground truth (circles)
        x_gt = orig_frame[indices, 0]
        y_gt = -orig_frame[indices, 1]
        valid_gt = ~np.isclose(orig_frame[indices], 0.0).all(axis=1)
        axes[3].scatter(x_gt[valid_gt], y_gt[valid_gt], c=color, s=50, 
                       marker='o', alpha=0.8, zorder=3, label=f'{part_name} (GT)')
        
        # Reconstructed (X markers)
        x_re = recon_frame[indices, 0]
        y_re = -recon_frame[indices, 1]
        valid_re = ~np.isclose(recon_frame[indices], 0.0).all(axis=1)
        axes[3].scatter(x_re[valid_re], y_re[valid_re], c=color, s=50, 
                       marker='x', alpha=0.8, zorder=4, linewidths=2)
        
        # Draw lines connecting GT to Recon for same keypoints
        for idx in range(len(indices)):
            if valid_gt[idx] and valid_re[idx]:
                axes[3].plot([x_gt[idx], x_re[idx]], [y_gt[idx], y_re[idx]], 
                           c='gray', alpha=0.3, linewidth=1, linestyle='--', zorder=1)
    
    axes[3].set_xlim(-3, 3)
    axes[3].set_ylim(-3, 3)
    axes[3].set_aspect('equal')
    axes[3].axis('off')
    
    # Add error metric
    mse = np.mean((orig_frame - recon_frame) ** 2)
    mae = np.mean(np.abs(orig_frame - recon_frame))
    
    fig.text(0.5, 0.02, f"MSE: {mse:.6f} | MAE: {mae:.6f}", 
             ha='center', fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved to {save_path}")
        plt.close()
    else:
        plt.show()
    
    return mse, mae


def visualize_multiple_frames(original, reconstructed, mask, sample_id, gloss, 
                              num_frames=8, save_path=None):
    """
    Visualize multiple frames in a grid.
    Row 1: Original
    Row 2: Reconstructed  
    Row 3: Error heatmap per keypoint
    """
    T = original.shape[0]
    frame_indices = np.linspace(0, T-1, num_frames, dtype=int)
    
    fig, axes = plt.subplots(3, num_frames, figsize=(2.5*num_frames, 8))
    fig.suptitle(f"Sample: {sample_id} | Gloss: {gloss}", fontsize=14, fontweight='bold')
    
    for i, fidx in enumerate(frame_indices):
        orig = original[fidx, :, :2]
        recon = reconstructed[fidx, :, :2]
        is_masked = mask[fidx].item() if isinstance(mask[fidx], torch.Tensor) else mask[fidx]
        
        # Row 1: Original
        plot_skeleton(axes[0, i], orig, f"t={fidx}", markersize=10, linewidth=1)
        if is_masked:
            axes[0, i].add_patch(plt.Rectangle((-3, -3), 6, 6, color='red', 
                                               alpha=0.1, zorder=0))
        
        # Row 2: Reconstructed
        plot_skeleton(axes[1, i], recon, "", markersize=10, linewidth=1)
        
        # Row 3: Per-keypoint error
        error = np.sqrt(np.sum((orig - recon) ** 2, axis=1))  # L2 error per keypoint
        
        # Color by error
        for part_name, (indices, color) in BODY_PARTS.items():
            x = orig[indices, 0]
            y = -orig[indices, 1]
            err = error[indices]
            valid = ~np.isclose(orig[indices], 0.0).all(axis=1)
            
            sc = axes[2, i].scatter(x[valid], y[valid], c=err[valid], 
                                    cmap='YlOrRd', s=20, vmin=0, vmax=0.5, zorder=3)
        
        axes[2, i].set_xlim(-3, 3)
        axes[2, i].set_ylim(-3, 3)
        axes[2, i].set_aspect('equal')
        axes[2, i].axis('off')
    
    # Labels
    axes[0, 0].set_ylabel("Original", fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel("Reconstructed", fontsize=12, fontweight='bold')
    axes[2, 0].set_ylabel("Error", fontsize=12, fontweight='bold')
    
    # Colorbar
    cbar = fig.colorbar(sc, ax=axes[2, :], orientation='horizontal', 
                        fraction=0.05, pad=0.1, label='L2 Error')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved to {save_path}")
        plt.close()
    else:
        plt.show()


def mask_frame(pose, frame_idx):
    """Mask a specific frame."""
    masked = pose.clone()
    masked[frame_idx] = 0.0
    mask = torch.zeros(pose.shape[0], dtype=torch.bool)
    mask[frame_idx] = True
    return masked, mask


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Single MAE Frame")
    
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--pose_config", type=str, default="configs/pose_encoder_config.json")
    parser.add_argument("--output_dir", type=str, default="output/mae_frame_vis")
    
    parser.add_argument("--sample_idx", type=int, default=0, help="Which sample to visualize")
    parser.add_argument("--frame_idx", type=int, default=None, help="Which frame (None = middle)")
    parser.add_argument("--mask_ratio", type=float, default=0.15, help="For multi-frame vis")
    
    parser.add_argument("--mode", type=str, default="single", 
                        choices=["single", "multi", "both"])
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("MAE Single Frame Visualization")
    print("=" * 60)
    
    # Load model
    print("Loading model...")
    pose_config = PoseEncoderConfig.from_json_file(args.pose_config)
    model = PoseEncoder(pose_config)
    
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
        print(f"✓ Loaded encoder from {model_path}")
    
    model.to(device)
    model.eval()
    
    # Load reconstruction head
    reconstruction_head = ReconstructionHead(
        hidden_dim=pose_config.hidden_dim,
        num_keypoints=86,
        num_channels=4
    ).to(device)
    
    head_path = os.path.join(args.model_dir, "reconstruction_head.pt")
    if os.path.exists(head_path):
        reconstruction_head.load_state_dict(torch.load(head_path, map_location=device))
        print(f"✓ Loaded reconstruction head from {head_path}")
    else:
        print("⚠ No reconstruction head found - using random init")
    
    reconstruction_head.eval()
    
    # Load sample
    print(f"Loading sample {args.sample_idx}...")
    sample = load_sample(args.data_dir, args.sample_idx)
    sample_id = sample['id']
    gloss = sample['gloss']
    pose = sample['pose']  # (T, 86, 4)
    T = pose.shape[0]
    
    print(f"  Sample ID: {sample_id}")
    print(f"  Gloss: {gloss}")
    print(f"  Frames: {T}")
    
    frame_idx = args.frame_idx if args.frame_idx is not None else T // 2
    print(f"  Visualizing frame: {frame_idx}")
    
    # Mask and reconstruct
    pose_input = pose.unsqueeze(0).to(device)  # (1, T, 86, 4)
    pose_mask = torch.ones(1, T, device=device)
    
    if args.mode in ["single", "both"]:
        # Mask single frame
        masked_pose, frame_mask = mask_frame(pose_input.squeeze(0), frame_idx)
        masked_pose = masked_pose.unsqueeze(0)
        
        with torch.no_grad():
            encoder_out = model(masked_pose, attention_mask=pose_mask)
            reconstructed = reconstruction_head(encoder_out)
        
        # Visualize
        mse, mae = visualize_single_frame(
            pose.numpy(),
            reconstructed.squeeze(0).cpu().numpy(),
            frame_idx,
            sample_id,
            gloss,
            is_masked=True,
            save_path=os.path.join(args.output_dir, f"frame_{sample_id}_{frame_idx}.png")
        )
        print(f"\n  Single Frame MSE: {mse:.6f}, MAE: {mae:.6f}")
    
    if args.mode in ["multi", "both"]:
        # Mask multiple frames randomly
        from visualize_mae import mask_frames
        masked_pose, frame_mask = mask_frames(pose_input, pose_mask, args.mask_ratio)
        
        with torch.no_grad():
            encoder_out = model(masked_pose, attention_mask=pose_mask)
            reconstructed = reconstruction_head(encoder_out)
        
        visualize_multiple_frames(
            pose.numpy(),
            reconstructed.squeeze(0).cpu().numpy(),
            frame_mask.squeeze(0).cpu(),
            sample_id,
            gloss,
            num_frames=8,
            save_path=os.path.join(args.output_dir, f"multi_{sample_id}.png")
        )
    
    print("\n" + "=" * 60)
    print(f"Visualizations saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
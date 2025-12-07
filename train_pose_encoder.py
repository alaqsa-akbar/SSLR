"""
Contrastive Pre-training for Sign Language Pose Encoder

Updated for new architecture:
- Grouped GCN (hands, face, body)
- Dual-stream (pose + velocity)
- Spatial Attention Pooling

Key features:
1. Caches UMT5 embeddings for unique SENTENCES (~1000 items)
2. Trains Pose Encoder to align with text embeddings
3. Uses InfoNCE loss with learnable temperature
4. CLIP-style linear projection (prevents projection head from dominating)

After pre-training, use the encoder weights for hybrid CTC+Attention training.

Usage:
    accelerate launch --num_processes 2 train_contrastive.py \
        --epochs 75 \
        --batch_size 64 \
        --lr 5e-5 \
        --output_dir output/pretrained_encoder

    # Single GPU:
    python train_contrastive.py \
        --epochs 75 \
        --batch_size 64 \
        --output_dir output/pretrained_encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, UMT5EncoderModel, UMT5Config
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
# PROJECTION HEADS (CLIP-style: simple linear, no nonlinearity)
# =============================================================================

class LinearProjection(nn.Module):
    """CLIP-style linear projection (no hidden layer, no activation)."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim, bias=False)
        # Initialize with small values
        nn.init.normal_(self.proj.weight, std=0.02)
    
    def forward(self, x):
        return self.proj(x)


# =============================================================================
# CONTRASTIVE DATASET
# =============================================================================

class ContrastiveDataset(Dataset):
    """
    Dataset for contrastive pre-training.
    Returns pose sequences and their corresponding sentence text.
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
        
        # Default augmentation config
        self.aug_config = {
            'noise_sigma': 0.02,
            'scale_range': (0.85, 1.15),
            'shift_range': 0.1,
            'rotation_prob': 0.5,
            'rotation_max_angle': 12,
            'speed_prob': 0.5,
            'speed_range': (0.85, 1.15),
            'frame_drop_prob': 0.3,
            'frame_drop_rate': 0.08,
            'frame_drop_max_consecutive': 1,
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
        """Apply data augmentation."""
        from scipy import interpolate
        
        # Speed perturbation
        if np.random.random() < self.aug_config['speed_prob']:
            T, N, C = pose.shape
            if T >= 4:
                speed = np.random.uniform(*self.aug_config['speed_range'])
                new_T = max(4, min(int(T / speed), int(T * 1.5)))
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
        
        # Frame dropping
        if np.random.random() < self.aug_config['frame_drop_prob'] and T >= 10:
            keep_mask = np.ones(T, dtype=bool)
            consecutive_drops = 0
            
            for t in range(2, T - 2):
                if np.random.random() < self.aug_config['frame_drop_rate'] and consecutive_drops < self.aug_config['frame_drop_max_consecutive']:
                    keep_mask[t] = False
                    consecutive_drops += 1
                else:
                    consecutive_drops = 0
            
            if keep_mask.sum() >= 0.8 * T:
                pose = pose[keep_mask]
                valid_mask = valid_mask[keep_mask]
        
        T, N, C = pose.shape
        
        # Rotation
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
        
        # Noise, scale, shift
        if self.aug_config['noise_sigma'] > 0:
            noise = np.random.normal(0, self.aug_config['noise_sigma'], pose.shape).astype(np.float32)
            pose[valid_mask] += noise[valid_mask]
        
        scale = np.random.uniform(*self.aug_config['scale_range'])
        pose[valid_mask] *= scale
        
        shift = np.random.uniform(-self.aug_config['shift_range'], self.aug_config['shift_range'], size=(1, 1, C)).astype(np.float32)
        pose[valid_mask] += shift.squeeze()
        
        return pose, valid_mask
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sample_id = row['id']
        sentence = str(row['gloss']).strip()  # This is actually the full sentence
        
        # Get raw keypoints
        raw_pose = self.keypoints.get(sample_id, {}).get('keypoints', np.zeros((10, 86, 2), dtype=np.float32))
        
        # Normalize
        pose, valid_mask = self._normalize(raw_pose)
        
        # Augment (if training)
        if self.use_augmentation and self.training:
            pose, valid_mask = self._apply_augmentation(pose, valid_mask)
        
        # Compute velocity
        velocity = self._compute_velocity(pose, valid_mask)
        pose_combined = np.concatenate([pose, velocity], axis=-1)
        
        # Truncate
        if pose_combined.shape[0] > self.max_frames:
            pose_combined = pose_combined[:self.max_frames]
        
        return {
            'id': sample_id,
            'pose': torch.tensor(pose_combined, dtype=torch.float32),
            'sentence': sentence,
            'num_frames': pose_combined.shape[0]
        }


def collate_fn_contrastive(batch):
    """Collate function for contrastive learning."""
    poses = [x['pose'] for x in batch]
    ids = [x['id'] for x in batch]
    sentences = [x['sentence'] for x in batch]
    
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
        'sentence': sentences
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_text_encoder(model_dir, device, accelerator):
    """Load and freeze UMT5 text encoder."""
    umt5_weights_path = os.path.join(model_dir, "umt5-base.bin")
    
    try:
        if os.path.exists(umt5_weights_path):
            if accelerator.is_main_process:
                print(f"Loading UMT5 weights from: {umt5_weights_path}")
            config = UMT5Config.from_pretrained(model_dir)
            config.d_model = 768
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
            num_layers=12, num_heads=12, dropout_rate=0.1
        )
        text_encoder = UMT5EncoderModel(config)
    
    text_encoder.eval()
    for param in text_encoder.parameters():
        param.requires_grad = False
    
    return text_encoder


def pool_text_embeddings(embeddings, attention_mask):
    """Mean pool text embeddings."""
    mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def cache_sentence_embeddings(text_encoder, tokenizer, train_csv, dev_csv, device, accelerator):
    """Cache text embeddings for all unique SENTENCES from the data."""
    if accelerator.is_main_process:
        print("Caching text embeddings for unique sentences...")
    
    # Load sentences from CSVs
    train_df = pd.read_csv(train_csv)
    dev_df = pd.read_csv(dev_csv)
    
    # Get unique sentences from both train and dev
    all_sentences = pd.concat([train_df['gloss'], dev_df['gloss']]).dropna().unique()
    unique_sentences = sorted([str(s).strip() for s in all_sentences])
    
    if accelerator.is_main_process:
        print(f"Found {len(unique_sentences)} unique sentences")
    
    embeddings = []
    batch_size = 32
    
    iterator = range(0, len(unique_sentences), batch_size)
    if accelerator.is_main_process:
        iterator = tqdm(iterator, desc="Caching embeddings")
    
    for i in iterator:
        batch_text = unique_sentences[i:i + batch_size]
        
        encoded = tokenizer(
            batch_text,
            padding='max_length',
            truncation=True,
            max_length=64,
            return_tensors='pt'
        )
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)
        
        with torch.no_grad():
            outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled = pool_text_embeddings(outputs.last_hidden_state, attention_mask)
            embeddings.append(pooled.cpu())
    
    cached_embeddings = torch.cat(embeddings, dim=0)
    sentence_to_idx = {sent: i for i, sent in enumerate(unique_sentences)}
    
    if accelerator.is_main_process:
        print(f"Cached {len(cached_embeddings)} embeddings, shape: {cached_embeddings.shape}")
    
    return cached_embeddings, sentence_to_idx


# =============================================================================
# MAIN TRAINING
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Contrastive Pre-training for Pose Encoder")
    
    # Data
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model_dir", type=str, default="models/umt5")
    parser.add_argument("--pose_config", type=str, default="configs/pose_encoder_config.json")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="output/pretrained_encoder")
    parser.add_argument("--wandb_project", type=str, default=None)
    
    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    
    # Model
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--learnable_temperature", action="store_true", default=True)
    
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
                init_kwargs={"wandb": {"name": f"contrastive-pretrain"}}
            )
        
        print("=" * 70)
        print("CONTRASTIVE PRE-TRAINING (SENTENCE-LEVEL)")
        print("=" * 70)
        print(f"Pose Config: {args.pose_config}")
        print(f"Projection Dim: {args.projection_dim}")
        print(f"Temperature: {args.temperature} (learnable: {args.learnable_temperature})")
        print(f"Batch Size: {args.batch_size}")
        print(f"Learning Rate: {args.lr}")
        print("=" * 70)
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    except:
        tokenizer = AutoTokenizer.from_pretrained("google/umt5-base")
    
    # Load and cache text encoder
    text_encoder = load_text_encoder(args.model_dir, device, accelerator)
    text_encoder.to(device)
    
    # Cache SENTENCE embeddings (not glosses!)
    train_csv = os.path.join(args.data_dir, "train.csv")
    dev_csv = os.path.join(args.data_dir, "dev.csv")
    
    cached_embeddings, sentence_to_idx = cache_sentence_embeddings(
        text_encoder, tokenizer, train_csv, dev_csv, device, accelerator
    )
    cached_embeddings = cached_embeddings.to(device)
    
    # Free text encoder memory
    del text_encoder
    torch.cuda.empty_cache()
    
    # Load datasets
    if accelerator.is_main_process:
        print("\nLoading datasets...")
    
    train_dataset = ContrastiveDataset(
        train_csv,
        os.path.join(args.data_dir, "data.pkl"),
        use_augmentation=True
    )
    
    dev_dataset = ContrastiveDataset(
        dev_csv,
        os.path.join(args.data_dir, "data.pkl"),
        use_augmentation=False
    )
    dev_dataset.set_training(False)
    
    # Build ID -> sentence index mapping
    def build_id_map(dataset):
        mapping = {}
        missing = 0
        for idx in range(len(dataset)):
            row = dataset.data.iloc[idx]
            sid = row['id']
            sentence = str(row['gloss']).strip()
            if sentence in sentence_to_idx:
                mapping[sid] = sentence_to_idx[sentence]
            else:
                mapping[sid] = 0  # Fallback (shouldn't happen)
                missing += 1
        if missing > 0:
            print(f"Warning: {missing} samples have unknown sentences")
        return mapping
    
    train_id_map = build_id_map(train_dataset)
    dev_id_map = build_id_map(dev_dataset)
    
    if accelerator.is_main_process:
        print(f"Train samples: {len(train_dataset)}")
        print(f"Dev samples: {len(dev_dataset)}")
        print(f"Unique sentences: {len(sentence_to_idx)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_contrastive,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_contrastive,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    # Initialize models
    if accelerator.is_main_process:
        print("\nInitializing models...")
    
    pose_config = PoseEncoderConfig.from_json_file(args.pose_config)
    pose_encoder = PoseEncoder(pose_config)
    
    # CLIP-style linear projections (simple, forces encoder to learn)
    pose_projection = LinearProjection(
        input_dim=pose_config.hidden_dim,
        output_dim=args.projection_dim
    )
    
    text_projection = LinearProjection(
        input_dim=768,  # UMT5 hidden dim
        output_dim=args.projection_dim
    )
    
    # Learnable temperature (CLIP-style)
    # Initialize to ln(1/0.07) ≈ 2.66
    log_temperature = nn.Parameter(torch.ones([]) * np.log(1 / args.temperature))
    
    # Count parameters
    num_pose_params = sum(p.numel() for p in pose_encoder.parameters())
    num_proj_params = sum(p.numel() for p in pose_projection.parameters()) + sum(p.numel() for p in text_projection.parameters())
    
    if accelerator.is_main_process:
        print(f"Pose encoder parameters: {num_pose_params:,}")
        print(f"Projection parameters: {num_proj_params:,}")
        print(f"Total trainable: {num_pose_params + num_proj_params:,}")
    
    # Optimizer with different LR for projections (optional, can help)
    optimizer = torch.optim.AdamW([
        {'params': pose_encoder.parameters(), 'lr': args.lr},
        {'params': pose_projection.parameters(), 'lr': args.lr},
        {'params': text_projection.parameters(), 'lr': args.lr},
        {'params': [log_temperature], 'lr': args.lr * 10}  # Temperature can learn faster
    ], weight_decay=args.weight_decay)
    
    # Scheduler
    num_training_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * num_training_steps),
        num_training_steps=num_training_steps
    )
    
    # Prepare for distributed training
    pose_encoder, pose_projection, text_projection, optimizer, train_loader, dev_loader, scheduler = accelerator.prepare(
        pose_encoder, pose_projection, text_projection, optimizer, train_loader, dev_loader, scheduler
    )
    
    # Move log_temperature to device
    log_temperature = log_temperature.to(device)
    
    # Training loop
    global_step = 0
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        pose_encoder.train()
        pose_projection.train()
        text_projection.train()
        
        total_loss = 0
        total_acc = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}") if accelerator.is_main_process else train_loader
        
        for batch in pbar:
            optimizer.zero_grad()
            
            # Encode poses
            pose = batch['pose']
            pose_mask = batch['pose_mask']
            
            # Forward through encoder
            pose_features = pose_encoder(pose, attention_mask=pose_mask)  # (B, T, H)
            
            # Pool over time (mean pooling with mask)
            mask_expanded = pose_mask.unsqueeze(-1)  # (B, T, 1)
            pose_pooled = (pose_features * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
            
            # Project to contrastive space (CLIP-style linear)
            pose_embeds = F.normalize(pose_projection(pose_pooled), p=2, dim=1)
            
            # Project cached text embeddings
            all_text_embeds = F.normalize(text_projection(cached_embeddings), p=2, dim=1)
            
            # Get target indices
            target_indices = torch.tensor(
                [train_id_map[sid] for sid in batch['id']],
                device=device
            )
            
            # Compute logits with learnable temperature
            logit_scale = log_temperature.exp().clamp(max=100)  # Clamp for stability
            logits = logit_scale * torch.matmul(pose_embeds, all_text_embeds.T)
            
            # Cross-entropy loss
            loss = F.cross_entropy(logits, target_indices)
            
            # Backward
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(
                list(pose_encoder.parameters()) + list(pose_projection.parameters()) + list(text_projection.parameters()),
                args.max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            
            # Metrics
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                acc = (preds == target_indices).float().mean().item()
            
            total_loss += loss.item()
            total_acc += acc
            num_batches += 1
            global_step += 1
            
            if accelerator.is_main_process and global_step % 20 == 0:
                temp = 1/logit_scale.item()
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{acc:.3f}",
                    'temp': f"{1/temp:.2f}"
                })
                if args.wandb_project:
                    accelerator.log({
                        'train/loss': loss.item(),
                        'train/acc': acc,
                        'train/temperature': temp,
                        'train/lr': scheduler.get_last_lr()[0]
                    }, step=global_step)
        
        avg_train_loss = total_loss / num_batches
        avg_train_acc = total_acc / num_batches
        
        if accelerator.is_main_process:
            current_temp = log_temperature.exp().item()
            print(f"\nEpoch {epoch+1}")
            print(f"  Train - Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.3f}, Temp: {current_temp:.2f}")
        
        # Validation
        if (epoch + 1) % args.eval_every == 0:
            pose_encoder.eval()
            pose_projection.eval()
            text_projection.eval()
            
            total_val_loss = 0
            total_val_acc = 0
            num_val_batches = 0
            
            with torch.no_grad():
                all_text_embeds = F.normalize(text_projection(cached_embeddings), p=2, dim=1)
                logit_scale = log_temperature.exp().clamp(max=100)
                
                for batch in dev_loader:
                    pose = batch['pose']
                    pose_mask = batch['pose_mask']
                    
                    pose_features = pose_encoder(pose, attention_mask=pose_mask)
                    mask_expanded = pose_mask.unsqueeze(-1)
                    pose_pooled = (pose_features * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
                    
                    pose_embeds = F.normalize(pose_projection(pose_pooled), p=2, dim=1)
                    
                    target_indices = torch.tensor(
                        [dev_id_map[sid] for sid in batch['id']],
                        device=device
                    )
                    
                    logits = logit_scale * torch.matmul(pose_embeds, all_text_embeds.T)
                    loss = F.cross_entropy(logits, target_indices)
                    
                    preds = logits.argmax(dim=1)
                    acc = (preds == target_indices).float().mean().item()
                    
                    total_val_loss += loss.item()
                    total_val_acc += acc
                    num_val_batches += 1
            
            avg_val_loss = total_val_loss / num_val_batches
            avg_val_acc = total_val_acc / num_val_batches
            
            if accelerator.is_main_process:
                print(f"  Val   - Loss: {avg_val_loss:.4f}, Acc: {avg_val_acc:.3f}")
                if args.wandb_project:
                    accelerator.log({
                        'val/loss': avg_val_loss,
                        'val/acc': avg_val_acc,
                    }, step=global_step)
                
                # Save best model
                if avg_val_acc > best_val_acc:
                    best_val_acc = avg_val_acc
                    
                    save_path = os.path.join(args.output_dir, "best_model")
                    os.makedirs(save_path, exist_ok=True)
                    
                    # Save encoder only (what we'll use for downstream)
                    unwrapped_encoder = accelerator.unwrap_model(pose_encoder)
                    unwrapped_encoder.save_pretrained(save_path)
                    
                    # Save config
                    pose_config.save_pretrained(save_path)
                    
                    # Save training info
                    with open(os.path.join(save_path, "training_info.json"), 'w') as f:
                        json.dump({
                            'epoch': epoch + 1,
                            'val_loss': avg_val_loss,
                            'val_acc': avg_val_acc,
                            'num_sentences': len(sentence_to_idx),
                            'projection_dim': args.projection_dim,
                            'final_temperature': log_temperature.exp().item(),
                        }, f, indent=2)
                    
                    print(f"  ✓ Saved best model (Acc: {avg_val_acc:.4f})")
        
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
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        print(f"Number of sentence classes: {len(sentence_to_idx)}")
        print(f"Models saved to: {args.output_dir}")
        print("=" * 70)
        print("\nTo use pretrained encoder for hybrid training:")
        print(f"  python train_final.py --pretrained_encoder {os.path.join(args.output_dir, 'best_model')}")
    
    if args.wandb_project:
        accelerator.end_training()


if __name__ == "__main__":
    train()
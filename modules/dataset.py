import torch
from torch.utils.data import Dataset
import pandas as pd
import pickle
import numpy as np
import json
from scipy import interpolate


class GlossTokenizer:
    """Simple whitespace tokenizer for glosses."""
    def __init__(self, vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.token_to_id = json.load(f)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        
        self.pad_token_id = self.token_to_id.get("<pad>", 0)
        self.sos_token_id = self.token_to_id.get("<sos>", 1)
        self.eos_token_id = self.token_to_id.get("<eos>", 2)
        self.unk_token_id = self.token_to_id.get("<unk>", 3)
        self.vocab_size = len(self.token_to_id)
    
    def encode_for_ctc(self, sentence):
        """Encode for CTC loss - NO special tokens."""
        if not isinstance(sentence, str):
            return []
        tokens = sentence.strip().split()
        ids = [self.token_to_id.get(t, self.unk_token_id) for t in tokens]
        return ids
    
    def encode_for_decoder(self, sentence):
        """Encode for decoder/cross-entropy loss - with EOS token."""
        if not isinstance(sentence, str):
            return []
        tokens = sentence.strip().split()
        ids = [self.token_to_id.get(t, self.unk_token_id) for t in tokens]
        ids.append(self.eos_token_id)
        return ids
        
    def encode(self, sentence, add_special_tokens=True):
        """Legacy encode method for backward compatibility."""
        if add_special_tokens:
            return self.encode_for_decoder(sentence)
        else:
            return self.encode_for_ctc(sentence)
    
    def decode(self, ids):
        """Decode token IDs back to text."""
        tokens = []
        for i in ids:
            i = int(i)
            if i in [self.pad_token_id, self.sos_token_id]:
                continue
            if i == self.eos_token_id:
                break 
            token = self.id_to_token.get(i, "")
            if token:
                tokens.append(token)
        return " ".join(tokens)


def normalize_keypoints(keypoints, invalid_value=-1.0):
    """
    Robust normalization for keypoints using global standardization.
    
    Args:
        keypoints: (T, N, 2) array of (x, y) coordinates
        invalid_value: Value used to mark invalid points (default: -1.0)
    
    Returns:
        normalized: (T, N, 2) normalized keypoints with invalid points set to 0
        valid_mask: (T, N) boolean mask of valid points
    """
    arr = keypoints.copy().astype(np.float32)
    T, N, C = arr.shape
    
    # Create validity mask
    valid_mask = ~np.isclose(arr, invalid_value, atol=1e-6).any(axis=-1)
    num_valid = valid_mask.sum()
    
    if num_valid == 0:
        return np.zeros_like(arr), valid_mask
    
    # Extract valid points for statistics
    valid_points = arr[valid_mask]
    
    # Compute global center and scale
    center = valid_points.mean(axis=0)
    std = valid_points.std()
    scale = max(1e-6, std)
    
    # Normalize
    arr[valid_mask] = (arr[valid_mask] - center) / scale
    arr[~valid_mask] = 0.0
    
    return arr, valid_mask


def compute_velocity(pose, valid_mask):
    """
    Compute velocity (temporal derivative) of keypoints.
    
    Args:
        pose: (T, N, 2) normalized keypoints
        valid_mask: (T, N) boolean mask of valid points
    
    Returns:
        velocity: (T, N, 2) velocity with invalid transitions set to 0
    """
    T, N, C = pose.shape
    velocity = np.zeros_like(pose)
    
    velocity[1:] = pose[1:] - pose[:-1]
    
    # Velocity valid only when both current and previous frames are valid
    valid_velocity_mask = np.zeros((T, N), dtype=bool)
    valid_velocity_mask[1:] = valid_mask[1:] & valid_mask[:-1]
    
    velocity[~valid_velocity_mask] = 0.0
    velocity[0] = 0.0
    
    return velocity


# =============================================================================
# AUGMENTATION FUNCTIONS
# =============================================================================

def augment_rotation(pose, valid_mask, max_angle_degrees=12):
    """
    Apply random rotation around the body center.
    
    Rotation is applied around the centroid of valid keypoints,
    simulating camera tilt or signer leaning.
    
    Args:
        pose: (T, N, 2) normalized keypoints
        valid_mask: (T, N) boolean mask
        max_angle_degrees: Maximum rotation angle (±)
    
    Returns:
        rotated_pose: (T, N, 2) rotated keypoints
    """
    if valid_mask.sum() == 0:
        return pose
    
    angle_deg = np.random.uniform(-max_angle_degrees, max_angle_degrees)
    angle_rad = np.radians(angle_deg)
    
    # Rotation matrix
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ], dtype=np.float32)
    
    # Compute center of rotation (mean of all valid points)
    valid_points = pose[valid_mask]
    center = valid_points.mean(axis=0)
    
    # Apply rotation
    rotated = pose.copy()
    # Center, rotate, uncenter
    centered = pose[valid_mask] - center
    rotated_points = centered @ rotation_matrix.T
    rotated[valid_mask] = rotated_points + center
    
    return rotated


def augment_speed(pose, valid_mask, speed_range=(0.85, 1.15)):
    """
    Apply speed perturbation via temporal interpolation.
    
    Speed < 1.0: Slower signing (more frames)
    Speed > 1.0: Faster signing (fewer frames)
    
    Args:
        pose: (T, N, 2) normalized keypoints
        valid_mask: (T, N) boolean mask
        speed_range: (min_speed, max_speed) tuple
    
    Returns:
        resampled_pose: (T_new, N, 2) resampled keypoints
        resampled_mask: (T_new, N) resampled mask
    """
    T, N, C = pose.shape
    
    if T < 4:  # Too short to interpolate
        return pose, valid_mask
    
    speed = np.random.uniform(*speed_range)
    new_T = int(T / speed)
    new_T = max(4, min(new_T, int(T * 1.5)))  # Clamp to reasonable range
    
    if new_T == T:
        return pose, valid_mask
    
    # Original time points
    t_orig = np.linspace(0, 1, T)
    t_new = np.linspace(0, 1, new_T)
    
    # Interpolate each keypoint
    resampled_pose = np.zeros((new_T, N, C), dtype=np.float32)
    
    for n in range(N):
        for c in range(C):
            # Use linear interpolation
            f = interpolate.interp1d(t_orig, pose[:, n, c], kind='linear', fill_value='extrapolate')
            resampled_pose[:, n, c] = f(t_new)
    
    # Interpolate mask (use nearest neighbor to keep binary)
    resampled_mask = np.zeros((new_T, N), dtype=bool)
    mask_float = valid_mask.astype(float)
    for n in range(N):
        f = interpolate.interp1d(t_orig, mask_float[:, n], kind='nearest', fill_value='extrapolate')
        resampled_mask[:, n] = f(t_new) > 0.5
    
    return resampled_pose, resampled_mask


def augment_frame_drop(pose, valid_mask, drop_prob=0.08, max_consecutive=1):
    """
    Randomly drop individual frames (conservative).
    
    This simulates minor tracking hiccups while preserving sign integrity.
    Never drops more than max_consecutive frames in a row.
    
    Args:
        pose: (T, N, 2) normalized keypoints
        valid_mask: (T, N) boolean mask
        drop_prob: Probability of dropping each frame
        max_consecutive: Maximum consecutive frames to drop
    
    Returns:
        dropped_pose: (T_new, N, 2) with some frames removed
        dropped_mask: (T_new, N) corresponding mask
    """
    T, N, C = pose.shape
    
    if T < 10:  # Too short, don't drop
        return pose, valid_mask
    
    # Decide which frames to keep
    keep_mask = np.ones(T, dtype=bool)
    consecutive_drops = 0
    
    for t in range(T):
        # Don't drop first or last few frames (often important)
        if t < 2 or t >= T - 2:
            continue
            
        if np.random.random() < drop_prob and consecutive_drops < max_consecutive:
            keep_mask[t] = False
            consecutive_drops += 1
        else:
            consecutive_drops = 0
    
    # Ensure we keep at least 80% of frames
    if keep_mask.sum() < 0.8 * T:
        return pose, valid_mask
    
    return pose[keep_mask], valid_mask[keep_mask]


def augment_spatial(pose, valid_mask, noise_sigma=0.02, scale_range=(0.85, 1.15), shift_range=0.1):
    """
    Apply spatial augmentations: noise, scaling, shifting.
    
    Args:
        pose: (T, N, 2) normalized keypoints
        valid_mask: (T, N) boolean mask
        noise_sigma: Std of Gaussian noise
        scale_range: (min_scale, max_scale) for random scaling
        shift_range: Maximum shift in each direction
    
    Returns:
        augmented_pose: (T, N, 2) augmented keypoints
    """
    aug_pose = pose.copy()
    T, N, C = pose.shape
    
    # 1. Gaussian noise
    if noise_sigma > 0:
        noise = np.random.normal(0, noise_sigma, pose.shape).astype(np.float32)
        aug_pose[valid_mask] += noise[valid_mask]
    
    # 2. Random scaling
    scale = np.random.uniform(*scale_range)
    aug_pose[valid_mask] *= scale
    
    # 3. Random shift
    shift = np.random.uniform(-shift_range, shift_range, size=(1, 1, C)).astype(np.float32)
    aug_pose[valid_mask] += shift.squeeze()
    
    return aug_pose


class SignLanguageDataset(Dataset):
    def __init__(
        self, 
        csv_file, 
        pkl_file, 
        tokenizer, 
        max_frames=None, 
        downsample_factor=1,
        mode="hybrid",
        use_velocity=True,
        use_augmentation=True,
        augmentation_config=None
    ):
        """
        Sign Language Dataset with comprehensive augmentation.
        
        Augmentations applied (in order):
        1. Speed perturbation (temporal resampling)
        2. Frame dropping (conservative, random individual frames)
        3. Rotation (around body center)
        4. Spatial (noise, scale, shift)
        5. Velocity recomputation (after all pose augmentations)
        
        Args:
            csv_file: Path to CSV with 'id' and 'gloss' columns
            pkl_file: Path to pickle file with keypoints
            tokenizer: GlossTokenizer instance
            max_frames: Maximum frames to keep (None = auto-detect)
            downsample_factor: Temporal downsampling factor
            mode: 'hybrid' for CTC+Attention
            use_velocity: Whether to include velocity features
            use_augmentation: Whether to apply data augmentation
            augmentation_config: Dict with augmentation parameters
        """
        self.data = pd.read_csv(csv_file)
        with open(pkl_file, 'rb') as f:
            self.keypoints = pickle.load(f)
        
        self.tokenizer = tokenizer
        self.downsample_factor = downsample_factor
        self.mode = mode
        self.use_velocity = use_velocity
        self.use_augmentation = use_augmentation
        self.training = True
        
        # Default augmentation config
        self.aug_config = {
            # Spatial augmentations
            'noise_sigma': 0.02,
            'scale_range': (0.85, 1.15),
            'shift_range': 0.1,
            
            # Rotation
            'rotation_prob': 0.5,          # Probability of applying rotation
            'rotation_max_angle': 12,      # Maximum rotation in degrees (±)
            
            # Speed perturbation
            'speed_prob': 0.5,             # Probability of applying speed change
            'speed_range': (0.85, 1.15),   # Speed factor range
            
            # Frame dropping
            'frame_drop_prob': 0.3,        # Probability of applying frame dropping
            'frame_drop_rate': 0.08,       # Per-frame drop probability
            'frame_drop_max_consecutive': 1,  # Max consecutive frames to drop
        }
        if augmentation_config:
            self.aug_config.update(augmentation_config)
        
        # Calculate max_frames
        if max_frames is None:
            self.max_frames = 0
            for sample_id in self.data['id']:
                if sample_id in self.keypoints:
                    raw_len = self.keypoints[sample_id]['keypoints'].shape[0]
                    downsampled_len = (raw_len + downsample_factor - 1) // downsample_factor
                    # Account for potential speed slowdown (1.15x)
                    max_possible = int(downsampled_len * 1.2)
                    if max_possible > self.max_frames:
                        self.max_frames = max_possible
            print(f"Auto-detected max_frames: {self.max_frames}")
        else:
            self.max_frames = max_frames
        
        self.feature_dim = 4 if use_velocity else 2
        
        # Filter samples without keypoints
        valid_ids = [sid for sid in self.data['id'] if sid in self.keypoints]
        if len(valid_ids) < len(self.data):
            print(f"Warning: {len(self.data) - len(valid_ids)} samples missing keypoints")
            self.data = self.data[self.data['id'].isin(valid_ids)].reset_index(drop=True)

    def __len__(self):
        return len(self.data)
    
    def set_training(self, training=True):
        """Set training mode to control augmentation."""
        self.training = training
    
    def _apply_augmentation(self, pose, valid_mask):
        """
        Apply all augmentations in the correct order.
        
        Order matters:
        1. Temporal augmentations first (speed, frame drop) - changes sequence length
        2. Spatial augmentations (rotation, noise, scale, shift)
        3. Velocity is recomputed AFTER all pose augmentations
        """
        # =====================================================================
        # 1. TEMPORAL AUGMENTATIONS (change sequence length)
        # =====================================================================
        
        # Speed perturbation
        if np.random.random() < self.aug_config['speed_prob']:
            pose, valid_mask = augment_speed(
                pose, valid_mask,
                speed_range=self.aug_config['speed_range']
            )
        
        # Frame dropping (conservative)
        if np.random.random() < self.aug_config['frame_drop_prob']:
            pose, valid_mask = augment_frame_drop(
                pose, valid_mask,
                drop_prob=self.aug_config['frame_drop_rate'],
                max_consecutive=self.aug_config['frame_drop_max_consecutive']
            )
        
        # =====================================================================
        # 2. SPATIAL AUGMENTATIONS
        # =====================================================================
        
        # Rotation
        if np.random.random() < self.aug_config['rotation_prob']:
            pose = augment_rotation(
                pose, valid_mask,
                max_angle_degrees=self.aug_config['rotation_max_angle']
            )
        
        # Noise, scale, shift
        pose = augment_spatial(
            pose, valid_mask,
            noise_sigma=self.aug_config['noise_sigma'],
            scale_range=self.aug_config['scale_range'],
            shift_range=self.aug_config['shift_range']
        )
        
        return pose, valid_mask

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sample_id = row['id']
        gloss_text = str(row['gloss'])
        
        # Get raw keypoints
        if sample_id in self.keypoints:
            raw_pose = self.keypoints[sample_id]['keypoints']
        else:
            raw_pose = np.zeros((10, 86, 2), dtype=np.float32)
        
        # 1. Downsample temporally
        if self.downsample_factor > 1:
            raw_pose = raw_pose[::self.downsample_factor]
        
        # 2. Normalize keypoints
        pose, valid_mask = normalize_keypoints(raw_pose, invalid_value=-1.0)
        
        # 3. Apply augmentation (only during training)
        if self.use_augmentation and self.training:
            pose, valid_mask = self._apply_augmentation(pose, valid_mask)
        
        # 4. Compute velocity AFTER all augmentations
        #    This ensures velocity matches the augmented positions
        if self.use_velocity:
            velocity = compute_velocity(pose, valid_mask)
            pose_combined = np.concatenate([pose, velocity], axis=-1)
        else:
            pose_combined = pose
        
        # 5. Convert to tensor
        pose_tensor = torch.tensor(pose_combined, dtype=torch.float32)
        
        # 6. Truncate if necessary
        if pose_tensor.shape[0] > self.max_frames:
            pose_tensor = pose_tensor[:self.max_frames]
        
        # Build result dict
        result = {
            'id': sample_id,
            'pose': pose_tensor,
            'gloss': gloss_text,
            'num_frames': pose_tensor.shape[0]
        }
        
        # Tokenize based on mode
        if self.mode == 'hybrid':
            ctc_ids = self.tokenizer.encode_for_ctc(gloss_text)
            result['ctc_ids'] = torch.tensor(ctc_ids, dtype=torch.long)
            
            dec_ids = self.tokenizer.encode_for_decoder(gloss_text)
            result['dec_ids'] = torch.tensor(dec_ids, dtype=torch.long)
        
        return result


def collate_fn_hybrid(batch):
    """
    Collate function for hybrid CTC + Attention mode.
    """
    poses = [x['pose'] for x in batch]
    ctc_ids = [x['ctc_ids'] for x in batch]
    dec_ids = [x['dec_ids'] for x in batch]
    ids = [x['id'] for x in batch]
    texts = [x['gloss'] for x in batch]
    
    num_keypoints = poses[0].shape[1]
    feature_dim = poses[0].shape[2]
    
    # Pad poses
    max_frames = max([p.shape[0] for p in poses])
    padded_poses = torch.zeros(len(poses), max_frames, num_keypoints, feature_dim)
    input_lengths = torch.zeros(len(poses), dtype=torch.long)
    
    for i, p in enumerate(poses):
        seq_len = p.shape[0]
        padded_poses[i, :seq_len] = p
        input_lengths[i] = seq_len
    
    # CTC Labels
    max_ctc_len = max([t.shape[0] for t in ctc_ids])
    padded_ctc = torch.zeros((len(ctc_ids), max_ctc_len), dtype=torch.long)
    ctc_lengths = torch.zeros(len(ctc_ids), dtype=torch.long)
    
    for i, t in enumerate(ctc_ids):
        label_len = t.shape[0]
        padded_ctc[i, :label_len] = t
        ctc_lengths[i] = label_len
    
    # Decoder Labels
    max_dec_len = max([t.shape[0] for t in dec_ids])
    padded_dec = torch.full((len(dec_ids), max_dec_len), -100, dtype=torch.long)
    dec_lengths = torch.zeros(len(dec_ids), dtype=torch.long)
    
    for i, t in enumerate(dec_ids):
        label_len = t.shape[0]
        padded_dec[i, :label_len] = t
        dec_lengths[i] = label_len
    
    return {
        'pose': padded_poses,
        'input_lengths': input_lengths,
        'ctc_labels': padded_ctc,
        'ctc_lengths': ctc_lengths,
        'dec_labels': padded_dec,
        'dec_lengths': dec_lengths,
        'ids': ids,
        'raw_text': texts
    }


# =============================================================================
# TESTING
# =============================================================================

def test_augmentations():
    """Test all augmentation functions."""
    print("=" * 60)
    print("AUGMENTATION TESTS")
    print("=" * 60)
    
    # Create synthetic pose data
    T, N, C = 100, 86, 2
    pose = np.random.randn(T, N, C).astype(np.float32) * 0.5
    valid_mask = np.ones((T, N), dtype=bool)
    valid_mask[10:15, :20] = False  # Some invalid regions
    
    print(f"\nOriginal shape: {pose.shape}")
    print(f"Valid points: {valid_mask.sum()} / {valid_mask.size}")
    
    # Test rotation
    print("\n--- Rotation Test ---")
    rotated = augment_rotation(pose, valid_mask, max_angle_degrees=15)
    print(f"Rotated shape: {rotated.shape}")
    # Check that invalid points are unchanged
    assert np.allclose(rotated[~valid_mask], pose[~valid_mask]), "Invalid points should not change"
    print("✓ Rotation preserves invalid points")
    
    # Test speed perturbation
    print("\n--- Speed Perturbation Test ---")
    for speed_range in [(0.8, 0.8), (1.2, 1.2), (0.9, 1.1)]:
        speed_pose, speed_mask = augment_speed(pose, valid_mask, speed_range=speed_range)
        print(f"Speed range {speed_range}: {pose.shape[0]} → {speed_pose.shape[0]} frames")
    print("✓ Speed perturbation works")
    
    # Test frame dropping
    print("\n--- Frame Drop Test ---")
    dropped_pose, dropped_mask = augment_frame_drop(pose, valid_mask, drop_prob=0.1, max_consecutive=1)
    print(f"After drop: {pose.shape[0]} → {dropped_pose.shape[0]} frames")
    assert dropped_pose.shape[0] >= 0.8 * pose.shape[0], "Should keep at least 80% of frames"
    print("✓ Frame dropping is conservative")
    
    # Test spatial augmentation
    print("\n--- Spatial Augmentation Test ---")
    spatial_aug = augment_spatial(pose, valid_mask, noise_sigma=0.02, scale_range=(0.9, 1.1), shift_range=0.1)
    print(f"Spatial augmented shape: {spatial_aug.shape}")
    print("✓ Spatial augmentation works")
    
    # Test full pipeline
    print("\n--- Full Pipeline Test ---")
    aug_config = {
        'noise_sigma': 0.02,
        'scale_range': (0.85, 1.15),
        'shift_range': 0.1,
        'rotation_prob': 1.0,
        'rotation_max_angle': 12,
        'speed_prob': 1.0,
        'speed_range': (0.85, 1.15),
        'frame_drop_prob': 1.0,
        'frame_drop_rate': 0.08,
        'frame_drop_max_consecutive': 1,
    }
    
    # Simulate what happens in __getitem__
    aug_pose = pose.copy()
    aug_mask = valid_mask.copy()
    
    # Speed
    aug_pose, aug_mask = augment_speed(aug_pose, aug_mask, speed_range=aug_config['speed_range'])
    # Frame drop
    aug_pose, aug_mask = augment_frame_drop(aug_pose, aug_mask, 
                                             drop_prob=aug_config['frame_drop_rate'],
                                             max_consecutive=aug_config['frame_drop_max_consecutive'])
    # Rotation
    aug_pose = augment_rotation(aug_pose, aug_mask, max_angle_degrees=aug_config['rotation_max_angle'])
    # Spatial
    aug_pose = augment_spatial(aug_pose, aug_mask,
                               noise_sigma=aug_config['noise_sigma'],
                               scale_range=aug_config['scale_range'],
                               shift_range=aug_config['shift_range'])
    # Velocity (recomputed after augmentation)
    velocity = compute_velocity(aug_pose, aug_mask)
    
    print(f"Final shape: {aug_pose.shape}")
    print(f"Velocity shape: {velocity.shape}")
    print("✓ Full pipeline works")
    
    print("\n" + "=" * 60)
    print("ALL AUGMENTATION TESTS PASSED!")
    print("=" * 60)


def test_dataset_integration():
    """Test that dataset loads with augmentation."""
    print("\n" + "=" * 60)
    print("DATASET INTEGRATION TEST")
    print("=" * 60)
    
    # This would require actual data files
    print("Skipping (requires data files)")
    print("To test: Create dataset and iterate through a few samples")


if __name__ == "__main__":
    test_augmentations()
    test_dataset_integration()
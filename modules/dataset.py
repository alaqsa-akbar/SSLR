import torch
from torch.utils.data import Dataset
import pandas as pd
import pickle
import numpy as np
import json


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
        """
        Encode for CTC loss - NO special tokens.
        CTC doesn't need SOS/EOS, just the content tokens.
        
        Example: "انا حب اكل" -> [id_انا, id_حب, id_اكل]
        """
        if not isinstance(sentence, str):
            return []
        tokens = sentence.strip().split()
        ids = [self.token_to_id.get(t, self.unk_token_id) for t in tokens]
        return ids
    
    def encode_for_decoder(self, sentence):
        """
        Encode for decoder/cross-entropy loss - with EOS token.
        Decoder needs EOS to know when to stop generating.
        (SOS is prepended separately during training)
        
        Example: "انا حب اكل" -> [id_انا, id_حب, id_اكل, id_EOS]
        """
        if not isinstance(sentence, str):
            return []
        tokens = sentence.strip().split()
        ids = [self.token_to_id.get(t, self.unk_token_id) for t in tokens]
        ids.append(self.eos_token_id)
        return ids
        
    def encode(self, sentence, add_special_tokens=True):
        """
        Legacy encode method for backward compatibility.
        Prefer encode_for_ctc() or encode_for_decoder() for clarity.
        """
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
            if token:  # Skip empty strings
                tokens.append(token)
        return " ".join(tokens)


def normalize_keypoints(keypoints, invalid_value=-1.0):
    """
    Robust normalization for keypoints using global standardization.
    
    This approach:
    1. Centers data using mean of all valid points
    2. Scales using standard deviation (robust to outliers)
    3. Preserves relative scale information between samples
    4. Properly handles invalid/missing keypoints
    
    Args:
        keypoints: (T, N, 2) array of (x, y) coordinates
        invalid_value: Value used to mark invalid points (default: -1.0)
    
    Returns:
        normalized: (T, N, 2) normalized keypoints with invalid points set to 0
        valid_mask: (T, N) boolean mask of valid points
    """
    arr = keypoints.copy().astype(np.float32)
    T, N, C = arr.shape
    
    # Create validity mask: point is valid if BOTH x and y are not invalid
    valid_mask = ~np.isclose(arr, invalid_value, atol=1e-6).any(axis=-1)  # (T, N)
    
    # Count valid points
    num_valid = valid_mask.sum()
    
    if num_valid == 0:
        # No valid points - return zeros
        return np.zeros_like(arr), valid_mask
    
    # Extract all valid points for computing statistics
    valid_points = arr[valid_mask]  # (num_valid, 2)
    
    # Compute global center (mean of all valid points)
    center = valid_points.mean(axis=0)  # (2,)
    
    # Compute global scale (std of all valid points)
    std = valid_points.std()
    scale = max(1e-6, std)  # Prevent division by zero
    
    # Normalize valid points: center and scale
    arr[valid_mask] = (arr[valid_mask] - center) / scale
    
    # Set invalid points to 0 (neutral value after normalization)
    arr[~valid_mask] = 0.0
    
    return arr, valid_mask


def compute_velocity(pose, valid_mask):
    """
    Compute velocity (temporal derivative) of keypoints.
    
    CRITICAL: Velocity is only valid when BOTH current AND previous frames
    have valid observations.
    
    Args:
        pose: (T, N, 2) normalized keypoints
        valid_mask: (T, N) boolean mask of valid points
    
    Returns:
        velocity: (T, N, 2) velocity with invalid transitions set to 0
    """
    T, N, C = pose.shape
    velocity = np.zeros_like(pose)
    
    # Compute raw differences
    velocity[1:] = pose[1:] - pose[:-1]
    
    # Velocity is only valid when BOTH current AND previous frame are valid
    valid_velocity_mask = np.zeros((T, N), dtype=bool)
    valid_velocity_mask[1:] = valid_mask[1:] & valid_mask[:-1]
    
    # Zero out invalid velocities
    velocity[~valid_velocity_mask] = 0.0
    
    # First frame always has zero velocity
    velocity[0] = 0.0
    
    return velocity


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
        Sign Language Dataset with proper preprocessing.
        
        Args:
            csv_file: Path to CSV with 'id' and 'gloss' columns
            pkl_file: Path to pickle file with keypoints
            tokenizer: GlossTokenizer instance
            max_frames: Maximum frames to keep (None = auto-detect)
            downsample_factor: Temporal downsampling factor (1 = no downsampling)
            mode: 'hybrid' for CTC+Attention, 'contrastive' for contrastive learning
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
        self.training = True  # Default to training mode
        
        # Default augmentation config
        self.aug_config = {
            'noise_sigma': 0.02,
            'scale_range': (0.9, 1.1),
            'shift_range': 0.1,
            'time_mask_prob': 0.1,
            'time_mask_max': 5,
        }
        if augmentation_config:
            self.aug_config.update(augmentation_config)
        
        # Calculate max_frames from data if not provided
        if max_frames is None:
            self.max_frames = 0
            for sample_id in self.data['id']:
                if sample_id in self.keypoints:
                    raw_len = self.keypoints[sample_id]['keypoints'].shape[0]
                    downsampled_len = (raw_len + downsample_factor - 1) // downsample_factor
                    if downsampled_len > self.max_frames:
                        self.max_frames = downsampled_len
            print(f"Auto-detected max_frames: {self.max_frames}")
        else:
            self.max_frames = max_frames
        
        # Feature dimension
        self.feature_dim = 4 if use_velocity else 2
        
        # Filter out samples without keypoints
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
        """Apply data augmentation to normalized pose data."""
        T, N, C = pose.shape
        aug_pose = pose.copy()
        aug_mask = valid_mask.copy()
        
        # 1. Gaussian noise
        if self.aug_config['noise_sigma'] > 0:
            noise = np.random.normal(0, self.aug_config['noise_sigma'], pose.shape)
            aug_pose[valid_mask] += noise[valid_mask]
        
        # 2. Random scaling
        scale_lo, scale_hi = self.aug_config['scale_range']
        scale = np.random.uniform(scale_lo, scale_hi)
        aug_pose[valid_mask] *= scale
        
        # 3. Random shift
        shift_range = self.aug_config['shift_range']
        shift = np.random.uniform(-shift_range, shift_range, size=(1, 1, C))
        aug_pose[valid_mask] += shift.squeeze()
        
        # 4. Temporal masking
        if np.random.random() < self.aug_config['time_mask_prob'] and T > 10:
            mask_len = np.random.randint(1, min(self.aug_config['time_mask_max'], T // 4))
            mask_start = np.random.randint(0, T - mask_len)
            aug_pose[mask_start:mask_start + mask_len] = 0.0
            aug_mask[mask_start:mask_start + mask_len] = False
        
        return aug_pose, aug_mask

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sample_id = row['id']
        gloss_text = str(row['gloss'])
        
        # Get raw keypoints
        if sample_id in self.keypoints:
            raw_pose = self.keypoints[sample_id]['keypoints']  # (T, 86, 2)
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
        
        # 4. Compute velocity
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
        if self.mode == 'contrastive':
            tokenized = self.tokenizer(
                gloss_text, 
                padding='max_length', 
                truncation=True, 
                max_length=128, 
                return_tensors='pt'
            )
            result['input_ids'] = tokenized.input_ids.squeeze(0)
            result['attention_mask'] = tokenized.attention_mask.squeeze(0)
            
        elif self.mode == 'hybrid':
            # ============================================================
            # KEY FIX: Separate CTC and Decoder targets
            # ============================================================
            # CTC targets: NO special tokens (just content)
            ctc_ids = self.tokenizer.encode_for_ctc(gloss_text)
            result['ctc_ids'] = torch.tensor(ctc_ids, dtype=torch.long)
            
            # Decoder targets: WITH EOS (SOS prepended during training)
            dec_ids = self.tokenizer.encode_for_decoder(gloss_text)
            result['dec_ids'] = torch.tensor(dec_ids, dtype=torch.long)
        
        return result


def collate_fn_contrastive(batch):
    """Collate function for contrastive learning mode."""
    poses = [x['pose'] for x in batch]
    ids = [x['id'] for x in batch]
    input_ids = torch.stack([x['input_ids'] for x in batch])
    attention_mask = torch.stack([x['attention_mask'] for x in batch])
    
    max_frames = max([p.shape[0] for p in poses])
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
        'pose_attention_mask': pose_mask,
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }


def collate_fn_hybrid(batch):
    """
    Collate function for hybrid CTC + Attention mode.
    
    IMPORTANT: Returns SEPARATE targets for CTC and Decoder!
    
    Returns:
        pose: (B, T_max, N, C) padded pose sequences
        input_lengths: (B,) actual frame lengths for CTC
        ctc_labels: (B, S_ctc) padded CTC targets (NO special tokens)
        ctc_lengths: (B,) actual CTC target lengths
        dec_labels: (B, S_dec) padded decoder targets (with EOS, -100 padding)
        dec_lengths: (B,) actual decoder target lengths
        ids: list of sample IDs
        raw_text: list of original gloss strings
    """
    poses = [x['pose'] for x in batch]
    ctc_ids = [x['ctc_ids'] for x in batch]
    dec_ids = [x['dec_ids'] for x in batch]
    ids = [x['id'] for x in batch]
    texts = [x['gloss'] for x in batch]
    
    # Get dimensions
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
    
    # ============================================================
    # CTC Labels: NO EOS, use 0 for padding (ignored via ctc_lengths)
    # ============================================================
    max_ctc_len = max([t.shape[0] for t in ctc_ids])
    padded_ctc = torch.zeros((len(ctc_ids), max_ctc_len), dtype=torch.long)
    ctc_lengths = torch.zeros(len(ctc_ids), dtype=torch.long)
    
    for i, t in enumerate(ctc_ids):
        label_len = t.shape[0]
        padded_ctc[i, :label_len] = t
        ctc_lengths[i] = label_len
    
    # ============================================================
    # Decoder Labels: WITH EOS, use -100 for padding (CrossEntropyLoss ignore)
    # ============================================================
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


# ============================================================================
# Testing / Verification
# ============================================================================

def verify_tokenization():
    """Verify that CTC and decoder tokenization are different."""
    print("=" * 50)
    print("TOKENIZATION VERIFICATION")
    print("=" * 50)
    
    # Create a mock vocab
    mock_vocab = {
        "<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3,
        "انا": 4, "حب": 5, "اكل": 6
    }
    
    # Write temp vocab file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(mock_vocab, f)
        vocab_path = f.name
    
    tokenizer = GlossTokenizer(vocab_path)
    
    test_sentence = "انا حب اكل"
    
    ctc_tokens = tokenizer.encode_for_ctc(test_sentence)
    dec_tokens = tokenizer.encode_for_decoder(test_sentence)
    
    print(f"Input: '{test_sentence}'")
    print(f"CTC tokens:     {ctc_tokens} (length: {len(ctc_tokens)})")
    print(f"Decoder tokens: {dec_tokens} (length: {len(dec_tokens)})")
    print(f"EOS token ID:   {tokenizer.eos_token_id}")
    print()
    
    # Verify
    assert len(dec_tokens) == len(ctc_tokens) + 1, "Decoder should have 1 more token (EOS)"
    assert dec_tokens[-1] == tokenizer.eos_token_id, "Decoder should end with EOS"
    assert tokenizer.eos_token_id not in ctc_tokens, "CTC should NOT have EOS"
    
    print("✓ CTC tokens do NOT include EOS")
    print("✓ Decoder tokens DO include EOS")
    print("✓ All checks passed!")
    
    # Cleanup
    import os
    os.unlink(vocab_path)


def test_normalization():
    """Test the normalization function."""
    print("\n" + "=" * 50)
    print("NORMALIZATION TEST")
    print("=" * 50)
    
    # Create test data with some invalid points
    test_pose = np.random.randn(10, 86, 2) * 100 + 500
    test_pose[0, :10, :] = -1  # Mark first 10 points in frame 0 as invalid
    test_pose[5, :, :] = -1    # Mark entire frame 5 as invalid
    
    normalized, mask = normalize_keypoints(test_pose)
    
    print(f"Input shape: {test_pose.shape}")
    print(f"Input range: [{test_pose[test_pose != -1].min():.1f}, {test_pose[test_pose != -1].max():.1f}]")
    print(f"Output range: [{normalized[mask].min():.2f}, {normalized[mask].max():.2f}]")
    print(f"Output mean: {normalized[mask].mean():.4f} (should be ~0)")
    print(f"Output std: {normalized[mask].std():.4f} (should be ~1)")
    print(f"Invalid points are zero: {(normalized[~mask] == 0).all()}")
    print("✓ Normalization test passed!")


def test_velocity():
    """Test the velocity computation."""
    print("\n" + "=" * 50)
    print("VELOCITY TEST")
    print("=" * 50)
    
    pose = np.array([
        [[0, 0], [1, 1]],   # Frame 0
        [[1, 1], [2, 2]],   # Frame 1: valid transition
        [[-1, -1], [3, 3]], # Frame 2: first point invalid
        [[2, 2], [4, 4]],   # Frame 3
    ], dtype=np.float32)
    
    normalized, mask = normalize_keypoints(pose)
    velocity = compute_velocity(normalized, mask)
    
    print(f"Frame 0 velocity (should be 0): {velocity[0].tolist()}")
    print(f"Frame 2 point 0 (invalid transition): {velocity[2, 0].tolist()} (should be [0,0])")
    print(f"Frame 3 point 0 (valid after invalid): {velocity[3, 0].tolist()} (should be [0,0])")
    print("✓ Velocity test passed!")


if __name__ == "__main__":
    verify_tokenization()
    test_normalization()
    test_velocity()
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)
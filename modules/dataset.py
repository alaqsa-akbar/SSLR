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
        
    def encode(self, sentence, add_special_tokens=True):
        if not isinstance(sentence, str):
            return []
        tokens = sentence.strip().split()
        ids = [self.token_to_id.get(t, self.unk_token_id) for t in tokens]
        
        # Ensure EOS is appended
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids
    
    def decode(self, ids):
        tokens = []
        for i in ids:
            i = int(i)
            if i in [self.pad_token_id, self.sos_token_id]:
                continue
            if i == self.eos_token_id:
                break 
            tokens.append(self.id_to_token.get(i, ""))
        return " ".join(tokens)

class SignLanguageDataset(Dataset):
    def __init__(self, csv_file, pkl_file, tokenizer, max_frames=None, downsample_factor=2, mode="hybrid"):
        self.data = pd.read_csv(csv_file)
        with open(pkl_file, 'rb') as f:
            self.keypoints = pickle.load(f)
        self.tokenizer = tokenizer
        self.downsample_factor = downsample_factor
        self.mode = mode
        
        if max_frames is None:
            self.max_frames = 0
            for sample_id in self.data['id']:
                if sample_id in self.keypoints:
                    l = self.keypoints[sample_id]['keypoints'].shape[0] // downsample_factor
                    if l > self.max_frames: self.max_frames = l
            print(f"Calculated max_frames: {self.max_frames}")
        else:
            self.max_frames = max_frames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sample_id = row['id']
        gloss_text = str(row['gloss'])
        
        if sample_id in self.keypoints:
            pose = self.keypoints[sample_id]['keypoints'] 
        else:
            pose = np.zeros((10, 86, 2), dtype=np.float32)
            
        # 1. Downsample
        pose = pose[::self.downsample_factor]
        
        # 2. Normalize & Augment (Re-ordered for correctness)
        mask = (pose != -1).all(axis=-1) # Identify valid points
        
        if mask.sum() > 0:
            valid = pose[mask]
            min_x, max_x = valid[:,0].min(), valid[:,0].max()
            min_y, max_y = valid[:,1].min(), valid[:,1].max()
            scale = max(max_x - min_x, max_y - min_y)
            if scale < 1e-6: scale = 1.0
            
            # Normalize to 0-1
            pose[mask, 0] = (pose[mask, 0] - min_x) / scale
            pose[mask, 1] = (pose[mask, 1] - min_y) / scale
            
            # === AUGMENTATION: Gaussian Noise ===
            # Adds 0.5% jitter. Crucial for preventing overfitting to specific signers.
            noise_sigma = 0.005 
            noise = np.random.normal(0, noise_sigma, pose.shape)
            # Only add noise to valid points
            pose[mask] += noise[mask]
            # ====================================

            # Restore -1 for invisible points (safety)
            pose = np.where(mask[:, :, None], pose, -1.0)
        
        # 3. Velocity Stream 
        # Calculate AFTER noise/norm so velocity reflects the augmented motion
        velocity = np.zeros_like(pose)
        velocity[1:] = pose[1:] - pose[:-1]
        # Mask velocity for invisible points
        velocity = np.where(mask[:, :, None], velocity, 0.0)

        # 4. Concatenate -> (T, 86, 4)
        pose_combined = torch.tensor(np.concatenate([pose, velocity], axis=-1), dtype=torch.float32)
        
        if pose_combined.shape[0] > self.max_frames:
            pose_combined = pose_combined[:self.max_frames]

        result = {
            'id': sample_id,
            'pose': pose_combined,
            'gloss': gloss_text
        }

        # --- Tokenize Labels ---
        if self.mode == 'contrastive':
            tokenized = self.tokenizer(
                gloss_text, padding='max_length', truncation=True, max_length=128, return_tensors='pt'
            )
            result['input_ids'] = tokenized.input_ids.squeeze(0)
            result['attention_mask'] = tokenized.attention_mask.squeeze(0)
            
        elif self.mode == 'hybrid':
            token_ids = self.tokenizer.encode(gloss_text)
            result['token_ids'] = torch.tensor(token_ids, dtype=torch.long)
            
        return result

def collate_fn_contrastive(batch):
    poses = [x['pose'] for x in batch]
    ids = [x['id'] for x in batch]
    input_ids = torch.stack([x['input_ids'] for x in batch])
    attention_mask = torch.stack([x['attention_mask'] for x in batch])
    
    max_frames = max([p.shape[0] for p in poses])
    padded_poses = torch.zeros(len(poses), max_frames, 86, 4)
    pose_mask = torch.zeros(len(poses), max_frames)
    
    for i, p in enumerate(poses):
        l = p.shape[0]
        padded_poses[i, :l] = p
        pose_mask[i, :l] = 1
        
    return {
        'id': ids,
        'pose': padded_poses,
        'pose_attention_mask': pose_mask,
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

def collate_fn_hybrid(batch):
    poses = [x['pose'] for x in batch]
    token_ids = [x['token_ids'] for x in batch]
    ids = [x['id'] for x in batch]
    texts = [x['gloss'] for x in batch]
    
    max_frames = max([p.shape[0] for p in poses])
    padded_poses = torch.zeros(len(poses), max_frames, 86, 4)
    input_lengths = torch.zeros(len(poses), dtype=torch.long)
    
    for i, p in enumerate(poses):
        l = p.shape[0]
        padded_poses[i, :l] = p
        input_lengths[i] = l
        
    max_label_len = max([t.shape[0] for t in token_ids])
    padded_labels = torch.full((len(token_ids), max_label_len), -100, dtype=torch.long)
    target_lengths = torch.zeros(len(token_ids), dtype=torch.long)
    
    for i, t in enumerate(token_ids):
        l = t.shape[0]
        padded_labels[i, :l] = t
        target_lengths[i] = l
        
    return {
        'pose': padded_poses,
        'input_lengths': input_lengths,
        'labels': padded_labels, 
        'target_lengths': target_lengths,
        'ids': ids,
        'raw_text': texts
    }
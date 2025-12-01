import torch
from torch.utils.data import Dataset
import pandas as pd
import pickle
import os
import numpy as np

class SignLanguageDataset(Dataset):
    def __init__(self, csv_file, pkl_file, tokenizer, max_length=128, max_frames=None):
        self.data = pd.read_csv(csv_file)
        with open(pkl_file, 'rb') as f:
            self.keypoints = pickle.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if max_frames is None:
            self.max_frames = 0
            for sample_id in self.data['id']:
                if sample_id in self.keypoints:
                    frames = self.keypoints[sample_id]['keypoints'].shape[0]
                    if frames > self.max_frames:
                        self.max_frames = frames
            print(f"Calculated max_frames from dataset: {self.max_frames}")
        else:
            self.max_frames = max_frames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sample_id = row['id']
        gloss = row['gloss']
        
        if sample_id in self.keypoints:
            pose = self.keypoints[sample_id]['keypoints']
        else:
            pose = np.zeros((10, 86, 2), dtype=np.float32)
            
        # Robust Normalization (0-centered, aspect ratio preserved)
        # 1. Center the pose
        # We use the bounding box center of the current frame or the whole sequence
        # Here we center the whole sequence based on its global min/max
        min_x, min_y = np.min(pose[:, :, 0]), np.min(pose[:, :, 1])
        max_x, max_y = np.max(pose[:, :, 0]), np.max(pose[:, :, 1])
        
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        pose[:, :, 0] -= center_x
        pose[:, :, 1] -= center_y
        
        # 2. Scale to [-1, 1] preserving aspect ratio
        # We divide by the maximum dimension (width or height)
        width = max_x - min_x
        height = max_y - min_y
        scale = max(width, height) / 2.0 # Divide by half to map to [-1, 1]
        
        if scale > 1e-6:
            pose = pose / scale
            
        pose = torch.tensor(pose, dtype=torch.float32)
        
        num_frames = pose.shape[0]
        if num_frames > self.max_frames:
            pose = pose[:self.max_frames]
        
        # Tokenize gloss
        # We need to tokenize the gloss for the decoder
        tokenized_gloss = self.tokenizer(
            gloss,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = tokenized_gloss.input_ids.squeeze(0)
        attention_mask = tokenized_gloss.attention_mask.squeeze(0)
        
        return {
            'id': sample_id,
            'pose': pose,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'gloss': gloss
        }

def collate_fn(batch):
    # Collate function to pad poses
    ids = [item['id'] for item in batch]
    poses = [item['pose'] for item in batch]
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    glosses = [item['gloss'] for item in batch]
    
    # Pad poses
    # poses is a list of (Frames, 84, 2)
    # We want (Batch, MaxFrames, 84, 2)
    
    max_frames = max([p.shape[0] for p in poses])
    padded_poses = torch.zeros(len(poses), max_frames, 86, 2)
    pose_attention_mask = torch.zeros(len(poses), max_frames) # 1 for valid, 0 for pad
    
    for i, p in enumerate(poses):
        frames = p.shape[0]
        padded_poses[i, :frames] = p
        pose_attention_mask[i, :frames] = 1
        
    return {
        'id': ids,
        'pose': padded_poses,
        'pose_attention_mask': pose_attention_mask,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'gloss': glosses
    }

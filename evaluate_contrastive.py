import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, UMT5EncoderModel, UMT5Config
from modules.pose_encoder import PoseEncoder, PoseEncoderConfig
from modules.dataset import SignLanguageDataset, collate_fn
from tqdm import tqdm
import os
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Contrastive Pose Encoder")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--model_dir", type=str, default="output/pose_encoder_contrastive", help="Path to trained model directory")
    parser.add_argument("--umt5_dir", type=str, default="models/umt5", help="Path to UMT5 directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_frames", type=int, default=100, help="Max frames")
    return parser.parse_args()

def load_pose_encoder(model_dir):
    print(f"Loading Pose Encoder from {model_dir}")
    try:
        # Try standard loading
        model = PoseEncoder.from_pretrained(model_dir)
    except Exception as e:
        print(f"Standard load failed: {e}")
        # Try safetensors manual load
        config = PoseEncoderConfig.from_pretrained(model_dir)
        model = PoseEncoder(config)
        if os.path.exists(os.path.join(model_dir, "model.safetensors")):
            from safetensors.torch import load_file
            state_dict = load_file(os.path.join(model_dir, "model.safetensors"))
            model.load_state_dict(state_dict)
        elif os.path.exists(os.path.join(model_dir, "pytorch_model.bin")):
            model.load_state_dict(torch.load(os.path.join(model_dir, "pytorch_model.bin")))
        else:
            raise ValueError("No model weights found!")
    return model

def evaluate_retrieval():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Tokenizer
    tokenizer_path = os.path.join(args.umt5_dir, "tokenizer.json")
    if os.path.exists(tokenizer_path):
        print(f"Loading tokenizer from {args.umt5_dir}")
        tokenizer = AutoTokenizer.from_pretrained(args.umt5_dir)
    else:
        print("Local tokenizer not found, downloading 'google/umt5-base'...")
        tokenizer = AutoTokenizer.from_pretrained("google/umt5-base")
        
    # Load UMT5 (Text Encoder)
    print("Loading UMT5...")
    umt5_bin = os.path.join(args.umt5_dir, "umt5-base.bin")
    if not os.path.exists(umt5_bin):
        umt5_bin = os.path.join(args.umt5_dir, "pytorch_model.bin")
        
    if os.path.exists(umt5_bin):
        print(f"Loading UMT5 weights from {umt5_bin}")
        # Try loading config, fallback to google/umt5-base if not found locally
        try:
            config = UMT5Config.from_pretrained(args.umt5_dir)
            if config.d_model != 768:
                print(f"Loaded config has d_model={config.d_model}, which is wrong for umt5-base. Using manual config.")
                raise ValueError("Wrong config dimensions")
        except:
            print("Could not load correct config, creating manual UMT5-Base config")
            config = UMT5Config(
                vocab_size=256384,
                d_model=768,
                d_kv=64,
                d_ff=2048,
                num_layers=12,
                num_heads=12,
                relative_attention_num_buckets=32,
                dropout_rate=0.1,
                layer_norm_epsilon=1e-6,
                initializer_factor=1.0,
                feed_forward_proj="gated-gelu",
                is_encoder_decoder=True
            )
            
        text_encoder = UMT5EncoderModel(config)
        state_dict = torch.load(umt5_bin, map_location="cpu")
        text_encoder.load_state_dict(state_dict, strict=False)
    else:
        print("Local UMT5 weights not found, downloading 'google/umt5-base'...")
        text_encoder = UMT5EncoderModel.from_pretrained("google/umt5-base")
    text_encoder.to(device)
    text_encoder.eval()
    
    # Load Pose Encoder
    pose_encoder = load_pose_encoder(args.model_dir)
    pose_encoder.to(device)
    pose_encoder.eval()
    
    # Load Dev Dataset
    # Load Train Data (Subset)
    dev_csv = os.path.join(args.data_dir, "train.csv")
    pkl_file = os.path.join(args.data_dir, "data.pkl")
    dataset = SignLanguageDataset(dev_csv, pkl_file, tokenizer, max_frames=args.max_frames)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    all_pose_embeddings = []
    all_text_embeddings = []
    
    print("Extracting Embeddings...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            if i >= 30: break # Limit to ~1000 samples for speed
            pose = batch['pose'].to(device)
            pose_mask = batch['pose_attention_mask'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Pose Embeddings
            p_emb = pose_encoder(pose, attention_mask=pose_mask)
            # Pooling
            p_mask = pose_mask.unsqueeze(-1).expand(p_emb.size()).float()
            p_sum = torch.sum(p_emb * p_mask, dim=1)
            p_counts = torch.clamp(p_mask.sum(1), min=1e-9)
            p_rep = p_sum / p_counts
            p_rep = nn.functional.normalize(p_rep, p=2, dim=1)
            
            # Text Embeddings
            t_out = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            t_emb = t_out.last_hidden_state
            # Pooling
            t_mask = attention_mask.unsqueeze(-1).expand(t_emb.size()).float()
            t_sum = torch.sum(t_emb * t_mask, dim=1)
            t_counts = torch.clamp(t_mask.sum(1), min=1e-9)
            t_rep = t_sum / t_counts
            t_rep = nn.functional.normalize(t_rep, p=2, dim=1)
            
            all_pose_embeddings.append(p_rep.cpu())
            all_text_embeddings.append(t_rep.cpu())
            
    all_pose_embeddings = torch.cat(all_pose_embeddings, dim=0)
    all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
    
    print(f"Computing Similarity Matrix ({len(all_pose_embeddings)} samples)...")
    # Similarity: (N, N)
    similarity = torch.matmul(all_pose_embeddings, all_text_embeddings.T)
    
    # Calculate Top-k Accuracy
    # For each pose, is the correct text (diagonal) in the top k?
    
    top1 = 0
    top5 = 0
    top10 = 0
    n = len(similarity)
    
    for i in range(n):
        scores = similarity[i]
        # Get indices of top scores
        _, indices = scores.topk(10)
        
        if i in indices[:1]:
            top1 += 1
        if i in indices[:5]:
            top5 += 1
        if i in indices[:10]:
            top10 += 1
            
    print(f"Top-1 Accuracy: {top1/n:.4f}")
    print(f"Top-5 Accuracy: {top5/n:.4f}")
    print(f"Top-10 Accuracy: {top10/n:.4f}")
    
    # Random Baseline
    print(f"Random Baseline (Top-1): {1/n:.4f}")

if __name__ == "__main__":
    evaluate_retrieval()

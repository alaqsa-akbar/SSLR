"""
Evaluation Script for Hybrid Sign Language Translation Model (Fixed Generation)

Computes:
- BLEU scores (1-4 gram)
- ROUGE scores (1, 2, L)
- WER (Word Error Rate)
- Exact Match Accuracy

Usage:
    python evaluate.py --model_dir output/sign_language_model_final/best_model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
import json
import numpy as np

# Import modules
from modules.dataset import SignLanguageDataset, GlossTokenizer, collate_fn_hybrid
from modules.pose_encoder import PoseEncoder, TinyAdvancedDecoder, PoseEncoderConfig

# --- Metric Imports ---
try:
    from sacrebleu.metrics import BLEU
    import jiwer
    from rouge_score import rouge_scorer
except ImportError:
    print("Please install metrics: pip install sacrebleu jiwer rouge-score")
    exit()

class HybridModelEval(nn.Module):
    def __init__(self, pose_encoder, decoder):
        super().__init__()
        self.encoder = pose_encoder
        self.decoder = decoder
        
    def forward(self, pose, input_lengths):
        # Encoder
        max_len = pose.shape[1]
        enc_mask = torch.arange(max_len, device=pose.device)[None, :] < input_lengths[:, None]
        enc_out = self.encoder(pose, attention_mask=enc_mask.float())
        return enc_out, enc_mask

    @torch.no_grad()
    def beam_search(self, enc_out, enc_mask, tokenizer, beam_width=5, max_len=60, 
                   repetition_penalty=1.2, length_penalty=1.0, no_repeat_ngram_size=0):
        """
        Performs Beam Search Decoding with Repetition Penalty.
        """
        device = enc_out.device
        # Start Token
        start_seq = torch.tensor([[tokenizer.sos_token_id]], dtype=torch.long, device=device)
        
        # Beam Structure: (score, sequence_tensor)
        # Initial beam: score 0.0, sequence [SOS]
        beams = [(0.0, start_seq)]
        completed_beams = []
        
        for step in range(max_len):
            candidates = []
            
            # Expand each beam
            for score, seq in beams:
                # If finished, keep it
                if seq[0, -1] == tokenizer.eos_token_id:
                    completed_beams.append((score, seq))
                    continue
                    
                # Forward Decoder
                logits = self.decoder(seq, enc_out, enc_mask=enc_mask.float())
                
                # Take the logits for the last step
                next_token_logits = logits[0, -1, :].clone() # (V)
                
                # --- APPLY PENALTIES HERE ---
                
                # 1. Repetition Penalty
                if repetition_penalty != 1.0:
                    for token_id in set(seq[0].tolist()):
                        if next_token_logits[token_id] < 0:
                            next_token_logits[token_id] *= repetition_penalty
                        else:
                            next_token_logits[token_id] /= repetition_penalty
                            
                # 2. No Repeat N-Gram
                # (Simple check: if adding a token creates a duplicate ngram)
                # Omitted for speed/complexity in pure python loop, but Repetition Penalty usually suffices.
                
                # ----------------------------

                # Get Log Probs
                next_prob = F.log_softmax(next_token_logits, dim=-1)
                
                # Get Top K candidates
                topk_probs, topk_ids = next_prob.topk(beam_width)
                
                for i in range(beam_width):
                    token_id = topk_ids[i].item()
                    token_prob = topk_probs[i].item()
                    
                    new_seq = torch.cat([seq, torch.tensor([[token_id]], device=device)], dim=1)
                    new_score = score + token_prob
                    candidates.append((new_score, new_seq))
            
            # Sort candidates by score (highest first)
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Prune to beam width
            beams = candidates[:beam_width]
            
            # Early stopping
            if len(completed_beams) >= beam_width:
                break
                
        if not completed_beams:
            completed_beams = beams
            
        # Sort final completed (apply Length Penalty)
        # Score = LogProb / (Length ^ alpha)
        completed_beams.sort(key=lambda x: x[0] / (x[1].shape[1] ** length_penalty), reverse=True)
        best_seq = completed_beams[0][1]
        
        # Decode to text
        token_list = best_seq[0].tolist()
        if token_list[0] == tokenizer.sos_token_id:
            token_list = token_list[1:]
        if tokenizer.eos_token_id in token_list:
            idx = token_list.index(tokenizer.eos_token_id)
            token_list = token_list[:idx]
            
        return tokenizer.decode(token_list)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Hybrid SLR Model")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--vocab_file", type=str, default="vocab.json")
    parser.add_argument("--output_file", type=str, default="evaluation_results.txt")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--pose_config", type=str, default="configs/pose_encoder_config.json")
    
    # Generation Params
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--repetition_penalty", type=float, default=3, help="Penalty for repeating tokens (>1.0)")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="Penalty for length (<1.0 prefers short)")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0) # Not fully implemented in simple beam above, relying on rep_penalty
    
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Tokenizer
    tokenizer = GlossTokenizer(args.vocab_file)
    
    # 2. Load Dataset
    csv_path = os.path.join(args.data_dir, f"{args.split}.csv")
    pkl_path = os.path.join(args.data_dir, "data.pkl")
    dataset = SignLanguageDataset(csv_path, pkl_path, tokenizer, mode="hybrid")
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn_hybrid, num_workers=4)
    
    # 3. Load Model
    print(f"Loading model from {args.model_dir}...")
    pose_config = PoseEncoderConfig.from_json_file(args.pose_config) 
    pose_encoder = PoseEncoder(pose_config)
    dec_config = PoseEncoderConfig.from_json_file(args.pose_config)
    decoder = TinyAdvancedDecoder(tokenizer.vocab_size, dec_config)
    
    model = HybridModelEval(pose_encoder, decoder)
    
    # Load Weights
    if os.path.isdir(args.model_dir):
        files = [f for f in os.listdir(args.model_dir) if f.endswith('.bin') or f.endswith('.pt')]
        if not files: raise FileNotFoundError(f"No weights found in {args.model_dir}")
        weights_path = os.path.join(args.model_dir, files[0])
    else:
        weights_path = args.model_dir
        
    print(f"Loading weights: {weights_path}")
    state_dict = torch.load(weights_path, map_location='cpu')
    
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."): k = k[7:]
        new_state[k] = v
        
    model.load_state_dict(new_state, strict=False)
    model.to(device)
    model.eval()
    
    # 4. Evaluation Loop
    predictions = []
    references = []
    
    print(f"Starting Beam Search (Beams={args.num_beams}, RepPenalty={args.repetition_penalty})...")
    for batch in tqdm(loader):
        pose = batch['pose'].to(device)
        input_lens = batch['input_lengths'].to(device)
        raw_text = batch['raw_text'][0]
        
        with torch.no_grad():
            enc_out, enc_mask = model(pose, input_lens)
            
            # Pass penalty args
            pred_text = model.beam_search(
                enc_out, enc_mask, tokenizer, 
                beam_width=args.num_beams, 
                repetition_penalty=args.repetition_penalty,
                length_penalty=args.length_penalty
            )
            
            predictions.append(pred_text)
            references.append(raw_text)
            
    # 5. Metrics
    print("\nCalculating metrics...")
    
    # BLEU
    bleu = BLEU()
    b1 = BLEU(max_ngram_order=1).corpus_score(predictions, [references]).score
    b2 = BLEU(max_ngram_order=2).corpus_score(predictions, [references]).score
    b3 = BLEU(max_ngram_order=3).corpus_score(predictions, [references]).score
    b4 = BLEU(max_ngram_order=4).corpus_score(predictions, [references]).score
    
    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    r1, r2, rl = [], [], []
    for p, r in zip(predictions, references):
        s = scorer.score(r, p)
        r1.append(s['rouge1'].fmeasure)
        r2.append(s['rouge2'].fmeasure)
        rl.append(s['rougeL'].fmeasure)
    avg_r1 = sum(r1)/len(r1) if r1 else 0
    avg_r2 = sum(r2)/len(r2) if r2 else 0
    avg_rl = sum(rl)/len(rl) if rl else 0
    
    # WER
    try:
        wer = jiwer.wer(references, predictions)
    except:
        wer = 1.0
        
    # Exact Match
    exact = sum([1 for p, r in zip(predictions, references) if p.strip()==r.strip()])
    acc = exact / len(predictions)
    
    # 6. Save & Print
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("-" * 60)
    print(f"BLEU-1: {b1:.2f}")
    print(f"BLEU-4: {b4:.2f}")
    print(f"ROUGE-L: {avg_rl:.4f}")
    print(f"WER: {wer:.4f}")
    print(f"Exact Match: {acc:.4f}")
    print("=" * 60)
    
    # Save Text
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write("EVALUATION RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"BLEU-1: {b1:.2f}\nBLEU-4: {b4:.2f}\n")
        f.write(f"WER: {wer:.4f}\nExact: {acc:.4f}\n")
        f.write("="*60 + "\n\n")
        for i, (p, r) in enumerate(zip(predictions, references)):
            f.write(f"[{i+1}]\nRef:  {r}\nPred: {p}\n{'-'*30}\n")
            
    # Save JSON
    json_path = args.output_file.replace('.txt', '.json')
    results = {
        'bleu1': b1, 'bleu4': b4, 'wer': wer, 'accuracy': acc,
        'rougeL': avg_rl,
        'predictions': [{'ref': r, 'pred': p} for r, p in zip(references, predictions)]
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
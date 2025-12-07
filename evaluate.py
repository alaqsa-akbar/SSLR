"""
Evaluation Script for Sign Language Recognition Model

Supports multiple decoding methods:
- ctc: CTC greedy decoding (fast)
- ctc_beam: CTC beam search (more accurate)
- attention: Attention decoder with greedy/beam search

Usage:
    # CTC decoding (recommended for speed)
    python evaluate.py --model_dir output/best_model --decode_method ctc
    
    # Attention decoding
    python evaluate.py --model_dir output/best_model --decode_method attention --num_beams 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import os
import json
from tqdm import tqdm
from collections import Counter

from modules.dataset import SignLanguageDataset, GlossTokenizer, collate_fn_hybrid
from modules.pose_encoder import PoseEncoder, TinyAdvancedDecoder, PoseEncoderConfig


class HybridModel(nn.Module):
    """Hybrid model matching the training architecture."""
    def __init__(self, pose_encoder, vocab_size, hidden_dim=768, num_decoder_layers=2):
        super().__init__()
        self.encoder = pose_encoder
        self.vocab_size = vocab_size
        self.ctc_head = nn.Linear(hidden_dim, vocab_size + 1)
        
        dec_config = pose_encoder.config
        dec_config.num_decoder_layers = num_decoder_layers
        self.decoder = TinyAdvancedDecoder(vocab_size, dec_config)
        
    def forward(self, pose, input_lengths, decoder_input_ids=None):
        max_len = pose.shape[1]
        device = pose.device
        enc_mask = torch.arange(max_len, device=device)[None, :] < input_lengths[:, None]
        
        enc_out = self.encoder(pose, attention_mask=enc_mask.float())
        ctc_logits = self.ctc_head(enc_out)
        
        dec_logits = None
        if decoder_input_ids is not None:
            dec_logits = self.decoder(decoder_input_ids, enc_out, enc_mask=enc_mask.float())
            
        return ctc_logits, dec_logits, enc_out, enc_mask


def ctc_greedy_decode(ctc_logits, input_lengths, tokenizer):
    """CTC greedy decoding."""
    batch_size = ctc_logits.shape[0]
    blank_id = tokenizer.vocab_size
    predictions = []
    
    best_paths = torch.argmax(ctc_logits, dim=-1)
    
    for b in range(batch_size):
        seq_len = input_lengths[b].item()
        path = best_paths[b, :seq_len].tolist()
        
        # Collapse repeats and remove blanks
        decoded = []
        prev = None
        for token in path:
            if token != blank_id and token != prev:
                decoded.append(token)
            prev = token
        
        predictions.append(tokenizer.decode(decoded))
    
    return predictions


def ctc_beam_decode(ctc_logits, input_lengths, tokenizer, beam_width=10):
    """CTC beam search decoding."""
    batch_size = ctc_logits.shape[0]
    blank_id = tokenizer.vocab_size
    predictions = []
    
    log_probs = F.log_softmax(ctc_logits, dim=-1)
    
    for b in range(batch_size):
        seq_len = input_lengths[b].item()
        lp = log_probs[b, :seq_len].cpu().numpy()
        
        # Simple beam search
        beams = [([], 0.0)]  # (prefix, score)
        
        for t in range(seq_len):
            new_beams = {}
            
            for prefix, score in beams:
                for c in range(lp.shape[1]):
                    new_score = score + lp[t, c]
                    
                    if c == blank_id:
                        new_prefix = tuple(prefix)
                    elif len(prefix) > 0 and prefix[-1] == c:
                        new_prefix = tuple(prefix)
                    else:
                        new_prefix = tuple(prefix + [c])
                    
                    if new_prefix not in new_beams or new_beams[new_prefix] < new_score:
                        new_beams[new_prefix] = new_score
            
            # Keep top beams
            beams = sorted(new_beams.items(), key=lambda x: -x[1])[:beam_width]
            beams = [(list(p), s) for p, s in beams]
        
        best_prefix = beams[0][0] if beams else []
        predictions.append(tokenizer.decode(best_prefix))
    
    return predictions


def attention_decode(model, enc_out, enc_mask, tokenizer, max_len=50, num_beams=1):
    """Attention decoder with greedy or beam search."""
    device = enc_out.device
    batch_size = enc_out.shape[0]
    
    if num_beams == 1:
        # Greedy decoding
        predictions = []
        for b in range(batch_size):
            generated = torch.tensor([[tokenizer.sos_token_id]], device=device)
            
            for _ in range(max_len):
                logits = model.decoder(generated, enc_out[b:b+1], enc_mask=enc_mask[b:b+1].float())
                next_token = logits[0, -1, :].argmax().item()
                
                if next_token == tokenizer.eos_token_id:
                    break
                    
                generated = torch.cat([generated, torch.tensor([[next_token]], device=device)], dim=1)
            
            predictions.append(tokenizer.decode(generated[0, 1:].tolist()))
        
        return predictions
    else:
        # Beam search
        predictions = []
        for b in range(batch_size):
            # Initialize beams: (sequence, score)
            beams = [([tokenizer.sos_token_id], 0.0)]
            
            for _ in range(max_len):
                all_candidates = []
                
                for seq, score in beams:
                    if seq[-1] == tokenizer.eos_token_id:
                        all_candidates.append((seq, score))
                        continue
                    
                    input_ids = torch.tensor([seq], device=device)
                    logits = model.decoder(input_ids, enc_out[b:b+1], enc_mask=enc_mask[b:b+1].float())
                    log_probs = F.log_softmax(logits[0, -1, :], dim=-1)
                    
                    top_probs, top_ids = log_probs.topk(num_beams)
                    
                    for prob, idx in zip(top_probs.tolist(), top_ids.tolist()):
                        all_candidates.append((seq + [idx], score + prob))
                
                # Select top beams
                beams = sorted(all_candidates, key=lambda x: -x[1])[:num_beams]
                
                # Check if all beams ended
                if all(seq[-1] == tokenizer.eos_token_id for seq, _ in beams):
                    break
            
            best_seq = beams[0][0]
            # Remove SOS and EOS
            if best_seq[0] == tokenizer.sos_token_id:
                best_seq = best_seq[1:]
            if best_seq and best_seq[-1] == tokenizer.eos_token_id:
                best_seq = best_seq[:-1]
            
            predictions.append(tokenizer.decode(best_seq))
        
        return predictions


def compute_wer(predictions, references):
    """Compute Word Error Rate."""
    total_errors = 0
    total_words = 0
    
    for pred, ref in zip(predictions, references):
        pred_words = pred.strip().split()
        ref_words = ref.strip().split()
        
        # Levenshtein distance
        m, n = len(ref_words), len(pred_words)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_words[i-1] == pred_words[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        total_errors += dp[m][n]
        total_words += m
    
    return total_errors / total_words if total_words > 0 else 0


def compute_bleu(predictions, references, max_n=4):
    """Compute BLEU scores."""
    def get_ngrams(words, n):
        return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    
    bleu_scores = {}
    
    for n in range(1, max_n + 1):
        total_matches = 0
        total_pred = 0
        
        for pred, ref in zip(predictions, references):
            pred_words = pred.strip().split()
            ref_words = ref.strip().split()
            
            pred_ngrams = Counter(get_ngrams(pred_words, n))
            ref_ngrams = Counter(get_ngrams(ref_words, n))
            
            matches = sum(min(pred_ngrams[ng], ref_ngrams[ng]) for ng in pred_ngrams)
            total_matches += matches
            total_pred += max(0, len(pred_words) - n + 1)
        
        precision = total_matches / total_pred if total_pred > 0 else 0
        bleu_scores[f"BLEU-{n}"] = precision * 100
    
    return bleu_scores


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SLR Model")
    
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to model directory or pytorch_model.bin")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--vocab_file", type=str, default="vocab.json")
    parser.add_argument("--pose_config", type=str, default="configs/pose_encoder_config.json")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"])
    
    parser.add_argument("--decode_method", type=str, default="ctc",
                        choices=["ctc", "ctc_beam", "attention"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=50)
    
    parser.add_argument("--output_file", type=str, default=None)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Decode method: {args.decode_method}")
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.vocab_file}...")
    tokenizer = GlossTokenizer(args.vocab_file)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Load dataset
    print(f"Loading dataset from {args.data_dir}/{args.split}.csv...")
    dataset = SignLanguageDataset(
        f"{args.data_dir}/{args.split}.csv",
        f"{args.data_dir}/data.pkl",
        tokenizer,
        mode='hybrid',
        use_augmentation=False
    )
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size if args.decode_method != "attention" else 1,
        shuffle=False,
        collate_fn=collate_fn_hybrid,
        num_workers=2
    )
    
    # Load model
    print(f"Loading model from {args.model_dir}...")
    config = PoseEncoderConfig.from_json_file(args.pose_config)
    pose_encoder = PoseEncoder(config)
    model = HybridModel(pose_encoder, tokenizer.vocab_size, config.hidden_dim, config.num_decoder_layers)
    
    # Load weights
    if os.path.isdir(args.model_dir):
        weights_path = os.path.join(args.model_dir, "pytorch_model.bin")
    else:
        weights_path = args.model_dir
    
    print(f"Loading weights from {weights_path}...")
    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    # Evaluate
    print(f"\nRunning evaluation on {len(dataset)} samples...")
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            pose = batch['pose'].to(device)
            input_lengths = batch['input_lengths'].to(device)
            references = batch['raw_text']
            
            ctc_logits, _, enc_out, enc_mask = model(pose, input_lengths)
            
            if args.decode_method == "ctc":
                predictions = ctc_greedy_decode(ctc_logits, input_lengths, tokenizer)
            elif args.decode_method == "ctc_beam":
                predictions = ctc_beam_decode(ctc_logits, input_lengths, tokenizer, args.num_beams)
            else:  # attention
                predictions = attention_decode(model, enc_out, enc_mask, tokenizer, 
                                               args.max_len, args.num_beams)
            
            all_predictions.extend(predictions)
            all_references.extend(references)
    
    # Compute metrics
    print("\nComputing metrics...")
    
    wer = compute_wer(all_predictions, all_references)
    bleu_scores = compute_bleu(all_predictions, all_references)
    
    exact_matches = sum(1 for p, r in zip(all_predictions, all_references) 
                       if p.strip() == r.strip())
    exact_match_rate = exact_matches / len(all_predictions)
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Decode Method: {args.decode_method}")
    print("-" * 60)
    print(f"BLEU-1: {bleu_scores['BLEU-1']:.2f}")
    print(f"BLEU-4: {bleu_scores['BLEU-4']:.2f}")
    print(f"WER: {wer:.4f}")
    print(f"Exact Match: {exact_match_rate:.4f} ({exact_matches}/{len(all_predictions)})")
    print("=" * 60)
    
    # Show samples
    print("\nSample Predictions (first 20):")
    for i in range(min(20, len(all_predictions))):
        match = "✓" if all_predictions[i].strip() == all_references[i].strip() else "✗"
        print(f"[{i+1}] {match}")
        print(f"Ref:  {all_references[i]}")
        print(f"Pred: {all_predictions[i]}")
        print("-" * 30)
    
    # Save results
    if args.output_file:
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("EVALUATION RESULTS\n")
            f.write("=" * 60 + "\n")
            f.write(f"Decode Method: {args.decode_method}\n")
            f.write(f"BLEU-1: {bleu_scores['BLEU-1']:.2f}\n")
            f.write(f"BLEU-4: {bleu_scores['BLEU-4']:.2f}\n")
            f.write(f"WER: {wer:.4f}\n")
            f.write(f"Exact Match: {exact_match_rate:.4f}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("All Predictions:\n")
            for i, (pred, ref) in enumerate(zip(all_predictions, all_references)):
                match = "✓" if pred.strip() == ref.strip() else "✗"
                f.write(f"[{i+1}] {match}\n")
                f.write(f"Ref:  {ref}\n")
                f.write(f"Pred: {pred}\n\n")
        
        # Also save JSON
        json_path = args.output_file.replace(".txt", ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "decode_method": args.decode_method,
                "bleu_1": bleu_scores['BLEU-1'],
                "bleu_4": bleu_scores['BLEU-4'],
                "wer": wer,
                "exact_match": exact_match_rate,
                "num_samples": len(all_predictions)
            }, f, indent=2)
        
        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
"""
Evaluation Script for Sign Language Translation Model

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
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, UMT5Config, UMT5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from modules.pose_encoder import PoseEncoder, PoseEncoderConfig
from modules.dataset import SignLanguageDataset, collate_fn
from tqdm import tqdm
import os
import argparse
import sacrebleu
from sacrebleu.metrics import BLEU
import json
import jiwer


class SignLanguageModel(nn.Module):
    """
    Sign Language Translation Model (Decoder-Only UMT5)
    
    This class matches the architecture from train_final.py
    """
    
    def __init__(self, pose_encoder: PoseEncoder, decoder: nn.Module, decoder_config: UMT5Config):
        super().__init__()
        self.pose_encoder = pose_encoder
        self.decoder = decoder
        self.config = decoder_config
        
        # LM head for token prediction
        self.lm_head = nn.Linear(decoder_config.d_model, decoder_config.vocab_size, bias=False)
        
    def forward(
        self,
        pose: torch.Tensor,
        pose_attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
        decoder_input_ids: torch.Tensor = None,
        decoder_attention_mask: torch.Tensor = None,
    ):
        # Encode pose
        encoder_hidden_states = self.pose_encoder(pose, attention_mask=pose_attention_mask)
        
        # Prepare decoder inputs from labels (shift right)
        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self._shift_right(labels)
        
        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=pose_attention_mask,
        )
        
        # Project to vocabulary
        sequence_output = decoder_outputs.last_hidden_state
        logits = self.lm_head(sequence_output)
        
        # Compute loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
        )
    
    def _shift_right(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Shift input ids right for decoder input."""
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id
        
        if decoder_start_token_id is None:
            decoder_start_token_id = pad_token_id
        
        shifted = input_ids.new_zeros(input_ids.shape)
        shifted[:, 1:] = input_ids[:, :-1].clone()
        shifted[:, 0] = decoder_start_token_id
        shifted.masked_fill_(shifted == -100, pad_token_id)
        
        return shifted
    
    @torch.no_grad()
    def generate(
        self,
        pose: torch.Tensor,
        pose_attention_mask: torch.Tensor,
        max_length: int = 128,
        num_beams: int = 4,
        early_stopping: bool = True,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        decoder_start_token_id: int = None,
    ):
        """Generate text from pose using greedy decoding."""
        batch_size = pose.size(0)
        device = pose.device
        
        if decoder_start_token_id is None:
            decoder_start_token_id = self.config.decoder_start_token_id or pad_token_id
        
        # Encode pose
        encoder_hidden_states = self.pose_encoder(pose, attention_mask=pose_attention_mask)
        
        # Initialize decoder input
        decoder_input_ids = torch.full(
            (batch_size, 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=device
        )
        
        # Greedy decoding
        for _ in range(max_length - 1):
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=pose_attention_mask,
            )
            
            logits = self.lm_head(decoder_outputs.last_hidden_state[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)
            
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
            
            if (next_token == eos_token_id).all():
                break
        
        return decoder_input_ids

    @classmethod
    def from_pretrained(cls, load_directory: str, device: torch.device = None):
        """Load model from saved directory."""
        if device is None:
            device = torch.device("cpu")
            
        # Load pose encoder
        pose_encoder_path = os.path.join(load_directory, "pose_encoder")
        pose_encoder = PoseEncoder.from_pretrained(pose_encoder_path)
        
        # Load decoder config and weights
        decoder_path = os.path.join(load_directory, "decoder")
        decoder_config = UMT5Config.from_pretrained(decoder_path)
        
        # Create decoder
        full_model = UMT5ForConditionalGeneration(decoder_config)
        decoder = full_model.decoder
        
        # Load decoder weights
        decoder_weights_path = os.path.join(decoder_path, "pytorch_model.bin")
        if os.path.exists(decoder_weights_path):
            decoder_weights = torch.load(decoder_weights_path, map_location=device)
            decoder.load_state_dict(decoder_weights)
        
        # Create model
        model = cls(pose_encoder, decoder, decoder_config)
        
        # Load LM head
        lm_head_path = os.path.join(load_directory, "lm_head.pt")
        if os.path.exists(lm_head_path):
            model.lm_head.load_state_dict(torch.load(lm_head_path, map_location=device))
        
        return model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Sign Language Model")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--model_dir", type=str, default="output/sign_language_model_final/best_model", 
                        help="Path to trained model directory")
    parser.add_argument("--tokenizer_dir", type=str, default=None, 
                        help="Path to tokenizer (defaults to model_dir)")
    parser.add_argument("--output_file", type=str, default="evaluation_results.txt", 
                        help="Path to save results")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--max_frames", type=int, default=1000, help="Max frames for input")
    parser.add_argument("--max_length", type=int, default=128, help="Max generation length")
    parser.add_argument("--num_beams", type=int, default=4, help="Beam size (currently uses greedy)")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"], 
                        help="Data split to evaluate")
    
    return parser.parse_args()


def load_model(model_dir: str, device: torch.device):
    """
    Load the trained SignLanguageModel.
    
    Handles both new decoder-only format and legacy full UMT5 format.
    """
    print(f"Loading model from {model_dir}")
    
    # Check if this is the new format (has decoder/ subdirectory)
    decoder_path = os.path.join(model_dir, "decoder")
    is_new_format = os.path.exists(decoder_path)
    
    if is_new_format:
        print("Detected new decoder-only format")
        model = SignLanguageModel.from_pretrained(model_dir, device=device)
    else:
        # Legacy format: full UMT5ForConditionalGeneration
        print("Detected legacy format (full UMT5)")
        model = load_legacy_model(model_dir, device)
    
    model.to(device)
    model.eval()
    
    return model


def load_legacy_model(model_dir: str, device: torch.device):
    """Load model saved in the old format (full UMT5)."""
    
    # Handle checkpoint subdirectories
    checkpoint_path = model_dir
    if "checkpoint" not in os.path.basename(model_dir) and "best_model" not in os.path.basename(model_dir):
        if os.path.exists(model_dir):
            # Look for best_model first
            best_model_path = os.path.join(model_dir, "best_model")
            if os.path.exists(best_model_path):
                checkpoint_path = best_model_path
            else:
                # Fall back to latest checkpoint
                checkpoints = [d for d in os.listdir(model_dir) if d.startswith("checkpoint-")]
                if checkpoints:
                    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
                    checkpoint_path = os.path.join(model_dir, checkpoints[-1])
    
    print(f"Using checkpoint: {checkpoint_path}")
    
    # Load Pose Encoder
    pose_encoder_path = os.path.join(checkpoint_path, "pose_encoder")
    try:
        pose_encoder = PoseEncoder.from_pretrained(pose_encoder_path)
    except Exception as e:
        print(f"Error loading pose encoder: {e}")
        raise
    
    # Load UMT5 (full model for legacy)
    umt5_path = os.path.join(checkpoint_path, "umt5")
    try:
        umt5_model = UMT5ForConditionalGeneration.from_pretrained(umt5_path)
    except Exception as e:
        print(f"Error loading UMT5: {e}")
        raise
    
    # Create legacy wrapper
    class LegacySignLanguageModel(nn.Module):
        def __init__(self, pose_encoder, umt5_model):
            super().__init__()
            self.pose_encoder = pose_encoder
            self.umt5_model = umt5_model
            self.config = umt5_model.config
            
        def generate(self, pose, pose_attention_mask, **kwargs):
            pose_embeddings = self.pose_encoder(pose, attention_mask=pose_attention_mask)
            encoder_outputs = BaseModelOutput(last_hidden_state=pose_embeddings)
            return self.umt5_model.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=pose_attention_mask,
                **kwargs
            )
    
    return LegacySignLanguageModel(pose_encoder, umt5_model)


def evaluate():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer_dir = args.tokenizer_dir or args.model_dir
    print(f"Loading tokenizer from {tokenizer_dir}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    except:
        print("Could not load tokenizer from model dir, trying 'google/umt5-base'")
        tokenizer = AutoTokenizer.from_pretrained("google/umt5-base")
    
    # Load model
    model = load_model(args.model_dir, device)
    
    # Load dataset
    data_csv = os.path.join(args.data_dir, f"{args.split}.csv")
    pkl_file = os.path.join(args.data_dir, "data.pkl")
    
    print(f"Loading {args.split} dataset...")
    dataset = SignLanguageDataset(data_csv, pkl_file, tokenizer, max_frames=args.max_frames)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=4
    )
    
    predictions = []
    references = []
    
    print("Starting Evaluation...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating"):
            pose = batch['pose'].to(device)
            pose_mask = batch['pose_attention_mask'].to(device)
            
            # Generate
            generated_ids = model.generate(
                pose, 
                pose_mask, 
                max_length=args.max_length,
                num_beams=args.num_beams,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            decoded_refs = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
            
            predictions.extend(decoded_preds)
            references.extend(decoded_refs)
    
    # ============ Calculate Metrics ============
    print("\nCalculating metrics...")
    
    # BLEU Scores
    bleu1 = BLEU(max_ngram_order=1)
    bleu2 = BLEU(max_ngram_order=2)
    bleu3 = BLEU(max_ngram_order=3)
    bleu4 = BLEU(max_ngram_order=4)
    
    b1_score = bleu1.corpus_score(predictions, [references]).score
    b2_score = bleu2.corpus_score(predictions, [references]).score
    b3_score = bleu3.corpus_score(predictions, [references]).score
    b4_score = bleu4.corpus_score(predictions, [references]).score
    
    # ROUGE Scores
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
        avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0
        avg_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0
        has_rouge = True
    except ImportError:
        print("Warning: rouge_score not installed, skipping ROUGE metrics")
        avg_rouge1 = avg_rouge2 = avg_rougeL = 0
        has_rouge = False
    
    # WER (Word Error Rate)
    try:
        wer = jiwer.wer(references, predictions)
    except Exception as e:
        print(f"Warning: Could not compute WER: {e}")
        wer = -1
    
    # Exact Match Accuracy
    exact_matches = sum(1 for p, r in zip(predictions, references) 
                       if p.strip().lower() == r.strip().lower())
    accuracy = exact_matches / len(predictions) if predictions else 0
    
    # ============ Print Results ============
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Dataset: {args.split} ({len(predictions)} samples)")
    print("-" * 60)
    print(f"BLEU-1: {b1_score:.2f}")
    print(f"BLEU-2: {b2_score:.2f}")
    print(f"BLEU-3: {b3_score:.2f}")
    print(f"BLEU-4: {b4_score:.2f}")
    print("-" * 60)
    if has_rouge:
        print(f"ROUGE-1: {avg_rouge1:.4f}")
        print(f"ROUGE-2: {avg_rouge2:.4f}")
        print(f"ROUGE-L: {avg_rougeL:.4f}")
        print("-" * 60)
    print(f"WER: {wer:.4f}")
    print(f"Exact Match Accuracy: {accuracy:.4f} ({exact_matches}/{len(predictions)})")
    print("=" * 60)
    
    # ============ Save Results ============
    results = {
        'split': args.split,
        'num_samples': len(predictions),
        'bleu1': b1_score,
        'bleu2': b2_score,
        'bleu3': b3_score,
        'bleu4': b4_score,
        'rouge1': avg_rouge1,
        'rouge2': avg_rouge2,
        'rougeL': avg_rougeL,
        'wer': wer,
        'accuracy': accuracy,
    }
    
    # Save to text file
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {args.model_dir}\n")
        f.write(f"Dataset: {args.split} ({len(predictions)} samples)\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("METRICS:\n")
        f.write(f"  BLEU-1: {b1_score:.2f}\n")
        f.write(f"  BLEU-2: {b2_score:.2f}\n")
        f.write(f"  BLEU-3: {b3_score:.2f}\n")
        f.write(f"  BLEU-4: {b4_score:.2f}\n")
        if has_rouge:
            f.write(f"  ROUGE-1: {avg_rouge1:.4f}\n")
            f.write(f"  ROUGE-2: {avg_rouge2:.4f}\n")
            f.write(f"  ROUGE-L: {avg_rougeL:.4f}\n")
        f.write(f"  WER: {wer:.4f}\n")
        f.write(f"  Accuracy: {accuracy:.4f}\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("SAMPLE PREDICTIONS:\n")
        f.write("=" * 60 + "\n\n")
        
        # Write all predictions
        for i, (pred, ref) in enumerate(zip(predictions, references), 1):
            f.write(f"[{i}]\n")
            f.write(f"  Reference:  {ref}\n")
            f.write(f"  Prediction: {pred}\n")
            f.write("-" * 50 + "\n")
    
    # Also save as JSON for programmatic access
    json_output = args.output_file.rsplit('.', 1)[0] + '.json'
    with open(json_output, 'w', encoding='utf-8') as f:
        results['predictions'] = [
            {'reference': ref, 'prediction': pred}
            for ref, pred in zip(references, predictions)
        ]
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to:")
    print(f"  - {args.output_file}")
    print(f"  - {json_output}")
    
    return results


if __name__ == "__main__":
    evaluate()
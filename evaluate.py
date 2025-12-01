import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, UMT5ForConditionalGeneration, UMT5Config
from modules.pose_encoder import PoseEncoder, PoseEncoderConfig
from modules.dataset import SignLanguageDataset, collate_fn
from tqdm import tqdm
import os
import argparse
import sacrebleu
from sacrebleu.metrics import BLEU
import pandas as pd
import pickle
import jiwer

from transformers.modeling_outputs import BaseModelOutput

class SignLanguageModel(nn.Module):
    def __init__(self, pose_encoder, umt5_model):
        super().__init__()
        self.pose_encoder = pose_encoder
        self.umt5_model = umt5_model
        
    def forward(self, pose, pose_attention_mask, labels=None, decoder_attention_mask=None):
        pose_embeddings = self.pose_encoder(pose, attention_mask=pose_attention_mask)
        # Forward can accept tuple or BaseModelOutput, let's be consistent
        outputs = self.umt5_model(
            encoder_outputs=(pose_embeddings,),
            attention_mask=pose_attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        )
        return outputs
    
    def generate(self, pose, pose_attention_mask, **kwargs):
        pose_embeddings = self.pose_encoder(pose, attention_mask=pose_attention_mask)
        encoder_outputs = BaseModelOutput(last_hidden_state=pose_embeddings)
        return self.umt5_model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=pose_attention_mask,
            **kwargs
        )

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Sign Language Model")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--model_dir", type=str, default="output/sign_language_model", help="Path to trained model directory")
    parser.add_argument("--tokenizer_dir", type=str, default="models/umt5", help="Path to tokenizer directory")
    parser.add_argument("--output_file", type=str, default="evaluation_results.txt", help="Path to save results")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--max_frames", type=int, default=1000, help="Max frames for input")
    parser.add_argument("--pose_config", type=str, default="configs/pose_encoder_config.json", help="Path to Pose Encoder config file")
    
    return parser.parse_args()

def load_model(args, device):
    # Load Tokenizer
    tokenizer_path = os.path.join(args.tokenizer_dir, "tokenizer.json")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except:
        print(f"Could not load tokenizer from {tokenizer_path}, trying 'google/umt5-base'")
        tokenizer = AutoTokenizer.from_pretrained("google/umt5-base")

    # Find latest checkpoint
    checkpoint_path = args.model_dir
    if "checkpoint" not in os.path.basename(args.model_dir):
        if os.path.exists(args.model_dir):
            checkpoints = [d for d in os.listdir(args.model_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split("-")[1]))
                checkpoint_path = os.path.join(args.model_dir, checkpoints[-1])
    
    print(f"Loading model from {checkpoint_path}")
    
    # Load Pose Encoder
    pose_encoder_path = os.path.join(checkpoint_path, "pose_encoder")
    try:
        pose_encoder = PoseEncoder.from_pretrained(pose_encoder_path)
    except:
        print("Could not load pose encoder from pretrained, initializing with config")
        pose_config = PoseEncoderConfig.from_json_file(args.pose_config)
        pose_encoder = PoseEncoder(pose_config)
        if os.path.exists(os.path.join(pose_encoder_path, "pytorch_model.bin")):
            pose_encoder.load_state_dict(torch.load(os.path.join(pose_encoder_path, "pytorch_model.bin")))
        elif os.path.exists(os.path.join(pose_encoder_path, "model.safetensors")):
            from safetensors.torch import load_file
            pose_encoder.load_state_dict(load_file(os.path.join(pose_encoder_path, "model.safetensors")))

    # Load UMT5
    umt5_path = os.path.join(checkpoint_path, "umt5")
    try:
        umt5_model = UMT5ForConditionalGeneration.from_pretrained(umt5_path)
    except:
        print("Could not load UMT5 from pretrained, initializing with config")
        config = UMT5Config.from_pretrained("google/umt5-base")
        umt5_model = UMT5ForConditionalGeneration(config)
        if os.path.exists(os.path.join(umt5_path, "pytorch_model.bin")):
             umt5_model.load_state_dict(torch.load(os.path.join(umt5_path, "pytorch_model.bin")))
        elif os.path.exists(os.path.join(umt5_path, "model.safetensors")):
            from safetensors.torch import load_file
            umt5_model.load_state_dict(load_file(os.path.join(umt5_path, "model.safetensors")))

    model = SignLanguageModel(pose_encoder, umt5_model)
    model.to(device)
    model.eval()
    
    return model, tokenizer

def evaluate():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, tokenizer = load_model(args, device)
    
    # Load Dev Data
    dev_csv = os.path.join(args.data_dir, "dev.csv")
    pkl_file = os.path.join(args.data_dir, "data.pkl")
    
    print("Loading Dev Dataset...")
    # We use the dataset class to handle loading and processing
    dev_dataset = SignLanguageDataset(dev_csv, pkl_file, tokenizer, max_frames=args.max_frames)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    predictions = []
    references = []
    
    print("Starting Evaluation...")
    with torch.no_grad():
        for batch in tqdm(dev_loader):
            pose = batch['pose'].to(device)
            pose_mask = batch['pose_attention_mask'].to(device)
            
            # Generate
            generated_ids = model.generate(
                pose, 
                pose_mask, 
                max_length=128, 
                num_beams=4, 
                early_stopping=True
            )
            
            decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Get references
            # We decode the input_ids directly. Tokenizer handles pad tokens.
            decoded_refs = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
            
            predictions.extend(decoded_preds)
            references.extend(decoded_refs)
            
    # Calculate BLEU Scores
    # BLEU-1
    bleu1 = BLEU(max_ngram_order=1)
    b1_score = bleu1.corpus_score(predictions, [references]).score
    
    # BLEU-2
    bleu2 = BLEU(max_ngram_order=2)
    b2_score = bleu2.corpus_score(predictions, [references]).score
    
    # BLEU-3
    bleu3 = BLEU(max_ngram_order=3)
    b3_score = bleu3.corpus_score(predictions, [references]).score
    
    # BLEU-4 (Standard)
    bleu4 = BLEU(max_ngram_order=4)
    b4_score = bleu4.corpus_score(predictions, [references]).score
    
    print(f"BLEU-1: {b1_score}")
    print(f"BLEU-2: {b2_score}")
    print(f"BLEU-3: {b3_score}")
    print(f"BLEU-4: {b4_score}")
    
    # Calculate ROUGE Scores
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
        
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)
    
    print(f"ROUGE-1: {avg_rouge1}")
    print(f"ROUGE-2: {avg_rouge2}")
    print(f"ROUGE-L: {avg_rougeL}")
    
    # Calculate WER
    wer = jiwer.wer(references, predictions)
    print(f"WER: {wer}")
    
    # Calculate Exact Match Accuracy
    exact_matches = 0
    for pred, ref in zip(predictions, references):
        if pred.strip().lower() == ref.strip().lower():
            exact_matches += 1
    accuracy = exact_matches / len(predictions) if len(predictions) > 0 else 0.0
    print(f"Accuracy: {accuracy:.4f}")
    
    # Save Results
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(f"BLEU-1: {b1_score}\n")
        f.write(f"BLEU-2: {b2_score}\n")
        f.write(f"BLEU-3: {b3_score}\n")
        f.write(f"BLEU-4: {b4_score}\n")
        f.write(f"ROUGE-1: {avg_rouge1}\n")
        f.write(f"ROUGE-2: {avg_rouge2}\n")
        f.write(f"ROUGE-L: {avg_rougeL}\n")
        f.write(f"WER: {wer}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        for pred, ref in zip(predictions, references):
            f.write(f"Ref:  {ref}\n")
            f.write(f"Pred: {pred}\n")
            f.write("-" * 50 + "\n")
            
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    evaluate()

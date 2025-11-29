import torch
import torch.nn as nn
from transformers import AutoTokenizer, UMT5ForConditionalGeneration, UMT5Config
from modules.pose_encoder import PoseEncoder, PoseEncoderConfig
import argparse
import os
import numpy as np
import pickle

class SignLanguageModel(nn.Module):
    def __init__(self, pose_encoder, umt5_model):
        super().__init__()
        self.pose_encoder = pose_encoder
        self.umt5_model = umt5_model
        
    def forward(self, pose, pose_attention_mask, labels=None, decoder_attention_mask=None):
        pose_embeddings = self.pose_encoder(pose, attention_mask=pose_attention_mask)
        outputs = self.umt5_model(
            encoder_outputs=(pose_embeddings,),
            attention_mask=pose_attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        )
        return outputs
    
    def generate(self, pose, pose_attention_mask, **kwargs):
        pose_embeddings = self.pose_encoder(pose, attention_mask=pose_attention_mask)
        return self.umt5_model.generate(
            encoder_outputs=(pose_embeddings,),
            attention_mask=pose_attention_mask,
            **kwargs
        )

def parse_args():
    parser = argparse.ArgumentParser(description="Inference for Sign Language Recognition")
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

    model = SignLanguageModel(pose_encoder, umt5_model)
    model.to(device)
    model.eval()
    
    return model, tokenizer

def preprocess_input(args, sample_id):
    with open(args.input_file, 'rb') as f:
        data = pickle.load(f)
    
    if sample_id not in data:
        raise ValueError(f"Sample ID {sample_id} not found in {args.input_file}")
        
    pose, pose_attention_mask = preprocess_input(args, args.sample_id)
    
    pose = pose.to(device)
    pose_attention_mask = pose_attention_mask.to(device)
    
    print("Generating...")
    with torch.no_grad():
        generated_ids = model.generate(
            pose, 
            pose_attention_mask, 
            max_length=128, 
            num_beams=4, 
            early_stopping=True
        )
    
    decoded_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Generated Gloss: {decoded_text}")

if __name__ == "__main__":
    main()

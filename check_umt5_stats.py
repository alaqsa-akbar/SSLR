import torch
from transformers import UMT5EncoderModel, UMT5Config, AutoTokenizer
import os

def check_stats():
    print("Loading UMT5...")
    try:
        config = UMT5Config.from_pretrained("models/umt5")
        if config.d_model != 768:
            config = UMT5Config(vocab_size=256384, d_model=768, d_kv=64, d_ff=2048, num_layers=12, num_heads=12, is_encoder_decoder=True)
    except:
        config = UMT5Config(vocab_size=256384, d_model=768, d_kv=64, d_ff=2048, num_layers=12, num_heads=12, is_encoder_decoder=True)
        
    model = UMT5EncoderModel(config)
    
    # Load weights if available to get realistic stats
    if os.path.exists("models/umt5/umt5-base.bin"):
        state_dict = torch.load("models/umt5/umt5-base.bin", map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print("Loaded local weights.")
    else:
        print("Using random weights (Stats might be slightly off but indicative of scale if LayerNorm is used).")

    model.eval()
    
    # Create dummy input
    input_ids = torch.randint(0, 256384, (4, 20)) # Batch 4, Seq 20
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        embeddings = outputs.last_hidden_state
        
    print(f"Embedding Shape: {embeddings.shape}")
    print(f"Mean: {embeddings.mean().item():.4f}")
    print(f"Std: {embeddings.std().item():.4f}")
    print(f"Mean Norm: {embeddings.norm(dim=-1).mean().item():.4f}")
    
    # Calculate MSE between two random vectors from this distribution
    # MSE = E[(x-y)^2] = Var(x-y) + Mean(x-y)^2 = Var(x)+Var(y) = 2*Var(x) (assuming mean 0)
    expected_random_mse = 2 * embeddings.var().item()
    print(f"Expected Random MSE: {expected_random_mse:.4f}")

if __name__ == "__main__":
    check_stats()

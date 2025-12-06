import pandas as pd
import json
import os

def build_vocab(csv_files, output_file="vocab.json"):
    special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]
    unique_glosses = set()
    
    print("Scanning files for glosses...")
    for f in csv_files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            # Assuming glosses are space-separated strings like "WOMAN WEAR RED"
            for sentence in df['gloss']:
                if isinstance(sentence, str):
                    # Split by space and add to set
                    for word in sentence.strip().split():
                        unique_glosses.add(word)
    
    # Sort for reproducibility
    sorted_glosses = sorted(list(unique_glosses))
    
    # Create mapping: ID 0-3 are reserved for specials
    vocab = {token: i for i, token in enumerate(special_tokens)}
    start_idx = len(special_tokens)
    
    for i, gloss in enumerate(sorted_glosses):
        vocab[gloss] = start_idx + i
        
    print(f"Total unique glosses found: {len(sorted_glosses)}")
    print(f"Final vocab size (with specials): {len(vocab)}")
    
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
        
    print(f"Saved vocabulary to {output_file}")

if __name__ == "__main__":
    # Point this to your actual data paths
    build_vocab(["data/train.csv", "data/dev.csv"])
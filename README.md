# Sign Language Recognition Project

This project implements a Sign Language Recognition model using a **Pose Encoder** and a **UMT5 Text Encoder/Decoder**. It translates sequences of pose keypoints into text glosses.

## Key Features

*   **Advanced Pose Encoder**:
    *   **Spatial GCN**: Graph Convolutional Network to aggregate features based on skeletal connectivity (Body, Hands, Face).
    *   **RoPE (Rotary Positional Embeddings)**: For robust relative position modeling.
    *   **RMSNorm**: For improved training stability.
    *   **Flash Attention**: Uses PyTorch's optimized `F.scaled_dot_product_attention` (SDPA).
*   **Robust Data Handling**:
    *   **Dynamic Normalization**: Automatically scales inputs to `[0, 1]` based on the maximum value per sample, handling varying resolutions.
    *   **Dynamic Cache**: RoPE cache resizes automatically to the input sequence length.
*   **Configuration**: Hyperparameters managed via `configs/pose_encoder_config.json`.

## Project Structure

- `configs/`: Configuration files (`pose_encoder_config.json`).
- `data/`: Dataset files (downloaded via script).
- `models/`:
    - `umt5/`: Pretrained UMT5 model files.
- `modules/`:
    - `pose_encoder.py`: GCN + Transformer implementation.
    - `dataset.py`: Data loading and dynamic normalization.
    - `umt5/`: UMT5 modeling code.
- `download_resources.py`: Script to download model and data.
- `train_pose_encoder.py`: Contrastive Pretraining script.
- `train_final.py`: Full Fine-tuning script.
- `evaluate.py`: Evaluation script (BLEU/ROUGE).
- `inference.py`: Single-sample inference script.
- `visualize_keypoints.py`: Interactive visualization tool.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    pip install huggingface_hub
    ```

2.  **Download Resources**:
    This script downloads the UMT5 model and your dataset from Hugging Face.
    ```bash
    python download_resources.py --dataset_repo_id <YOUR_HF_USERNAME>/<DATASET_NAME>
    ```

## Training

The training process consists of two stages.

### 1. Contrastive Pretraining

Aligns the Pose Encoder with the frozen UMT5 Encoder using InfoNCE loss.

```bash
python train_pose_encoder.py \
    --epochs 50 \
    --batch_size 64 \
    --pose_config configs/pose_encoder_config.json
```

### 2. Full Fine-tuning

Trains the Pose Encoder and UMT5 Decoder for sign-to-text generation.

```bash
python train_final.py \
    --epochs 50 \
    --batch_size 32 \
    --contrastive_model_dir output/pose_encoder_contrastive \
    --pose_config configs/pose_encoder_config.json
```

## Evaluation

Evaluate the trained model on the development set using BLEU and ROUGE scores.

```bash
python evaluate.py \
    --model_dir output/sign_language_model \
    --output_file evaluation_results.txt \
    --batch_size 8
```

## Inference

Generate glosses from a specific pose sequence.

```bash
python inference.py \
    --model_dir output/sign_language_model/checkpoint-best \
    --input_file data/data.pkl \
    --sample_id <YOUR_SAMPLE_ID>
```

## External Training Guide (RunPod/Vast.ai)

To train on a remote GPU instance:

1.  **Zip the Project**: Compress the entire folder (exclude `venv`, `__pycache__`, `runs`, `output`).
2.  **Upload**: Transfer the zip file to your instance.
3.  **Install & Download**:
    ```bash
    unzip project.zip
    cd project
    pip install -r requirements.txt
    python download_resources.py --dataset_repo_id <YOUR_REPO_ID>
    ```
4.  **Run Training**: Execute the training commands listed above.
5.  **Retrieve Results**: Download the `output/` folder back to your local machine for evaluation.

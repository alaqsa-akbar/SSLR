#!/bin/bash

# Activate venv
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    echo "Warning: Virtual environment not found. Running with system python."
fi

# 1. Pre-train Pose Encoder (Contrastive Loss)
echo "Starting Pose Encoder Pre-training (Contrastive)..."
accelerate launch train_pose_encoder.py --epochs 50 --batch_size 32 --gradient_accumulation_steps 4 --pose_config configs/pose_encoder_config.json --output_dir output/pose_encoder_contrastive

# 2. Train Final Model (Loads Contrastive Encoder)
echo "Starting Final Model Training..."
accelerate launch train_final.py --epochs 50 --batch_size 32 --gradient_accumulation_steps 4 --patience 10 --contrastive_model_dir output/pose_encoder_contrastive --output_dir output/sign_language_model_final

# 3. Evaluate
echo "Starting Evaluation..."
python evaluate.py --model_dir output/sign_language_model_final

echo "Pipeline Completed!"

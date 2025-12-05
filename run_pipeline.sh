#!/bin/bash
set -e  # Exit on error

echo "=============================================="
echo "Sign Language Translation Training Pipeline"
echo "=============================================="

# Configuration
DATA_DIR="data"
MODEL_DIR="models/umt5"
OUTPUT_BASE="output"

# Stage 1: Contrastive Pre-training
CONTRASTIVE_OUTPUT="${OUTPUT_BASE}/pose_encoder_contrastive"

# Stage 2: Final Model Training  
FINAL_OUTPUT="${OUTPUT_BASE}/sign_language_model_final"

# ============================================
# Stage 1: Pre-train Pose Encoder (Contrastive)
# ============================================
echo ""
echo "[Stage 1/3] Contrastive Pre-training..."
echo "=============================================="

accelerate launch --num_processes 2 train_pose_encoder.py \
    --data_dir ${DATA_DIR} \
    --model_dir ${MODEL_DIR} \
    --output_dir ${CONTRASTIVE_OUTPUT} \
    --epochs 50 \
    --batch_size 64 \
    --queue_size 16384 \
    --projection_dim 256 \
    --lr 1e-4 \
    --temperature 0.07 \
    --warmup_ratio 0.1 \
    --pose_config configs/pose_encoder_config.json \
    --save_every 10 \
    --eval_every 1

echo "Contrastive pre-training complete!"
echo "Model saved to: ${CONTRASTIVE_OUTPUT}"

# ============================================
# Stage 2: Train Final Translation Model
# ============================================
echo ""
echo "[Stage 2/3] Final Model Training..."
echo "=============================================="

accelerate launch --num_processes 2 train_final.py \
    --data_dir ${DATA_DIR} \
    --model_dir ${MODEL_DIR} \
    --contrastive_model_dir ${CONTRASTIVE_OUTPUT}/best_model/pose_encoder \
    --output_dir ${FINAL_OUTPUT} \
    --epochs 50 \
    --batch_size 32 \
    --gradient_accumulation_steps 4 \
    --lr 5e-5 \
    --patience 10 \
    --warmup_ratio 0.1 \
    --pose_config configs/pose_encoder_config.json \
    --save_every 5 \
    --eval_every 1 \
    --num_beams 4

echo "Final model training complete!"
echo "Model saved to: ${FINAL_OUTPUT}"

# ============================================
# Stage 3: Evaluate
# ============================================
echo ""
echo "[Stage 3/3] Evaluation..."
echo "=============================================="

python evaluate.py \
    --model_dir ${FINAL_OUTPUT}/best_model \
    --data_dir ${DATA_DIR} \
    --batch_size 16 \
    --num_samples 20 \
    --output_file ${FINAL_OUTPUT}/evaluation_results.json

echo ""
echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo ""
echo "Results:"
echo "  - Contrastive model: ${CONTRASTIVE_OUTPUT}"
echo "  - Final model: ${FINAL_OUTPUT}/best_model"
echo "  - Evaluation: ${FINAL_OUTPUT}/evaluation_results.json"
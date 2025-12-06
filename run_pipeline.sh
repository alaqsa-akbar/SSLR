#!/bin/bash
set -e  # Exit on error

echo "=============================================="
echo "Sign Language Translation Training Pipeline"
echo "=============================================="

# Configuration
DATA_DIR="data"
MODEL_DIR="models/umt5" # Only needed for Stage 1 caching
OUTPUT_BASE="output"
VOCAB_FILE="vocab.json"

# Stage 1: Contrastive Pre-training Output
CONTRASTIVE_OUTPUT="${OUTPUT_BASE}/pose_encoder_contrastive"

# Stage 2: Final Model Training Output
FINAL_OUTPUT="${OUTPUT_BASE}/sign_language_model_final"

# ============================================
# Stage 0: Build Vocabulary (One-time setup)
# ============================================
if [ ! -f "$VOCAB_FILE" ]; then
    echo ""
    echo "[Stage 0/3] Building Vocabulary..."
    python build_vocab.py 
else
    echo ""
    echo "[Stage 0/3] Vocabulary found ($VOCAB_FILE)."
fi

# ============================================
# Stage 1: Pre-train Pose Encoder (Contrastive)
# ============================================
# echo ""
# echo "[Stage 1/3] Contrastive Pre-training..."
# echo "=============================================="

# accelerate launch --num_processes 2 --mixed_precision bf16 train_pose_encoder.py \
#     --data_dir ${DATA_DIR} \
#     --model_dir ${MODEL_DIR} \
#     --output_dir ${CONTRASTIVE_OUTPUT} \
#     --epochs 25 \
#     --batch_size 64 \
#     --lr 1e-5 \
#     --projection_dim 256 \
#     --learnable_temperature \
#     --temperature 0.07 \
#     --warmup_ratio 0.1 \
#     --pose_config configs/pose_encoder_config.json \
#     --save_every 10 \
#     --eval_every 1 \
#     --wandb_project "slr"

# echo "Contrastive pre-training complete!"
# echo "Model saved to: ${CONTRASTIVE_OUTPUT}"

# ============================================
# Stage 2: Train Final Hybrid Model
# ============================================
echo ""
echo "[Stage 2/3] Final Model Training (Hybrid)..."
echo "=============================================="

# We point to the BEST model from stage 1
PRETRAINED_PATH="${CONTRASTIVE_OUTPUT}/best_model/pose_encoder"

accelerate launch --num_processes 2 --mixed_precision bf16 train_final.py \
    --data_dir ${DATA_DIR} \
    --contrastive_model_dir ${PRETRAINED_PATH} \
    --vocab_file ${VOCAB_FILE} \
    --output_dir ${FINAL_OUTPUT} \
    --epochs 150 \
    --batch_size 32 \
    --lr 3e-5 \
    --lambda_ctc 0.7 \
    --patience 15 \
    --warmup_ratio 0.1 \
    --save_every 10 \
    --eval_every 1 \
    --pose_config configs/pose_encoder_config.json \
    --fresh_start \
    --wandb_project "slr"

echo "Final model training complete!"
echo "Model saved to: ${FINAL_OUTPUT}"

# ============================================
# Stage 3: Evaluate
# ============================================
echo ""
echo "[Stage 3/3] Evaluation..."
echo "=============================================="

# Evaluate on DEV set first
# Note: Pointing to pytorch_model.bin ensures we load the weights directly
python evaluate.py \
    --model_dir ${FINAL_OUTPUT}/best_model/pytorch_model.bin \
    --data_dir ${DATA_DIR} \
    --vocab_file ${VOCAB_FILE} \
    --split dev \
    --batch_size 1 \
    --num_beams 1 \
    --output_file ${FINAL_OUTPUT}/evaluation_results.txt \
    --repetition_penalty 1.5

echo ""
echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo ""
echo "Results:"
echo "  - Contrastive model: ${CONTRASTIVE_OUTPUT}"
echo "  - Final model: ${FINAL_OUTPUT}/best_model"
echo "  - Evaluation: ${FINAL_OUTPUT}/evaluation_results.txt"
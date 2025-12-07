#!/bin/bash
set -e  # Exit on error

echo "=============================================="
echo "Sign Language Recognition Training Pipeline"
echo "=============================================="
echo "Using: Grouped GCN + Dual-Stream + Transformer"
echo "=============================================="

# Configuration
DATA_DIR="data"
OUTPUT_BASE="output"
VOCAB_FILE="vocab.json"
POSE_CONFIG="configs/pose_encoder_config.json"

# Output directories
FINAL_OUTPUT="${OUTPUT_BASE}/sign_language_model_final"

# ============================================
# Stage 0: Build Vocabulary (One-time setup)
# ============================================
if [ ! -f "$VOCAB_FILE" ]; then
    echo ""
    echo "[Stage 0/2] Building Vocabulary..."
    python build_vocab.py 
else
    echo ""
    echo "[Stage 0/2] Vocabulary found ($VOCAB_FILE)."
fi

# ============================================
# Stage 1: Train Hybrid Model (CTC + Attention)
# ============================================
echo ""
echo "[Stage 1/2] Training Hybrid Model..."
echo "=============================================="
echo "Architecture:"
echo "  - Grouped GCN (hands, face, body)"
echo "  - Dual-stream (pose + velocity)"
echo "  - Transformer temporal encoder"
echo "  - CTC + Attention decoder"
echo "=============================================="

# Single GPU training
# python train_final.py \
#     --data_dir ${DATA_DIR} \
#     --vocab_file ${VOCAB_FILE} \
#     --pose_config ${POSE_CONFIG} \
#     --output_dir ${FINAL_OUTPUT} \
#     --epochs 150 \
#     --batch_size 32 \
#     --lr 1e-4 \
#     --lambda_ctc 0.3 \
#     --patience 20 \
#     --warmup_ratio 0.1 \
#     --weight_decay 0.01 \
#     --max_grad_norm 1.0 \
#     --save_every 10 \
#     --eval_every 1 \
#     --fresh_start

# For multi-GPU training, uncomment below and comment above:
# accelerate launch --num_processes 2 train_final.py \
#     --data_dir ${DATA_DIR} \
#     --vocab_file ${VOCAB_FILE} \
#     --pose_config ${POSE_CONFIG} \
#     --output_dir ${FINAL_OUTPUT} \
#     --epochs 150 \
#     --batch_size 32 \
#     --lr 5e-5 \
#     --lambda_ctc 0.3 \
#     --patience 20 \
#     --warmup_ratio 0.1 \
#     --save_every 10 \
#     --eval_every 1 \
#     --wandb_project "slr" \
#     --fresh_start

# echo ""
# echo "Training complete! Model saved to: ${FINAL_OUTPUT}"

# ============================================
# Stage 2: Evaluate
# ============================================
echo ""
echo "[Stage 2/2] Evaluation..."
echo "=============================================="

# Evaluate with CTC decoding (faster, good baseline)
echo "Running CTC evaluation..."
python evaluate.py \
    --model_dir ${FINAL_OUTPUT}/best_model \
    --data_dir ${DATA_DIR} \
    --vocab_file ${VOCAB_FILE} \
    --pose_config ${POSE_CONFIG} \
    --split dev \
    --batch_size 32 \
    --decode_method ctc \
    --output_file ${FINAL_OUTPUT}/evaluation_ctc.txt

# Evaluate with Attention decoding (slower, potentially better)
echo ""
echo "Running Attention evaluation..."
python evaluate.py \
    --model_dir ${FINAL_OUTPUT}/best_model \
    --data_dir ${DATA_DIR} \
    --vocab_file ${VOCAB_FILE} \
    --pose_config ${POSE_CONFIG} \
    --split dev \
    --batch_size 1 \
    --decode_method attention \
    --num_beams 5 \
    --output_file ${FINAL_OUTPUT}/evaluation_attention.txt

echo ""
echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo ""
echo "Results:"
echo "  - Model: ${FINAL_OUTPUT}/best_model"
echo "  - CTC Evaluation: ${FINAL_OUTPUT}/evaluation_ctc.txt"
echo "  - Attention Evaluation: ${FINAL_OUTPUT}/evaluation_attention.txt"
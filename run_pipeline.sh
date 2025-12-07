#!/bin/bash
set -e  # Exit on error

echo "=============================================="
echo "Sign Language Recognition Training Pipeline"
echo "=============================================="
echo "Using: Grouped GCN + Dual-Stream + Transformer"
echo "=============================================="

# ============================================
# Configuration
# ============================================
DATA_DIR="data"
OUTPUT_BASE="output"
VOCAB_FILE="vocab.json"
POSE_CONFIG="configs/pose_encoder_config.json"
UMT5_DIR="models/umt5"

# Output directories
PRETRAIN_OUTPUT="${OUTPUT_BASE}/pretrained_encoder"
FINAL_OUTPUT="${OUTPUT_BASE}/sign_language_model_final"

# Training options
USE_PRETRAIN=false        # Set to true to use contrastive pre-training
MULTI_GPU=false           # Set to true for multi-GPU training
NUM_GPUS=2                # Number of GPUs for multi-GPU training

# Hyperparameters
PRETRAIN_EPOCHS=75
PRETRAIN_BATCH_SIZE=64
PRETRAIN_LR=1e-4

TRAIN_EPOCHS=150
TRAIN_BATCH_SIZE=32
TRAIN_LR=5e-5
LAMBDA_CTC=0.3
PATIENCE=25

# ============================================
# Parse command line arguments
# ============================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --pretrain)
            USE_PRETRAIN=true
            shift
            ;;
        --multi-gpu)
            MULTI_GPU=true
            shift
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--pretrain] [--multi-gpu] [--num-gpus N] [--skip-train] [--skip-eval]"
            exit 1
            ;;
    esac
done

if [ "$MULTI_GPU" = true ]; then
    echo "Multi-GPU training enabled with ${NUM_GPUS} GPUs."
else
    echo "Single-GPU training."
fi

# ============================================
# Stage 0: Build Vocabulary (One-time setup)
# ============================================
if [ ! -f "$VOCAB_FILE" ]; then
    echo ""
    echo "[Stage 0] Building Vocabulary..."
    python build_vocab.py 
else
    echo ""
    echo "[Stage 0] Vocabulary found ($VOCAB_FILE)."
fi

# ============================================
# Stage 1: Contrastive Pre-training (Optional)
# ============================================
if [ "$USE_PRETRAIN" = true ]; then
    echo ""
    echo "[Stage 1] Contrastive Pre-training..."
    echo "=============================================="
    echo "Learning pose-to-text alignment"
    echo "  - Encoder learns: 'this pose = this gloss'"
    echo "  - Uses cached UMT5 embeddings for efficiency"
    echo "=============================================="
    
    if [ "$MULTI_GPU" = true ]; then
        accelerate launch --num_processes ${NUM_GPUS} train_pose_encoder.py \
            --data_dir ${DATA_DIR} \
            --pose_config ${POSE_CONFIG} \
            --model_dir ${UMT5_DIR} \
            --output_dir ${PRETRAIN_OUTPUT} \
            --epochs ${PRETRAIN_EPOCHS} \
            --batch_size ${PRETRAIN_BATCH_SIZE} \
            --lr ${PRETRAIN_LR} \
            --projection_dim 256 \
            --temperature 0.07 \
            --learnable_temperature \
            --save_every 10 \
            --eval_every 1 \
            --wandb_project slr
    else
        python train_pose_encoder.py \
            --data_dir ${DATA_DIR} \
            --pose_config ${POSE_CONFIG} \
            --model_dir ${UMT5_DIR} \
            --output_dir ${PRETRAIN_OUTPUT} \
            --epochs ${PRETRAIN_EPOCHS} \
            --batch_size ${PRETRAIN_BATCH_SIZE} \
            --lr ${PRETRAIN_LR} \
            --projection_dim 256 \
            --temperature 0.07 \
            --learnable_temperature \
            --save_every 10 \
            --eval_every 1
    fi
    
    echo ""
    echo "Pre-training complete! Encoder saved to: ${PRETRAIN_OUTPUT}/best_model"
    
    PRETRAINED_ENCODER="${PRETRAIN_OUTPUT}/best_model"
else
    echo ""
    echo "[Stage 1] Skipping contrastive pre-training (use --pretrain to enable)"
    PRETRAINED_ENCODER=""
fi

# ============================================
# Stage 2: Train Hybrid Model (CTC + Attention)
# ============================================
if [ "$SKIP_TRAIN" != true ]; then
    echo ""
    echo "[Stage 2] Training Hybrid Model..."
    echo "=============================================="
    echo "Architecture:"
    echo "  - Grouped GCN (hands, face, body)"
    echo "  - Dual-stream (pose + velocity)"
    echo "  - Spatial Attention Pooling"
    echo "  - Transformer temporal encoder"
    echo "  - CTC + Attention decoder"
    if [ -n "$PRETRAINED_ENCODER" ]; then
        echo "  - Using pretrained encoder: ${PRETRAINED_ENCODER}"
    fi
    echo "=============================================="
    
    # Build the training command
    TRAIN_CMD="--data_dir ${DATA_DIR} \
        --vocab_file ${VOCAB_FILE} \
        --pose_config ${POSE_CONFIG} \
        --output_dir ${FINAL_OUTPUT} \
        --epochs ${TRAIN_EPOCHS} \
        --batch_size ${TRAIN_BATCH_SIZE} \
        --lr ${TRAIN_LR} \
        --lambda_ctc ${LAMBDA_CTC} \
        --patience ${PATIENCE} \
        --warmup_ratio 0.1 \
        --weight_decay 0.01 \
        --max_grad_norm 1.0 \
        --save_every 10 \
        --eval_every 1"
    
    # Add pretrained encoder if available
    if [ -n "$PRETRAINED_ENCODER" ]; then
        TRAIN_CMD="${TRAIN_CMD} --pretrained_encoder ${PRETRAINED_ENCODER}"
    else
        TRAIN_CMD="${TRAIN_CMD} --fresh_start"
    fi
    
    if [ "$MULTI_GPU" = true ]; then
        TRAIN_CMD="${TRAIN_CMD} --wandb_project slr"
        accelerate launch --num_processes ${NUM_GPUS} train_final.py ${TRAIN_CMD}
    else
        python train_final.py ${TRAIN_CMD}
    fi
    
    echo ""
    echo "Training complete! Model saved to: ${FINAL_OUTPUT}"
else
    echo ""
    echo "[Stage 2] Skipping training (--skip-train flag set)"
fi

# ============================================
# Stage 3: Evaluate
# ============================================
if [ "$SKIP_EVAL" != true ]; then
    echo ""
    echo "[Stage 3] Evaluation..."
    echo "=============================================="
    
    # Check if model exists
    if [ ! -d "${FINAL_OUTPUT}/best_model" ]; then
        echo "Error: Model not found at ${FINAL_OUTPUT}/best_model"
        echo "Run training first or check the path."
        exit 1
    fi
    
    # Evaluate with CTC decoding
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
    
    # Evaluate with Attention decoding
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
        --output_file ${FINAL_OUTPUT}/evaluation_attention.txt
else
    echo ""
    echo "[Stage 3] Skipping evaluation (--skip-eval flag set)"
fi

# ============================================
# Summary
# ============================================
echo ""
echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo ""
echo "Configuration used:"
echo "  - Contrastive pre-training: ${USE_PRETRAIN}"
echo "  - Multi-GPU: ${MULTI_GPU}"
echo ""
echo "Output locations:"
if [ "$USE_PRETRAIN" = true ]; then
    echo "  - Pretrained encoder: ${PRETRAIN_OUTPUT}/best_model"
fi
echo "  - Final model: ${FINAL_OUTPUT}/best_model"
if [ "$SKIP_EVAL" != true ]; then
    echo "  - CTC Evaluation: ${FINAL_OUTPUT}/evaluation_ctc.txt"
    echo "  - Attention Evaluation: ${FINAL_OUTPUT}/evaluation_attention.txt"
fi
echo ""
echo "=============================================="
echo ""
echo "Quick commands:"
echo ""
echo "  # Run with pre-training:"
echo "  ./run_pipeline.sh --pretrain"
echo ""
echo "  # Run with multi-GPU:"
echo "  ./run_pipeline.sh --multi-gpu --num-gpus 2"
echo ""
echo "  # Run with both:"
echo "  ./run_pipeline.sh --pretrain --multi-gpu"
echo ""
echo "  # Only evaluate existing model:"
echo "  ./run_pipeline.sh --skip-train"
echo ""
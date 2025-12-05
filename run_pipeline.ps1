# Activate venv
.\venv\Scripts\Activate.ps1

# 1. Pre-train Pose Encoder (Contrastive Loss)
Write-Host "Starting Pose Encoder Pre-training (Contrastive)..."
accelerate launch train_pose_encoder.py --epochs 50 --batch_size 8 --gradient_accumulation_steps 4 --pose_config configs/pose_encoder_config.json --output_dir output/pose_encoder_contrastive

# 2. Train Final Model (Loads Contrastive Encoder)
Write-Host "Starting Final Model Training..."
accelerate launch train_final.py --epochs 50 --batch_size 4 --gradient_accumulation_steps 4 --patience 10 --contrastive_model_dir output/pose_encoder_contrastive --output_dir output/sign_language_model_final

# 3. Evaluate
Write-Host "Starting Evaluation..."
python evaluate.py --model_dir output/sign_language_model_final

Write-Host "Pipeline Completed!"

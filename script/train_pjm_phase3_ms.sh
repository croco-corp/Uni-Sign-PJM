#!/bin/bash
# Faza 3 - pelny fine-tune na PJM multi-speaker split (random, 24k train).
# Start z CSL-News stage-2 checkpoint (NIE z phase2, zeby nie dziedziczyc mode-collapse "I I I").
# Visual rozmrozony, mT5 rozmrozony, bez LoRA. Wszystkie parametry trenowane jednoczesnie.

export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/lib/ollama/cuda_v12:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export PATH="$(dirname "$0")/../.venv/bin:$PATH"

cd "$(dirname "$0")/.."

output_dir=out/pjm_phase3_full_ms
mkdir -p "$output_dir"

# bs=8, ga=2 -> efektywny batch 16 (jak w phase2), ale dwa razy mniejsze zuzycie VRAM per-step
# bo full fine-tune trzyma grady + stany Adama dla wszystkich parametrow (visual + mT5 + pose_proj).
# lr=3e-4 uniform, bo Uni-Sign repo (stage2, stage3) uzywa 3e-4 dla calego modelu naraz.
# epochs=15 zamiast 20 (stage3), bo PJM MS ma 24k probek vs CSL_Daily ~18k, wiec 15*~3000step/ep = ~45k kroki starczy.
deepspeed --include localhost:0 --master_port 29514 fine_tuning.py \
  --batch-size 8 \
  --gradient-accumulation-steps 2 \
  --epochs 15 \
  --warmup-epochs 1 \
  --opt AdamW \
  --lr 3e-4 \
  --weight-decay 0.01 \
  --finetune out/csl_news_stage2/csl_stage2_weight.pth \
  --dataset PJM \
  --task SLT \
  --pjm_split ms \
  --bertscore --num_examples 5 \
  --output_dir "$output_dir" \
  --wandb --wandb_project uni-sign-pjm --wandb_dir ./wandb_logs

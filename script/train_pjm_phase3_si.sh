#!/bin/bash
# Faza 3 - pelny fine-tune na PJM signer-independent split (11k/11k/11k).
# Analogicznie do train_pjm_phase3_ms.sh, tylko inny split i output_dir.
# Start z CSL-News stage-2 checkpoint. Visual + mT5 + pose_proj wszystkie rozmrozone.

export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/lib/ollama/cuda_v12:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export PATH="$(dirname "$0")/../.venv/bin:$PATH"

cd "$(dirname "$0")/.."

output_dir=out/pjm_phase3_full_si
mkdir -p "$output_dir"

# Te same hyperparams co phase3 ms poza epokami - to jest ablacja split.
# SI train = 11k probek vs MS = 24k, wiec 30 epok => ~21k krokow ~= MS (22k krokow @ 15 epok).
# Equal-steps fair comparison: czy SI przegra to nie przez undertrain tylko przez signer shift.
deepspeed --include localhost:0 --master_port 29514 fine_tuning.py \
  --batch-size 8 \
  --gradient-accumulation-steps 2 \
  --epochs 30 \
  --warmup-epochs 2 \
  --opt AdamW \
  --lr 3e-4 \
  --weight-decay 0.01 \
  --finetune out/csl_news_stage2/csl_stage2_weight.pth \
  --dataset PJM \
  --task SLT \
  --pjm_split si \
  --bertscore --num_examples 5 \
  --output_dir "$output_dir" \
  --wandb --wandb_project uni-sign-pjm --wandb_dir ./wandb_logs

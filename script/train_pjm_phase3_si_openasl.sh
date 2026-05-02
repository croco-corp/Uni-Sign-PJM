#!/bin/bash
# Faza 3 ablacja: start z OpenASL pose-only SLT checkpoint (YouTube-ASL pretrain -> OpenASL finetune)
# zamiast CSL-News stage-2. Hipoteza: ASL blizsze PJM niz CSL (wplywy europejskie, wizualnie podobni signerzy).
# Identyczny recipe co train_pjm_phase3_si.sh (CSL base) dla fair porownania zrodla pretrainingu.

export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/lib/ollama/cuda_v12:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export PATH="$(dirname "$0")/../.venv/bin:$PATH"

cd "$(dirname "$0")/.."

output_dir=out/pjm_phase3_full_si_openasl
mkdir -p "$output_dir"

# Wszystko identyczne co train_pjm_phase3_si.sh poza --finetune.
# 30 epok (SI ~11k train -> ~21k krokow, rowne steps MS).
deepspeed --include localhost:0 --master_port 29514 fine_tuning.py \
  --batch-size 8 \
  --gradient-accumulation-steps 2 \
  --epochs 30 \
  --warmup-epochs 2 \
  --opt AdamW \
  --lr 3e-4 \
  --weight-decay 0.01 \
  --finetune pretrained_weight/openasl_pose_only_slt.pth \
  --dataset PJM \
  --task SLT \
  --pjm_split si \
  --bertscore --num_examples 5 \
  --output_dir "$output_dir" \
  --wandb --wandb_project uni-sign-pjm --wandb_dir ./wandb_logs

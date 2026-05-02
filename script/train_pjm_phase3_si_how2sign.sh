#!/bin/bash
# Faza 3 ablacja: SI z How2Sign pose-only SLT checkpoint jako startpoint.
# Trzecia noga ablacji zrodla pretrainingu na splicie signer-independent.

export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/lib/ollama/cuda_v12:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export PATH="$(dirname "$0")/../.venv/bin:$PATH"

cd "$(dirname "$0")/.."

output_dir=out/pjm_phase3_full_si_how2sign
mkdir -p "$output_dir"

# 30 epok (SI ~11k train) dla rownego compute z MS.
deepspeed --include localhost:0 --master_port 29514 fine_tuning.py \
  --batch-size 8 \
  --gradient-accumulation-steps 2 \
  --epochs 30 \
  --warmup-epochs 2 \
  --opt AdamW \
  --lr 3e-4 \
  --weight-decay 0.01 \
  --finetune pretrained_weight/how2sign_pose_only_slt.pth \
  --dataset PJM \
  --task SLT \
  --pjm_split si \
  --bertscore --num_examples 5 \
  --output_dir "$output_dir" \
  --wandb --wandb_project uni-sign-pjm --wandb_dir ./wandb_logs

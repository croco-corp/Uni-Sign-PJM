#!/bin/bash
# Faza 3 ablacja: MS z How2Sign pose-only SLT checkpoint (YouTube-ASL pretrain -> How2Sign SLT finetune).
# Trzecia noga ablacji zrodla pretrainingu (CSL / OpenASL / How2Sign).
# H2S: 79h instructional ASL - mniejsze i waskosc domenowa vs OpenASL (288h open-domain).

export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/lib/ollama/cuda_v12:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export PATH="$(dirname "$0")/../.venv/bin:$PATH"

cd "$(dirname "$0")/.."

output_dir=out/pjm_phase3_full_ms_how2sign
mkdir -p "$output_dir"

# Ten sam recipe co MS CSL/OpenASL (15 epok, bs=8 ga=2, lr=3e-4).
deepspeed --include localhost:0 --master_port 29514 fine_tuning.py \
  --batch-size 8 \
  --gradient-accumulation-steps 2 \
  --epochs 15 \
  --warmup-epochs 1 \
  --opt AdamW \
  --lr 3e-4 \
  --weight-decay 0.01 \
  --finetune pretrained_weight/how2sign_pose_only_slt.pth \
  --dataset PJM \
  --task SLT \
  --pjm_split ms \
  --bertscore --num_examples 5 \
  --output_dir "$output_dir" \
  --wandb --wandb_project uni-sign-pjm --wandb_dir ./wandb_logs

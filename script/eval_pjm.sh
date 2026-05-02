#!/bin/bash
# Faza 1 - zero-shot eval of CSL-News ckpt on PJM.
# Usage: bash script/eval_pjm.sh [ms|si|filtered]   (default: ms)

export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/lib/ollama/cuda_v12:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export PATH="$(dirname "$0")/../.venv/bin:$PATH"

split=${1:-ms}
output_dir=out/pjm_phase1_eval_${split}
mkdir -p "$output_dir"

deepspeed --include localhost:0 --master_port 29511 fine_tuning.py \
  --eval \
  --batch-size 16 \
  --finetune out/csl_news_stage2/csl_stage2_weight.pth \
  --dataset PJM \
  --task SLT \
  --pjm_split "$split" \
  --output_dir "$output_dir" \
  --bertscore \
  --wandb --wandb_project uni-sign-pjm \
  --wandb_dir ./wandb_logs

#!/bin/bash
# Zero-shot eval: openasl pose-only SLT checkpoint na PJM (MS + SI) - BEZ fine-tune na PJM.
# Cel: pokazac ile "czystego" cross-lingual transferu daje openasl encoder.

export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/lib/ollama/cuda_v12:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export PATH="$(dirname "$0")/../.venv/bin:$PATH"

cd "$(dirname "$0")/.."

for split in ms si; do
  output_dir=out/pjm_openasl_zeroshot_${split}
  mkdir -p "$output_dir"
  echo "=== zero-shot openasl eval split=$split start $(date) ==="
  deepspeed --include localhost:0 --master_port 29511 fine_tuning.py \
    --eval \
    --batch-size 16 \
    --finetune pretrained_weight/openasl_pose_only_slt.pth \
    --dataset PJM \
    --task SLT \
    --pjm_split "$split" \
    --output_dir "$output_dir" \
    --bertscore \
    --num_examples 10 \
    --wandb --wandb_project uni-sign-pjm --wandb_dir ./wandb_logs
  echo "=== zero-shot openasl eval split=$split done $(date) ==="
  sleep 15
done

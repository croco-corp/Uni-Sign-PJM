#!/bin/bash
# Uczciwy eval: best_checkpoint.pth z phase3 SI (CSL-News base, epoka 12 - min dev loss)
# Porownanie z final_test z log.txt ktore bylo z epoki 29 (overfitted).

export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/lib/ollama/cuda_v12:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export PATH="$(dirname "$0")/../.venv/bin:$PATH"

cd "$(dirname "$0")/.."

output_dir=out/pjm_phase3_si_best_eval
mkdir -p "$output_dir"

deepspeed --include localhost:0 --master_port 29511 fine_tuning.py \
  --eval \
  --batch-size 16 \
  --finetune out/pjm_phase3_full_si/best_checkpoint.pth \
  --dataset PJM \
  --task SLT \
  --pjm_split si \
  --output_dir "$output_dir" \
  --bertscore \
  --num_examples 10 \
  --wandb --wandb_project uni-sign-pjm --wandb_dir ./wandb_logs

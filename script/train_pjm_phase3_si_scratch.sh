#!/bin/bash
# Faza 3 ablacja: PJM SI bez transferu (Uni-Sign from scratch).
# Identyczne hyperparams co train_pjm_phase3_ms_scratch.sh poza --pjm_split si i output_dir.
# Cel: zmierzyć wartość transferu w Uni-Sign na SI split (kontrast z {CSL-News, OpenASL, How2Sign} na SI).

export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/lib/ollama/cuda_v12:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export PATH="$(dirname "$0")/../.venv/bin:$PATH"

cd "$(dirname "$0")/.."

output_dir=out/pjm_phase3_full_si_scratch
mkdir -p "$output_dir"

deepspeed --include localhost:0 --master_port 29515 fine_tuning.py \
  --batch-size 8 \
  --gradient-accumulation-steps 2 \
  --epochs 15 \
  --warmup-epochs 1 \
  --opt AdamW \
  --lr 3e-4 \
  --weight-decay 0.01 \
  --dataset PJM \
  --task SLT \
  --pjm_split si \
  --bertscore --num_examples 5 \
  --output_dir "$output_dir" \
  --wandb --wandb_project uni-sign-pjm --wandb_dir ./wandb_logs

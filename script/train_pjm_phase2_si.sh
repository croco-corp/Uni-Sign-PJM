#!/bin/bash
# Faza 2 - LoRA fine-tune na PJM signer-independent split (11k train, test speakers disjoint).
# Start z CSL-News stage-2 checkpoint, visual frozen, tylko LoRA na mT5 (q,k,v,o).

export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/lib/ollama/cuda_v12:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export PATH="$(dirname "$0")/../.venv/bin:$PATH"

cd "$(dirname "$0")/.."

output_dir=out/pjm_phase2_lora_si
mkdir -p "$output_dir"

deepspeed --include localhost:0 --master_port 29513 fine_tuning.py \
  --batch-size 16 \
  --gradient-accumulation-steps 1 \
  --epochs 10 \
  --warmup-epochs 1 \
  --lr 3e-4 \
  --finetune out/csl_news_stage2/csl_stage2_weight.pth \
  --dataset PJM \
  --task SLT \
  --pjm_split si \
  --lora --lora_target "q,k,v,o" \
  --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 \
  --freeze_visual \
  --bertscore --num_examples 5 \
  --output_dir "$output_dir" \
  --wandb --wandb_project uni-sign-pjm --wandb_dir ./wandb_logs

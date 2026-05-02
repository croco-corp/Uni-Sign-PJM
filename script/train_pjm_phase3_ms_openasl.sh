#!/bin/bash
# Faza 3 ablacja: MS split z OpenASL pose-only SLT checkpoint jako startpointem.
# Hipoteza: ASL blizej PJM niz CSL (wplywy europejskie, wizualnie podobni signerzy).
# Porownanie z pjm_phase3_full_ms (CSL-News base, BLEU-4=2.51).

export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/lib/ollama/cuda_v12:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export PATH="$(dirname "$0")/../.venv/bin:$PATH"

cd "$(dirname "$0")/.."

output_dir=out/pjm_phase3_full_ms_openasl
mkdir -p "$output_dir"

# Identyczne hyperparams co train_pjm_phase3_ms.sh poza --finetune (fair ablacja zrodla pretrainingu).
# 15 epok, bs=8, ga=2, lr=3e-4, AdamW, wd=0.01.
deepspeed --include localhost:0 --master_port 29514 fine_tuning.py \
  --batch-size 8 \
  --gradient-accumulation-steps 2 \
  --epochs 15 \
  --warmup-epochs 1 \
  --opt AdamW \
  --lr 3e-4 \
  --weight-decay 0.01 \
  --finetune pretrained_weight/openasl_pose_only_slt.pth \
  --dataset PJM \
  --task SLT \
  --pjm_split ms \
  --bertscore --num_examples 5 \
  --output_dir "$output_dir" \
  --wandb --wandb_project uni-sign-pjm --wandb_dir ./wandb_logs

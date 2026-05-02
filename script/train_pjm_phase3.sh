export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/lib/ollama/cuda_v12:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export PATH="$(dirname "$0")/../.venv/bin:$PATH"

output_dir=out/pjm_finetuning

# Stage 2 CSL-News checkpoint as starting point
ckpt_path=out/csl_news_stage2/csl_stage2_weight.pth

deepspeed --include localhost:0 --master_port 29511 fine_tuning.py \
  --batch-size 8 \
  --gradient-accumulation-steps 4 \
  --epochs 20 \
  --opt AdamW \
  --lr 3e-4 \
  --output_dir $output_dir \
  --finetune $ckpt_path \
  --dataset PJM \
  --task SLT \
  --wandb \
  --wandb_project uni-sign-pjm \
  --wandb_dir ./wandb_logs

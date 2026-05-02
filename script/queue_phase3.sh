#!/bin/bash
# Kolejka faza 3: grzecznie ubij faza 2 (SIGTERM, poczekaj na flush checkpointu),
# potem odpal faza 3 MS (full fine-tune z CSL-News base).
# Uruchom w tle:
#   nohup bash script/queue_phase3.sh > phase3_queue.log 2>&1 &
#   echo $! > phase3_queue.pid
#   disown

cd "$(dirname "$0")/.."

echo "=== Phase3 queue start $(date) ==="

# Graceful kill phase 2 jesli wciaz zyje
if [[ -f phase2_queue.pid ]]; then
  p2_pid=$(cat phase2_queue.pid)
  if kill -0 "$p2_pid" 2>/dev/null; then
    echo "[$(date)] Phase 2 queue alive (pid=$p2_pid), sending SIGTERM to group"
    # Ubij cale drzewo: queue_phase2.sh + train_pjm_phase2_ms.sh + deepspeed + fine_tuning
    pkill -TERM -P "$p2_pid" 2>/dev/null
    kill -TERM "$p2_pid" 2>/dev/null
    pkill -TERM -f "train_pjm_phase2_" 2>/dev/null
    pkill -TERM -f "fine_tuning.py" 2>/dev/null
    # Poczekaj do 60s az procesy zejda (pozwoli deepspeed zapisac ckpt jesli byl w trakcie)
    for i in $(seq 1 60); do
      if ! pgrep -f "fine_tuning.py" > /dev/null; then
        echo "[$(date)] Phase 2 processes exited cleanly after ${i}s"
        break
      fi
      sleep 1
    done
    # Jesli wciaz zyja - SIGKILL
    if pgrep -f "fine_tuning.py" > /dev/null; then
      echo "[$(date)] Phase 2 still alive, SIGKILL"
      pkill -KILL -f "fine_tuning.py" 2>/dev/null
      pkill -KILL -f "train_pjm_phase2_" 2>/dev/null
      kill -KILL "$p2_pid" 2>/dev/null
      sleep 3
    fi
  else
    echo "[$(date)] Phase 2 queue pid=$p2_pid not running, skipping kill"
  fi
else
  echo "[$(date)] No phase2_queue.pid file, skipping phase 2 kill"
fi

# Upewnij sie, ze GPU jest wolne
echo "=== GPU status before phase3 ==="
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

# Preflight: krotki dry-run (kilka krokow) aby wykryc OOM zanim zostawimy 15-epokowy run
# Jesli OOM zaraz na starcie - lepiej wiedziec w ciagu 5 min niz po godzinie
echo "=== Phase3 preflight dry-run start $(date) ==="
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/lib/ollama/cuda_v12:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export PATH="$(pwd)/.venv/bin:$PATH"

dry_dir=/tmp/phase3_dryrun_$$
mkdir -p "$dry_dir"
deepspeed --include localhost:0 --master_port 29515 fine_tuning.py \
  --batch-size 8 \
  --gradient-accumulation-steps 2 \
  --epochs 1 \
  --warmup-epochs 0 \
  --opt AdamW \
  --lr 3e-4 \
  --weight-decay 0.01 \
  --finetune out/csl_news_stage2/csl_stage2_weight.pth \
  --dataset PJM \
  --task SLT \
  --pjm_split ms \
  --quick_break 5 \
  --output_dir "$dry_dir" \
  > "$dry_dir/dryrun.log" 2>&1
dry_status=$?
echo "=== Phase3 preflight dry-run done $(date) (exit=$dry_status) ==="

# quick_break 5 => po 5 krokach zapisuje ckpt i wyjdzie (kod w train_one_epoch).
# Jesli OOM => exit != 0 => nie odpalamy pelnego treningu.
if [[ $dry_status -ne 0 ]]; then
  echo "[$(date)] PREFLIGHT FAILED (exit=$dry_status). Last 80 lines of dryrun log:"
  tail -n 80 "$dry_dir/dryrun.log"
  echo "[$(date)] Skipping full phase3 run. Fix OOM/other issue and restart queue manually."
  # Zostaw katalog do debug
  exit 2
fi

# Wyczysc po preflight, zwolnij GPU
rm -rf "$dry_dir"
sleep 5
echo "=== GPU status after preflight ==="
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

echo "=== Phase3 MS start $(date) ==="
bash script/train_pjm_phase3_ms.sh
p3_status=$?
echo "=== Phase3 MS done $(date) (exit=$p3_status) ==="

exit 0

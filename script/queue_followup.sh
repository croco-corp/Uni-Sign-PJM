#!/bin/bash
# Kolejka po queue_si_openasl:
#   1. zero-shot eval openasl na PJM (MS + SI) ~10-15 min
#   2. download how2sign checkpoint (~2 min)
#   3. train how2sign MS (~8h)
#   4. train how2sign SI (~8h, 30 epok)
# Uruchom w tle:
#   nohup bash script/queue_followup.sh > queue_followup.log 2>&1 &
#   echo $! > queue_followup.pid
#   disown

cd "$(dirname "$0")/.."

echo "=== queue_followup start $(date) ==="

# --- Czekaj na zakonczenie queue_si_openasl ---
qpid=$(cat queue_si_openasl.pid 2>/dev/null)
if [[ -n "$qpid" ]] && kill -0 "$qpid" 2>/dev/null; then
  echo "[$(date)] czekam na queue_si_openasl (pid=$qpid)"
  while kill -0 "$qpid" 2>/dev/null; do sleep 60; done
  echo "[$(date)] queue_si_openasl zakonczony"
  sleep 30
else
  echo "[$(date)] queue_si_openasl juz zakonczony albo nie byl odpalony"
fi

# Safety: zadne fine_tuning.py procesy
if pgrep -f "fine_tuning.py" > /dev/null; then
  echo "[$(date)] UWAGA: fine_tuning.py wciaz dziala, czekam 120s"
  sleep 120
  if pgrep -f "fine_tuning.py" > /dev/null; then
    echo "[$(date)] BLAD: fine_tuning.py wciaz dziala, przerywam"
    exit 1
  fi
fi

# --- 1. Zero-shot eval openasl ---
echo "=== zero-shot openasl eval start $(date) ==="
bash script/eval_pjm_openasl_zeroshot.sh
zs_exit=$?
echo "=== zero-shot openasl eval done $(date) (exit=$zs_exit) ==="
sleep 30

# --- 2. Download how2sign ---
if [[ ! -f pretrained_weight/how2sign_pose_only_slt.pth ]]; then
  echo "=== download how2sign start $(date) ==="
  wget -c https://huggingface.co/ZechengLi19/Uni-Sign/resolve/main/how2sign_pose_only_slt.pth \
    -O pretrained_weight/how2sign_pose_only_slt.pth
  dl_exit=$?
  echo "=== download how2sign done $(date) (exit=$dl_exit) ==="
  if [[ $dl_exit -ne 0 ]]; then
    echo "[$(date)] BLAD: download how2sign nieudany, przerywam"
    exit 2
  fi
else
  echo "[$(date)] how2sign_pose_only_slt.pth juz istnieje, skip download"
fi

# Safety przed trainami
sleep 30
if pgrep -f "fine_tuning.py" > /dev/null; then
  echo "[$(date)] UWAGA: fine_tuning.py wciaz dziala, czekam 60s"
  sleep 60
fi

# --- 3. How2Sign MS ---
echo "=== how2sign MS start $(date) ==="
bash script/train_pjm_phase3_ms_how2sign.sh
h2s_ms_exit=$?
echo "=== how2sign MS done $(date) (exit=$h2s_ms_exit) ==="
sleep 30

if pgrep -f "fine_tuning.py" > /dev/null; then
  echo "[$(date)] UWAGA: fine_tuning.py wciaz dziala, czekam 60s"
  sleep 60
fi

# --- 4. How2Sign SI ---
echo "=== how2sign SI start $(date) ==="
bash script/train_pjm_phase3_si_how2sign.sh
h2s_si_exit=$?
echo "=== how2sign SI done $(date) (exit=$h2s_si_exit) ==="

echo "=== queue_followup done $(date) ==="
exit 0

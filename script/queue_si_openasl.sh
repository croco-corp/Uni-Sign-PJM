#!/bin/bash
# Kolejka: czeka az obecny MS openasl zakonczy prace, potem odpala SI openasl.
# Uruchom w tle:
#   nohup bash script/queue_si_openasl.sh > queue_si_openasl.log 2>&1 &
#   echo $! > queue_si_openasl.pid
#   disown

cd "$(dirname "$0")/.."

echo "=== queue_si_openasl start $(date) ==="

ms_pid=$(cat phase3_ms_openasl.pid 2>/dev/null)
if [[ -z "$ms_pid" ]]; then
  echo "[$(date)] BRAK phase3_ms_openasl.pid - odpalam SI od razu"
elif ! kill -0 "$ms_pid" 2>/dev/null; then
  echo "[$(date)] pid=$ms_pid juz nie dziala - odpalam SI od razu"
else
  echo "[$(date)] czekam na zakonczenie MS openasl (pid=$ms_pid)"
  while kill -0 "$ms_pid" 2>/dev/null; do
    sleep 60
  done
  echo "[$(date)] MS openasl zakonczony"
  sleep 30
fi

# Safety check - zadne fine_tuning.py procesy nie zyja
if pgrep -f "fine_tuning.py" > /dev/null; then
  echo "[$(date)] UWAGA: inne fine_tuning.py wciaz dziala, czekam jeszcze 120s"
  sleep 120
  if pgrep -f "fine_tuning.py" > /dev/null; then
    echo "[$(date)] BLAD: fine_tuning.py wciaz dziala po 120s, przerywam kolejke"
    exit 1
  fi
fi

echo "=== GPU status przed SI openasl ==="
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

echo "=== SI openasl start $(date) ==="
bash script/train_pjm_phase3_si_openasl.sh
exit_status=$?
echo "=== SI openasl done $(date) (exit=$exit_status) ==="
exit $exit_status

#!/bin/bash
# Faza 2 - kolejka MS -> SI LoRA fine-tune.
# Uruchom w tle:
#   nohup bash script/queue_phase2.sh > phase2_queue.log 2>&1 &
#   echo $! > phase2_queue.pid
#   disown

cd "$(dirname "$0")/.."

echo "=== MS start $(date) ==="
bash script/train_pjm_phase2_ms.sh
ms_status=$?
echo "=== MS done $(date) (exit=$ms_status) ==="

echo "=== SI start $(date) ==="
bash script/train_pjm_phase2_si.sh
si_status=$?
echo "=== SI done $(date) (exit=$si_status) ==="

exit 0

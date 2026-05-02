#!/bin/bash
# Faza 3 (full fine-tune) zachowana w train_pjm_phase3.sh.
# Ten skrypt jest wpiety w kolejke queue.log jako etap po extract.
# Obecnie uruchamia kolejno Faze 1 (zero-shot eval): najpierw ms, potem si.
# Faza 2 (LoRA) — robimy recznie z rana.

cd "$(dirname "$0")/.."

echo "=== Faza 1 eval MS start $(date) ==="
bash script/eval_pjm.sh ms
ms_status=$?
echo "=== Faza 1 eval MS done $(date) (exit=$ms_status) ==="

echo "=== Faza 1 eval SI start $(date) ==="
bash script/eval_pjm.sh si
si_status=$?
echo "=== Faza 1 eval SI done $(date) (exit=$si_status) ==="

exit 0

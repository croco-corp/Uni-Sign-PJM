#!/bin/bash
# Evaluate best_checkpoint.pth on the test split for all 6 PJM phase3 runs.
# Results land in out/eval_<run_name>/log.txt under "test_bleu4" etc.
# Run from the Uni-Sign directory: bash scripts/eval_best_checkpoints.sh

set -e
cd "$(dirname "$0")/.."

source .venv/bin/activate

declare -A RUNS
RUNS[pjm_phase3_full_ms]="ms"
RUNS[pjm_phase3_full_si]="si"
RUNS[pjm_phase3_full_ms_openasl]="ms"
RUNS[pjm_phase3_full_si_openasl]="si"
RUNS[pjm_phase3_full_ms_how2sign]="ms"
RUNS[pjm_phase3_full_si_how2sign]="si"

for run in pjm_phase3_full_ms pjm_phase3_full_si \
           pjm_phase3_full_ms_openasl pjm_phase3_full_si_openasl \
           pjm_phase3_full_ms_how2sign pjm_phase3_full_si_how2sign; do

    ckpt="out/${run}/best_checkpoint.pth"
    split="${RUNS[$run]}"
    outdir="out/eval_${run}"

    if [ ! -f "$ckpt" ]; then
        echo "SKIP $run — checkpoint not found: $ckpt"
        continue
    fi

    if [ -f "${outdir}/log.txt" ]; then
        echo "SKIP $run — already evaluated (${outdir}/log.txt exists)"
        continue
    fi

    echo "=== Evaluating $run (split=$split) ==="
    mkdir -p "$outdir"

    # Find a free TCP port (start at 29501, increment if busy)
    PORT=29501
    while ss -tln 2>/dev/null | awk '{print $4}' | grep -qE ":$PORT$"; do
        PORT=$((PORT + 1))
    done
    echo "  using master port: $PORT"
    torchrun --nproc_per_node=1 --master_port=$PORT fine_tuning.py \
        --eval \
        --finetune "$ckpt" \
        --dataset PJM \
        --task SLT \
        --pjm_split "$split" \
        --bertscore \
        --num_examples 5 \
        --batch-size 8 \
        --num_workers 4 \
        --output_dir "$outdir" \
        --wandb \
        --wandb_project uni-sign-eval \
        2>&1 | tee "${outdir}/log.txt"

    echo "=== Done $run ==="
done

echo ""
echo "=== Summary ==="
for run in pjm_phase3_full_ms pjm_phase3_full_si \
           pjm_phase3_full_ms_openasl pjm_phase3_full_si_openasl \
           pjm_phase3_full_ms_how2sign pjm_phase3_full_si_how2sign; do
    log="out/eval_${run}/log.txt"
    if [ -f "$log" ]; then
        bleu=$(grep -o '"test_bleu4": [0-9.]*' "$log" | tail -1 | grep -o '[0-9.]*$')
        rouge=$(grep -o '"test_rouge": [0-9.]*' "$log" | tail -1 | grep -o '[0-9.]*$')
        echo "  $run: BLEU-4=$bleu  ROUGE-L=$rouge"
    else
        echo "  $run: no results"
    fi
done

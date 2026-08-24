#!/usr/bin/env bash
# Evaluate ONE MLLM checkpoint on all 4 benchmarks -> 1-row CSV (append-safe).
# Benchmarks: MathVision / MathVerse / MathVista / We-Math (MM-UPT protocol).
#
# Usage:
#   bash eval/run_eval_all.sh --model <ckpt> --tag <name> [--csv <path>] \
#        [--prompt answer|boxed] [--gpu 0] [--limit N] [--out_dir <dir>]
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root

OUT_ROOT="${OUT_ROOT:-./data/mllm_eval}"
# GPU defaults to whatever the caller already selected. CUDA_VISIBLE_DEVICES does NOT
# nest: line ~51 re-exports it for the python child, and that child resolves the
# value against PHYSICAL devices, not against this shell's already-filtered view.
# A bare "0" default therefore moved the engine onto physical GPU0 no matter how the
# caller invoked the script -- someone else's card on a shared box.
MODEL="" TAG="" CSV="" PROMPT="answer" GPU="${CUDA_VISIBLE_DEVICES:-0}" LIMIT="0" OUTDIR=""
while [ $# -gt 0 ]; do case "$1" in
  --model) MODEL="$2"; shift 2;;
  --tag) TAG="$2"; shift 2;;
  --csv) CSV="$2"; shift 2;;
  --prompt) PROMPT="$2"; shift 2;;
  --gpu) GPU="$2"; shift 2;;
  --limit) LIMIT="$2"; shift 2;;
  --out_dir) OUTDIR="$2"; shift 2;;
  *) echo "unknown arg $1"; exit 1;;
esac; done
[ -n "$MODEL" ] || { echo "need --model"; exit 1; }
[ -n "$TAG" ] || TAG="$(basename "$MODEL")"
TS="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${OUTDIR:-work_dirs/eval/${TAG}_${TS}}"
CSV="${CSV:-$OUTDIR/results.csv}"
mkdir -p "$OUTDIR"

# Sanity-check the target GPU. Default is warn-only: refusing would break a caller that
# legitimately shares a card, and that policy is not this script's to impose. Set
# EVAL_STRICT_GPU=1 to make another user's card fatal instead. A comma-separated
# CUDA_VISIBLE_DEVICES is left alone entirely.
if [ -n "$GPU" ] && [ -z "${GPU//[0-9]/}" ]; then
  _ngpu=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  if [ "$GPU" -ge "$_ngpu" ]; then
    echo "[eval] --gpu $GPU is out of range (0..$((_ngpu-1)))" >&2; exit 2
  fi
  _used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
  _uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
  _others=0
  while IFS=, read -r _u _p; do
    [ "$(echo "$_u" | tr -d ' ')" = "$_uuid" ] || continue
    _owner=$(ps -o user= -p "$(echo "$_p" | tr -d ' ')" 2>/dev/null)
    [ -n "$_owner" ] && [ "$_owner" != "$(id -un)" ] && _others=1
  done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader)
  echo "[eval] physical GPU $GPU, ${_used} MiB used, other users on it: $_others"
  if [ "$_others" = "1" ]; then
    echo "[eval] WARNING: GPU $GPU is held by another user." >&2
    if [ "${EVAL_STRICT_GPU:-0}" = "1" ]; then
      echo "[eval] EVAL_STRICT_GPU=1 -> refusing to share." >&2; exit 3
    fi
  fi
fi

# benchmark -> jsonl path (mathvista uses local testmini.jsonl; others data.jsonl)
declare -A DATA=(
  [mathvision]="$OUT_ROOT/mathvision/data.jsonl"
  [mathverse]="$OUT_ROOT/mathverse/data.jsonl"
  [mathvista]="$OUT_ROOT/mathvista/testmini.jsonl"
  [wemath]="$OUT_ROOT/wemath/data.jsonl"
  [corecognition]="$OUT_ROOT/corecognition/data.jsonl"
)
ORDER=(mathvision mathverse mathvista wemath corecognition)
[ -n "${ONLY:-}" ] && ORDER=($ONLY)   # e.g. ONLY=corecognition for a single bench

for b in "${ORDER[@]}"; do
  d="${DATA[$b]}"
  if [ ! -f "$d" ]; then echo "!! missing data for $b: $d (run: python eval/prepare_benchmarks.py $b)"; continue; fi
  if [ -f "$OUTDIR/$b.json" ] && python -c "
import json,sys
o=json.load(open('$OUTDIR/$b.json'))
sys.exit(0 if o.get('n',0)>0 and len(o.get('samples',[]))==o['n'] else 1)" 2>/dev/null; then
    echo "==== [$b] complete, skipping ===="; continue
  fi
  echo "==== [$b] ===="
  CUDA_VISIBLE_DEVICES="$GPU" python eval/eval_mllm.py \
    --model "$MODEL" --data "$d" --image_dir "$(dirname "$d")" \
    --out "$OUTDIR/$b.json" --prompt "$PROMPT" --limit "$LIMIT" \
    --max_tokens "${MAX_TOKENS:-16384}" --max_model_len "${MAX_MODEL_LEN:-24576}" \
    --gpu_mem "${GPU_MEM:-0.92}" 2>&1 | tee -a "$OUTDIR/run.log"
done

# aggregate 4 json -> 1 CSV row (append-safe)
python eval/aggregate_row.py --tag "$TAG" --model "$MODEL" --out_dir "$OUTDIR" --csv "$CSV"
echo ">>> DONE  $TAG  -> $CSV"

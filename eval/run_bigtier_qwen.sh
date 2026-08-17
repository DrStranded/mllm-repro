#!/usr/bin/env bash
# Big-tier Qwen2.5-VL-7B column, all four cells, one GPU, sequential.
#
#   CUDA_VISIBLE_DEVICES=2 bash eval/run_bigtier_qwen.sh --out_root /path/to/out
#
# Each cell writes <out_root>/<tag>/{5 bench json, results.csv}.  Resumable:
# a cell whose json already has n rows is skipped, so re-running after a crash
# picks up where it stopped.
#
# Protocol (frozen 2026-08-17, do not change without re-running the whole table):
#   temperature 0 · max_tokens 16384 · max_model_len 24576 · prompt boxed
#   rule-based grading only (no LLM judge) · 5 benchmarks, AVG5
set -euo pipefail

OUT_ROOT="${OUT_ROOT:-work_dirs/eval_bigtier}"
GT_CKPT="${GT_CKPT:-}"          # local path or HF id of the mmupt-retrained GT
LIMIT="${LIMIT:-0}"             # 0 = full benchmarks; set small for a smoke
while [ $# -gt 0 ]; do
  case "$1" in
    --out_root) OUT_ROOT="$2"; shift 2;;
    --gt_ckpt)  GT_CKPT="$2";  shift 2;;
    --limit)    LIMIT="$2";    shift 2;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

# vLLM spawns its EngineCore in a child process. CUDA is not fork-safe, so the
# default fork start method makes the child fail with "CUDA driver
# initialization failed" whenever the parent has already touched CUDA. Always spawn.
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

HF=q1716523669
declare -a CELLS=(
  "base-qwenvl7b|Qwen/Qwen2.5-VL-7B-Instruct"
  "ttrl-q7b-mmupt|$HF/mllm-mmr1-ttrl-qwen25vl7b-mmupt-full/best"
  "co-q7b-x-i8b|$HF/mllm-cogrpo-heter-qwen25vl-7b-x-internvl35-8b-mmr1-mmupt-groupA-qwen25vl-7b"
)
[ -n "$GT_CKPT" ] && CELLS+=("gt-q7b-mmupt|$GT_CKPT")

echo "[bigtier] $(date --iso-8601=seconds)  out_root=$OUT_ROOT  cells=${#CELLS[@]}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -2

for spec in "${CELLS[@]}"; do
  TAG="${spec%%|*}"; MODEL="${spec#*|}"
  echo "===== [$TAG] $MODEL"
  OUT="$OUT_ROOT/$TAG"; mkdir -p "$OUT"
  # every cell uses boxed: the whole table is scored under one prompt so the
  # cells stay comparable (see doc: eval prompt confound).
  LIMIT="$LIMIT" bash eval/run_eval_all.sh \
      --model "$MODEL" --tag "$TAG" --prompt boxed --gpu 0 \
      --limit "$LIMIT" --out_dir "$OUT" || { echo "[bigtier] $TAG FAILED"; continue; }
  echo "[bigtier] $TAG done  $(date --iso-8601=seconds)"
done

echo "[bigtier] all cells attempted. Per-cell results.csv under $OUT_ROOT/<tag>/"
echo "[bigtier] AVG5 summary:"
python - "$OUT_ROOT" <<'PYEOF'
import json, os, sys
root = sys.argv[1]
B = ["mathvision", "mathverse", "mathvista", "wemath", "corecognition"]
for tag in sorted(os.listdir(root)):
    d = os.path.join(root, tag)
    if not os.path.isdir(d):
        continue
    acc = []
    for b in B:
        f = os.path.join(d, b + ".json")
        if os.path.exists(f):
            acc.append(json.load(open(f))["accuracy"] * 100)
    status = f"{sum(acc)/5:6.2f}" if len(acc) == 5 else f"  {len(acc)}/5 benches"
    print(f"  {tag:24s} {status}")
PYEOF

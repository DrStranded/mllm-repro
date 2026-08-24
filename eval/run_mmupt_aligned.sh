#!/usr/bin/env bash
# One cell under the MM-UPT-aligned protocol (see EVAL_ALIGNED.md): generation at
# T=0.2 / top_p=1.0 on the paper's four benchmarks, then the aligned judge pass, on
# one GPU. The two knobs that differ from the frozen internal protocol are exactly
# temperature and top_p (verified against the MM-UPT repo: their SamplingParams sets
# temperature=0.2 and leaves top_p at the vLLM default of 1.0).
#
# Usage:
#   GPU=4 TAG=co-q7b-old-aligned MODEL=<hf-id-or-local-path> \
#     bash eval/run_mmupt_aligned.sh
#
# MODEL must resolve to a directory/repo whose top level carries config.json; for
# repos that keep weights under best/ or endpoint/, snapshot_download the subfolder
# first and pass the local path.
set -uo pipefail
GPU="${GPU:?set GPU}"
TAG="${TAG:?set TAG}"
MODEL="${MODEL:?set MODEL}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
OUT="${OUT:-$REPO/work_dirs/eval_bigtier}"
E="$REPO/data/mllm_eval"
cd "$REPO"

for B in mathvision mathverse mathvista wemath; do
  D="$E/$B/data.jsonl"; [ "$B" = mathvista ] && D="$E/$B/testmini.jsonl"
  CUDA_VISIBLE_DEVICES=$GPU python eval/eval_mllm.py --model "$MODEL" \
    --data "$D" --image_dir "$E/$B" \
    --out "$OUT/$TAG/$B.json" --prompt boxed \
    --temperature 0.2 --top_p 1.0 --max_tokens 16384 --max_model_len 24576 \
    --gpu_mem "${GPU_MEM:-0.72}" \
    && echo "[$TAG] $B OK" || echo "[$TAG] $B FAILED"
done

CUDA_VISIBLE_DEVICES=$GPU python eval/judge_aligned.py \
  "$OUT/$TAG/mathvision.json:$E/mathvision/data.jsonl" \
  "$OUT/$TAG/mathverse.json:$E/mathverse/data.jsonl" \
  "$OUT/$TAG/mathvista.json:$E/mathvista/testmini.jsonl" \
  "$OUT/$TAG/wemath.json:$E/wemath/data.jsonl"

python - "$OUT/$TAG" <<'PY'
import json, os, sys
d = sys.argv[1]
BS = ["mathvision", "mathverse", "mathvista", "wemath"]
js = [100 * json.load(open(f"{d}/{b}.json"))["accuracy_judged_pred"] for b in BS]
print(f"[{os.path.basename(d)}] judged: " +
      "  ".join(f"{b} {v:.2f}" for b, v in zip(BS, js)) +
      f"   AVG4 {sum(js)/4:.2f}")
PY
echo "######## ${TAG}_ALIGNED_DONE ########"

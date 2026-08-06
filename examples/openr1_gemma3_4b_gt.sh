#!/usr/bin/env bash
# open_r1 / mmr1 · gemma3_4b · google/gemma-3-4b-it · gt (stock GRPO) · stack B (torch2.9/vllm0.11.2/tf4.57.0/ds0.18), ZeRO-3 (no offload).
# HP identical to the source small-tier launcher (trainers/dp-scripts/); only NAS activation / hardcoded secrets / wandb-online removed.
# Requires the mllm-repro env active + MLLM_ENV_READY=1. Gemma is gated: accept the license on its HF page + hf read token.
# NOTE small tier: ZeRO-3 WITHOUT optimizer offload, EB=64 via bs=1 x ga=8 x 8gpu, vllm_mem 0.45, attn=sdpa.
# Dataset is chosen by MLLM_PRE_DIR (openr1_8k vs mmr1_8k) -- see README section 3 step D.
# smoke: MAX_STEPS=1 MAX_SAMPLES=64 bash examples/openr1_gemma3_4b_gt.sh
set -euo pipefail
[ "${MLLM_ENV_READY:-0}" = "1" ] || { echo "[mllm-repro] ERROR: env not activated. Enter the image/conda env and 'export MLLM_ENV_READY=1' (see README)." >&2; exit 1; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"; cd "$REPO_ROOT"
# HF cache: local, overridable, no NAS default.
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"; mkdir -p "$HF_HUB_CACHE"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets}"; mkdir -p "$HF_DATASETS_CACHE"
# wandb: OFFLINE by default (no API key needed); `wandb sync <dir>` later to upload.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-mllm-co-grpo-dp}"
export DISABLE_MLFLOW_INTEGRATION=TRUE
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN (hf read-scope token ; Gemma also needs license acceptance)}"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
# Data/eval paths: required, no NAS default (see setup/prepare_data.sh + bundled data/mathvista).
export MAX_SAMPLES="${MAX_SAMPLES:-8000}"
export MLLM_PRE_DIR="${MLLM_PRE_DIR:?set MLLM_PRE_DIR = preprocessed openr1_8k or mmr1_8k dir (see setup/prepare_data.sh)}"
export MLLM_EVAL_PATH="${MLLM_EVAL_PATH:?set MLLM_EVAL_PATH = mathvista/testmini_150.jsonl (bundled under data/mathvista)}"
export MLLM_EVAL_IMAGE_DIR="${MLLM_EVAL_IMAGE_DIR:?set MLLM_EVAL_IMAGE_DIR = mathvista image root (data/mathvista)}"
DATASET="lmms-lab/multimodal-open-r1-8k-verified"
MODEL="google/gemma-3-4b-it"
VLLM_MEM="${VLLM_MEM:-0.45}"
TS="$(date +%Y%m%d_%H%M%S)"; RUN="openr1_gemma3_4b_gt_${TS}"
BASE_OUT="work_dirs/mllm-co-grpo-dp/$RUN"; mkdir -p "$BASE_OUT"
MAXSTEPS_ARG=""; [ -n "${MAX_STEPS:-}" ] && MAXSTEPS_ARG="--max_steps ${MAX_STEPS}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" accelerate launch \
    --config_file trainers/accelerate_zero3.yaml \
    --num_processes 8 --main_process_port 19415 --gradient_accumulation_steps ${GA:-8} \
    trainers/train_mllm_single.py \
    --model_name_or_path "$MODEL" --train_dataset "$DATASET" \
    --output_dir "$BASE_OUT" --run_config "$RUN" \
    --learning_rate 1e-6 \
    --per_device_train_batch_size ${BS:-1} --gradient_accumulation_steps ${GA:-8} \
    --num_train_epochs 1 ${MAXSTEPS_ARG} \
    --lr_scheduler_type cosine_with_min_lr --lr_scheduler_kwargs '{"min_lr_rate": 0.1}' --warmup_ratio 0.03 \
    --gradient_checkpointing --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --max_completion_length 1024 --num_generations 8 --temperature 1.0 \
    --use_vllm --vllm_mode colocate --vllm_max_model_length 4096 \
    --vllm_gpu_memory_utilization "$VLLM_MEM" --vllm_importance_sampling_mode token_truncate \
    --logging_steps 1 --save_strategy steps --save_steps ${SAVE_STEPS:-20} \
    --eval_strategy steps --eval_steps ${EVAL_STEPS:-20} \
    --num_generations_eval 1 --per_device_eval_batch_size 1 \
    --adam_beta2 0.95 --beta 0 --loss_type bnpo --scale_rewards group \
    --seed 42 --data_seed 42 --report_to wandb --wandb_project mllm-co-grpo-dp \
    --attn_implementation sdpa --bf16 true 2>&1 | tee -a "$BASE_OUT/train.log"

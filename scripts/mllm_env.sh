#!/usr/bin/env bash
# scripts/mllm_env.sh — minimal env helper for the mllm-repro repo.
# The Docker image already exports all of these; source this ONLY for a
# bare-metal / non-container run:   source scripts/mllm_env.sh
# NO secrets and NO site-specific (NAS) paths. Supply HF_TOKEN / WANDB_API_KEY
# via your shell or a .env (never commit them).

# HuggingFace cache: single root. NEVER set TRANSFORMERS_CACHE (silently
# redirects HF lookups to an empty dir -> breaks offline mode).
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
unset TRANSFORMERS_CACHE 2>/dev/null || true
# Block a mismatched ~/.local flash_attn/torch from leaking in.
export PYTHONNOUSERSITE=1
# Env already provisioned -> launch scripts skip any site-specific activator.
export MLLM_ENV_READY=1
# W&B default OFF. Flip to online + set WANDB_API_KEY yourself.
export WANDB_MODE="${WANDB_MODE:-offline}"
echo ">>> mllm env ready (HF_HOME=$HF_HOME, WANDB_MODE=$WANDB_MODE)"

# syntax=docker/dockerfile:1
# ============================================================================
# mllm-repro, reproducibility image for the MLLM co-GRPO-DP experiments.
#
# STACK B (the FROZEN, PROVEN Anvil env `mllm-cogrpodp-v2`; see DOCKER_SLIM_PLAN
# §11.1 / §11.9):
#     torch 2.9.x + cu128   (provided by the pytorch base image, NOT reinstalled)
#     vllm  0.11.2
#     transformers 4.57.0
#     deepspeed 0.18.0
#     flash-attn 2.8.3      (built from source in stage 1, wheel copied into stage 2)
#     trl   DrStranded fork @ 9881fe1 (token_truncate importance-sampling mode)
#
# The full 244-package closure is pinned in docker/constraints.txt (a real
# `pip freeze` of the working Anvil env). We replay it with --no-deps so pip runs
# NO resolver and can never clobber the base image's torch/CUDA stack.
#
# This is a build-and-iterate runbook image for a Claude Code agent, not a
# push-button appliance. Every version / flag below is deliberate, read the
# inline comments before changing anything.
# ============================================================================


# ----------------------------------------------------------------------------
# Stage 1: build flash-attn 2.8.3 from source against the SAME torch the runtime
# stage uses (so the ABI matches). We use the -devel base so nvcc is present.
# ----------------------------------------------------------------------------
FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel AS fa-build

ARG FA_VERSION=2.8.3
# MAX_JOBS caps parallel nvcc, flash-attn units are RAM-hungry (~3-5 GB each).
# Lower to 2 if the builder OOMs; raise via --build-arg MAX_JOBS=8 on a big host.
ARG MAX_JOBS=4
# flash-attn 2.8.3 IGNORES TORCH_CUDA_ARCH_LIST, it reads FLASH_ATTN_CUDA_ARCHS.
# 80 = A100 (sm_80), 90 = H100 (sm_90). Add "100" here for Blackwell if needed.
ENV MAX_JOBS=${MAX_JOBS} \
    FLASH_ATTENTION_FORCE_BUILD=TRUE \
    FLASH_ATTN_CUDA_ARCHS="80;90"

# git: some flash-attn build paths touch submodules. build-essential: host g++
# for nvcc. The pytorch base ships nvcc but not always a system g++.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir packaging ninja wheel setuptools

# --no-build-isolation → build against the base image's torch (2.9.0), NOT a
# fresh PEP517 torch. --no-deps → produce only the flash-attn wheel.
RUN pip wheel --no-build-isolation --no-deps "flash-attn==${FA_VERSION}" -w /wheels \
    && ls -la /wheels


# ----------------------------------------------------------------------------
# Stage 2: runtime image.
# Kept on the -devel base (NOT -runtime) on purpose: DeepSpeed's cpu_adam JIT-
# compiles at the first optimizer step and needs nvcc + g++ (§11.5 #4). Keeping
# the toolchain means the first step just works instead of crashing. (Alternative
# noted below: prebuild with DS_BUILD_CPU_ADAM=1.)
# ----------------------------------------------------------------------------
FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel

# Stack-B pins. These MUST stay at the frozen values (§11.1), do NOT bump to the
# aspirational torch2.10 / vllm0.18 / tf4.57.6 stack, which was never validated.
ARG TRANSFORMERS_VERSION=4.57.0
ARG VLLM_VERSION=0.11.2
# Matches the `trl @ git+...@<sha>` line in docker/constraints.txt.
ARG TRL_REF=9881fe1e14fb2a628165a75dcd309e75ef7930d4

# --- Environment (§11.5 #3/#4/#5/#6 + the TRANSFORMERS_CACHE trap) ------------
#  PYTHONNOUSERSITE=1  → block ~/.local site-packages leaking a mismatched
#                        flash_attn/torch into the env (a real Anvil footgun).
#  HF_HOME=/cache/hf   → single HF cache root (bind-mount it at runtime).
#  MLLM_ENV_READY=1    → tells the launch scripts the env is already provisioned,
#                        so they skip any site-specific activator step.
#  WANDB_MODE=offline  → default to local logging; runtime .env can flip it.
#  We deliberately DO NOT set TRANSFORMERS_CACHE (it silently redirects HF
#  lookups to an empty dir → offline LocalEntryNotFoundError), and we deliberately
#  DO NOT set HF_HUB_ENABLE_HF_TRANSFER (hf_transfer is NOT in the closure; =1
#  would hard-fail every download, §11.5 #3).
ENV PYTHONNOUSERSITE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/hf \
    WANDB_MODE=offline \
    MLLM_ENV_READY=1 \
    DISABLE_MLFLOW_INTEGRATION=TRUE \
    VLLM_WORKER_MULTIPROC_METHOD=spawn

# System deps:
#   git            → pip install trl from git+https.
#   tini           → PID 1 init (reaps the many vllm/ray/accelerate children).
#   build-essential→ DeepSpeed cpu_adam JIT (see stage banner above).
#   libglib2.0-0   → opencv-python-headless runtime dep (§11.5 #3), needed by the
#                    image preprocessing path.
#   ca-certificates→ TLS for pip/git/HF over the (online) Google egress proxy.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git tini build-essential libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /data /cache/hf /outputs /workspace
WORKDIR /workspace

# --- Python packages ---------------------------------------------------------
# constraints.txt is a COMPLETE pip-freeze lockfile (every transitive dep pinned),
# so it can be replayed with --no-deps: pip installs each exact version and runs
# no dependency resolution, which is what keeps it from touching the base torch.
COPY docker/constraints.txt /tmp/constraints.txt

# Strip the packages the base image OWNS and that must NOT be reinstalled:
#   torch / torchvision / torchaudio → already in the base; their frozen "+cu128"
#     local-version tags aren't even on PyPI, so any attempt to fetch them fails.
#   trl @ git+...  → a direct-URL line is illegal inside a constraints file AND we
#     install it explicitly below.
# Everything else (incl. the nvidia-*-cu12 CUDA wheels, triton, vllm, xformers,
# flashinfer) is kept and installed at its frozen version.
RUN grep -vE '^(torch|torchvision|torchaudio)==|^trl @' /tmp/constraints.txt \
        > /tmp/constraints.build.txt

# transformers + vllm FIRST, with --no-deps: both hard-pin an exact torch, and
# --no-deps stops pip from dragging that pin in and clobbering the base torch.
RUN pip install --no-cache-dir --no-deps \
        "transformers==${TRANSFORMERS_VERSION}" \
        "vllm==${VLLM_VERSION}"

# ...then the rest of the frozen closure, also --no-deps (complete lockfile → no
# resolver, base torch/CUDA left untouched). If a runtime smoke test later reports
# a missing CUDA lib, add that one pinned nvidia-*-cu12 wheel from constraints.txt.
RUN pip install --no-cache-dir --no-deps -r /tmp/constraints.build.txt

# flash-attn: the wheel compiled in stage 1 (ABI-matched to this base's torch).
COPY --from=fa-build /wheels/*.whl /tmp/wheels/
RUN pip install --no-cache-dir --no-deps /tmp/wheels/*.whl

# trl: the DrStranded fork (the token_truncate importance-sampling mode the mllm
# trainers require). --no-deps; SHA matches the constraints.txt trl line.
RUN pip install --no-cache-dir --no-deps \
        "trl @ git+https://github.com/DrStranded/trl.git@${TRL_REF}"

# Belt-and-suspenders: the grader is the qwen-sympy 2-hop wrapper, NOT math-verify;
# verify.py asserts math-verify is absent. Nothing above installs it (--no-deps),
# but drop it if some base layer smuggled it in.
RUN pip uninstall -y math-verify latex2sympy2_extended 2>/dev/null || true

# --- Repo code (slim subset; no paper docs / results / work_dirs) ------------
# NOTE: examples/ and setup/ are populated by their owning tasks (cleaned launch
# scripts + resolve_flash_attn.sh / prefetch / prepare_data). They must be non-
# empty before a real build if you intend to run the examples.
COPY trainers/ /workspace/trainers/
COPY tools/    /workspace/tools/
COPY eval/     /workspace/eval/
COPY examples/ /workspace/examples/
COPY setup/    /workspace/setup/
COPY tools/verify.py /workspace/tools/verify.py

# --- Verification: RUNTIME ONLY, never at build time -------------------------
# Do NOT run verify.py during build (§11.5 #2): its check2/5/6 import vllm and
# trip a qwen3_vl AutoConfig assertion / GPU paths that ABORT the build. Run it
# once on a GPU node after the image is up:
#
#     docker run --rm --gpus all mllm-repro:th2.9-cu128-vllm0.11.2 python tools/verify.py
#
# (Alternative to keeping build-essential for DeepSpeed cpu_adam: prebuild it with
#  `DS_BUILD_CPU_ADAM=1 pip install --no-deps --force-reinstall deepspeed==0.18.0`
#  and you may then base stage 2 on -runtime. Kept toolchain = simpler + JIT works.)

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]

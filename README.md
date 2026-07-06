# mllm-repro — big-tier MLLM RL reproduction runbook

Private internal reproducibility repo. **Audience: a Claude Code agent that reads and
executes this file top-to-bottom.** Every step is an explicit command. Paths are
repo-relative unless stated; run everything from the repo root unless told otherwise.
Do not improvise stack versions or hyperparameters — they are frozen and load-bearing
(see §4). When a step says "override via env", export the variable *before* the
`bash examples/...` call so it wins over the launcher's built-in default.

---

## 1. What this reproduces

Nine "big-tier" (7B/8B/12B) multimodal RL runs — the §5.6 scale row of the paper. Same
recipe as the small-model main table, only the models are larger. Three training methods
× three models:

- **GT** — stock GRPO against the dataset's ground-truth answer.
- **TTRL** — self-label majority-vote reward (no ground truth).
- **co-learn** — two *different* models train jointly, each rewarding the other with its
  pseudo-labels over a file rendezvous (heterogeneous cross-supervision). A "co-learn"
  run is therefore a **model pair**, run 4+4 on one node.

Models: `Qwen/Qwen2.5-VL-7B-Instruct`, `OpenGVLab/InternVL3_5-8B-HF`,
`google/gemma-3-12b-it` (Gemma has no 8B; 12B is the nearest same-tier size).

| # | example script (in `examples/`) | method | model(s) | attn | GPU layout |
|---|---|---|---|---|---|
| 1 | `qwen25vl7b_gt.sh`                     | GT       | Qwen2.5-VL-7B                | FA2        | 8 |
| 2 | `qwen25vl7b_ttrl.sh`                   | TTRL     | Qwen2.5-VL-7B                | FA2        | 8 |
| 3 | `internvl35_8b_gt.sh`                  | GT       | InternVL3.5-8B               | FA2        | 8 |
| 4 | `internvl35_8b_ttrl.sh`               | TTRL     | InternVL3.5-8B               | FA2        | 8 |
| 5 | `gemma3_12b_gt.sh`                     | GT       | Gemma3-12B **(gated)**       | **sdpa**   | 8 |
| 6 | `gemma3_12b_ttrl.sh`                  | TTRL     | Gemma3-12B **(gated)**       | **sdpa**   | 8 |
| 7 | `heter_qwen25vl7b_x_internvl35_8b.sh` | co-learn | Qwen-7B × InternVL-8B        | FA2 × FA2  | 4+4 |
| 8 | `heter_qwen25vl7b_x_gemma3_12b.sh`    | co-learn | Qwen-7B × Gemma-12B **(gated)** | FA2 × sdpa | 4+4 |
| 9 | `heter_internvl35_8b_x_gemma3_12b.sh` | co-learn | InternVL-8B × Gemma-12B **(gated)** | FA2 × sdpa | 4+4 |

> Exact filenames may differ slightly — run `ls examples/` and match on method+model.
> **#5, #6, #8, #9 use Gemma → they need an accepted license + `HF_TOKEN` (§3 step B).**

**Both datasets.** The recipe is dataset-agnostic; each experiment can be run on either
training set:

- `open_r1` = `lmms-lab/multimodal-open-r1-8k-verified` (**default**, baked into every launcher)
- `mmr1` = the MMR1 math set (source id pinned in `setup/prepare_data.sh`)

Selection is by **which preprocessed dir the launcher loads**, via `MLLM_PRE_DIR`
(`openr1_8k` vs `mmr1_8k`) — see §3 step D and the mmr1 example at the end of §3.
Grader口径 differs by dataset: mmr1 → `GRADE_MCQ_MAP=1`, open_r1 → `GRADE_MCQ_MAP=0` (§7).

Do **not** hand-tune hyperparameters. The launchers hard-code the frozen recipe (kept
identical across GT/TTRL/co-learn for a fair comparison): `lr=1e-6`, `num_generations=8`,
`max_completion_length=1024`, `temperature=1.0`, `beta=0`, `loss_type=bnpo`,
`scale_rewards=group`, 1 epoch, `seed=42`, effective batch **EB=64**
(single-model `bs=2 × ga=4 × 8gpu`; co-learn `bs=2 × ga=8 × 4gpu`), and
`--vllm_importance_sampling_mode token_truncate` (mandatory — without it importance
weights collapse to ~1e-5). Model-specific, also baked in and **not** to be changed:
Gemma → `--attn_implementation sdpa` (FA2 crashes it in this config) + EOS/ZeRO-3 init
fix; InternVL → the `-HF` variant + FA2. Only the env knobs in §8 are meant to be tuned.

---

## 2. Hardware floor

- **8×80GB GPUs on a single node.** Non-negotiable. Single-model runs use all 8; co-learn
  splits 4+4 (model A = GPU0-3, model B = GPU4-7) and exchanges pseudo-labels over a
  file rendezvous. 4×80GB is **not** enough for any of the nine.
- **Large host RAM.** All runs use full-parameter ZeRO-3 with **optimizer CPU offload**
  (`trainers/accelerate_zero3_offload.yaml`). The fp32 Adam m/v/master states live in
  host RAM; for the 12B models that is on the order of ~150GB resident. **Provision
  ≥256GB host RAM.** Without offload, 12B full-param 4+4 co-learn OOMs (model states
  ~48GB/card + vLLM ~24GB + activations > 80GB). With offload, GPU model-state drops to
  ~12GB/card and it fits.
- **Shared memory / IPC.** ZeRO offload + NCCL + vLLM need real shared memory. Launch the
  container with `--ipc=host --shm-size=32g` (Docker) or the apptainer equivalent (§7).
  A default 64MB `/dev/shm` will hang or crash NCCL/dataloaders.
- **Measured peak VRAM** (1-step smoke, bf16, grad-checkpointing, ZeRO-3+offload): Qwen-7B ×
  InternVL-8B co-learn ≈ **63.7GB/80** (~17GB headroom). The **Gemma-12B pairs (#8/#9)
  are the tightest** — 1-step passes but completions lengthen during training and VRAM
  keeps climbing; watch GPU4-7 and drop `VLLM_MEM=0.40` if it approaches the limit (§8).
- Arch/driver requirements: see §5.

---

## 3. Quickstart — run these IN ORDER

Do steps A→H. A/B are one-time setup; F/G/H repeat per experiment.

### A. Build or pull the image

```bash
# Build from the pinned Dockerfile (frozen stack + toolchain for cpu_adam JIT):
docker build -t mllm-repro:th2.9-cu128-vllm0.11.2 .
#   (Dockerfile + docker-compose.yml are at the repo root; build context = the repo root)

# — or pull from the private registry (internal GHCR; ask the repo owner for the ref):
# docker pull <internal-registry>/mllm-repro:th2.9-cu128-vllm0.11.2
```

The image bakes the frozen stack (§4), installs `docker/constraints.txt`, resolves
flash-attn (§4), pre-compiles DeepSpeed CPU-Adam so the first optimizer step does not
JIT-compile at runtime, and sets `MLLM_ENV_READY=1` so the launchers skip the (absent)
ByteDance NAS activation and use the container's Python directly.

Run an interactive container (mount host cache/data/outputs so nothing re-downloads and
checkpoints survive the container):

```bash
docker run --rm -it --gpus all --ipc=host --shm-size=32g \
  -e HF_TOKEN="${HF_TOKEN:?set HF_TOKEN in your host shell}" \
  -e HF_HOME=/cache/hf -e HF_HUB_CACHE=/cache/hf/hub -e HF_DATASETS_CACHE=/cache/hf/datasets \
  -v /host/hf_cache:/cache/hf \
  -v /host/mllm_data:/data \
  -v "$PWD:/workspace" -w /workspace \
  mllm-repro:th2.9-cu128-vllm0.11.2 bash
```

Everything below runs **inside** that container from `/workspace`.

### B. Accept the Gemma license + set HF_TOKEN  (required for #5, #6, #8, #9 only)

`google/gemma-3-12b-it` is gated. On the Hugging Face website, accept the Gemma license
with the account that owns your token, then export a **read**-scope token:

```bash
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN}"        # never hard-code a token in a file
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
huggingface-cli whoami                              # sanity: should print your username
```

Skip this if you are only running the Qwen/InternVL experiments (#1-#4, #7).

### C. Prefetch model weights

```bash
bash setup/prefetch_models.sh
```

Downloads the 5 model repos (Qwen2.5-VL-7B, InternVL3.5-8B, Gemma3-12B, and the two
peers) into `HF_HUB_CACHE`. Gemma downloads only if `HF_TOKEN` is set and the license is
accepted (step B). On an online host this is optional but strongly recommended (it turns
the first training step from a multi-hundred-GB download into a cache hit); on an
offline/air-gapped compute node it is **mandatory** and must be done from a login node
(§6).

### D. Prepare data

```bash
bash setup/prepare_data.sh          # preprocesses BOTH train sets + fetches eval benchmarks
```

This does two things:

1. **Training sets** → preprocesses open_r1 and mmr1 into
   `$PRE_ROOT/{openr1_8k,mmr1_8k}` (via `tools/preprocess_mllm_dataset.py`).
   The launchers read these through `MLLM_PRE_DIR`.
2. **Eval benchmarks** → `eval/prepare_benchmarks.py` fetches MathVision
   (`MathLLMs/MathVision`, 3040), MathVerse (`AI4Math/MathVerse` testmini, 3940), and
   We-Math (`We-Math/We-Math`, 1740) into `$OUT_ROOT/{mathvision,mathverse,wemath}/`.
   MathVista (1000, `testmini.jsonl`) ships in the repo at `data/mathvista/` — including
   the 150-example in-loop subset used to pick the best checkpoint during training — so it
   is **not** downloaded.

Set the local roots once (these override the launchers' baked-in NAS defaults — see §8;
NAS paths are never reachable here):

```bash
export PRE_ROOT=/data                         # any writable local path
export OUT_ROOT=$PRE_ROOT/mllm_eval           # eval benchmark root (used by run_eval_all.sh)
export MLLM_PRE_DIR=$PRE_ROOT/openr1_8k     # default training set
export MLLM_EVAL_PATH=$PWD/data/mathvista/testmini_150.jsonl # in-loop val (bundled)
export MLLM_EVAL_IMAGE_DIR=$PWD/data/mathvista
```

### E. Validate with a smoke run (do this before any full run)

```bash
bash examples/smoke.sh              # 1-2 steps, tiny sample; asserts no OOM / IS≈1.0 / clean exit
```

`smoke.sh` is `MAX_STEPS=1 MAX_SAMPLES=64` over a representative config. It must exit
cleanly with importance-sampling mean ≈ 1.0 and no `inhomogeneous shape` (Gemma) before
you commit a multi-hour full run. To smoke a *specific* experiment instead:

```bash
MAX_STEPS=1 MAX_SAMPLES=64 bash examples/<exp>.sh
```

For a Gemma pair, confirm the smoke log shows `sdpa`, the Gemma EOS id, the ZeRO-3
init-fix, and transformers 4.57.x — those four are the load-bearing Gemma fixes.

### F. Run a full experiment

```bash
bash examples/<exp>.sh              # e.g. examples/openr1_qwen25vl7b_gt.sh
```

One command, self-contained. A full run is ~1 epoch (≈1000 steps at EB=64). Checkpoints
land in `work_dirs/mllm-co-grpo-dp/<run>/` (co-learn writes `model_a/` and `model_b/`).
The in-loop MathVista-150 eval runs every `EVAL_STEPS` and selects the best checkpoint.

To run **mmr1** instead of the default open_r1, point `MLLM_PRE_DIR` at the mmr1 dir:

```bash
MLLM_PRE_DIR=$PRE_ROOT/mmr1_8k bash examples/openr1_qwen25vl7b_gt.sh
```

### G. Final evaluation (post-training, 4 benchmarks)

Pick the best checkpoint (highest in-loop MathVista-150), then evaluate it greedily on all
four benchmarks:

```bash
bash eval/run_eval_all.sh \
  --model work_dirs/mllm-co-grpo-dp/<run>/best_model \
  --tag <exp_name> --csv work_dirs/eval/results.csv \
  --prompt answer                    # trained ckpt → "answer"; untrained base → "boxed"
```

Greedy (T=0), `--max_tokens` (default 4096), rule-based grader (`eval/grade.py` — `<answer>` →
`\boxed` → MCQ letter↔value from the question's options, **no LLM judge**). It writes one CSV row
(`tag,model,mathvision,mathverse,mathvista,wemath,avg`). If **We-Math** stalls (long outputs oversubscribe the vLLM KV cache), lower `--limit` or run `eval/eval_mllm.py` directly with a smaller `--max_tokens` — see §8. The grader maps MCQ letter↔value automatically (grade.py), so no `GRADE_MCQ_MAP` flag is needed.

### H. Compare to expected results

```bash
cat EXPECTED_RESULTS.md
```

Match your per-benchmark and `avg` numbers against `EXPECTED_RESULTS.md` within its stated
tolerance band. The target is "reproduces within tolerance", **not** bit-identical numbers
— RL has real run-to-run variance and the reference numbers were produced on different
(pod) hardware; the tolerance band absorbs that.

---

## 4. Stack note (frozen)

This repo pins the **frozen, proven Anvil environment** — a real machine on which every
one of these runs was smoke-verified — not an aspirational target:

- `torch==2.9.0+cu128`
- `vllm==0.11.2`
- `transformers==4.57.0`
- `deepspeed==0.18.0`

Full pin list: `docker/constraints.txt` (a `pip freeze` of that working env, so the
transitive deps that older plans worried about — `cachetools`, `py-cpuinfo`, `pylatexenc`,
`latex2sympy2`, `mathruler`, etc. — are already present). Do **not** retarget torch 2.10:
that stack was never validated on hardware and its flash-attn wheel does not exist.

**flash-attn is not committed** (the 952MB wheel is excluded). It is resolved at image
build by `setup/resolve_flash_attn.sh`, in order: (1) if `import flash_attn` already works
at a compatible version, use it; (2) else pip-install the matching official prebuilt wheel
(torch 2.9 / cu12 / py3.12 has one); (3) last resort, compile from source with
`FLASH_ATTN_CUDA_ARCHS="80;90"` (FA 2.8.3 ignores `TORCH_CUDA_ARCH_LIST`). Gemma runs on
**sdpa** and does not need flash-attn; Qwen/InternVL use FA2.

---

## 5. Support matrix

| requirement | value |
|---|---|
| NVIDIA driver | **≥ R570** (needed for the CUDA 12.8 runtime in the `+cu128` stack) |
| GPU arch | **sm80 (A100 80GB)** or **sm90 (H100)** — the FA resolver builds/selects for `80;90` |
| GPUs / node | **8×80GB, single node** (see §2) |
| CUDA toolkit in image | 12.8 (`-devel` base; ships nvcc/g++ for DeepSpeed CPU-Adam) |

Anything below 80GB per card, fewer than 8 cards, or sm < 80 is out of scope. cu128 vs the
vLLM 0.11.2 cu129 minor-version skew has run fine in practice but is not separately
certified.

---

## 6. Offline / air-gapped HPC note

On the intended internal target (public egress via proxy, same as the reference pod) the
online flow in §3 works as-is; prefetch is just a speed/cache convenience.

If your **compute** nodes have no external network (classic HPC), do the fetch on a
**login** node first, then run compute offline:

```bash
# On a login node (has network):
bash setup/prefetch_models.sh          # populates HF_HUB_CACHE
bash setup/prepare_data.sh             # populates $OUT_ROOT + $PRE_ROOT/*

# On the compute node (no network) — force offline so nothing tries to reach the hub:
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Make sure the same `HF_HOME`/`HF_HUB_CACHE`/`HF_DATASETS_CACHE`/`OUT_ROOT`/`PRE_ROOT`
paths are visible (shared filesystem or bind mount) from both nodes. Known cache traps to
avoid on such setups: unset any inherited `TRANSFORMERS_CACHE` (it silently redirects
lookups to an empty dir), and set `PYTHONNOUSERSITE=1` (a stray `~/.local` torch stub can
shadow the env's torch).

---

## 7. Apptainer / Singularity alternative

If Docker is unavailable (common on HPC), build a SIF from the same image and run with
`--nv`:

```bash
apptainer build mllm-repro.sif docker://<internal-registry>/mllm-repro:th2.9-cu128-vllm0.11.2
# (or: apptainer build mllm-repro.sif docker-daemon://mllm-repro:th2.9-cu128-vllm0.11.2
#  to convert a locally-built image without a registry round-trip)

apptainer exec --nv --cleanenv \
  --env HF_TOKEN="${HF_TOKEN:?set HF_TOKEN}" \
  --env HF_HOME=/cache/hf --env HF_HUB_CACHE=/cache/hf/hub --env HF_DATASETS_CACHE=/cache/hf/datasets \
  --env OUT_ROOT=/data/mllm_eval --env PRE_ROOT=/data --env MLLM_ENV_READY=1 \
  -B /host/hf_cache:/cache/hf \
  -B /host/mllm_data:/data \
  -B "$PWD":/workspace \
  mllm-repro.sif bash -lc 'cd /workspace && bash examples/<exp>.sh'
```

Notes:
- `--nv` exposes the GPUs; `--cleanenv` gives a reproducible environment (so pass the vars
  you need explicitly with `--env`, as above — including `MLLM_ENV_READY=1`).
- **Bind everything the run touches**: the HF cache, the data root (`/data`), and
  `work_dirs/` (via the `$PWD:/workspace` bind, so outputs persist on the host).
- **co-learn (#7-#9) needs a writable rendezvous dir.** By default it lives under the run's
  output dir (`work_dirs/mllm-co-grpo-dp/<run>/rdv`), so the `/workspace` bind already
  covers it. If you relocate it, bind that path too and make sure **both** 4-GPU groups
  see the same filesystem, or the pseudo-label exchange will deadlock.
- Apptainer maps `/dev/shm` from the host — ensure the host has ample shared memory (the
  `--ipc=host`/`--shm-size` requirement from §2).

---

## 8. Troubleshooting

- **OOM (usually a Gemma-12B pair, #8/#9)** → lower the vLLM memory fraction:
  `VLLM_MEM=0.40 bash examples/phase4_heter_qwen25vl7b_x_gemma3_12b_openr1.sh`. This gives the backward
  pass more headroom; leave everything else alone. Gemma pairs are the tightest and VRAM
  climbs as completions lengthen — watch GPU4-7. If still tight, that is the only knob to
  turn; do not change `bs`/`ga` without re-deriving `GA` to keep EB=64 (otherwise the
  GT/TTRL/co-learn comparison is no longer fair).
- **First optimizer step crashes compiling CPU-Adam** → DeepSpeed's `cpu_adam` JIT-builds a
  C++/CUDA extension and needs `g++`/`nvcc`. The provided image ships the toolchain and
  pre-compiles it. If you are *not* using the image, install a C++/CUDA toolchain and set
  `DS_BUILD_CPU_ADAM=1` before first run.
- **`401`/gated-repo error on Gemma** → the license is not accepted for your token, or
  `HF_TOKEN` is unset/expired. Redo §3 step B (accept license on HF, export a read-scope
  token, `huggingface-cli whoami`). Only #5/#6/#8/#9 need it.
- **We-Math eval hangs / times out at ~4h** → lower `--limit`/`--max_tokens` (this repo's `eval/eval_mllm.py` has no `--max_num_seqs` flag; for the proper fix add `max_num_seqs=64` to its `LLM(...)`). We-Math
  has long outputs that oversubscribe the vLLM KV cache → preempt/recompute → throughput
  collapses (~226→6 it/s). A bounded running batch is
  metric-neutral under greedy decoding. (MathVision/MathVerse finish fine because their
  outputs are short.)
- **A collapsed TTRL/self checkpoint makes an eval bench return NA / never finish** → it is
  emitting runaway long text; call `eval/eval_mllm.py` directly with `--max_tokens 1024` (the real answer is early; `run_eval_all.sh` does not forward this flag). The grader maps MCQ letter↔value automatically (`grade.py`), so no `GRADE_MCQ_MAP` is needed.
- **`inhomogeneous shape` / Gemma processor crash** → transformers too old for the Gemma3
  multimodal processor. The frozen stack is `transformers==4.57.0` (in the image); if a
  Gemma smoke run hits this, verify the container's transformers version rather than
  patching around it.
- **NCCL/dataloader hang at startup** → shared memory too small. Ensure `--ipc=host
  --shm-size=32g` (Docker) or a large host `/dev/shm` (apptainer). See §2.
- **Launcher tries to `source` a `./PATH` NAS path or writes to a NAS default** → the
  image sets `MLLM_ENV_READY=1` (launchers then skip NAS activation), and you overrode
  `MLLM_PRE_DIR`/`OUT_ROOT`/`MLLM_EVAL_*` to local paths in §3 step D. If you see a NAS
  path in a log, one of those exports is missing — re-export before re-running.

---

_Frozen stack: torch 2.9.0+cu128 / vllm 0.11.2 / transformers 4.57.0 / deepspeed 0.18.0.
Wandb defaults to offline. Never commit a live `HF_TOKEN` or wandb key — inject at runtime._

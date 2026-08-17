<h1 align="center">
    Co-RL: Vision-Language Experiments
</h1>

<p align="center">
    Vision-language experiments of "Decoupled Co-Learning: Diversity as the Engine of Label-Free Self-Supervised RL" on open-r1 and MMR1.
</p>

Methods: GT-Reward, TTRL, and cross-family co-learning. Models: Qwen2.5-VL-3B
and 7B, InternVL3.5-2B and 8B, Gemma-3-4B and 12B. One launcher per paper run
under `examples/`, plus the four-benchmark evaluation suite under `eval/`.

## ⚙️ Configuration

Build the pinned environment with Docker:

```bash
docker build -f docker/Dockerfile -t mllm-repro .
docker run --rm --gpus all --ipc=host --shm-size=32g \
    -v $PWD:/workspace -w /workspace mllm-repro bash
python tools/verify.py    # asserts the stack matches the paper runs
```

Or install directly: `pip install -r requirements.txt` on Python 3.11 with
CUDA 12.8. The freeze is a full pip freeze of the
environment behind the paper numbers (torch 2.9.0+cu128, vllm 0.11.2,
transformers 4.57.0, flash-attn 2.8.3, patched TRL). Export `MLLM_VIT_ATTN_FIX=1` for every Qwen2.5-VL **training** run (the
trainers read it; `eval_mllm.py` does not).

Prepare models and data once:

```bash
export HF_TOKEN=...                 # Gemma models are gated
bash setup/prefetch_models.sh
bash setup/prepare_data.sh          # writes data/mllm_pre/{mmr1_8k, openr1_8k}
```

Each launcher selects its training set through `MLLM_PRE_DIR`.

## 🚀 Training

A one-step smoke:

```bash
MAX_STEPS=1 MAX_SAMPLES=64 bash examples/openr1_qwen25vl3b_gt.sh
```

Small tier, 8 GPUs per run:

```bash
bash examples/openr1_[model]_[method].sh
# model  in {qwen25vl3b, internvl35_2b, gemma3_4b}
# method in {gt, ttrl}
```

Big tier co-learning, 4+4 GPUs on one node:

```bash
bash examples/phase4_heter_qwen25vl7b_x_gemma3_12b_openr1.sh
bash examples/phase4_heter_internvl35_8b_x_gemma3_12b_openr1.sh
bash examples/phase4_heter_qwen25vl7b_x_internvl35_8b_openr1.sh
```

Every run uses an effective batch of 64 prompts per step, 8 rollouts per
prompt, 1 epoch, seed 42.

## 📊 Evaluation

Score a checkpoint on MathVision, MathVerse, MathVista, and We-Math with
greedy decoding and rule-based grading:

```bash
python eval/prepare_benchmarks.py all
bash eval/run_eval_all.sh --model [checkpoint] --prompt answer
```

Use `--prompt boxed` for the whole table so cells stay comparable, and always
export `VLLM_WORKER_MULTIPROC_METHOD=spawn` — vLLM runs its engine in a child
process and CUDA is not fork-safe, so the default `fork` makes the child die
with "CUDA driver initialization failed".

Single-GPU deployment (incl. Blackwell / sm_120 notes and the frozen protocol):
see [`SINGLE_GPU_EVAL.md`](SINGLE_GPU_EVAL.md). To score the whole big-tier
Qwen column on one card:

```bash
CUDA_VISIBLE_DEVICES=2 bash eval/run_bigtier_qwen.sh --out_root work_dirs/eval_bigtier
```

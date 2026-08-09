# mllm-repro

Vision-language experiments of the Co-RL paper: GT-Reward, TTRL, and
cross-family co-learning on open-r1 and MMR1, at the small tier (2B to 4B) and
the big tier (7B to 12B). One launcher per paper run under `examples/`, plus
the four-benchmark evaluation suite under `eval/`.

## Environment

Build the pinned stack with Docker:

```bash
docker build -t mllm-repro .
docker run --rm --gpus all --ipc=host --shm-size=32g \
    -v $PWD:/workspace -w /workspace mllm-repro bash
```

The image replays `docker/constraints.txt`, a full pip freeze of the
environment that produced the paper numbers (torch 2.9.0+cu128, vllm 0.11.2,
transformers 4.57.0, deepspeed 0.18.0, flash-attn 2.8.3, and a patched TRL
fork). Export `MLLM_VIT_ATTN_FIX=1` for every Qwen2.5-VL run; its vision tower
has head dimension 80 and vLLM 0.11.2 otherwise routes it to a kernel that
rejects it.

Verify before training:

```bash
python verify.py
```

## Data

```bash
export HF_TOKEN=...                 # Gemma models are gated
bash setup/prefetch_models.sh       # model weights into the HF cache
bash setup/prepare_data.sh          # preprocess both training sets, build eval data
```

Preprocessing writes `mmr1_8k/` and `openr1_8k/` under `data/mllm_pre/`. Every
launcher selects its training set through `MLLM_PRE_DIR`, and its in-loop
evaluation set through `MLLM_EVAL_PATH` (MathVista, bundled under
`data/mathvista/`).

## Running

A one-step smoke of any launcher:

```bash
MAX_STEPS=1 MAX_SAMPLES=64 bash examples/openr1_qwen25vl3b_gt.sh
```

Small tier, 8 GPUs per run:

```bash
bash examples/openr1_<model>_<method>.sh
# model  in {qwen25vl3b, internvl35_2b, gemma3_4b}
# method in {gt, ttrl}
```

Big tier co-learning, 4+4 GPUs on one node:

```bash
bash examples/phase4_heter_qwen25vl7b_x_gemma3_12b_openr1.sh
bash examples/phase4_heter_internvl35_8b_x_gemma3_12b_openr1.sh
bash examples/phase4_heter_qwen25vl7b_x_internvl35_8b_openr1.sh
```

Every run uses an effective batch of 64 prompts per optimizer step, 8 rollouts
per prompt, completion length 1024, lr 1e-6, 1 epoch, seed 42. The same
launcher trains on either dataset; point `MLLM_PRE_DIR` at `mmr1_8k` or
`openr1_8k`.

Reference numbers for every run are in `EXPECTED_RESULTS.md`.

## Evaluation

Score a checkpoint on MathVision, MathVerse, MathVista, and We-Math
(greedy decoding, rule-based grading, no LLM judge):

```bash
python eval/prepare_benchmarks.py all
bash eval/run_eval_all.sh --model <checkpoint> --prompt answer
```

Use `--prompt answer` for trained checkpoints (matches the training format)
and `--prompt boxed` for untrained base models.

## Layout

```
examples/     one launcher per paper run
trainers/     training entries and the co-learning trainer
eval/         four-benchmark evaluation suite
setup/        model prefetch and dataset preprocessing
data/         bundled evaluation sets
docker/       dependency freeze
tools/        dataset preprocessing utilities
```

## License

License TBD.

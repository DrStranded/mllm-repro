# MM-UPT-aligned evaluation protocol — authoritative definition

This file is the single authority for the **aligned protocol**: the setting used to
compare our checkpoints against MM-UPT (arXiv 2505.22453) on their terms. It exists
*alongside* the frozen internal protocol (rule-based grading; authority: Co-RL repo,
`mllm/README`), never replacing it. The two protocols answer different questions and
their numbers must never share a table row:

| | frozen internal | MM-UPT-aligned (this file) |
|---|---|---|
| purpose | method-vs-method comparisons among our own runs | external comparison vs MM-UPT |
| grading | mathruler rules (`eval/grade.py`) | Qwen2.5-32B-Instruct judge (`eval/judge_aligned.py`) |
| decoding | T=0 greedy, top_p 0.95 | **T=0.2, top_p 1.0** |
| benches | five, AVG5 (incl. CoreCognition) | the paper's four, AVG4 |

## The protocol

| item | value | provenance |
|---|---|---|
| benchmarks | MathVision 3040 / MathVerse 3940 / MathVista 1000 / We-Math 1740, full splits, AVG4 | paper App. A.2 (sizes match exactly) |
| prompt | boxed + helpful-assistant system ("put your final answer within \boxed{}") | paper App. A.2 |
| max_tokens | 16384 | their repo README |
| temperature | 0.2 | their repo README `SamplingParams` |
| top_p | 1.0 | their `SamplingParams` sets none → vLLM default |
| images | long edge capped 1024 (ours) vs max_pixels≈1.0–1.2M (theirs; their two configs disagree: 1204224 vs 1003520) | small mechanical residual, settled by calibration |
| grading | Qwen2.5-32B-Instruct judges question + extracted answer; no extraction = wrong | paper App. A.2 ("comparing the extracted responses") |
| judge prompt wording | ours (`eval/judge_aligned.py`); theirs is unreleased | residual, settled by calibration |

Run one cell with `GPU=<n> TAG=<tag> MODEL=<ckpt> bash eval/run_mmupt_aligned.sh`.

## Judge variants — measured, do not re-litigate

Same Qwen2.5-VL-7B base outputs, three graders (paper's printed base row: 49.47):

| grader | AVG4 |
|---|---|
| rules only | 47.34 |
| judge sees question + **last 6000 chars of response** | 51.92 |
| judge sees question + **extracted answer** (aligned) | 51.54 |

A **tail-reading judge scores +0.38 higher** than the extraction judge — it credits
answers the extractor missed. The paper grades extractions, so the extraction judge is
the aligned one. A judge **without the question collapses below the rule score**:
gold `A` vs pred `55` cannot be checked against the option table. Never judge without
the question.

## Calibration evidence — what makes comparisons legal

Two-sided validation against the paper, both measured with this exact pipeline:

- **Their flagship row reproduces.** Official ckpt `WaltonFuture/Qwen2.5-VL-7B-MM-UPT-MMR1`
  scored here: 27.93 / 45.10 / 72.30 / 70.00 = **AVG4 53.83** vs their printed 53.17
  (**+0.66**). An earlier measurement of this same ckpt at top_p=0.95 gave 54.05, so
  the top_p knob alone is worth −0.22 — small, but the anchor must be measured under
  the exact protocol, and the 53.83 figure is the one to quote.
- **Their base row does not fully close.** Our base measurement lands **+1.86** above
  their printed 49.47, uniform across all four benches (+1.61 … +2.17), surviving the
  temperature and top_p alignment. Attribution: unpublished harness components (judge
  prompt wording, library versions). Their paper itself warns its scores "may differ
  from those in the original papers due to variations in evaluation protocols".

**Consequence — the only legal comparison shape:** same-ruler, self-measured on both
sides (our measurement of their checkpoint vs our measurement of ours). Comparing our
measured numbers against their *printed* numbers inherits a ~+1.9 tailwind and is
forbidden in tables; printed numbers may be quoted in prose with the protocol caveat.

## Cells measured under this protocol (2026-08-24)

All rows below are self-measured with this exact pipeline, Qwen2.5-VL-7B side, and
are therefore mutually comparable. Every co-RL row is the Qwen-side weight — a single
model at inference; the peer only supplies training signal.

| cell | MathVision | MathVerse | MathVista | We-Math | AVG4 |
|---|---|---|---|---|---|
| Qwen2.5-VL-7B base | 27.04 | 45.58 | 68.20 | 64.48 | 51.33 |
| MM-UPT official ckpt (15 ep × n=10) | 27.93 | 45.10 | 72.30 | 70.00 | **53.83** |
| co-RL ×InternVL-8B, beta=0, endpoint | 28.16 | 46.24 | 71.20 | 68.28 | **53.47** |
| co-RL ×Gemma-12B, mmupt recipe | 27.63 | 46.37 | 70.70 | 66.32 | 52.76 |
| GT-GRPO (ground-truth reward, beta=0) | 28.09 | 47.31 | 66.90 | 68.62 | 52.73 |
| co-RL ×InternVL-8B, mmupt recipe | 27.76 | 46.95 | 67.90 | 67.82 | 52.61 |
| TTRL (majority-vote self-reward, beta=0) | 27.53 | 43.55 | 69.50 | 65.80 | 51.60 |
| co-RL ×InternVL-8B, beta=0, best-by-val (s580) | 26.64 | 44.95 | 67.40 | 64.89 | 50.97 |

Reading the table:

- co-RL at 1 epoch lands 0.36 under MM-UPT's 15-epoch official ckpt — inside the noise
  floor (two full sweeps of adjacent steps of one model differ by 0.20 AVG, up to
  ±0.91 on a single bench).
- co-RL (unsupervised) **beats GT-GRPO** (supervised, +0.74) and beats TTRL by +1.87,
  while TTRL clears base by only 0.27. Same base, data, recipe and budget across those
  three; the only difference is where the reward comes from.
- beta=0 co-RL (53.47) > both mmupt-recipe co-RL cells (52.76 / 52.61): the mmupt
  recipe's KL penalty (`--beta 0.01`) acts as a leveler, compressing methods toward base.
- best-by-val selection costs 2.50 AVG4 versus the endpoint on the same run. Report
  endpoints.

Training-budget note for the comparison narrative: MM-UPT runs 15 epochs × rollout
n=10 (their `examples/config.yaml`); the beta=0 co-RL row is 1 epoch × 8 rollouts —
roughly 1/10 of the generation budget counting both co-trained models, ~1/23 counting
only the published side.

Raw per-item outputs for every cell live in `work_dirs/eval_bigtier/<tag>/<bench>.json`
(git-ignored; archived to a private HF dataset for audit).

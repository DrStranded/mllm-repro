# EXPECTED_RESULTS, acceptance criterion for the mllm repro (small + big tier)

This file is the **acceptance target** for a reproduction run. After you train one of
the 15 experiments (9 big-tier 7B/8B/12B + 6 small-tier 2B/3B/4B) and evaluate it with
`eval/run_eval_all.sh`, compare your 4-benchmark row against the numbers here. A run
"reproduces" if it lands inside the **tolerance band** (see the last section) *and*
preserves the qualitative ordering (`trained > base`, `co-learn ≈ GT`, both `≥ TTRL`
within noise).

> **Small-tier reference numbers** live in the shipped CSVs (see the honesty note below) -
> those rows are exactly the small tier, so #10–#15 have real targets to hit. The big tier
> is the sparser half.

> **Read the honesty note first.** The three shipped result CSVs
> (`RESULTS_bestval_4bench.csv`, `RESULTS_endpoint_4bench.csv`, `RESULTS_ALL_mllm.csv`)
> contain **only small-tier (InternVL-2B / Qwen-VL-3B) rows**, none of the 9 big-tier
> experiments are in them. Most big-tier runs were done on ByteDance pods and were not
> captured into those CSVs. What big-tier numbers we *do* have come from pod eval
> staging (`work_dirs/eval/scale_new/`) and cover only **5 of the ~11 model outputs**.
> The rest are marked **"to be filled after first repro"**, your first clean run *is*
> the reference for those, so record it.

---

## 1. Eval protocol (what a "number" means here)

All numbers below are produced by this repo's `eval/` layer. To be comparable they
**must** be reproduced the same way:

- **Benchmarks (4):**
  - **MathVision**, HF `MathLLMs/MathVision`, 3040 items (fetched at runtime).
  - **MathVerse**, HF `AI4Math/MathVerse`, `testmini` 3940 items (fetched at runtime).
  - **MathVista**, local `data/mathvista/testmini.jsonl`, 1000 items (the only bench
    bundled in-repo). An in-loop 150-item subset is used *during training* to pick the
    best checkpoint; the final number uses the full 1000.
  - **We-Math**, HF `We-Math/We-Math`, 1740 items (fetched at runtime).
- **`avg` = the simple arithmetic mean of the 4 per-benchmark accuracies** (confirmed in
  `eval/aggregate_row.py`: `sum(accs)/len(accs)`, then ×100). It is *not* item-weighted.
  If a bench is missing it is dropped from the mean and shown as `NA`, a partial `avg`
  is not comparable to a full one.
- **Decoding:** greedy, **temperature = 0** (deterministic sampling). Collapsed / runaway
  checkpoints are capped at `--max_tokens 1024`.
- **Grader:** rule-based **`mathruler`** (`eval/grade.py`) with MCQ valueoption repair.
  **No LLM judge.** This matters: a rule grader is stricter and less forgiving of format
  drift than an LLM judge, so these numbers are systematically a few points below what an
  LLM-judged leaderboard would report for the same model (see tolerance section).
- **We-Math gotcha:** run with `--max_num_seqs 64`, otherwise the KV cache thrashes and
  the job wall-times out (~4h) before finishing.
- **"best-val" vs "endpoint":** `best_model` = checkpoint chosen by the in-loop
  MathVista-150 score; `checkpoint-N` (endpoint) = the last saved step. They differ by
  1–2 points and, on collapse-prone runs, by a lot (e.g. a Qwen-VL-3B TTRL endpoint fell
  to avg 14.3 vs best-val 27.1). **Report the best-val number** unless you are
  deliberately studying collapse.

---

## 2. The 9 big-tier experiments (7B / 8B / 12B)

Matrix from `DOCKER_SLIM_PLAN.md §1` (dataset = **open_r1**; the pipeline is
dataset-agnostic and can also run mmr1). Co-learn runs produce **two** model outputs
(`model_a` = Group A, `model_b` = Group B), so each pair has two eval rows.

### 2a. Capture status

| # | Model(s) | Method | Attn | Captured big-tier avg | Status |
|---|----------|--------|------|-----------------------|--------|
| 1 | Qwen2.5-VL-7B-Instruct | GT | FA2 |, | **to be filled after first repro** |
| 2 | Qwen2.5-VL-7B-Instruct | TTRL | FA2 |, | **to be filled after first repro** |
| 3 | InternVL3_5-8B-HF | GT | FA2 |, | **to be filled after first repro** |
| 4 | InternVL3_5-8B-HF | TTRL | FA2 | **54.16** | captured |
| 5 | gemma-3-12b-it | GT | sdpa | **45.17** | captured |
| 6 | gemma-3-12b-it | TTRL | sdpa | **44.45** | captured |
| 7 | Qwen-VL-7B × InternVL-8B | co-learn | FA2×FA2 |, | **to be filled** (both sides) |
| 8 | Qwen-VL-7B × Gemma-12B | co-learn | FA2×sdpa | **44.81** (Gemma side `model_b`) | partial, Qwen-7B side to be filled |
| 9 | InternVL-8B × Gemma-12B | co-learn | sdpa×sdpa | **47.56** (Gemma side `model_b`) | partial, Intern-8B side to be filled |

### 2b. Per-benchmark for the captured big-tier rows

Numbers are absolute % accuracy (mathvista is 1-decimal because n=1000). Anchor these
when you re-run exps 4/5/6 and the Gemma sides of 8/9.

| Exp | Model output | MathVision | MathVerse | MathVista | We-Math | **avg** |
|-----|--------------|-----------:|----------:|----------:|--------:|--------:|
| #4  | InternVL3_5-8B-HF · TTRL | 35.07 | 41.24 | 68.6 | 71.72 | **54.16** |
| #5  | gemma-3-12b-it · GT | 30.89 | 33.63 | 56.9 | 59.25 | **45.17** |
| #6  | gemma-3-12b-it · TTRL | 27.93 | 36.37 | 54.7 | 58.79 | **44.45** |
| #8  | (Qwen7B×Gemma12B) co-learn, **Gemma side** | 30.66 | 35.41 | 54.4 | 58.79 | **44.81** |
| #9  | (Intern8B×Gemma12B) co-learn, **Gemma side** | 32.01 | 35.91 | 55.6 | 66.72 | **47.56** |

**Provenance & confidence for these 5 rows (be honest with yourself):**
- They are **not** in the three shipped `RESULTS_*` CSVs. They were recovered from pod
  eval staging under `work_dirs/eval/scale_new/*/results.csv`. The model paths are
  pod-side HF-hub layouts (`q1716523669/mllm-open-r1-...`), i.e. **pushed / endpoint**
  checkpoints, not confirmed to be the in-loop best-val checkpoint.
- The eval environment that produced them was the **pod** (torch 2.10 / vllm 0.18), not
  the frozen Anvil stack this repo ships (torch 2.9.0 / vllm 0.11.2). Cross-arch kernel
  differences alone can move a bench ±1–2 pts.
- For the two co-learn rows (#8, #9) we only have the **Gemma partner** (`model_b`); the
  Qwen-7B (#8) and InternVL-8B (#9) partners were never captured.
- **No big-tier `base` row was captured** (`scale_new/gemma12b_base/results.csv` is
  empty), so exact basetrained deltas at 12B cannot be stated from this repo. Use the
  small-tier deltas in §3 for the *shape* of the improvement.
- **Treat these 5 as directional anchors, not gold acceptance targets.** The intended
  final acceptance numbers should be a protocol-consistent rerun on the shipped stack;
  when you produce one, overwrite these with your captured values and note the stack.
- (A stray `scale_new_t4096/gemma12b_ttrl` row exists with MathVision 28.59 and the other
  three benches `NA`, that is an incomplete rerun, ignore it.)

---

## 3. Small-tier directional reference (co-learn ≈ GT)

The core paper claim is **co-learn ≈ GT** (heterogeneous co-learning matches the
ground-truth-reward upper bound, and both beat `base` and self-labeling `TTRL`), and the
project note is that this **holds from 3B 7B/8B**. The big-tier repro should reproduce
the same *ordering*, even if absolute values differ. The cleanest small-tier evidence is
the curated `paper_artifacts/results_tables/mllm_main_MASTER.csv` (best-val, same 4-bench
protocol). Reproduced here as % accuracy:

**open_r1 (the big-tier training set most relevant analog):**

| Arm (tier) | base | GT | TTRL | co-learn |
|------------|-----:|---:|-----:|---------:|
| InternVL3_5-**2B** | 31.90 | 45.20 | 44.99 | **45.40** |
| Qwen2.5-VL-**3B** | 31.65 | 42.97 | 42.47 | **43.89** |
| gemma-3-**4B** | 29.96 | 39.74 | 38.60 | 39.28 |

**mmr1:**

| Arm (tier) | base | GT | TTRL | co-learn |
|------------|-----:|---:|-----:|---------:|
| Qwen2.5-VL-**3B** | 37.24 | 41.03 | 37.97 | **41.12** |
| InternVL3_5-**2B** | 43.11 | 44.65 | 45.30 | 45.15 |
| gemma-3-**4B** |, |, | 38.68 |, |

Reading of the reference:
- On **open_r1**, `co-learn ≥ GT` for InternVL-2B and Qwen-VL-3B, and both clear `base`
  by ~+11–13 pts and beat `TTRL`. That is the "co-learn ≈ GT" headline your big-tier run
  should echo (e.g. exp #7 Qwen7B×Intern8B co-learn should sit at/above the #1/#3 GT runs
  once those are captured).
- **TTRL is the fragile arm**, it matches GT on stable arms but *collapses* on others
  (Qwen-VL-3B mmr1: 37.97, below base; and endpoint TTRL can implode entirely, see §1).
  So do not be alarmed if a big-tier TTRL run underperforms its GT twin.
- The gemma-4B small arm is the weakest and its heterogeneous pairing is lossy
  (`gemma_xQ` co-learn = 35.29 in MASTER), which is consistent with the modest 12B Gemma
  co-learn numbers in §2b (#8/#9 Gemma side ≈ 44–48).

---

## 4. Tolerance band & acceptance criterion

RL fine-tuning is **nondeterministic run-to-run** (seed, data order, GPU/kernel
nondeterminism, best-ckpt selection sensitivity), and this repo's stack (torch 2.9.0 /
vllm 0.11.2, FA2 for Qwen/Intern, **sdpa** for Gemma) differs from the pod stack that
produced the §2b anchors. So exact-match is not the goal.

- **Per-benchmark tolerance:** **±1–3 absolute points** is normal and expected. Bigger
  swings on a single bench (esp. We-Math, which is the most variance-prone) are
  acceptable if the **avg** still lands in band.
- **Per-exp `avg` tolerance:** target **within ±2–3 absolute points** of the anchor
  (for exps #4/#5/#6 and the Gemma sides of #8/#9). For the not-yet-captured exps
  (#1/#2/#3/#7 and the partner sides of #8/#9), the **first successful repro sets the
  reference**, record its 4-bench row into this file with the stack it ran on.
- **Do not compare these to public leaderboards.** The eval protocol here (greedy T=0,
  rule-based `mathruler` grader, this repo's prompt template, no LLM judge) differs from
  official leaderboard protocols by roughly **~1–3%** on its own, in addition to model
  differences. A "perfect" reproduction will still not match a leaderboard number.

**A repro run PASSES if all of:**
1. **Ordering holds:** `trained > base`, `co-learn ≈ GT` (co-learn within ~1–2 avg pts of
   its GT twin, not below it by more than noise), and neither is beaten by `TTRL` by more
   than noise. (An individual TTRL run collapsing is expected, not a failure of the repro.)
2. **In band (where an anchor exists):** per-exp `avg` within ±2–3 abs pts of §2b, and no
   single bench off by more than ~3 pts without a compensating explanation.
3. **Protocol matched:** greedy T=0, `mathruler` grader, full-size benches (MathVista 1000
   / MathVision 3040 / MathVerse 3940 / We-Math 1740), We-Math run with `--max_num_seqs 64`,
   best-val checkpoint reported.

If (1) holds but (2) is outside band on the pod-anchored rows, that is most likely the
torch2.9/vllm0.11.2-vs-pod stack gap or best-vs-endpoint checkpoint mismatch, note it and
prefer the ordering criterion, which is the paper's actual claim.

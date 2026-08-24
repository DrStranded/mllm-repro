#!/usr/bin/env python3
"""MM-UPT-aligned judge pass: Qwen2.5-32B-Instruct scores the EXTRACTED answer.

This is the grading half of the aligned protocol (see EVAL_ALIGNED.md, the
authoritative definition). The MM-UPT paper (arXiv 2505.22453, App. A.2) grades by
"comparing the extracted responses with ground truth answers" with
Qwen2.5-32B-Instruct, so the judge here sees the question plus the boxed
extraction -- never the response tail. Samples with no extraction score wrong.

Three judge variants were measured on the same Qwen2.5-VL-7B base outputs before
this one was fixed as the aligned grader (AVG4 over the paper's four benchmarks;
paper's published base row is 49.47):

    rules only (frozen internal protocol)                47.34
    judge sees question + last 6000 chars of response    51.92   <- tail-reading
    judge sees question + extracted answer (THIS FILE)   51.54       judges score
                                                                     +0.38 higher
A judge WITHOUT the question text collapses below the rule score: gold 'A' with
pred '55' cannot be verified against the option table, so every letter<->value
match that mathruler credits is lost. Never judge without the question.

The residual +~1.9 over the paper's printed base row is uniform across benches
and attributable to unpublished parts of their harness (judge prompt wording,
library versions); the paper's own flagship row reproduces at +0.88 under this
pipeline, which is the calibration evidence that makes self-measured comparisons
legal. All external comparisons must therefore be same-ruler: our measurement of
their checkpoint vs our measurement of ours.

Usage (one 96GB GPU; the sample files must carry per-item "pred"):
    python eval/judge_aligned.py <bench.json>:<data.jsonl> [...]
Writes accuracy_judged_pred and per-sample ok_judged_pred back into each json.
"""
import json
import sys

TEMPLATE = """You are checking a model's answer to a math question.

Question: {q}

Reference answer: {gold}

Model's answer: {pred}

Does the model's answer match the reference answer? For multiple choice, the
option letter or that option's value both count. Numeric answers count if
mathematically equal. Reply with exactly one word: Correct or Incorrect."""

JUDGE_MODEL = "Qwen/Qwen2.5-32B-Instruct"


def main():
    pairs = [a.split(":", 1) for a in sys.argv[1:]]
    if not pairs:
        print(__doc__)
        sys.exit(1)
    from vllm import LLM, SamplingParams
    llm = LLM(model=JUDGE_MODEL, dtype="bfloat16", tensor_parallel_size=1,
              gpu_memory_utilization=0.92, max_model_len=8192, enforce_eager=True)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0, max_tokens=8)

    for jpath, dpath in pairs:
        d = json.load(open(jpath))
        S = d["samples"]
        qs = [json.loads(l)["problem"] for l in open(dpath)]
        assert len(qs) == len(S), f"{jpath}: {len(S)} samples vs {len(qs)} rows"
        todo = [i for i, s in enumerate(S) if s.get("pred")]
        prompts = [tok.apply_chat_template(
            [{"role": "user", "content": TEMPLATE.format(
                q=qs[i][:2000], gold=S[i]["gold"], pred=S[i]["pred"])}],
            tokenize=False, add_generation_prompt=True) for i in todo]
        outs = llm.generate(prompts, sp)
        ok = 0
        for i, o in zip(todo, outs):
            v = o.outputs[0].text.strip().lower().startswith("correct")
            S[i]["ok_judged_pred"] = v
            ok += v
        d["accuracy_judged_pred"] = ok / len(S)
        d["judge_model"] = JUDGE_MODEL
        json.dump(d, open(jpath, "w"), indent=1)
        print(f"[judge-aligned] {jpath.split('/')[-1]}: rule {d['accuracy']:.4f} "
              f"-> judged {ok/len(S):.4f}", flush=True)
    print("######## JUDGE_ALIGNED_DONE ########")


if __name__ == "__main__":
    main()

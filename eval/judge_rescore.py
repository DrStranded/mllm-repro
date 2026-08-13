#!/usr/bin/env python3
"""LLM-judge rescoring pass over eval_mllm.py outputs (MM-UPT paper protocol).

MM-UPT scores benchmarks with an LLM judge (Qwen2.5-32B-Instruct, paper §4.1),
not pure rules. This script adds that layer on top of our rule-based pass:
samples that followed the answer format but failed rule matching are re-judged
(equivalent forms: 3/2 vs 1.5, option letter vs value); samples with NO
recognized answer format stay wrong - format compliance is part of the task;
rule-credited samples are left alone, so the judge can only recover
false negatives (missed extractions, equivalent-but-differently-written
answers), never inflate.

Reads one or more <bench>.json files produced by eval_mllm.py (which must have
been run with full-sample saving), asks the judge whether the response's final
answer matches the gold, and writes back:
    accuracy_judged, n_correct_judged, judge_model, per-sample "ok_judged".

Usage (needs 2 GPUs for the 32B judge):
    python eval/judge_rescore.py --judge Qwen/Qwen2.5-32B-Instruct --tp 2 \
        --files out/tag1/mathvision.json out/tag1/mathverse.json ...
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

JUDGE_TEMPLATE = """You are checking a model's answer to a question.

Question: {q}

Reference answer: {gold}

Model response (may contain reasoning; judge only its FINAL answer):
{resp}

Does the model's final answer match the reference answer? For multiple choice,
the option letter or the option's value both count. For yes/no questions,
judge the stated yes/no. Numeric answers count if mathematically equal.

Strict rule: the response must explicitly COMMIT to a final answer. If it is
cut off mid-reasoning, trails off, or never clearly states a final answer,
reply Incorrect even if the partial work points toward the correct value.
Reply with exactly one word: Correct or Incorrect."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--judge", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--max_resp_chars", type=int, default=6000,
                    help="tail of the response shown to the judge")
    args = ap.parse_args()

    todo = []   # (file_idx, sample_idx, prompt)
    metas = []
    for fi, path in enumerate(args.files):
        o = json.load(open(path))
        data_root = os.path.dirname(os.path.abspath(o["data"]))
        qs = [json.loads(l)["problem"] for l in open(o["data"])] \
            if os.path.exists(o["data"]) else None
        metas.append(o)
        if len(o.get("samples", [])) != o.get("n"):
            sys.exit(f"{path}: samples truncated ({len(o.get('samples', []))} of "
                     f"{o.get('n')}) - rerun eval_mllm.py with full saving")
        for si, s in enumerate(o["samples"]):
            if s["ok"]:
                s["ok_judged"] = True
                continue
            if s.get("pred") is None:
                # No recognized answer format (<answer> tag or \boxed{}): counts as
                # wrong, full stop. Format compliance is part of the task.
                s["ok_judged"] = False
                continue
            q = qs[si] if qs else ""
            resp = s["resp"][-args.max_resp_chars:]
            todo.append((fi, si, JUDGE_TEMPLATE.format(q=q, gold=s["gold"], resp=resp)))

    print(f"[judge] {len(todo)} rule-failed samples across {len(args.files)} files")
    if todo:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.judge)
        def _build(eager):
            return LLM(model=args.judge, dtype="bfloat16", tensor_parallel_size=args.tp,
                       gpu_memory_utilization=0.92, max_model_len=8192, enforce_eager=eager)
        try:
            llm = _build(False)
        except Exception:
            llm = _build(True)
        prompts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                           tokenize=False, add_generation_prompt=True)
                   for _, _, p in todo]
        sp = SamplingParams(temperature=0, max_tokens=8)
        outs = llm.generate(prompts, sp)
        for (fi, si, _), o in zip(todo, outs):
            verdict = o.outputs[0].text.strip().lower()
            metas[fi]["samples"][si]["ok_judged"] = verdict.startswith("correct")

    for fi, path in enumerate(args.files):
        o = metas[fi]
        nj = sum(1 for s in o["samples"] if s.get("ok_judged"))
        o["n_correct_judged"] = nj
        o["accuracy_judged"] = round(nj / o["n"], 4) if o["n"] else 0.0
        o["judge_model"] = args.judge
        json.dump(o, open(path, "w"), indent=2, ensure_ascii=False)
        print(f"[judge] {path}: rule {o['accuracy']:.4f} -> judged {o['accuracy_judged']:.4f} "
              f"(+{o['n_correct_judged'] - o['n_correct']})")


if __name__ == "__main__":
    main()

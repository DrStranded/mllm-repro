#!/usr/bin/env python3
"""Test-time (self-consistency) ensemble for the MLLM benchmarks — majority vote over K samples per model.

Inputs are eval_mllm.py outputs produced with --n K (each sample carries `resps`, K responses).
For N models, the first K_i = ceil(total / N) responses of each are pooled (total = 8 by default),
answers are extracted with the frozen v2 rule extractor and grouped into equivalence buckets using the
same mathruler grader that scores the main table. MCQ answers are normalised to the option letter via
the question's option list (so "B" and "60°" vote together), exactly as grade() credits them.
The largest bucket wins; ties break deterministically and order-independently (lexicographically
smallest canonical key). The winner's first response becomes `resp`, so judge_rescore.py can be run
on the output unchanged and report `accuracy_judged` under the v2 ruler.

Usage:
  python eval/ensemble_vote.py --files a.json[,b.json] --total 8 --out voted.json
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grade import extract_pred, grade, _option_letter, parse_choices  # frozen v2 extractor + grader
from mathruler.grader import grade_answer


def canon(p: str) -> str:
    return "".join(str(p).split()).replace("$", "").replace("\\left", "").replace("\\right", "").lower()


def mcq_key(pred: str, choices: dict) -> str | None:
    """Map a prediction to an option letter when the question is MCQ; None if it is not an option."""
    if not choices:
        return None
    L = _option_letter(pred)
    if L in choices:
        return L
    for letter, val in choices.items():
        try:
            if grade_answer(pred, val):
                return letter
        except Exception:
            pass
    return None


def same(a: str, b: str) -> bool:
    if canon(a) == canon(b):
        return True
    try:
        return bool(grade_answer(a, b)) or bool(grade_answer(b, a))
    except Exception:
        return False


def load_questions(run: dict) -> list:
    """Question text per row, from the benchmark jsonl the run was generated on (same order)."""
    data = run.get("data")
    if data and os.path.exists(data):
        return [json.loads(l).get("problem", "") for l in open(data) if l.strip()]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True, help="comma-separated eval_mllm.py outputs (same benchmark, same row order)")
    ap.add_argument("--total", type=int, default=8, help="total votes pooled across models")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runs = [json.load(open(f)) for f in args.files.split(",")]
    n_models = len(runs)
    k = -(-args.total // n_models)
    n = len(runs[0]["samples"])
    assert all(len(r["samples"]) == n for r in runs), "row count mismatch between models"
    questions = load_questions(runs[0])
    if len(questions) != n:
        print(f"[vote] WARNING: {len(questions)} questions for {n} rows; MCQ letter/value merging disabled", flush=True)
        questions = [None] * n

    out_samples, n_correct, n_valid_total, n_mcq = [], 0, 0, 0
    for i in range(n):
        gold = runs[0]["samples"][i]["gold"]
        q = questions[i]
        choices = parse_choices(q) if q else {}
        n_mcq += bool(choices)
        pool = []  # (pred, resp)
        for r in runs:
            s = r["samples"][i]
            assert s["gold"] == gold, f"gold mismatch at row {i}"
            for resp in (s.get("resps") or [s["resp"]])[:k]:
                pred = extract_pred(resp)
                if pred is not None and str(pred).strip():
                    pool.append((str(pred).strip(), resp))
        buckets = []  # [key, count, first_resp, first_pred]
        for pred, resp in pool:
            mk = mcq_key(pred, choices)
            if mk is not None:
                for b in buckets:
                    if b[0] == "opt:" + mk:
                        b[1] += 1
                        break
                else:
                    buckets.append(["opt:" + mk, 1, resp, pred])
                continue
            for b in buckets:
                if not b[0].startswith("opt:") and same(b[3], pred):
                    b[1] += 1
                    break
            else:
                buckets.append([canon(pred), 1, resp, pred])
        if buckets:
            buckets.sort(key=lambda b: (-b[1], b[0]))
            key, cnt, resp, pred = buckets[0]
        else:
            key, cnt, resp, pred = "", 0, "", None
        ok = bool(pred is not None and grade(pred, gold, q))
        n_correct += ok
        n_valid_total += len(pool)
        out_samples.append({"gold": gold, "pred": pred, "ok": ok, "resp": resp,
                            "vote_count": cnt, "n_valid": len(pool), "n_buckets": len(buckets)})

    res = {"model": "ensemble:" + "+".join(os.path.basename(r["model"].rstrip("/")) for r in runs),
           "models": [r["model"] for r in runs], "data": runs[0]["data"], "prompt": runs[0]["prompt"],
           "ensemble": {"n_models": n_models, "k_per_model": k, "total": args.total, "n_mcq_questions": n_mcq,
                        "tie_break": "largest bucket, then lexicographically smallest canonical key"},
           "n": n, "n_correct": n_correct, "accuracy": round(n_correct / n, 4) if n else 0.0,
           "avg_valid_votes": round(n_valid_total / n, 2) if n else 0.0, "samples": out_samples}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"[vote] {res['model']}  N={n_models} K={k}  accuracy={res['accuracy']} ({n_correct}/{n})  "
          f"avg_valid_votes={res['avg_valid_votes']}  mcq={n_mcq}  -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

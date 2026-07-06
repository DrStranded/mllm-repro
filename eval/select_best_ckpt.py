#!/usr/bin/env python3
"""Print the best-by-val checkpoint path for an MLLM training run.

Priority:
  1. <run>/best_model  (BestKeeperCallback's hardlink of the global-best ckpt)
  2. argmax of eval_reward over checkpoint-*/trainer_state.json log_history

Usage:
  python select_best_ckpt.py <run_dir>
  # co-learn runs have model_a/ and model_b/ subdirs; pass the subdir you want.
"""
import os, sys, json, glob


def best_ckpt(run):
    bm = os.path.join(run, "best_model")
    if os.path.isdir(bm) and glob.glob(os.path.join(bm, "*.safetensors")):
        return bm, None
    cks = sorted((int(d.rsplit("-", 1)[1]) for d in glob.glob(os.path.join(run, "checkpoint-*"))
                  if os.path.isdir(d)))
    if not cks:
        return None, None
    ts = os.path.join(run, f"checkpoint-{cks[-1]}", "trainer_state.json")
    ev = []
    if os.path.exists(ts):
        for e in json.load(open(ts)).get("log_history", []):
            if "eval_reward" in e:
                ev.append((e["step"], e["eval_reward"]))
    if not ev:
        return os.path.join(run, f"checkpoint-{cks[-1]}"), None
    bstep, bval = max(ev, key=lambda x: x[1])
    ck = os.path.join(run, f"checkpoint-{bstep}")
    if not os.path.isdir(ck):  # best step rotated out and no best_model — fall back to nearest surviving
        ck = os.path.join(run, f"checkpoint-{min(cks, key=lambda c: abs(c - bstep))}")
        return ck, ("WARN: best step %d rotated out; using nearest surviving %s (val %.4f)" % (bstep, ck, bval))
    return ck, ("best step %d val %.4f" % (bstep, bval))


if __name__ == "__main__":
    run = sys.argv[1].rstrip("/")
    ck, note = best_ckpt(run)
    if ck is None:
        sys.stderr.write(f"no checkpoint found in {run}\n")
        sys.exit(1)
    if note:
        sys.stderr.write(note + "\n")
    print(ck)

#!/usr/bin/env python3
"""Download + convert the 4 MLLM math benchmarks to a unified jsonl+images layout.

Unified per-example schema (matches the project's `_load_local_eval_jsonl`):
    {"problem": <question text with choices inlined>, "image": "images/<id>.png",
     "solution": <gold answer string>, "qtype": "mcq"|"free"}

Splits follow MM-UPT (full test/testmini):
    MathVision 3040 | MathVista 1000 | MathVerse 3940 (5 versions) | We-Math 1740

Output: <OUT_ROOT>/<bench>/{data.jsonl, images/}  (image path relative to data.jsonl).
MathVista is already local; we just re-point to it.

Usage:
    python prepare_benchmarks.py <bench>      # bench = mathvision|mathverse|wemath|mathvista|all
    OUT_ROOT=/path python prepare_benchmarks.py all
"""
import os, sys, json

OUT_ROOT = os.environ.get(
    "OUT_ROOT", "./data/mllm_eval"
)
MATHVISTA_LOCAL = "./data/mathvista"
MAX_SIDE = 1024  # cap long side (same as training _cap_image; avoids vLLM blowup)


def cap_image(img):
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_SIDE:
        s = MAX_SIDE / max(w, h)
        from PIL import Image
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BICUBIC)
    return img


def _write(bench, rows_iter, total=None):
    out_dir = os.path.join(OUT_ROOT, bench)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    jsonl = os.path.join(out_dir, "data.jsonl")
    n = 0
    with open(jsonl, "w") as f:
        for i, (img, problem, solution, qtype) in enumerate(rows_iter):
            rel = f"images/{i}.png"
            cap_image(img).save(os.path.join(out_dir, rel))
            f.write(json.dumps({"problem": problem, "image": rel,
                                "solution": str(solution), "qtype": qtype},
                               ensure_ascii=False) + "\n")
            n += 1
            if n % 200 == 0:
                print(f"  {bench}: {n}{'/'+str(total) if total else ''}", flush=True)
    print(f"✅ {bench}: {n} examples -> {jsonl}")


def _strip_img_tags(s):
    import re
    return re.sub(r"<image\d*>", "", s or "").strip()


def mathvision():
    from datasets import load_dataset
    ds = load_dataset("MathLLMs/MathVision", split="test")
    def rows():
        for ex in ds:
            q = _strip_img_tags(ex["question"])
            opts = ex.get("options") or []
            qtype = "free"
            if opts:  # MCQ: inline choices as (A)/(B)/...
                letters = [chr(65 + j) for j in range(len(opts))]
                q += "\nChoices:\n" + "\n".join(f"({l}) {o}" for l, o in zip(letters, opts))
                qtype = "mcq"
            yield ex["decoded_image"], q, ex["answer"], qtype
    _write("mathvision", rows(), total=len(ds))


def mathverse():
    from datasets import load_dataset
    ds = load_dataset("AI4Math/MathVerse", "testmini", split="testmini")  # 3940 = 5 versions
    def rows():
        for ex in ds:
            qtype = "mcq" if ex.get("question_type") == "multi-choice" else "free"
            yield ex["image"], ex["question"], ex["answer"], qtype
    _write("mathverse", rows(), total=len(ds))


def wemath():
    from datasets import load_dataset
    ds = load_dataset("We-Math/We-Math", split="testmini")  # 1740
    def rows():
        for ex in ds:
            q = ex["question"].strip() + "\nChoices: " + ex["option"].strip()
            yield ex["image_path"], q, ex["answer"], "mcq"
    _write("wemath", rows(), total=len(ds))


def mathvista():
    """Already local in the project's format; just symlink/copy the dir reference."""
    out_dir = os.path.join(OUT_ROOT, "mathvista")
    os.makedirs(OUT_ROOT, exist_ok=True)
    if os.path.islink(out_dir) or os.path.exists(out_dir):
        print(f"✅ mathvista: already at {out_dir}")
        return
    os.symlink(MATHVISTA_LOCAL, out_dir)
    # normalize: project file is testmini.jsonl; eval expects data.jsonl
    src = os.path.join(MATHVISTA_LOCAL, "testmini.jsonl")
    print(f"✅ mathvista: symlinked {MATHVISTA_LOCAL} (use testmini.jsonl, {sum(1 for _ in open(src))} rows)")


FNS = {"mathvision": mathvision, "mathverse": mathverse, "wemath": wemath, "mathvista": mathvista}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    targets = list(FNS) if which == "all" else [which]
    for b in targets:
        print(f"==== {b} ====")
        FNS[b]()

#!/usr/bin/env python3
"""Merge the per-benchmark json outputs into one append-safe CSV row.

Frozen protocol (2026-08-17): the avg column is AVG5 over all five benchmarks,
and the score is the RULE-BASED accuracy. `accuracy_judged`, if a file still
carries it from an older run, is deliberately ignored — the judge over-credits
long rambling responses (it reads only the tail and credits a correct value
that appears mid-stream), which inflates verbose models by several points."""
import os, json, argparse, csv

BENCHES = ["mathvision", "mathverse", "mathvista", "wemath", "corecognition"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--upsert", action="store_true",
                    help="replace any existing row with the same --tag instead of appending "
                         "(so calling after every benchmark keeps ONE live, current row)")
    args = ap.parse_args()

    accs, row = [], {"tag": args.tag, "model": args.model}
    for b in BENCHES:
        p = os.path.join(args.out_dir, f"{b}.json")
        if os.path.exists(p):
            j = json.load(open(p))
            a = j.get("accuracy")          # rule-based only, never accuracy_judged
            row[b] = round(a * 100, 2) if a is not None else "NA"
            if a is not None:
                accs.append(a)
        else:
            row[b] = "NA"
    # AVG5: only report it when all five benchmarks are present, so a partial
    # run can never be mistaken for a complete cell.
    row["avg"] = round(sum(accs) / len(BENCHES) * 100, 2) if len(accs) == len(BENCHES) else "NA"

    cols = ["tag", "model"] + BENCHES + ["avg"]
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    if args.upsert:
        # read existing rows, drop same-tag, rewrite all + this row -> one live current row
        existing = []
        if os.path.exists(args.csv):
            with open(args.csv, newline="") as f:
                for r in csv.DictReader(f):
                    if r.get("tag") != args.tag:
                        existing.append(r)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in existing:
                w.writerow({k: r.get(k, "") for k in cols})
            w.writerow(row)
    else:
        new = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            if new:
                w.writeheader()
            w.writerow(row)
    print("CSV row:", {k: row[k] for k in cols})


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Grade generated answers (scripts/eval_generation.py JSONL) against V3 reference answers with
a local judge model, and print mean correctness per strategy (correct 1, partial 0.5).

Usage (GPU, ~6 GB VRAM in 4-bit):
    python scripts/judge_answers.py data/eval/generation_hr.jsonl [more.jsonl ...]
Writes <input>.judged.jsonl next to each input (or under --out-dir).
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from mira.evaluation.judge import SCORES, Judge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out-dir", help="Where to write *.judged.jsonl (default: next to the input)")
    args = parser.parse_args()

    judge = Judge()
    for path in map(Path, args.inputs):
        rows = [json.loads(line) for line in path.open()]
        verdicts = judge.grade([{"question": r["query"], "reference": r["reference"], "answer": r["answer"]} for r in rows])
        out = Path(args.out_dir or path.parent) / f"{path.stem}.judged.jsonl"
        with out.open("w") as f:
            for row, (verdict, reason) in zip(rows, verdicts):
                f.write(json.dumps({**row, "verdict": verdict, "verdict_reason": reason}) + "\n")

        by_strategy = defaultdict(list)
        for row, (verdict, _) in zip(rows, verdicts):
            by_strategy[row["strategy"]].append(verdict)
        print(f"\n{path.name} (judge: correct=1, partial=0.5)")
        print(f"{'strategy':12s}{'n':>5s}{'score':>8s}  verdicts")
        for strategy, vs in by_strategy.items():
            score = mean(SCORES.get(v, 0.0) for v in vs)
            print(f"{strategy:12s}{len(vs):5d}{score:8.3f}  {dict(Counter(vs))}")
        print(f"Saved {out}")


if __name__ == "__main__":
    main()

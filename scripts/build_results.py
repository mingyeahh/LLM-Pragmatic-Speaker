#!/usr/bin/env python3
"""Phase 2/3: assemble the per-game result tables (LLM scores + RSA speaker)."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

import pandas as pd

from rsa_speaker.pipeline import build_final_table, build_first_step


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topk-dir", default="data/llm_scores/topk")
    parser.add_argument("--logic-dir", default="data/llm_scores/logical")
    parser.add_argument("--mf-dir", required=True, help="meaning-function table dir")
    parser.add_argument("--out-dir", required=True, help="final result dir")
    parser.add_argument("--merged-dir", default=None,
                        help="optional dir to also cache the merged LLM-score tables")
    parser.add_argument("--n-objects", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=1.0, help="RSA rationality parameter")
    args = parser.parse_args()

    topk_dir, logic_dir, mf_dir = Path(args.topk_dir), Path(args.logic_dir), Path(args.mf_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_dir = Path(args.merged_dir) if args.merged_dir else None
    if merged_dir:
        merged_dir.mkdir(parents=True, exist_ok=True)

    for mf_file in sorted(mf_dir.glob("*.csv")):
        stem = mf_file.stem
        topk_df = pd.read_csv(topk_dir / f"{stem}.csv", sep="|")
        getprob_df = pd.read_csv(logic_dir / f"{stem}.csv", sep="|")
        first_step = build_first_step(topk_df, getprob_df, args.n_objects)
        if merged_dir:
            first_step.to_csv(merged_dir / f"{stem}.csv")

        mf_table = pd.read_csv(mf_file)
        final = build_final_table(first_step, mf_table, args.n_objects, args.alpha)
        final.to_csv(out_dir / f"{stem}.csv")
        print(f"[result] {stem}")
    print("done.")


if __name__ == "__main__":
    main()

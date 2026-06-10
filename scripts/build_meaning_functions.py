#!/usr/bin/env python3
"""Phase 2: build meaning-function tables M(u, o) for every reference game."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import time
from pathlib import Path

import pandas as pd

from rsa_speaker.meaning_functions import rule_based_meaning_table


def utterance_union(topk_dir: Path, logic_dir: Path, stem: str) -> list[str]:
    topk = pd.read_csv(topk_dir / f"{stem}.csv", sep="|")
    logic = pd.read_csv(logic_dir / f"{stem}.csv", sep="|")
    return sorted(set(topk["Sequence"]).union(logic["Sequence"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mf", choices=["rule", "prompt"], help="meaning function")
    parser.add_argument("--corpus-dir", default="data/corpus")
    parser.add_argument("--topk-dir", default="data/llm_scores/topk")
    parser.add_argument("--logic-dir", default="data/llm_scores/logical")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="meta")
    parser.add_argument("--model-dir", default="model")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    topk_dir, logic_dir = Path(args.topk_dir), Path(args.logic_dir)
    stems = sorted(f.stem for f in logic_dir.glob("*.csv"))

    model = None
    if args.mf == "prompt":
        from rsa_speaker.llm import load_model

        model = load_model(args.model, model_dir=args.model_dir)

    for stem in stems:
        start = time.time()
        utterances = utterance_union(topk_dir, logic_dir, stem)
        corpus_file = f"{args.corpus_dir}/{stem}.xml"
        if args.mf == "rule":
            df = rule_based_meaning_table(corpus_file, utterances)
        else:
            from rsa_speaker.meaning_functions import prompt_based_meaning_table

            df = prompt_based_meaning_table(corpus_file, utterances, model)
        df.to_csv(out_dir / f"{stem}.csv")
        print(f"[{args.mf}] {stem} ({time.time() - start:.1f}s)")
    print("done.")


if __name__ == "__main__":
    main()

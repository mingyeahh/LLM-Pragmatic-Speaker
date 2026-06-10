#!/usr/bin/env python3
"""Phase 2: vanilla-LLM scores for the logically-constructed utterances."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import time
from pathlib import Path

import pandas as pd

from rsa_speaker.llm import load_model
from rsa_speaker.scoring import score_utterances
from rsa_speaker.utterances import read_logical_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--utt-dir", default="data/logical_utterances")
    parser.add_argument("--out-dir", default="data/llm_scores/logical")
    parser.add_argument("--model", default="meta")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--n-objects", type=int, default=7)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = {f.stem for f in out_dir.glob("*.csv")}
    files = [f for f in sorted(Path(args.utt_dir).glob("*.txt")) if f.stem not in done]

    model = load_model(args.model, model_dir=args.model_dir)
    for txt in files:
        start = time.time()
        corpus_file, utterances = read_logical_file(txt)
        df = pd.DataFrame({"Sequence": utterances})
        scored = score_utterances(corpus_file, df, model, args.n_objects)
        scored.to_csv(out_dir / f"{txt.stem}.csv", sep="|")
        print(f"[score] {txt.stem} ({time.time() - start:.1f}s)")
    print("done.")


if __name__ == "__main__":
    main()

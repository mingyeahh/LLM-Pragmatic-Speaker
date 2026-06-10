#!/usr/bin/env python3
"""Phase 1: build the utterance space (logical alternatives or top-k beam search)."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

from rsa_speaker.utterances import generate_topk, write_logical_file


def run_logic(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for xml in sorted(Path(args.corpus_dir).glob("*.xml")):
        write_logical_file(str(xml), out_dir / f"{xml.stem}.txt")
        print(f"[logic] {xml.stem}")
    print("done.")


def run_topk(args: argparse.Namespace) -> None:
    from rsa_speaker.llm import load_model

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = {f.stem for f in out_dir.glob("*.csv")}
    files = [f for f in sorted(Path(args.corpus_dir).glob("*.xml")) if f.stem not in done]

    model = load_model(args.model, model_dir=args.model_dir)
    for xml in files:
        df = generate_topk(str(xml), model, args.beam_width, args.n_objects)
        df.to_csv(out_dir / f"{xml.stem}.csv", sep="|")
        print(f"[topk] {xml.stem}")
    print("done.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="method", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--corpus-dir", default="data/corpus")
    common.add_argument("--n-objects", type=int, default=7)

    p_logic = sub.add_parser("logic", parents=[common], help="logical alternatives")
    p_logic.add_argument("--out-dir", default="data/logical_utterances")
    p_logic.set_defaults(func=run_logic)

    p_topk = sub.add_parser("topk", parents=[common], help="LLM beam-search alternatives")
    p_topk.add_argument("--out-dir", default="data/llm_scores/topk")
    p_topk.add_argument("--model", default="meta")
    p_topk.add_argument("--model-dir", default="model")
    p_topk.add_argument("--beam-width", type=int, default=50)
    p_topk.set_defaults(func=run_topk)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

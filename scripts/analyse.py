#!/usr/bin/env python3
"""Phase 3: correlation analysis and meaning-function evaluation."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

import pandas as pd

from rsa_speaker.analysis import (
    correlation_table,
    evaluate_meaning_function,
    plot_correlation_histograms,
    plot_score_scatter,
    summarise_correlations,
)


def run_correlation(args: argparse.Namespace) -> None:
    corr = correlation_table(args.results_dir, args.n_objects)
    summary = summarise_correlations(corr)
    print(summary.round(3).to_string())

    if args.out_csv:
        corr.to_csv(args.out_csv, index=False)
        print(f"saved per-game correlations -> {args.out_csv}")

    if args.fig_dir:
        import matplotlib.pyplot as plt

        fig_dir = Path(args.fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        for metric in ("PCC", "SRCC"):
            ax = plot_correlation_histograms(corr, metric)
            ax.figure.savefig(fig_dir / f"hist_{metric}.png", dpi=150, bbox_inches="tight")
            plt.close(ax.figure)
        ax = plot_score_scatter(args.results_dir, args.n_objects)
        ax.figure.savefig(fig_dir / "scatter.png", dpi=150, bbox_inches="tight")
        plt.close(ax.figure)
        print(f"saved figures -> {fig_dir}")


def run_meaning_function(args: argparse.Namespace) -> None:
    generated = pd.read_csv(args.generated, index_col=0)
    ground_truth = pd.read_csv(args.ground_truth, index_col=0)
    metrics = evaluate_meaning_function(generated, ground_truth, args.threshold)
    print(f"accuracy={metrics['accuracy']:.3f} "
          f"precision={metrics['precision']:.3f} "
          f"recall={metrics['recall']:.3f}")
    if metrics["mismatches"]:
        print(f"{len(metrics['mismatches'])} mismatches, e.g. {metrics['mismatches'][:5]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="task", required=True)

    p_corr = sub.add_parser("correlation", help="LLM vs RSA correlation analysis")
    p_corr.add_argument("--results-dir", required=True)
    p_corr.add_argument("--n-objects", type=int, default=7)
    p_corr.add_argument("--out-csv", default=None)
    p_corr.add_argument("--fig-dir", default=None)
    p_corr.set_defaults(func=run_correlation)

    p_mf = sub.add_parser("meaning-function", help="meaning function vs ground truth")
    p_mf.add_argument("--generated", required=True)
    p_mf.add_argument("--ground-truth", required=True)
    p_mf.add_argument("--threshold", type=float, default=0.5)
    p_mf.set_defaults(func=run_meaning_function)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

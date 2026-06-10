"""Comparing LLM and RSA scores: correlations, meaning-function eval, figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import distance


def _safe(fn, a: pd.Series, b: pd.Series) -> float:
    try:
        value = fn(a, b)
        return float(getattr(value, "statistic", value))
    except (ValueError, FloatingPointError):
        return float("nan")


def correlation_table(results_dir: str | Path, n_objects: int = 7) -> pd.DataFrame:
    """Per (game, object) PCC/SRCC/JSD between LLM and RSA, split by utterance type."""
    results_dir = Path(results_dir)
    rows = []
    for file in sorted(results_dir.glob("*.csv")):
        df = pd.read_csv(file)
        for o in range(1, n_objects + 1):
            seq_type = df[f"seq_type{o}"]
            p_llm, p_rsa = df[f"p_llm_{o}"], df[f"p_rsa_{o}"]
            masks = {
                "": slice(None),
                "_logic": (seq_type == "logic").to_numpy(),
                "_topk": (seq_type == "topk").to_numpy(),
            }
            row = {"file": file.stem, "obj": o}
            for suffix, mask in masks.items():
                a, b = p_llm[mask], p_rsa[mask]
                row[f"PCC{suffix}"] = _safe(stats.pearsonr, a, b)
                row[f"SRCC{suffix}"] = _safe(stats.spearmanr, a, b)
                row[f"JSD{suffix}"] = _safe(distance.jensenshannon, a, b)
            rows.append(row)
    return pd.DataFrame(rows)


def summarise_correlations(corr_df: pd.DataFrame) -> pd.DataFrame:
    summary = {}
    for label, suffix in [("Logic", "_logic"), ("Top-k", "_topk"), ("All", "")]:
        summary[label] = {
            "PCC_mean": corr_df[f"PCC{suffix}"].mean(),
            "PCC_sd": corr_df[f"PCC{suffix}"].std(),
            "SRCC_mean": corr_df[f"SRCC{suffix}"].mean(),
            "SRCC_sd": corr_df[f"SRCC{suffix}"].std(),
        }
    return pd.DataFrame(summary).T


def evaluate_meaning_function(
    generated: pd.DataFrame, ground_truth: pd.DataFrame, threshold: float = 0.5
) -> dict:
    """Accuracy/precision/recall of a meaning function against human labels."""
    pred = generated.drop(columns="Sequence").to_numpy() >= threshold
    gold = ground_truth.drop(columns="alt_utt").to_numpy() == 1

    tp = int(np.sum(gold & pred))
    fp = int(np.sum(~gold & pred))
    fn = int(np.sum(gold & ~pred))
    tn = int(np.sum(~gold & ~pred))

    cols = generated.columns[1:].to_list()
    seqs = generated["Sequence"].to_list()
    mismatches = [(cols[c], seqs[r]) for r, c in np.argwhere(pred != gold)]

    return {
        "accuracy": (tp + tn) / (tp + fp + fn + tn),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "recall": tp / (tp + fn) if (tp + fn) else float("nan"),
        "mismatches": mismatches,
    }


def plot_correlation_histograms(corr_df: pd.DataFrame, metric: str = "PCC", ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    bins = np.linspace(-1, 1, 41)
    ax.hist(corr_df[f"{metric}_logic"].dropna(), bins=bins, alpha=0.6, label="Logic")
    ax.hist(corr_df[f"{metric}_topk"].dropna(), bins=bins, alpha=0.6, label="Top-k")
    ax.hist(corr_df[metric].dropna(), bins=bins, alpha=0.4, label="All")
    ax.set_xlabel(metric)
    ax.set_ylabel("Reference game count")
    ax.legend()
    return ax


def plot_score_scatter(results_dir: str | Path, n_objects: int = 7, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    for seq_type, colour in [("logic", "tab:red"), ("topk", "tab:blue")]:
        xs, ys = [], []
        for file in Path(results_dir).glob("*.csv"):
            df = pd.read_csv(file)
            for o in range(1, n_objects + 1):
                mask = df[f"seq_type{o}"] == seq_type
                xs.extend(df.loc[mask, f"p_rsa_{o}"].tolist())
                ys.extend(df.loc[mask, f"p_llm_{o}"].tolist())
        ax.scatter(xs, ys, s=2, c=colour, label=seq_type, alpha=0.5)
    ax.set_xlabel("RSA model probability")
    ax.set_ylabel("Vanilla LLM probability")
    ax.legend()
    return ax

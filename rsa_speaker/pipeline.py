"""Assembling the per-game result tables (LLM + RSA scores)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rsa_speaker.rsa import pragmatic_speaker


def build_first_step(
    topk_df: pd.DataFrame, getprob_df: pd.DataFrame, n_objects: int = 7
) -> pd.DataFrame:
    """Merge top-k and logical utterances (with LLM scores) into one wide table."""
    final_df = pd.DataFrame()
    for o in range(n_objects):
        topk = topk_df.loc[topk_df["Object"] == o][["Sequence", "p_llm"]].rename(
            columns={"Sequence": f"S{o + 1}", "p_llm": f"p_llm_{o + 1}"}
        )
        topk[f"seq_type{o + 1}"] = "topk"

        logic = getprob_df[["Sequence", f"o{o + 1}_llm"]].rename(
            columns={"Sequence": f"S{o + 1}", f"o{o + 1}_llm": f"p_llm_{o + 1}"}
        )
        logic[f"seq_type{o + 1}"] = "logic"

        column = pd.concat(
            [topk.reset_index(drop=True), logic.reset_index(drop=True)], ignore_index=True
        )
        final_df = pd.concat([final_df, column], axis=1)
        final_df[f"p_rsa_{o + 1}"] = None
    return final_df.reset_index(drop=True)


def _coerce_logprob(value):
    if isinstance(value, str):
        return float(value[1:].split(",")[0])
    return value


def build_final_table(
    first_step_df: pd.DataFrame,
    mf_table: pd.DataFrame,
    n_objects: int = 7,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """Attach RSA speaker probabilities and column-normalise both models' scores."""
    speaker = pragmatic_speaker(mf_table, [f"o{i}" for i in range(1, n_objects + 1)], alpha=alpha)
    df = first_step_df.copy()

    for i in range(1, n_objects + 1):
        rsa_i = speaker[["Sequence", f"o{i}"]]
        rsa_i = rsa_i[rsa_i["Sequence"].isin(df[f"S{i}"])]
        rsa_i = rsa_i.rename(columns={"Sequence": f"S{i}", f"o{i}": f"p_rsa_{i}"})
        merged = df.merge(rsa_i, on=f"S{i}", how="left", suffixes=("_a", "_b"))
        df[f"p_rsa_{i}"] = merged[f"p_rsa_{i}_b"]

    llm_cols = [f"p_llm_{i}" for i in range(1, n_objects + 1)]
    rsa_cols = [f"p_rsa_{i}" for i in range(1, n_objects + 1)]

    df[llm_cols] = df[llm_cols].apply(lambda c: c.map(_coerce_logprob))
    df[llm_cols] = np.exp(df[llm_cols].astype(float))
    df[llm_cols] = df[llm_cols] / df[llm_cols].sum(axis=0)
    df[rsa_cols] = df[rsa_cols] / df[rsa_cols].sum(axis=0)
    return df

"""RSA literal listener and pragmatic speaker."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame / frame.sum(axis=0)
    out[~np.isfinite(out)] = 0
    return out


def literal_listener(mf: pd.DataFrame, object_cols: list[str]) -> pd.DataFrame:
    """L0(o | u): normalise each utterance over the objects."""
    out = mf.copy()
    out[object_cols] = out[object_cols].apply(_normalise, axis=1)
    return out


def pragmatic_speaker(
    mf: pd.DataFrame,
    object_cols: list[str],
    sequence_col: str = "Sequence",
    alpha: float = 1.0,
) -> pd.DataFrame:
    """S1(u | o) ∝ (L0(o | u) / |u|)^alpha, normalised over utterances."""
    listener = literal_listener(mf, object_cols)
    cost = listener[sequence_col].map(len).to_numpy().reshape(-1, 1)
    scores = (listener[object_cols] / cost) ** alpha
    out = listener.copy()
    out[object_cols] = _normalise(scores)
    return out

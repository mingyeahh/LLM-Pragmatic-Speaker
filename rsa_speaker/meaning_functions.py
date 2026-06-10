"""Meaning functions M(u, o): prompt-based and rule-based."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from rsa_speaker.corpus import parse_furniture, parse_furniture_features
from rsa_speaker.llm import call_llm, vocab_size

if TYPE_CHECKING:
    from llama_cpp import Llama


FEW_SHOT_PROMPT = """
Does the description apply to the object?

1.
Description: That's a green sofa facing left.
Object: small, green sofa facing to the left
Yes

2.
Description: That's a green sofa facing left.
Object: small, grey sofa facing to the left
No

3.
Description: A fan.
Object:  large, grey fan facing backwards
Yes

"""


def prompt_based_meaning(description: str, obj: str, model: "Llama") -> float:
    """P(Yes | Yes or No) for whether description applies to obj."""
    prompt = FEW_SHOT_PROMPT + f"4.\nDescription: {description}\nObject: {obj}\n"
    result = call_llm(prompt, model, max_tokens=1, logprobs=vocab_size(model))
    top = result["choices"][0]["logprobs"]["top_logprobs"][-1]
    p_yes, p_no = np.exp(top["Yes"]), np.exp(top["No"])
    return float(p_yes / (p_yes + p_no))


def prompt_based_meaning_table(
    corpus_file: str, utterances: list[str], model: "Llama"
) -> pd.DataFrame:
    objects = object_descriptions(corpus_file)
    rows = []
    for utt in utterances:
        row = {"Sequence": utt}
        for i, obj in enumerate(objects):
            row[f"o{i + 1}"] = prompt_based_meaning(utt, obj, model)
        rows.append(row)
    return pd.DataFrame(rows)


FULL_FEATURE = {
    "colour": {"blue", "red", "green", "grey"},
    "orientation": {"left", "right", "front", "back"},
    "type": {"chair", "sofa", "desk", "fan"},
    "size": {"large", "small"},
}

# surface forms normalised to the world vocabulary before matching
SYNONYMS = {
    "table": "desk",
    "forward": "forwards",
    "front": "forwards",
    "facing upwards": "forwards",
    "back": "backwards",
    "backward": "backwards",
    "opposite direction": "backwards",
    "facing the wall": "backwards",
    "away": "backwards",
    "big": "large",
    "tiny": "small",
    "little": "small",
    "gray": "grey",
}

# extra surface forms that also count as a contradicting feature
CONTRA_SYNONYMS = {
    "desk": {"table"},
    "front": {"forward", "forwards", "facing upwards"},
    "back": {"backwards", "backward", "opposite direction", "facing the wall", "away"},
    "large": {"big"},
    "small": {"tiny", "little"},
    "grey": {"gray"},
}

# fragments that describe nothing meaningfully
EXTREME_CASES = {"a", "a ", "a large", "a large,", "a small", "a small,"}


def contradicting_features(object_features: dict[str, str]) -> list[str]:
    own = set(object_features.values())
    full = {f for group in FULL_FEATURE.values() for f in group}
    contra = list(full - own)
    for feature in list(contra):
        contra.extend(CONTRA_SYNONYMS.get(feature, set()))
    return contra


def rule_based_meaning(utterance: str, contra_features: list[str]) -> int:
    """1 if the utterance is consistent with the object, else 0."""
    if utterance in EXTREME_CASES:
        return 0
    for surface, canonical in SYNONYMS.items():
        if surface in utterance:
            utterance = utterance.replace(surface, canonical)
    return 0 if any(feature in utterance for feature in contra_features) else 1


def rule_based_meaning_table(corpus_file: str, utterances: list[str]) -> pd.DataFrame:
    object_features = parse_furniture_features(corpus_file)
    rows = []
    for utt in utterances:
        row = {"Sequence": utt}
        for i, feats in enumerate(object_features):
            row[f"o{i + 1}"] = rule_based_meaning(utt, contradicting_features(feats))
        rows.append(row)
    return pd.DataFrame(rows)


def object_descriptions(corpus_file: str) -> list[str]:
    return [str(obj) for obj in parse_furniture(corpus_file)]

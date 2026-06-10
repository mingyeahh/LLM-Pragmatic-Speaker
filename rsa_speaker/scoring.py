"""Vanilla-LLM scoring of utterances via summed token log-probabilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from rsa_speaker.corpus import parse_furniture, produce_prompt
from rsa_speaker.llm import call_llm, vocab_size

if TYPE_CHECKING:
    from llama_cpp import Llama


def _tokenise(utterance: str, model: "Llama") -> list[str]:
    token_ids = model.tokenize(bytes(utterance, "utf-8"))
    return [model.detokenize([t]).decode("utf-8") for t in token_ids]


def sequence_logprob(
    prompt: str,
    utterance: str,
    model: "Llama",
    prefix_logprob: dict[str, float],
) -> float:
    """Summed token log-prob of utterance given prompt.

    prefix_logprob caches cumulative log-probs of seen prefixes; seed it with
    {"": 0.0} and reuse it across utterances sharing a prefix for the same prompt.
    """
    tokens = _tokenise(utterance, model)

    i = 0
    while i < len(tokens) and "".join(tokens[: i + 1]) in prefix_logprob:
        i += 1
    prefix = "".join(tokens[:i])

    for token in tokens[i:]:
        result = call_llm(
            prompt + prefix, model, stop=[], max_tokens=1, logprobs=vocab_size(model)
        )
        # the model occasionally returns an empty distribution; retry until non-empty
        while result["choices"][0]["logprobs"]["top_logprobs"] == []:
            result = call_llm(
                prompt + prefix, model, stop=[], max_tokens=1, logprobs=vocab_size(model)
            )
        token_logp = result["choices"][0]["logprobs"]["top_logprobs"][-1][token]
        prefix_logprob[prefix + token] = prefix_logprob[prefix] + token_logp
        prefix += token
    return prefix_logprob[prefix]


def score_utterances(
    corpus_file: str,
    utterances: pd.DataFrame,
    model: "Llama",
    n_objects: int = 7,
) -> pd.DataFrame:
    """Add an o{i}_llm log-prob column per object to utterances (needs a Sequence column)."""
    objects = parse_furniture(corpus_file)
    df = utterances.copy()
    for o in range(n_objects):
        prefix_logprob: dict[str, float] = {"": 0.0}
        col = f"o{o + 1}_llm"
        df[col] = 0.0
        prompt = produce_prompt(objects, o) + "\n"
        for row in df.itertuples():
            df.loc[row.Index, col] = sequence_logprob(
                prompt, row.Sequence, model, prefix_logprob
            )
    return df

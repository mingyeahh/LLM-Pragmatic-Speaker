"""Beam search over the local LLM to sample top-k utterances."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rsa_speaker.llm import call_llm, vocab_size

if TYPE_CHECKING:
    from llama_cpp import Llama

# partial generation -> (cumulative log-prob, finished?)
BeamItem = dict[str, tuple[float, bool]]


def _is_finished(text: str) -> bool:
    return "\n" in text or "." in text


def extend_sequence(
    base_prompt: str, generated: str, model: "Llama", beam_width: int
) -> BeamItem:
    res = call_llm(base_prompt + generated, model, stop=[], max_tokens=1, logprobs=beam_width)
    if res["choices"][0]["text"] == "":
        return {generated: (0.0, True)}
    next_tokens = res["choices"][0]["logprobs"]["top_logprobs"][-1]
    return {
        generated + tok: (logp, _is_finished(generated + tok))
        for tok, logp in next_tokens.items()
    }


def extend_sequence_set(
    base_prompt: str, beam: BeamItem, model: "Llama", beam_width: int
) -> BeamItem:
    out: BeamItem = {}
    for generated, (cum_logp, finished) in beam.items():
        if finished:
            out[generated] = (cum_logp, True)
        else:
            for ext, (logp, ext_finished) in extend_sequence(
                base_prompt, generated, model, beam_width
            ).items():
                out[ext] = (cum_logp + logp, ext_finished)
    return out


def beam_search(
    prompt: str,
    model: "Llama",
    beam_width: int,
    seed_words: list[str] | None = None,
) -> dict[str, float]:
    if seed_words is None:
        seed_words = ["a"]

    seed = call_llm(prompt, model, max_tokens=1, logprobs=vocab_size(model))
    seed_logprobs = seed["choices"][0]["logprobs"]["top_logprobs"][0]
    beam: BeamItem = {word: (seed_logprobs[word], False) for word in seed_words}

    while True:
        beam = extend_sequence_set(prompt, beam, model, beam_width)
        if len(beam) > beam_width:
            top = sorted(beam.items(), key=lambda kv: kv[1][0], reverse=True)[:beam_width]
            beam = dict(top)
        if all(finished for _, finished in beam.values()):
            break

    # strip terminators, keep the best-scoring copy of each unique utterance
    cleaned = [(gen.split("\n")[0].split(".")[0], logp) for gen, (logp, _) in beam.items()]
    out: dict[str, float] = {}
    for utt, logp in cleaned:
        out[utt] = max(out.get(utt, logp), logp)
    return out

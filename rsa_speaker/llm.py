"""Loading and calling the local Llama model via llama-cpp-python."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llama_cpp import Llama

MODEL_REGISTRY = {
    "xs": "ggml-model-IQ3_XS.gguf",
    "f16": "ggml-model-F16.gguf",
    "meta": "Meta-Llama-3-8B-Instruct.Q8_0.gguf",
}

DEFAULT_MODEL_DIR = Path("model")


def load_model(
    model_key: str = "meta",
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    n_ctx: int = 1000,
    n_gpu_layers: int = -1,
    verbose: bool = False,
) -> "Llama":
    from llama_cpp import Llama

    if model_key not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model key {model_key!r}; choose from {sorted(MODEL_REGISTRY)}"
        )
    model_path = Path(model_dir) / MODEL_REGISTRY[model_key]
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. See the README for download instructions."
        )
    # logits_all lets us read the log-prob of arbitrary tokens, not just the argmax.
    return Llama(
        model_path=str(model_path),
        verbose=verbose,
        logits_all=True,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
    )


def call_llm(
    prompt: str,
    model: "Llama",
    stop: list[str] | None = None,
    max_tokens: int = 32,
    logprobs: int = 1,
    echo: bool = False,
) -> dict[str, Any]:
    if stop is None:
        stop = ["\n"]
    return model(prompt, max_tokens=max_tokens, stop=stop, logprobs=logprobs, echo=echo)


def vocab_size(model: "Llama") -> int:
    return int(model.metadata["llama.vocab_size"])

# Are LLMs good pragmatic speakers?

Reference implementation for the NeurIPS 2024 Behavioral ML workshop paper
**[Are LLMs good pragmatic speakers?](https://arxiv.org/abs/2411.01562)**
(Mingyue Jian & N. Siddharth, University of Edinburgh).

We ask whether a vanilla LLM (Llama-3-8B-Instruct) behaves like a *pragmatic speaker*
by comparing the scores it assigns to candidate referential utterances against the
scores produced by a [Rational Speech Act (RSA)](https://www.problang.org/) model.
The comparison is run as a **reference game** built from the
[TUNA furniture corpus](https://aclanthology.org/W06-1420/): given a room of seven
objects, score every candidate description of a target object under both models and
measure their agreement.

---

## How it works

The evaluation pipeline has three phases (Figure 3 / Appendix C of the paper):

```
Phase 1 — build spaces            Phase 2 — score                 Phase 3 — compare
┌────────────────────────┐        ┌─────────────────────────┐     ┌───────────────────┐
│ meaning space O        │        │ vanilla LLM  p(u|O,oₜ)  │      │ PCC / SRCC / JSD  │
│  (object descriptions) │        │                         │     │ per reference game│
│ utterance space U:     │ ─────► │ RSA speaker S₁(u|o):    │ ──► │                   │
│  • logical rules       │        │  literal listener L₀    │     │ histograms (Fig 1)│
│  • top-k (beam search) │        │  × meaning function M   │     │ summary (Table 3) │
└────────────────────────┘        └─────────────────────────┘     └───────────────────┘
```

Two **meaning functions** `M(u, o) ∈ [0, 1]` are compared (Section 2.1):

- **Rule-based** — feature exclusion: an utterance describes an object unless it
  mentions a feature that contradicts it (with synonym normalisation).
- **Prompt-based** — a 3-shot prompt asks the LLM whether a description applies to an
  object; `M = P(Yes | Yes ∪ No)`.

## Repository layout

```
rsa_speaker/                 # the library
├── corpus.py                # parse TUNA XML → objects; build the reference-game prompt
├── llm.py                   # load / call the local GGUF Llama model
├── utterances.py            # utterance space U: logical rules + top-k (beam search)
├── beam_search.py           # beam search over the LLM
├── meaning_functions.py     # rule-based & prompt-based meaning functions M(u,o)
├── rsa.py                   # literal listener L₀ + pragmatic speaker S₁ (Eqs. 1–2)
├── scoring.py               # vanilla-LLM retroactive utterance scoring
├── pipeline.py              # assemble per-game result tables (LLM + RSA)
└── analysis.py              # PCC / SRCC / JSD, meaning-function eval, figures

scripts/                     # command-line entry points (one per pipeline step)
├── construct_utterances.py  # Phase 1
├── score_llm.py             # Phase 2 (vanilla LLM, logical utterances)
├── build_meaning_functions.py  # Phase 2 (M tables)
├── build_results.py         # Phase 2/3 (final per-game tables)
└── analyse.py               # Phase 3 (correlation + figures)

data/                        # corpus + all generated artifacts (see data/README.md)
```

## Installation

Requires Python ≥ 3.10.

```bash
git clone https://github.com/mingyeahh/LLM-RSA-reasoning.git
cd LLM-RSA-reasoning

python -m venv .venv && source .venv/bin/activate
pip install -e .          # core (parsing + analysis)
pip install -e ".[llm]"   # also installs llama-cpp-python for the scoring phases
```

`llama-cpp-python` is only needed for the **generation/scoring** phases. Corpus
parsing and the **analysis** of the included results run without it.

### Model

The scoring phases use a GGUF build of Meta-Llama-3-8B-Instruct. Download a GGUF file
(e.g. [`bartowski/Meta-Llama-3.1-8B-Instruct-GGUF`](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF))
into `model/` and reference it by alias in `rsa_speaker/llm.py:MODEL_REGISTRY`
(default alias `meta` → `Meta-Llama-3-8B-Instruct.Q8_0.gguf`).

### Data

All inputs and generated artifacts for the 420 furniture reference games are in
`data/`, ready to use (no unzip step):

```
data/
├── corpus/                  TUNA furniture trials (one XML per reference game)
├── logical_utterances/      logically-constructed utterance space U
├── llm_scores/              vanilla-LLM scores (topk/ and logical/)
├── llm_scores_merged/       top-k + logical scores merged per game
├── meaning_function/        M(u,o) tables (rule_based/ and prompt_based/)
├── results/                 final per-game tables (rule_based/ and prompt_based/)
└── meaning_function_eval/   human-labelled sets for evaluating M(u,o)
```

The unused TUNA *people* domain is omitted; the full corpus is available from the
[TUNA project](http://www.csd.abdn.ac.uk/research/tuna/). See
[`data/README.md`](data/README.md) for the full layout.

## Reproducing the results

With the included `data/`, you can regenerate the headline numbers without a model:

```bash
# Table 3 summary + Figure 1 for the rule-based RSA model
python scripts/analyse.py correlation \
    --results-dir data/results/rule_based --fig-dir figures/rule_based

# ... and for the prompt-based RSA model
python scripts/analyse.py correlation \
    --results-dir data/results/prompt_based --fig-dir figures/prompt_based

# Meaning-function accuracy vs human labels (Table 2)
python scripts/analyse.py meaning-function \
    --generated "data/meaning_function_eval/logic/s1t7-topk.csv" \
    --ground-truth "data/meaning_function_eval/ground truth/s1t7-topk.csv"
```

### Running the full pipeline from scratch

```bash
# Phase 1 — utterance space
python scripts/construct_utterances.py logic    # logical alternatives (no model)
python scripts/construct_utterances.py topk --model meta   # top-k via beam search

# Phase 2 — scoring
python scripts/score_llm.py --model meta                       # vanilla-LLM scores
python scripts/build_meaning_functions.py rule  --out-dir data/meaning_function/rule_based
python scripts/build_meaning_functions.py prompt --out-dir data/meaning_function/prompt_based --model meta

# Phase 2/3 — assemble result tables (per meaning function)
python scripts/build_results.py --mf-dir data/meaning_function/rule_based   --out-dir data/results/rule_based
python scripts/build_results.py --mf-dir data/meaning_function/prompt_based --out-dir data/results/prompt_based

# Phase 3 — analysis
python scripts/analyse.py correlation --results-dir data/results/rule_based
```

`build_results.py` takes an `--alpha` flag to sweep the RSA rationality parameter
(Appendix E uses `α ∈ {0.2, 0.6, 1.0, 1.4, 1.8, 3.0}`).

## Library quickstart

```python
from rsa_speaker import parse_furniture, produce_prompt
from rsa_speaker.utterances import logical_alternatives
from rsa_speaker.meaning_functions import rule_based_meaning_table
from rsa_speaker.rsa import pragmatic_speaker

corpus = "data/corpus/s1t7.xml"
objects = parse_furniture(corpus)

# Enumerate the logical utterance space for this room
utterances = list(logical_alternatives(objects))

# Meaning-function table M(u, o), then the RSA pragmatic speaker S₁(u | o)
mf = rule_based_meaning_table(corpus, utterances)
speaker = pragmatic_speaker(mf, [f"o{i}" for i in range(1, 8)], alpha=1.0)
```

## Notes on this release

This is a cleaned-up release of the original research code. Behaviour-preserving
changes worth flagging:

- The retroactive log-prob accumulator is seeded with `0.0` (a correct sum) rather
  than the original `1.0`; final scores are column-normalised so results are
  unchanged.
- `produce_prompt`, which was inadvertently commented out in the original, is
  restored exactly as the Appendix B template.

## Citation

```bibtex
@inproceedings{jian2024llms,
  title     = {Are LLMs good pragmatic speakers?},
  author    = {Jian, Mingyue and Siddharth, N.},
  booktitle = {NeurIPS 2024 Workshop on Behavioral Machine Learning},
  year      = {2024},
  url       = {https://arxiv.org/abs/2411.01562}
}
```

## License

Code released under the [MIT License](LICENSE). The TUNA corpus
(`data/corpus.zip`) is © van Deemter, van der Sluis & Gatt (2006) under its own terms.

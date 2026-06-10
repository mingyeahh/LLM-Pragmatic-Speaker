# Data

All inputs and generated artifacts for the 420 TUNA furniture reference games.
Files are named by trial id (e.g. `s1t7`). Most CSVs use `|` as the separator
(noted below) because some utterances contain commas. The unused TUNA *people*
domain is omitted.

| Path | What | Produced by |
|------|------|-------------|
| `corpus/*.xml` | One room of 7 objects per trial (the reference games); furniture singular from TUNA | van Deemter et al. (2006) |
| `logical_utterances/*.txt` | Logically-constructed utterance space `U`. Line 1 is the source corpus path; the rest are utterances | `construct_utterances.py logic` |
| `llm_scores/topk/*.csv` (`\|`) | Top-k utterances + LLM log-probs (`Sequence`, `Object`, `p_llm`) | `construct_utterances.py topk` |
| `llm_scores/logical/*.csv` (`\|`) | Logical utterances + per-object LLM log-probs (`o{i}_llm`) | `score_llm.py` |
| `llm_scores_merged/*.csv` | Per-game wide table merging the top-k + logical LLM scores | `build_results.py` (intermediate) |
| `meaning_function/rule_based/*.csv` | Rule-based `M(u, o)` table (`Sequence`, `o1..o7`) | `build_meaning_functions.py rule` |
| `meaning_function/prompt_based/*.csv` | Prompt-based `M(u, o)` table | `build_meaning_functions.py prompt` |
| `results/rule_based/*.csv` | Final per-game table, RSA = rule-based MF (`p_llm_{i}`, `p_rsa_{i}`, `seq_type{i}`) | `build_results.py` |
| `results/prompt_based/*.csv` | Final per-game table, RSA = prompt-based MF | `build_results.py` |
| `meaning_function_eval/ground truth/*.csv` | Human-labelled `M(u, o)` (0/1) for evaluating the meaning functions | manual annotation |
| `meaning_function_eval/ground truth_selected/*.csv` | The objects/utterances selected for annotation | — |
| `meaning_function_eval/logic/*.csv` | Rule-based MF predictions on the eval sets | `build_meaning_functions.py rule` |
| `meaning_function_eval/{topk,getprob}_{3,6}shot/*.csv` | Prompt-based MF predictions at 3-/6-shot for threshold tuning (Table 2) | `build_meaning_functions.py prompt` |

## Column conventions

In a `results/*` table, object `i ∈ {1..7}` has four columns:

- `S{i}` — the candidate utterance
- `p_llm_{i}` — vanilla-LLM probability (exp of summed log-prob, column-normalised)
- `p_rsa_{i}` — RSA pragmatic-speaker probability (column-normalised)
- `seq_type{i}` — `topk` or `logic` (how the utterance was constructed)

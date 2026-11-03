# ARR — Ambiguity Resolution Rate

ARR is the project's end-to-end usefulness metric: it doesn't ask "does the
clarifying question *look* good?" (that's what the LLM judges score), it asks
**"if a user actually answered this clarifying question, would the system then
land on the correct answer?"**

Implemented in [`evaluation/auto_scorer.py`](../evaluation/auto_scorer.py) as
`arr()` (boolean) and its softer sibling `answer_match_score()`
(`"exact" | "close" | "far"`).

## How it's calculated

For a query the dataset marks as ambiguous, with gold question `Qg` and gold
answer `Ag`:

1. The system under test sees the original query and produces its own output
   `Q` (a clarifying question for Debate-Lite/ORACLE; a direct answer for the
   baselines — see *Notes* below).
2. We **simulate the user's reply** by feeding the dataset's `Ag` back as if
   the user had just answered `Q`. This avoids needing a live human in the
   loop for all 500 queries.
3. A fresh GPT call (`_ARR_SYSTEM` prompt in `auto_scorer.py`) is asked to
   answer the *original* query using only
   `{original query, Q, simulated reply = Ag}` as context — i.e. it has to
   actually use the clarification, not just recall the answer from training
   data.
4. Both the gold answer and the model's final answer are normalized
   (lower-cased, whitespace-collapsed, trailing period stripped — see
   `_normalize()`), and we check whether the normalized gold answer is a
   **substring** of the normalized final answer.
   - substring found → `arr = True`, `answer_match = "exact"`
   - not found, but `BERTScore(final, gold) >= 0.7` → `answer_match = "close"` (`arr` still `False`)
   - neither → `answer_match = "far"`, `arr = False`

So `arr` is the strict pass/fail version of `answer_match_score`; the two are
always computed from the same final answer for consistency.

## Worked example — ARR = True ("exact")

| | |
|---|---|
| Query | "Who has conceded the most goals in the premiership?" |
| Gold question | "Which **team** has conceded the most goals in the premiership?" |
| Gold answer (`Ag`) | "Tottenham Hotspur" |
| ORACLE's question (`Q`) | "Are you asking about the all-time record, a specific season, and are you interested in a **team** or an individual goalkeeper?" |

The ARR prompt becomes:

```
Original question: "Who has conceded the most goals in the premiership?"
Clarifying question asked: "Are you asking about the all-time record, a
  specific season, and are you interested in a team or an individual
  goalkeeper?"
User replied: "Tottenham Hotspur"

Now answer the original question in one short sentence.
```

GPT answers with a sentence that contains "Tottenham Hotspur" → the normalized
gold answer is a substring of the normalized response →
**`arr = True`, `answer_match = "exact"`**.

(Row id `2034739379280412436` in `evaluation/results_enriched.csv`.)

## Worked example — ARR = False ("far")

| | |
|---|---|
| Query | "How many ligue 1 titles does psg have?" |
| Gold question | "How many ligue 1 titles does psg have **as of 2017**?" |
| Gold answer (`Ag`) | "6" |
| ORACLE's question (`Q`) | "Are you looking for PSG's total Ligue 1 titles up to and including the 2023-2024 season, or up to a specific earlier season?" |

ORACLE's question correctly identifies *the kind* of ambiguity (a time
cutoff) — arguably a *better* question than the gold one, since it asks the
user to name the cutoff rather than assuming "as of 2017". But the simulated
reply is just the bare gold answer, `"6"`, which doesn't actually answer "which
season cutoff do you mean?". With no cutoff information to work with, GPT
answers using a more current cutoff, so its final number doesn't match the
gold "6" in the same framing → **`arr = False`, `answer_match = "far"`**.

(Row id `-8652199953083038138` in `evaluation/results_enriched.csv`.)

This pair of examples is the cleanest illustration of ARR's main limitation
(see *Notes* below): a good clarifying question can still score `False` when
the simulated reply — the dataset's gold answer — doesn't carry enough
information to resolve the very ambiguity the question raised.

## Aggregate results

`arr = True` rate over queries where the system produced a non-empty
clarifying question / response and a gold answer exists:

| System | ARR | n |
|---|---:|---:|
| Baseline GPT | 62.5% | 224 |
| Baseline Claude | 59.4% | 224 |
| Debate-Lite | 86.2% | 65 |
| ORACLE | 84.9% | 53 |

(`n` differs across systems: Debate-Lite/ORACLE only ask a clarifying question
on the subset of queries they themselves flag as ambiguous, while the
baselines always answer directly — see *Notes*.)

## Notes & limitations

- **Simulated user, not a real one.** ARR assumes the gold answer alone is a
  faithful stand-in for what a real user would say in response to the
  clarifying question. The PSG example above shows this assumption can break:
  the gold answer answers the *original* query, not necessarily the
  *clarifying* one.
- **Substring matching is strict.** `_normalize()` only lower-cases,
  collapses whitespace, and strips a trailing period — paraphrases,
  abbreviations, or differently-formatted numbers/dates can cause a correct
  answer to be scored `"far"`. `answer_match_score`'s `"close"` tier
  (BERTScore ≥ 0.7) exists specifically to catch near-misses like this.
- **Baselines don't ask clarifying questions.** The same `_apply_scores` /
  `compute_auto_score` path is run uniformly over all four systems for
  comparability, so for `baseline_gpt`/`baseline_claude` the "clarifying
  question" slot in the ARR prompt is filled with the baseline's **direct
  answer text** rather than an actual question. Their ARR numbers should
  therefore be read as "does feeding the model's own direct answer back as
  context, plus the gold answer, still let it recover the right answer" — a
  rough self-consistency check rather than a true clarification-loop
  measurement. Only the Debate-Lite and ORACLE numbers measure what ARR is
  actually meant to measure.

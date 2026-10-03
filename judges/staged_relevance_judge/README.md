# StagedRelevanceJudge

`StagedRelevanceJudge` is an LLM-based judge for the Auto-Judge framework that extends the
[`TinyJudge`](https://github.com/trec-auto-judge/auto-judge-starter-kit/tree/main/judges/tinyjudge)
idea with a 3-stage cascade and four preconfigured "cascade user model" measures that aggregate
the stages differently depending on how deeply a user is assumed to read.

## Stages

For every response, three separate LLM requests are made:

1. **`FIRST_SENTENCE_RELEVANT`** — Is the first sentence (or leading `segments`) of the response
   relevant to the query? This approximates what a user sees in a preview/snippet.
2. **`OVERALL_RELEVANT`** — Is the response as a whole relevant to the query? This approximates
   what a user concludes after reading the full response.
3. **`EXPERT_ENDORSED`** — Would a long-time domain expert, with deep expertise on the topic, have
   written the response roughly the way it is? This approximates what a highly engaged,
   expert-level user concludes after close scrutiny.

Each measure is a graded LLM judgment on a 0-3 scale (0 = not at all, 3 = fully), matching common
graded relevance judgment conventions. Each prompt asks the LLM to briefly reason about its
judgment, then end the response with a final line containing only the score. Parsing is lenient
about the exact shape of that last line in two ways:

1. An optional `Score:`-style label, an integer or decimal value (e.g. `3`, `Score: 3`, `2.0`,
   `2 - mostly relevant`), and trailing explanation text are all tolerated, as long as the numeric
   score appears at the very start of the line (after the optional label); or
2. if that fails, a standalone numeric score at the very end of the line is also accepted, to
   tolerate LLMs that ignore the "final line" instruction and append the score to the end of their
   reasoning on the same line (e.g. `"...making it highly relevant. 3"`).

Either way the parsed value must be within the valid 0-3 range. Anything else (no score on the
last line, a score buried in the middle of reasoning text or glued to other digits, an
out-of-range value, an empty/garbled response, or an LLM call failure) is treated as a parsing
failure and scored as `0` rather than guessing from reasoning text that may itself mention
unrelated numbers.

## Aggregation: four cascade user models

The three stages are combined into four separate `CASCADE_*_USER` measures, each representing a
different population of users split by how far they read:

| Measure                        | stage1 / stage2 / stage3 stop fractions | Persona                                             |
|---------------------------------|------------------------------------------|------------------------------------------------------|
| `CASCADE_IMPATIENT_USER`        | 0.7 / 0.2 / 0.1                           | Mostly bounces after skimming just the first sentence |
| `CASCADE_CASUAL_READER_USER`    | 0.2 / 0.6 / 0.2                           | Usually reads the whole response, rarely scrutinizes it like an expert |
| `CASCADE_EXPERT_USER`           | 0.1 / 0.2 / 0.7                           | Mostly reads with expert-level scrutiny (e.g. professional/research use cases) |
| `CASCADE_BALANCED_USER`         | 1/3 / 1/3 / 1/3                           | Equal weight across all three reading depths          |

All four are computed from the same three stage judgments and reported together on every run
(no `--variant` needed), so a single pass can be compared across different assumptions about user
reading behavior.

A user who reaches stage *k* is only satisfied if **all** stages up to and including *k* were
graded highly (a user who found the first sentence irrelevant does not become satisfied by the
overall response). The 0-3 grades are normalized to 0.0-1.0, combined as a product ("cascade"),
and each persona's score is its fraction-weighted sum, rescaled back to the 0-3 grade scale so all
measures share the same range:

```text
n1, n2, n3 = s1 / 3, s2 / 3, s3 / 3
CASCADE_*_USER = 3 * (w1 * n1 + w2 * (n1 * n2) + w3 * (n1 * n2 * n3))
```

where `s1`, `s2`, `s3` are the three 0-3 stage grades and `w1`, `w2`, `w3` are that persona's
stop fractions (normalized to sum to 1).

## Requirements

Install the project with the minima-llm dependency (already part of the `[all]` extra used by
the repository's root `Dockerfile`):

```bash
uv pip install -e ".[minima-llm,test]"
```

StagedRelevanceJudge expects these environment variables:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`
- `CACHE_DIR` for the Minima LLM cache

## Run locally

```bash
auto-judge run \
    --workflow judges/staged_relevance_judge/workflow.yml \
    --rag-responses data/kiddie/runs/repgen/ \
    --rag-topics data/kiddie/topics/kiddie-topics.jsonl \
    --out-dir ./output-staged-relevance-judge/
```

Meta-evaluate the produced leaderboard:

```bash
auto-judge-evaluate meta-evaluate \
    --truth-leaderboard data/kiddie/eval/kiddie_fake.eval.ir_measures.txt \
    --truth-format ir_measures \
    --truth-header \
    --eval-format ir_measures \
    --on-missing default \
    output-staged-relevance-judge/staged_relevance_judge.eval.txt
```

## Tests

```bash
python -m pytest -q judges/staged_relevance_judge/test_staged_relevance_judge.py
```

## TIRA code submission

Run a dry run from the repository root to verify the submission before uploading real credentials:

```bash
tira-cli code-submission \
    --dry-run \
    --path . \
    --cache-behaviour deterministic \
    --mount-cache '$CACHE_DIR=EMPTY_DIR' \
    --task trec-auto-judge \
    --dataset kiddie-20260605-training \
    --forward-environment-variable OPENAI_API_KEY OPENAI_BASE_URL OPENAI_MODEL \
    --command 'auto-judge run --workflow /auto-judge/judges/staged_relevance_judge/workflow.yml --rag-responses $inputDataset/runs/*/ --rag-topics $inputDataset/topics/*.jsonl --out-dir $outputDir'
```

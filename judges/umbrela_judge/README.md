# UmbrelaJudge

`UmbrelaJudge` is an LLM-based judge for the Auto-Judge framework built directly on the prompts
from [UMBRELA](https://github.com/castorini/umbrela), the open-source reproduction of Microsoft
Bing's "DNA" LLM relevance judge from Thomas et al.,
["Large Language Models can Accurately Predict Searcher Preferences"](https://arxiv.org/abs/2309.10621).
UMBRELA was built to produce query-passage relevance judgments (qrels) that closely match human TREC
assessors; this judge reuses its exact prompt templates (copied verbatim into `prompt_templates/`)
but applies them to RAG evaluation by treating each full RAG response as the "passage" being graded
for relevance to its topic's "query" (title).

## Relevance scale

Every (topic, run) response is graded on UMBRELA's standard 0-3 scale:

| Grade | Meaning |
|-------|---------|
| 0 | The response has nothing to do with the query |
| 1 | The response seems related to the query but does not answer it |
| 2 | The response has some answer for the query, but it may be unclear or hidden amongst extraneous information |
| 3 | The response is dedicated to the query and contains the exact answer |

## Prompt styles

Two UMBRELA prompt styles are available via the `prompt_style` setting:

- **`bing`** (default) — the original step-by-step "DNA" prompt: the LLM reasons about search
  intent, content match, and trustworthiness before giving a final score formatted as
  `##final score: N`.
- **`basic`** — UMBRELA's simpler prompt, without explicit step-by-step reasoning instructions.

Set `few_shot: true` to use UMBRELA's own built-in worked examples (only available for
`prompt_style: "bing"`; UMBRELA's few-shot `basic` template requires externally supplied examples
that are out of scope for this self-contained judge).

A `basic` variant is predefined in `workflow.yml` so both prompt styles can be run without editing
settings by hand:

```bash
auto-judge run --workflow judges/umbrela_judge/workflow.yml --variant basic ...   # "basic" prompt
auto-judge run --workflow judges/umbrela_judge/workflow.yml --all-variants ...    # both "default" (bing) and "basic"
```

`filebase: "umbrela_judge-{_name}"` resolves `{_name}` to the config name ("default" or "basic"),
so each variant's output files (e.g. `umbrela_judge-default.eval.txt`,
`umbrela_judge-basic.eval.txt`) don't overwrite each other.

Score parsing is a close adaptation of UMBRELA's own parser
(`umbrela.utils.common_utils.parse_fewshot_response`): it scans the *entire* response text for the
*last* occurrence of any recognized score pattern (`##final score: N`, `Score: N`, `category N`, a
standalone trailing digit, etc.), matching UMBRELA's strategy of preferring a restated final answer
over an incidental number mentioned earlier in the reasoning. Anything unparseable is scored `0`
with `UMBRELA_VALID_RESPONSE_RATE` recording the parse failure, rather than guessing.

## Measures

| Measure | Description |
|---------|--------------|
| `UMBRELA_RELEVANCE` | Mean UMBRELA graded relevance score (0-3) across topics. |
| `UMBRELA_RELEVANT` | Fraction of responses graded >= 2 (the standard graded-to-binary relevance cutoff), 0.0-1.0. |
| `UMBRELA_VALID_RESPONSE_RATE` | Fraction of LLM judgments successfully parsed into a 0-3 category; a prompt/LLM reliability diagnostic, not a quality measure. |

## Qrels

`create_qrels()` can run the same UMBRELA grading and produce standard `Qrels` rows
(`topic_id`, `doc_id=run_id`, `grade`), since generating relevance judgments is UMBRELA's original
purpose. This is **disabled by default** (`create_qrels: false` in `workflow.yml`), so no
`.qrels.txt` file is written. `judge()` always grades responses itself and never depends on it, so
enabling it only adds an extra output artifact. To enable it, set `create_qrels: true` and
uncomment `qrels_class` in `workflow.yml`.

## Requirements

Install the project with the minima-llm dependency (already part of the `[all]` extra used by
the repository's root `Dockerfile`):

```bash
uv pip install -e ".[minima-llm,test]"
```

UmbrelaJudge expects these environment variables:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`
- `CACHE_DIR` for the Minima LLM cache

## Run locally

```bash
auto-judge run \
    --workflow judges/umbrela_judge/workflow.yml \
    --rag-responses data/kiddie/runs/repgen/ \
    --rag-topics data/kiddie/topics/kiddie-topics.jsonl \
    --out-dir ./output-umbrela-judge/
```

Meta-evaluate the produced leaderboard:

```bash
auto-judge-evaluate meta-evaluate \
    --truth-leaderboard data/kiddie/eval/kiddie_fake.eval.ir_measures.txt \
    --truth-format ir_measures \
    --truth-header \
    --eval-format ir_measures \
    --on-missing default \
    output-umbrela-judge/umbrela_judge-default.eval.txt
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
    --command 'auto-judge run --workflow /auto-judge/judges/umbrela_judge/workflow.yml --rag-responses $inputDataset/runs/*/ --rag-topics $inputDataset/topics/*.jsonl --out-dir $outputDir'
```

## Tests

```bash
python -m pytest -q judges/umbrela_judge/test_umbrela_judge.py
```

## Attribution

Prompt templates in `prompt_templates/` are copied verbatim from
[castorini/umbrela](https://github.com/castorini/umbrela)
(`src/umbrela/prompts/prompt_templates/qrel_{zeroshot,fewshot}_{bing,basic}.yaml`, Apache-2.0
licensed), which itself reproduces the "DNA" prompt from Thomas et al. (SIGIR 2024).

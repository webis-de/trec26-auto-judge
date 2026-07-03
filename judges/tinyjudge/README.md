# TinyJudge

`TinyJudge` is a minimal LLM-based example judge for the Auto-Judge framework.

It asks an LLM whether the first sentence of each response is relevant to the topic and writes a leaderboard with one measure:

- `FIRST_SENTENCE_RELEVANT`: binary relevance score from the LLM (`0` or `1`)

This example is intentionally small. It is useful for understanding how to wire an LLM-backed judge, configure caching, and prepare a TIRA code submission.

## Requirements

Install the project with the TinyJudge dependencies:

```bash
uv pip install -e ".[minima-llm,evaluate]"
```

TinyJudge expects these environment variables:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`
- `CACHE_DIR` for the Minima LLM cache

`CACHE_DIR` is especially important for repeatability and for resuming runs without additional API costs for already cached LLM requests.

## Run locally from an existing cache

When a AutoJudge system was executed in TIRA, we can use the cache of the previous execution to run an AutoJudge system without making real LLM requests. For this, we can download a published cache of a previous execution of the TinyJudge on the kiddie dataset (for other datasets and judges it would be analogously). We can download and unzip the cache via:

```bash
wget https://www.tira.io/task/trec-auto-judge/user/webis/dataset/kiddie-20260605-training/download/2026-06-23-15-44-28.zip
unzip -j 2026-06-23-15-44-28.zip '2026-06-23-15-44-28/CACHE_DIR/*' -d example-cache-kiddie
```

We can verify the cache is as expected:

```bash
md5sum example-cache-kiddie/minima_llm.db 
cbf7afc38051de5d41ac67481094f62d  example-cache-kiddie/minima_llm.db
```

Now, we can set environment variables (we do not need an valid api key and base url as we use the previously downloaded cache) as:

```bash
export OPENAI_API_KEY=empty
export OPENAI_BASE_URL=empty
export OPENAI_MODEL=llama-3.1-8b-instant
export CACHE_DIR=example-cache-kiddie
```

To run the TinyJudge on the kiddie dataset from the cache:

```bash
auto-judge run \
    --workflow judges/tinyjudge/workflow.yml \
    --rag-responses data/kiddie/runs/repgen/ \
    --rag-topics data/kiddie/topics/kiddie-topics.jsonl \
    --out-dir ./output-tinyjudge/
```

Meta-evaluate the produced leaderboard:

```bash
auto-judge-evaluate meta-evaluate \
    --truth-leaderboard data/kiddie/eval/kiddie_fake.eval.ir_measures.txt \
    --truth-format ir_measures \
    --truth-header \
    --eval-format ir_measures \
    --on-missing default \
    output-tinyjudge/tinyjudge.eval.txt
```

Example output:

```text
         Judge TruthMeasure             EvalMeasure  kendall  pearson  spearman  tauap_b  kendall@10
tinyjudge.eval    RELEVANCE FIRST_SENTENCE_RELEVANT 0.547723 0.702559  0.632456 0.416667    0.547723
```

The kiddie dataset is synthetic, so these numbers are not meaningful, but still can help to easily verify that a judge produces a valid output.

## Run with a real LLM endpoint

To execute TinyJudge without a precomputed cache, point the environment
variables at a real OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...
export OPENAI_MODEL=...
export CACHE_DIR=./cache-tinyjudge
```

Then run the same `auto-judge run` command shown above.

## Submit to TIRA

For code submission, make sure Docker or Podman and `tira-cli` are installed.

Export the LLM configuration first (they are only used on your machine):

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...
export OPENAI_MODEL=...
```

Then run the dry run from the repository root:

```bash
tira-cli code-submission \
    --dry-run \
    --path . \
    --cache-behaviour deterministic \
    --mount-cache '$CACHE_DIR=EMPTY_DIR' \
    --task trec-auto-judge \
    --dataset kiddie-20260605-training \
    --forward-environment-variable OPENAI_API_KEY OPENAI_BASE_URL OPENAI_MODEL \
    --command 'auto-judge run --workflow /auto-judge/judges/tinyjudge/workflow.yml --rag-responses $inputDataset/runs/*/ --rag-topics $inputDataset/topics/*.jsonl --out-dir $outputDir'
```

If everything worked, the output should look like this:

<img width="1807" height="259" alt="Screenshot_20260703_165606" src="https://github.com/user-attachments/assets/409ce157-53b9-48ee-bd36-a01d46f9d352" />

If the dry run succeeds, remove `--dry-run` to submit the AutoJudge system.

`--cache-behaviour deterministic` tells TIRA that repeated runs with the same cache should produce the same output, which is useful to know for reproducibility.

For the full submission workflow, see the
[TIRA participant documentation](https://docs.tira.io/participants/participate.html#prepare-your-submission).

## Run the published TinyJudge

TinyJudge is already published on TIRA and can be executed locally through `tira-cli`.

Using the bundled cache:

```bash
OPENAI_API_KEY=empty OPENAI_BASE_URL=empty OPENAI_MODEL=llama-3.1-8b-instant \
tira-cli run local \
    --approach trec-auto-judge/webis/tinyjudge \
    --input kiddie-20260605-training \
    --forward-environment-variable OPENAI_API_KEY OPENAI_BASE_URL OPENAI_MODEL \
    --mount-cache "CACHE_DIR=example-cache-kiddie"
```

-The output should look like:

<img width="957" height="389" alt="Screenshot_20260703_170402" src="https://github.com/user-attachments/assets/1bcf0fb9-dda4-4819-9d6d-25e52c7fd62b" />

Using a real LLM endpoint with an empty cache:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...
export OPENAI_MODEL=...
```

```bash
tira-cli run local \
    --approach trec-auto-judge/webis/tinyjudge \
    --input kiddie-20260605-training \
    --forward-environment-variable OPENAI_API_KEY OPENAI_BASE_URL OPENAI_MODEL \
    --mount-cache "CACHE_DIR=EMPTY_DIR"
```

The output should look like:

<img width="957" height="389" alt="Screenshot_20260703_170630" src="https://github.com/user-attachments/assets/124ab00c-6312-4fbc-ac9a-b3a37fde2257" />

The produced output directory also contains the cache, which can be reused for later runs.

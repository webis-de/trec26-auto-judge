Then run the dry run from the repository root:

This is configured as dev-container.

In the dev-container, please run:

```bash
CACHE_DIR=auto-nuggetizer-cache PYTHONPATH=../.. auto-judge run \
    --workflow workflow.yml \
    --rag-responses ../../data/kiddie/runs/repgen/ \
    --rag-topics ../../data/kiddie/topics/*.jsonl \
    --variant documents \
    --out-dir fsa
```


```bash
tira-cli code-submission \
    --dry-run \
    --path . \
    --file judges/auto_nuggetizer/Dockerfile \
    --cache-behaviour deterministic \
    --mount-cache '$CACHE_DIR=EMPTY_DIR' \
    --task trec-auto-judge \
    --dataset kiddie-20260605-training \
    --forward-environment-variable OPENAI_API_KEY OPENAI_BASE_URL OPENAI_MODEL \
    --command 'auto-judge run --variant documents-and-responses --workflow /auto-judge/judges/auto_nuggetizer/workflow.yml --rag-responses $inputDataset/runs/*/ --rag-topics $inputDataset/topics/*.jsonl --out-dir $outputDir'
```


```bash
tira-cli code-submission \
    --dry-run \
    --path . \
    --file judges/auto_nuggetizer/Dockerfile \
    --cache-behaviour deterministic \
    --mount-cache '$CACHE_DIR=EMPTY_DIR' \
    --task trec-auto-judge \
    --dataset kiddie-20260605-training \
    --forward-environment-variable OPENAI_API_KEY OPENAI_BASE_URL OPENAI_MODEL \
    --command 'auto-judge run --variant responses --workflow /auto-judge/judges/auto_nuggetizer/workflow.yml --rag-responses $inputDataset/runs/*/ --rag-topics $inputDataset/topics/*.jsonl --out-dir $outputDir'
```


```bash
tira-cli code-submission \
    --dry-run \
    --path . \
    --file judges/auto_nuggetizer/Dockerfile \
    --cache-behaviour deterministic \
    --mount-cache '$CACHE_DIR=EMPTY_DIR' \
    --task trec-auto-judge \
    --dataset kiddie-20260605-training \
    --forward-environment-variable OPENAI_API_KEY OPENAI_BASE_URL OPENAI_MODEL \
    --command 'auto-judge run --variant documents --workflow /auto-judge/judges/auto_nuggetizer/workflow.yml --rag-responses $inputDataset/runs/*/ --rag-topics $inputDataset/topics/*.jsonl --out-dir $outputDir'
```
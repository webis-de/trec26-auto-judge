# JevJudge

Like TinyJudge, but the judgments come from the TypeSafe "jev" API
(`typesafe_sdk.AsyncTypeSafeClient.system_one`) rather than a chat LLM.
For each response, one request asks two questions about the leading segment(s):

- `FIRST_SENTENCE_RELEVANT`: `Noul` probability (0-1) that the text is relevant to the query.
- `FIRST_SENTENCE_RELEVANCE_GRADE`: `Score` (0-3) expected relevance grade.

Requires `TYPESAFE_API_KEY` in the environment.

```bash
auto-judge run --workflow judges/jev_judge/workflow.yml \
    --rag-responses data/kiddie/runs/repgen/ \
    --rag-topics data/kiddie/topics/kiddie-topics.jsonl \
    --out-dir ./output-kiddie/
```

TIRA dry run:

```bash
tira-cli code-submission \
    --dry-run \
    --path . \
    --task trec-auto-judge \
    --dataset kiddie-20260605-training \
    --forward-environment-variable TYPESAFE_API_KEY \
    --allow-network \
    --command 'auto-judge run --workflow /auto-judge/judges/jev_judge/workflow.yml --rag-responses $inputDataset/runs/*/ --rag-topics $inputDataset/topics/*.jsonl --out-dir $outputDir'
```

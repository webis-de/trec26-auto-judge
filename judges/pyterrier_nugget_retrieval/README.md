# PyTerrier Nugget Retrieval Judge

This judge indexes the submitted response from every system as one document per
topic. Each nugget in the supplied nugget bank is issued as a separate
PyTerrier query. For every weighting model, retrieval scores are min-max
normalized across systems for that nugget and aggregated per response as:

- `mean`: average coverage across all nuggets
- `min`: coverage of the least-represented nugget
- `max`: coverage of the best-represented nugget

All measures range from 0.0 to 1.0, and higher is better. The judge fails when
the nugget-bank input is missing or a response topic has no non-empty nuggets.

## Example nuggets

```
wget https://www.tira.io/task/trec-auto-judge/user/webis/dataset/kiddie-20260605-training/download/2026-07-14-08-08-51.zip
unzip 2026-07-14-08-08-51.zip 2026-07-14-08-08-51/output/auto_nuggetizer.nuggets.jsonl
```

## TIRA dry run

```
tira-cli code-submission \
            --dry-run \
            --path . \
            --file judges/pyterrier_nugget_retrieval/Dockerfile \
            --task trec-auto-judge \
            --dataset kiddie-20260605-training \
            --mount-dir '$NUGGETS=trec-auto-judge/webis/prefnugget-queryonly' \
            --command 'auto-judge run --workflow /auto-judge/judges/pyterrier_nugget_retrieval/workflow.yml --rag-responses $inputDataset/runs/*/ --rag-topics $inputDataset/topics/*.jsonl --out-dir $outputDir --nugget-banks $NUGGETS/*.nuggets.jsonl'
```

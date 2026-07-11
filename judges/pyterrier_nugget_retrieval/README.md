
```
tira-cli code-submission \
            --dry-run \
            --path . \
            --file judges/pyterrier_nugget_retrieval/Dockerfile \
            --task trec-auto-judge \
            --dataset kiddie-20260605-training \
            --mount-dir '$NUGGETS=trec-auto-judge/webis/prefnugget-queryonly' \
            --command 'auto-judge run --workflow /auto-judge/judges/pyterrier_nugget_retrieval/workflow.yml --rag-responses $inputDataset/runs/*/ --rag-topics $inputDataset/topics/*.jsonl --out-dir $outputDir --nugget-banks $NUGGETS'
```


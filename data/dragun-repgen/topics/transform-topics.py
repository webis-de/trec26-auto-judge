#!/usr/bin/env python3
import json

with open("trec-2025-dragun-topics.jsonl", "r") as f_in, open("topics.jsonl", "w") as f_out:
    for l in f_in:
        l = json.loads(l)
        m = {"original_topic": l}
        m["id"] = l["docid"]
        m["request_id"] = l["docid"]
        m["title"] = l["title"]
        m["narrative"] = l["title"]
        f_out.write(json.dumps(m) + "\n")

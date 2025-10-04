# make_train_qas.py (very simple pseudo)
import json, sys
from collections import defaultdict

def load_passages(path):
    return [json.loads(l) for l in open(path, "r", encoding="utf-8")]

if __name__ == "__main__":
    passages = load_passages(sys.argv[1])
    # for each original source (source_id), use title as query and positives = all chunks from same source
    src_map = defaultdict(list)
    for p in passages:
        src_map[p["meta"]["source_id"]].append(p["id"])
    out = []
    # you should provide a CSV with title->source_id mapping or reuse passages meta
    for p in passages:
        title = p["meta"].get("title")
        if not title: continue
        pos = src_map[p["meta"]["source_id"]]
        out.append({"query": title, "positive_ids": pos, "answers": []})
    with open("datasets/train_qas.jsonl","w",encoding="utf-8") as f:
        for item in out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("Wrote train_qas.jsonl with", len(out))

# csv_to_passages.py
import csv
import json
import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_csv_texts(path: str, title_col="Title", content_col="Content") -> List[dict]:
    items = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i,row in enumerate(reader):
            title = row.get(title_col, "").strip()
            content = row.get(content_col, "").strip()
            if not content and not title:
                continue
            # combine title + content to keep context
            text = (title + "\n\n" + content) if title else content
            items.append({"source_id": str(i), "title": title, "text": text})
    return items

def chunk_texts(items: List[dict], chunk_size=700, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    passages = []
    for it in items:
        chunks = splitter.split_text(it["text"])
        for idx, c in enumerate(chunks):
            pid = f"{it['source_id']}_p{idx}"
            meta = {"source_id": it["source_id"], "title": it["title"], "chunk_index": idx}
            passages.append({"id": pid, "text": c, "meta": meta})
    return passages

def write_jsonl(passages: List[dict], out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Wrote {len(passages)} passages to {out_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python datasets/csv_to_passages.py datasets/formatted_posts.csv datasets/data/passages.jsonl")
        sys.exit(1)
    in_csv = sys.argv[1]
    out_jsonl = sys.argv[2]
    items = load_csv_texts(in_csv)
    passages = chunk_texts(items, chunk_size=700, chunk_overlap=150)
    write_jsonl(passages, out_jsonl)

# build_faiss.py
import json
import sys
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

def load_passages(path):
    passages = [json.loads(l) for l in open(path, "r", encoding="utf-8")]
    texts = [p["text"] for p in passages]
    ids = [p["id"] for p in passages]
    return texts, ids

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python datasets/build_faiss.py datasets/passages.jsonl datasets/out_faiss.index datasets/out_id_map.json")
        sys.exit(1)
    passages_path = sys.argv[1]
    # tải từ HF Hub
    model = SentenceTransformer("AITeamVN/Vietnamese_Embedding_v2")

    # save về local path
    model.save("D:/Documents/HuggingFace/Vietnamese_Embedding_v2")

    # lần sau chỉ cần load local
    model = SentenceTransformer("D:/Documents/HuggingFace/Vietnamese_Embedding_v2")
    out_faiss = sys.argv[2]
    out_idmap = sys.argv[3]

    texts, ids = load_passages(passages_path)
    print(f"Loaded {len(texts)} passages")
    
    embs = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)

    # normalize for cosine (use inner product)
    faiss.normalize_L2(embs)
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs)
    faiss.write_index(index, out_faiss)
    print(f"Saved FAISS index to {out_faiss}")

    import json
    Path(out_idmap).parent.mkdir(parents=True, exist_ok=True)
    with open(out_idmap, "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False)
    print(f"Wrote id map to {out_idmap}")

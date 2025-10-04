# query_faiss.py
import sys, json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def load_id_map(idmap_path):
    return json.load(open(idmap_path, "r", encoding="utf-8"))

def load_passage_by_idmap(passages_jsonl):
    d = {}
    for line in open(passages_jsonl, "r", encoding="utf-8"):
        j = json.loads(line)
        d[j["id"]] = j["text"]
    return d

def load_queries(qas_jsonl):
    return [json.loads(line) for line in open(qas_jsonl, "r", encoding="utf-8")]

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: python datasets/query_faiss.py datasets/synthetic_qas.jsonl datasets/out_faiss.index datasets/out_id_map.json datasets/passages.jsonl 5")
        sys.exit(1)

    qas_path = sys.argv[1]
    faiss_idx = sys.argv[2]
    id_map = sys.argv[3]
    passages_jsonl = sys.argv[4]
    top_k = int(sys.argv[5])

    # Load FAISS index + id map + passages
    index = faiss.read_index(faiss_idx)
    ids = load_id_map(id_map)
    id_to_text = load_passage_by_idmap(passages_jsonl)

    model = SentenceTransformer("D:/Documents/HuggingFace/Vietnamese_Embedding_v2")
    
     # Load queries
    qas = load_queries(qas_path)

    for qi, qa in enumerate(qas[:10]):   # chỉ lấy 10 query đầu tiên để demo
        query = qa["query"]
        qvec = model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(qvec)
        D, I = index.search(qvec, top_k)

        print(f"\nQ{qi+1}: {query}")
        for rank, idx in enumerate(I[0]):
            pid = ids[idx]
            score = float(D[0][rank])
            text = id_to_text.get(pid, "")
            print(f"  {rank+1}. id={pid}  score={score:.4f}")
            print("     ", text[:200].replace("\n", " "), "...")

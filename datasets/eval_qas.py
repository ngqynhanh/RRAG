# eval_qas.py
import sys, json
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics import recall_score
import torch

model = SentenceTransformer("AITeamVN/Vietnamese_Embedding_v2")

def load_id_map(idmap_path):
    return json.load(open(idmap_path, "r", encoding="utf-8"))

def load_passages(passages_jsonl):
    data = {}
    for line in open(passages_jsonl,"r",encoding="utf-8"):
        j = json.loads(line)
        data[j["id"]] = j
    return data

if __name__ == "__main__":
    if len(sys.argv)<6:
        print("Usage: python datasets/eval_qas.py datasets/synthetic_qas.jsonl datasets/out_faiss.index datasets/out_id_map.json datasets/passages.jsonl 5")
        sys.exit(1)
    qas_path, faiss_idx, id_map_path, passages_path, top_k = sys.argv[1:6]
    top_k = int(top_k)

    ids = load_id_map(id_map_path)
    passages = load_passages(passages_path)
    index = faiss.read_index(faiss_idx)
    model = SentenceTransformer(model)

    with open(qas_path,"r",encoding="utf-8") as f:
        qas = [json.loads(l) for l in f]

    for qi, qa in enumerate(qas[:10]):  # chỉ in 10 query đầu tiên
        query = qa["query"]
        qvec = model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(qvec)
        D,I = index.search(qvec, top_k)
        print(f"\nQ{qi+1}: {query}")
        for rank, idx in enumerate(I[0]):
            pid = ids[idx]
            text = passages[pid]["text"]
            score = float(D[0][rank])
            print(f"  {rank+1}. {pid}  score={score:.4f}")
            print("     ", text[:150].replace("\n"," "),"...")



import os
import re
import json
import random
from pathlib import Path
from openai import OpenAI

# -----------------------------
# Cấu hình
# -----------------------------
MODEL    = os.getenv("MODEL_ID", "qwen/qwen-2.5-72b-instruct")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY","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")
if not OPENROUTER_API_KEY:
    raise RuntimeError("Thiếu OPENROUTER_API_KEY trong .env")

client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

# -----------------------------
# 1. Load passages tổng hợp
# -----------------------------
def load_passages(path):
    passages = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            passages.append(json.loads(line))
    return passages

# -----------------------------
# 2. Load graph info (CSV / TXT)
# -----------------------------
def load_graph_info(path):
    """
    Trả về dict {layer_name: set([entity_name])}
    """
    layer_dict = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # Regex tìm {name: ..., layer: ...}
            matches = re.findall(r"{name: (.+?), layer: (.+?)}", line)
            for name, layer in matches:
                layer_dict.setdefault(layer, set()).add(name.strip())
    return layer_dict

# -----------------------------
# 3. Tìm passage liên quan theo keyword
# -----------------------------
def find_passages_for_entity(entity_name, passages, top_k=3):
    matched = []
    for p in passages:
        if entity_name.lower() in p["text"].lower():
            matched.append({
                "id": p.get("id",""),
                "title": p.get("meta",{}).get("title",""),
                "text": p["text"],
                "hasanswer": True,
                "isgold": True
            })
    return matched[:top_k]

# -----------------------------
# 4. Sinh câu hỏi từ entity + layer
# -----------------------------
def generate_question(entity_text, layer):
    prompt = f"""
Bạn là trợ lý y tế, tạo câu hỏi ngắn từ thông tin sau:

Thông tin: {entity_text}
Layer: {layer}

Hãy tạo ra:
1. Câu hỏi ngắn, dễ hiểu.
2. Output dạng JSON:
{{
    "question": "..."
}}
"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    out = resp.choices[0].message.content
    try:
        parsed = json.loads(out[out.index("{"):out.rindex("}")+1])
    except:
        parsed = {"question": None}
    return parsed["question"]

# -----------------------------
# 5. Xây ctxs (positive + negatives)
# -----------------------------
def build_ctxs(passages, pos_idx, num_neg=2):
    pos = passages[pos_idx]
    negs = random.sample([p for i,p in enumerate(passages) if i!=pos_idx], num_neg)
    ctxs = []

    ctxs.append({
        "id": pos["id"],
        "title": pos.get("title",""),
        "text": pos["text"],
        "hasanswer": True,
        "isgold": True
    })
    for n in negs:
        ctxs.append({
            "id": n["id"],
            "title": n.get("title",""),
            "text": n["text"],
            "hasanswer": False,
            "isgold": False
        })
    return ctxs

# -----------------------------
# 6. MAIN
# -----------------------------
if __name__ == "__main__":
    passages = load_passages("datasets/passages.jsonl")
    layer_dict = load_graph_info("datasets/neo4j_query_table_data_2025-10-5.txt")

    output_path = Path("datasets/qas_synthetic.jsonl")
    with open(output_path, "w", encoding="utf-8") as fout:
        for layer, entities in layer_dict.items():
            for entity in entities:
                # 1. Tìm passages liên quan
                ctxs = find_passages_for_entity(entity, passages, top_k=3)
                if not ctxs:
                    continue

                # 2. Gom text passages để làm input cho Q&A
                combined_text = " ".join([c["text"] for c in ctxs])

                # 3. Sinh câu hỏi
                question = generate_question(entity, layer)
                if not question:
                    continue

                # 4. Tạo JSON cuối cùng
                example = {
                    "question": question,
                    "answers": [combined_text],  # answer gom từ các passage chứa entity
                    "ctxs": ctxs
                }

                fout.write(json.dumps(example, ensure_ascii=False) + "\n")

    print("Saved synthetic QAs to", output_path)

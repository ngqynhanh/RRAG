import os
import re
import traceback
import pandas as pd
import json
import random
from pathlib import Path
from transformers import pipeline
from dotenv import load_dotenv
from openai import OpenAI

MODEL    = os.getenv("MODEL_ID", "qwen/qwen-2.5-72b-instruct")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("Thiếu OPENROUTER_API_KEY trong .env")

client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

# 1. Load passages
def load_passages(path):
    passages = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            passages.append(json.loads(line))
    return passages

# 2. Dùng model sinh Q&A từ passage
def generate_qa_from_passage(client, passage_text):
    prompt = f"""
    Đoạn văn: {passage_text}

    1. Xác định câu hỏi ngắn phù hợp với đoạn văn.
    2. Trả lời ngắn gọn.
    3. Gán nhãn câu hỏi vào một trong 15 lớp sau: 
    [Definition, Symptom, Cause, Treatment, Prevention, RiskFactor, Disease, SubDisease, Population, Detail, Complication, Advice, Prevention, Topic, SubTopic].
    Nếu đoạn văn không thuộc lớp nào rõ ràng, hoặc không có thông tin phù hợp (ví dụ Application không tồn tại), hãy trả về "None".
    
    Output JSON:
    {{
    "question": "...",
    "answer": "...",
    "label": "..."
    }}

    Ví dụ output:
    {{
    "question": "Triệu chứng nào gợi ý bạn nên đi khám phụ khoa?",
    "answers": ["Khi xuất hiện các triệu chứng như đau bụng kinh dữ dội, khí hư bất thường, rối loạn kinh nguyệt hoặc đau khi quan hệ tình dục, bạn nên đi khám bác sĩ phụ khoa."],
    "labels": ["symptom"]
    }}
    
"""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    out = resp.choices[0].message.content
    try:
        parsed = json.loads(out[out.index("{"):out.rindex("}")+1])
    except:
        parsed = {"question": None, "answer": None, "label": None}
    return parsed["question"], parsed["answer"], parsed["label"]

# 3. Xây ctxs (positive + negatives)
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

# MAIN
if __name__ == "__main__":
    passages = load_passages("datasets/passages.jsonl")

    out_path = Path("datasets/synthetic_qas.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for i, passage in enumerate(passages[:50]):  # ví dụ sinh 50 query
            q, a, l = generate_qa_from_passage(client, passage["text"])
            if not q or not a:
                continue
            ctxs = build_ctxs(passages, i, num_neg=3)

            example = {
                "question": q,
                "answers": [a],
                "labels": [l] if l else [],
                "ctxs": ctxs
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print("Saved synthetic dataset to", out_path)
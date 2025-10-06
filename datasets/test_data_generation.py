import pandas as pd
import json
import requests
from collections import defaultdict
from underthesea import word_tokenize
import ast
from openai import OpenAI
from dotenv import load_dotenv
import os

# -----------------------------
# 1. Load filtered entity file
# -----------------------------
df = pd.read_csv("datasets/entity_full_mapping_filtered.csv")

# -----------------------------
# 2. Load passages data
# -----------------------------
passages_file = "datasets/passages.jsonl"
passages = []
with open(passages_file, "r", encoding="utf-8") as f:
    for line in f:
        passages.append(json.loads(line))

# Map source_id -> list of passages
source_to_passages = defaultdict(list)
for p in passages:
    source_id = str(p["meta"]["source_id"])
    source_to_passages[source_id].append(p)

# -----------------------------
# 3. OpenRouter API setup
# -----------------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

def call_llm(prompt, max_tokens=300):
    resp = client.chat.completions.create(
        model="qwen/qwen-2.5-72b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=max_tokens
    )
    print("Kết nối được với OPENROUTER")
    return resp.choices[0].message.content.strip()

# -----------------------------
# 4. Function: compute_token_match
# -----------------------------
def compute_token_match(answer_text, all_contexts):
    answer_tokens = set(word_tokenize(answer_text.lower()))
    context_tokens = set()
    for ctx in all_contexts:
        context_tokens.update(word_tokenize(ctx['text'].lower()))
    if not answer_tokens:
        return 0.0
    match_count = len(answer_tokens & context_tokens)
    return round(match_count / len(answer_tokens), 2)

# -----------------------------
# 5. Build dataset
# -----------------------------
output_path = "D:\\Work\\Side Projects\\rag-paper\\rag-system\\RRAG\\datasets\\qas_synthetic_vi.jsonl"
output_data = []
layer_cols = [c for c in df.columns if c not in ["entity_name","entity_type","article_ids"]]

with open(output_path, "w", encoding="utf-8") as fout:
    total_q = 0
    for idx, row in df.iterrows():
        entity_name = row["entity_name"]
        article_ids = ast.literal_eval(row["article_ids"])
        layers_dict = {c: row[c] for c in layer_cols if pd.notna(row[c]) and row[c] != ""}

        # -----------------
        # Generate question
        # -----------------
        try:
            prompt_q = f"Dựa vào entity '{entity_name}' với các layer: {json.dumps(layers_dict, ensure_ascii=False)}, hãy viết 1 câu hỏi y tế ngắn gọn, rõ ràng."
            question = call_llm(prompt_q)
        except Exception as e:
            print(f"[Warning] Không tạo được question cho {entity_name}: {e}")
            question = f"Câu hỏi về {entity_name}"

        # -----------------
        # Generate answer
        # -----------------
        try:
            prompt_a = f"Dựa vào entity '{entity_name}' với các layer: {json.dumps(layers_dict, ensure_ascii=False)}, hãy viết câu trả lời chi tiết, đúng sự thật, không thêm thông tin ngoài layer."
            answer = call_llm(prompt_a, max_tokens=500)
        except Exception as e:
            print(f"[Warning] Không tạo được answer cho {entity_name}: {e}")
            answer = "Không có thông tin"

        # -----------------
        # Build ctxs list
        # -----------------
        ctxs_list = []
        for art_id in article_ids:
            source_id = str(int(art_id) - 1)  # rule
            for p in source_to_passages.get(source_id, []):
                ctxs_list.append({
                    "id": p["id"],
                    "title": p["meta"]["title"],
                    "text": p["text"],
                    "score": 1.0,
                    "hasanswer": p.get("hasanswer", False),
                    "isgold": p.get("isgold", False)
                })

        # fallback nếu ctxs trống
        if not ctxs_list:
            ctxs_list.append({
                "id": "",
                "title": "",
                "text": "Không có dữ liệu tham khảo",
                "score": 0.0,
                "hasanswer": False,
                "isgold": False
            })

        # -----------------
        # Compute token_match_score
        # -----------------
        token_match_score = compute_token_match(answer, ctxs_list)

        # -----------------
        # Write to file
        # -----------------
        example = {
            "question": question,
            "answers": [answer] if answer else [],
            "ctxs": ctxs_list,
            "score": token_match_score
        }
        fout.write(json.dumps(example, ensure_ascii=False) + "\n")
        total_q += 1
        print(f"[Info] Tạo data thành công cho entity '{entity_name}' ({total_q} câu hỏi)")

print(f"Done! Dataset with token match score saved tại: {output_path} (tổng câu hỏi: {total_q})")

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
def search_passages(entity_name, passages, top_k=3):
    """
    Search passages có chứa entity_name (hoặc từ khóa liên quan).
    Trả về tối đa top_k passages.
    """
    results = []
    for p in passages:
        if entity_name.lower() in p['text'].lower():  # kiểm tra có chứa entity
            results.append(p)
    return results[:top_k]  # lấy top_k

def compute_token_match(answer_text, passage_text):
    """
    Tính token match score giữa answer và passage đơn lẻ
    """
    answer_tokens = set(word_tokenize(answer_text.lower()))
    passage_tokens = set(word_tokenize(passage_text.lower()))
    if not answer_tokens:
        return 0.0
    match_count = len(answer_tokens & passage_tokens)
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
            prompt_q = f"""Dựa vào entity '{entity_name}' với các layer: {json.dumps(layers_dict, ensure_ascii=False)}, hãy viết 1 câu hỏi y tế ngắn gọn, rõ ràng.
            ví dụ: 
            Triệu chứng của bệnh tiểu đường là gì?
            mang thai nên ăn gì để tốt cho sức khỏe
            bệnh viêm gan b có lây không

            KHÔNG DƯỢC GHI NHƯ SAU: Dựa vào thông tin được cung cấp, đây là một câu hỏi y tế ngắn gọn và rõ ràng: bệnh viêm gan b có lây không
            """
            question = call_llm(prompt_q)
        except Exception as e:
            print(f"[Warning] Không tạo được question cho {entity_name}: {e}")
            question = f"{entity_name} là gì?"

        # -----------------
        # Generate answer
        # -----------------
        try:
            prompt_a = f"""Dựa vào entity '{entity_name}' với các layer: {json.dumps(layers_dict, ensure_ascii=False)}, hãy viết câu trả lời chi tiết, đúng sự thật, không thêm thông tin ngoài layer."""
            answer = call_llm(prompt_a, max_tokens=500)
        except Exception as e:
            print(f"[Warning] Không tạo được answer cho {entity_name}: {e}")
            answer = "Không có thông tin"

        # -----------------
        # Build ctxs list with top-k positive & negative passages
        # -----------------
        ctxs_list = []

        # Lấy tất cả passages của source_id
        all_passages = []
        for art_id in article_ids:
            source_id = str(int(art_id) - 1)
            all_passages.extend(source_to_passages.get(source_id, []))

        # Tính token score cho từng passage
        for p in all_passages:
            token_score = compute_token_match(answer, p['text'])
            has_answer = token_score > 0
            is_gold = has_answer
            p['_token_score'] = token_score
            p['_has_answer'] = has_answer
            p['_is_gold'] = is_gold

        # Top 3 positive passages
        positives = [p for p in all_passages if p['_has_answer']]
        positives = sorted(positives, key=lambda x: x['_token_score'], reverse=True)[:3]

        # Top 2 negative passages
        negatives = [p for p in all_passages if not p['_has_answer']]
        negatives = sorted(negatives, key=lambda x: x['_token_score'], reverse=True)[:2]

        # Gộp lại
        final_passages = positives + negatives

        # Chuẩn hóa ctxs_list
        ctxs_list = [
            {
                "id": p["id"],
                "title": p["meta"]["title"],
                "text": p["text"],
                "score": round(p['_token_score'], 2),
                "hasanswer": p['_has_answer'],
                "isgold": p['_is_gold']
            }
            for p in final_passages
        ]

        # -----------------
        # Write one record per question/entity
        # -----------------
        example = {
            "question": question,
            "answers": [answer] if answer else [],
            "ctxs": ctxs_list,
            "score": round(sum([c['score'] for c in ctxs_list])/len(ctxs_list), 2) if ctxs_list else 0.0
        }
        fout.write(json.dumps(example, ensure_ascii=False) + "\n")
        total_q += 1
        print(f"[Info] Tạo data thành công cho entity '{entity_name}' ({total_q} câu hỏi)")

print(f"Done! Dataset with token match score saved tại: {output_path} (tổng câu hỏi: {total_q})")

import pandas as pd
import json, ast, os, re, unicodedata
from collections import defaultdict
from openai import OpenAI
from dotenv import load_dotenv
from underthesea import word_tokenize

# --- Helper: remove accents (để match không dấu) ---
def normalize(text):
    text = unicodedata.normalize('NFD', text)
    text = re.sub(r'[\u0300-\u036f]', '', text)
    return text.lower().strip()

# --- Load data ---
entity_df = pd.read_csv("datasets/data/3.csv")
questions_df = pd.read_csv("datasets/questions.csv")

# passages mapping
passages = [json.loads(line) for line in open("datasets/passages.jsonl", "r", encoding="utf-8")]
source_to_passages = defaultdict(list)
for p in passages:
    sid = str(p["meta"]["source_id"])
    source_to_passages[sid].append(p)

# --- Setup LLM ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1")

def call_llm(prompt, max_tokens=400):
    resp = client.chat.completions.create(
        model="qwen/qwen-2.5-72b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.08,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()

# --- Compute token overlap ---
def compute_token_match(ans, text):
    ans_tokens = set(word_tokenize(ans.lower()))
    txt_tokens = set(word_tokenize(text.lower()))
    if not ans_tokens:
        return 0.0
    return round(len(ans_tokens & txt_tokens) / len(ans_tokens), 2)

# --- Detect entity in question ---
def detect_entity(question, entity_names):
    q_norm = normalize(question)
    best_match = None
    for name in entity_names:
        if normalize(name) in q_norm:
            if best_match is None or len(name) > len(best_match):
                best_match = name
    return best_match

# --- Get passages from article IDs ---
def get_passages(article_ids):
    all_p = []
    for aid in article_ids:
        sid = str(int(aid) - 1)
        all_p.extend(source_to_passages.get(sid, []))
    return all_p

# --- Main generation ---
output_path = "datasets/qas_synthetic_vi.jsonl"
total = 0

entity_names = entity_df["entity_name"].dropna().tolist()

with open(output_path, "a", encoding="utf-8") as fout:
    for _, qrow in questions_df.iterrows():
        question = qrow["question"]
        labels = qrow["labels"].split("|")
        has_ctx = int(qrow["has_context"])

        # Tìm entity trong câu hỏi
        entity_name = detect_entity(question, entity_names)
        if not entity_name:
            print(f"[⚠️] Không tìm thấy entity cho: {question}")
            continue

        entity_row = entity_df[entity_df["entity_name"] == entity_name].iloc[0]
        article_ids = ast.literal_eval(entity_row["article_ids"])

        # Ghép layer text theo labels
        layer_texts = []
        for lb in labels:
            if lb in entity_row and not pd.isna(entity_row[lb]) and entity_row[lb] != "":
                layer_texts.append(f"- {lb}: {entity_row[lb]}")
        combined_layer = "\n".join(layer_texts) if layer_texts else "Không có thông tin."

        # Context (nếu có)
        context_text = ""
        if has_ctx:
            ctx_passages = get_passages(article_ids)
            ctx_combined = " ".join([p["text"] for p in ctx_passages[:3]])
            context_text = f"\n\nNgữ cảnh liên quan:\n{ctx_combined[:2000]}"

        # Prompt cho LLM
        prompt = f"""
Câu hỏi: "{question}"
Thông tin y khoa từ cơ sở dữ liệu:
{combined_layer}
{context_text}

Hãy viết câu trả lời chi tiết, rõ ràng, tự nhiên và đúng sự thật bằng tiếng Việt.
Chỉ dựa vào thông tin cung cấp, không thêm chi tiết ngoài dữ kiện này.
"""
        try:
            answer = call_llm(prompt)
        except Exception as e:
            print(f"[Error] Lỗi sinh câu trả lời cho {question}: {e}")
            answer = "Không có thông tin."

        # Match passages
        all_pass = get_passages(article_ids)
        for p in all_pass:
            p["_score"] = compute_token_match(answer, p["text"])
            p["_has_ans"] = p["_score"] > 0.15
            p["_is_gold"] = p["_has_ans"]

        pos = sorted([p for p in all_pass if p["_has_ans"]], key=lambda x: x["_score"], reverse=True)[:3]
        neg = sorted([p for p in all_pass if not p["_has_ans"]], key=lambda x: x["_score"], reverse=True)[:2]
        final_pass = pos + neg

        ctxs = [
            {
                "id": p["id"],
                "title": p["meta"]["title"],
                "text": p["text"],
                "score": p["_score"],
                "hasanswer": p["_has_ans"],
                "isgold": p["_is_gold"]
            }
            for p in final_pass
        ]

        example = {"question": question, "answers": [answer], "ctxs": ctxs}
        fout.write(json.dumps(example, ensure_ascii=False) + "\n")
        total += 1
        print(f"[✓] {total}. {question} → {entity_name}")

print(f"\n✅ Hoàn tất! Tổng {total} câu hỏi được lưu vào {output_path}")

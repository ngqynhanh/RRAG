import json, random
from openai import OpenAI

# OpenAI client (hoặc thay bằng API của LLM bạn dùng)
client = OpenAI(api_key="YOUR_API_KEY")

ALL_FIELDS = [
    "disease","subdisease","topic","subtopic",
    "complication","treatment","application","cause",
    "definition","detail","population","advice",
    "riskfactor","prevention","symptom"
]

PROMPT_TEMPLATE = """
Bạn là trợ lý y khoa. Hãy sinh ra một câu hỏi ngắn (tối đa 1 câu)
thuộc lớp: {label}.
Câu hỏi phải dựa trên đoạn văn sau đây, và người đọc khi hỏi câu đó
có thể tìm được câu trả lời trong đoạn văn.

Đoạn văn:
\"\"\"{passage}\"\"\"

Chỉ cần xuất ra đúng câu hỏi.
"""

def load_passages(path):
    passages = []
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            passages.append(json.loads(line))
    return passages

def generate_query(label, passage):
    prompt = PROMPT_TEMPLATE.format(label=label, passage=passage["text"][:600])
    resp = client.chat.completions.create(
        model="gpt-4o-mini",   # hoặc model khác bạn có
        messages=[{"role":"user","content":prompt}],
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()

def main(passages_path, out_path, num_labels=3):
    passages = load_passages(passages_path)
    out = []

    for p in passages[:50]:   # lấy 50 passage đầu để demo
        chosen_labels = random.sample(ALL_FIELDS, num_labels)
        for lbl in chosen_labels:
            q = generate_query(lbl, p)
            record = {
                "query": q,
                "label": lbl,
                "context_ids": [p["id"]],
                "has_ctx": 1
            }
            out.append(record)

    with open(out_path,"w",encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r,ensure_ascii=False)+"\n")

    print(f"Generated {len(out)} queries -> {out_path}")

if __name__=="__main__":
    main("datasets/passages.jsonl","datasets/synthetic_queries.jsonl")

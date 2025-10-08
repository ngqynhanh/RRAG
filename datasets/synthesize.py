import json
from openai import OpenAI
from dotenv import load_dotenv
import os

# -----------------------------
# 1. OpenRouter setup
# -----------------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

def rephrase_answer(answer_text):
    """Dùng LLM để viết lại answer mượt mà, tự nhiên nhưng vẫn giữ nội dung."""
    prompt = f"Hãy viết lại câu trả lời sau cho mượt mà, rõ ràng, tự nhiên, vẫn giữ nguyên thông tin: {answer_text}"
    resp = client.chat.completions.create(
        model="qwen/qwen-2.5-72b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.08,
        max_tokens=500
    )
    return resp.choices[0].message.content.strip()

# -----------------------------
# 2. Load JSON
# -----------------------------
input_file = "datasets/qas_vi_existing_questions.jsonl"  # JSONL file cũ
output_file = "datasets/qas_rag_rephrased.jsonl"

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        record = json.loads(line)
        answers = record.get("answers", [])
        new_answers = []
        for ans in answers:
            try:
                ans_smooth = rephrase_answer(ans)
            except Exception as e:
                print(f"[Warning] Rephrase lỗi: {e}, giữ nguyên answer")
                ans_smooth = ans
            new_answers.append(ans_smooth)
        record["answers"] = new_answers
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[Info] Rephrased question: {record['question']}")

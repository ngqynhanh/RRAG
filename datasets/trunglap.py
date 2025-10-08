import json
from collections import defaultdict

# Đọc file JSONL
file_path = "datasets/qas_synthetic_vi.jsonl"
data = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

# Tìm duplicate dựa trên 'question'
question_map = defaultdict(list)
for idx, item in enumerate(data):
    question_map[item["question"]].append(idx)

# Liệt kê các câu hỏi trùng lặp
duplicates = {q: idxs for q, idxs in question_map.items() if len(idxs) > 1}

print(f"Tìm thấy {len(duplicates)} câu hỏi trùng lặp.\n")

for q, idxs in duplicates.items():
    print(f"Câu hỏi: {q}")
    print(f"Dòng trùng lặp: {idxs}\n")

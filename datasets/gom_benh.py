from collections import defaultdict
def assign_disease(passages, disease_list):
    groups = defaultdict(list)
    for p in passages:
        text = p["text"].lower()
        matched = None
        for d in disease_list:
            if d.lower() in text:
                matched = d
                break
        key = matched if matched else "other"
        groups[key].append(p)
    return groups

def main():
    import json
    with open("diseases.json") as f:
        disease_list = json.load(f)
    with open("passages.json") as f:
        passages = json.load(f)
    grouped = assign_disease(passages, disease_list)
    with open("grouped_passages.json", "w") as f:
        json.dump(grouped, f, ensure_ascii=False, indent=2)
import re
import pandas as pd
from collections import defaultdict

# -----------------------------
# 1. Load article_id data
# -----------------------------
articles_df = pd.read_csv("datasets/chunk_data.csv")   # CSV: article_id, chunk_id, root_name, root_type, chunk_text

# -----------------------------
# 2. Load Neo4j query data
# -----------------------------
neo4j_file = "datasets/chunk_labeled_data.txt"
with open(neo4j_file, "r", encoding="utf-8") as f:
    neo4j_lines = f.readlines()

# -----------------------------
# 3. Parse Neo4j relationships
# -----------------------------
pattern = re.compile(
    r"\(:([A-Za-z]+) \{name: ([^,]+), layer: [^\}]+\}\)-\[:HAS_([A-Z]+)\]->\(:([A-Za-z]+) \{name: ([^,]+), layer: [^\}]+\}\)"
)

# mapping root_name -> {layer -> list(target_name)}
mapping = defaultdict(lambda: defaultdict(list))
entity_types = dict()  # entity_name -> type (topic, subtopic, disease, subdisease)
all_entity_names = set()

for line in neo4j_lines:
    match = pattern.search(line)
    if match:
        root_type, root_name, relation, target_type, target_name = match.groups()
        mapping[root_name][relation.lower()].append(target_name)
        entity_types[root_name.strip()] = root_type.lower()  # lưu type của root
        all_entity_names.add(root_name.strip())

# -----------------------------
# 4. Map article_id cho entity_name theo substring match
# -----------------------------
final_map = []

for entity_name in all_entity_names:
    layers_dict = defaultdict(list)
    article_ids = set()
    entity_type = None

    for root_name, layer_targets in mapping.items():
        if entity_name in root_name:  # substring match
            # gộp các target của các layer
            for layer, targets in layer_targets.items():
                layers_dict[layer].extend(targets)
            # gán article_id nếu root_name chứa entity_name
            ids = articles_df.loc[articles_df['root_name'].str.contains(re.escape(root_name)), 'article_id'].tolist()
            article_ids.update(ids)
            # lấy entity_type từ root_name (giữ lần đầu match)
            if not entity_type:
                entity_type = entity_types[root_name]

    # Gộp target names mỗi layer thành string
    layer_data = {layer: "; ".join(targets) for layer, targets in layers_dict.items()}

    final_map.append({
        "entity_name": entity_name,
        "entity_type": entity_type,
        "layers_targets": layer_data,
        "article_ids": list(article_ids)
    })

# -----------------------------
# 5. Convert to DataFrame và mở rộng layer columns
# -----------------------------
final_df = pd.DataFrame(final_map)

# Mở rộng layer columns
all_layers = set()
for ld in final_df['layers_targets']:
    all_layers.update(ld.keys())

for layer in all_layers:
    final_df[layer] = final_df['layers_targets'].apply(lambda x: x.get(layer, ""))

final_df = final_df.drop(columns=['layers_targets'])

# -----------------------------
# 6. Lưu CSV
# -----------------------------
final_df.to_csv("entity_full_mapping.csv", index=False)
print("Mapping done! Output saved to entity_full_mapping.csv")

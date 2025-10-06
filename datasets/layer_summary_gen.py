import re
import pandas as pd
from collections import defaultdict

# -----------------------------
# 1. Load article_id data
# -----------------------------
articles_df = pd.read_csv("datasets/chunk_data.csv")  # CSV file with columns: article_id, chunk_id, root_name, root_type, chunk_text

# Map: root_name -> list of article_ids
name_to_article = defaultdict(list)
for _, row in articles_df.iterrows():
    name_to_article[row['root_name']].append(row['article_id'])

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

# Map root_name -> layer -> list of target names
mapping = defaultdict(lambda: defaultdict(list))

for line in neo4j_lines:
    match = pattern.search(line)
    if match:
        root_type, root_name, relation, target_type, target_name = match.groups()
        mapping[root_name][relation.lower()].append(target_name)

# -----------------------------
# 4. Build final DataFrame
# -----------------------------
final_map = []

for root_name, layers_dict in mapping.items():
    articles = name_to_article.get(root_name, [])
    layer_data = {}
    for layer, names in layers_dict.items():
        layer_data[layer] = "; ".join(names)  # gộp target names thành string
    final_map.append({
        "entity_name": root_name,
        "layers_targets": layer_data,
        "article_ids": articles
    })

# Convert to DataFrame
final_df = pd.DataFrame(final_map)

# Optional: expand layer columns
all_layers = set()
for ld in final_df['layers_targets']:
    all_layers.update(ld.keys())

for layer in all_layers:
    final_df[layer] = final_df['layers_targets'].apply(lambda x: x.get(layer, ""))

# Remove the nested dict column
final_df = final_df.drop(columns=['layers_targets'])

# Save to CSV
final_df.to_csv("entity_layer_article_mapping_expanded.csv", index=False)
print("Mapping done! Output saved to entity_layer_article_mapping_expanded.csv")

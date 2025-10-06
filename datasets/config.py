import pandas as pd

# Đọc file CSV
df = pd.read_csv("entity_full_mapping.csv")  # thay bằng file của bạn

# Xóa những dòng mà article_ids là rỗng list '[]' hoặc NaN
df_filtered = df[~df['article_ids'].isin(['[]', ''])]

# Lưu lại file mới
df_filtered.to_csv("entity_full_mapping_filtered.csv", index=False)
print("Done! File đã loại bỏ các dòng article_ids rỗng.")

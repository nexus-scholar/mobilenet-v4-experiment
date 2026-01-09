print("Starting check...")
import pandas as pd
df = pd.read_csv("data/unified_dataset.csv")
# Count how many rows have label -1
missing_labels = df[df['label'] == -1]
print(f"Rows with missing labels: {len(missing_labels)}")
# print(missing_labels['class_name'].unique())


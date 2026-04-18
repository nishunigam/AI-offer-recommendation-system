import pandas as pd
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(BASE_DIR, '../features/features.csv')
json_path = os.path.join(BASE_DIR, '../features/features.json')

df = pd.read_csv(csv_path)

data = {}

for _, row in df.iterrows():
    data[int(row['user_id'])] = row.to_dict()

with open(json_path, 'w') as f:
    json.dump(data, f)

print("features.json created at:", json_path)
import pandas as pd
import json

# Load the dataset
with open('../arxiv-metadata-oai-snapshot.json', 'r') as file:
    data = [json.loads(line) for line in file]

# Convert to DataFrame
df = pd.DataFrame(data)

# Select a subset for demonstration purposes
df = df[['title', 'abstract']].dropna().head(1000)

# Combine title and abstract
df['text'] = df['title'] + ". " + df['abstract']

df = df[['title', 'abstract']]
df.to_csv("arvix_head.csv",index=False)


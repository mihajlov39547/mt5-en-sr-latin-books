import os
import pandas as pd
import unicodedata
import json

# Paths
serbian_folder = "translated"
output_csv = "serbian_corpus.csv"
output_jsonl = "serbian_corpus.jsonl"

# Function to clean text
def clean_text(text):
    text = text.strip()  # Remove leading/trailing whitespace
    text = unicodedata.normalize("NFKC", text)  # Normalize Unicode characters
    text = text.replace("„", '"').replace("“", '"')  # Standardize quotes
    text = text.replace("’", "'")  # Normalize apostrophes
    text = " ".join(text.split())  # Remove multiple spaces
    return text

# Collect Serbian text
data = []
for filename in os.listdir(serbian_folder):
    if filename.endswith(".txt"):
        sr_file = os.path.join(serbian_folder, filename)

        with open(sr_file, "r", encoding="utf-8") as sr_f:
            sr_sentences = sr_f.readlines()

            for sr in sr_sentences:
                sr = clean_text(sr)
                if sr and len(sr) > 20:  # Avoid very short texts
                    data.append({"text": sr})

# Save dataset to CSV
df = pd.DataFrame(data)
df.to_csv(output_csv, index=False, encoding="utf-8")
print(f"✅ Serbian dataset saved to {output_csv}")

# Save dataset to JSONL (better format for training)
with open(output_jsonl, "w", encoding="utf-8") as jsonl_f:
    for entry in data:
        jsonl_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
print(f"✅ Serbian dataset saved to {output_jsonl}")

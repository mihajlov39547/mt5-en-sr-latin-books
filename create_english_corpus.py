import os
import pandas as pd
import unicodedata
import json

# Paths
english_folder = "prepared"
output_dir = "data"
output_csv = os.path.join(output_dir, "english_corpus.csv")
output_jsonl = os.path.join(output_dir, "english_corpus.jsonl")

os.makedirs(output_dir, exist_ok=True)

# Function to clean text
def clean_text(text):
    text = text.strip()  # Remove leading/trailing whitespace
    text = unicodedata.normalize("NFKC", text)  # Normalize Unicode characters
    text = text.replace(""", '"').replace(""", '"')  # Standardize quotes
    text = text.replace("'", "'")  # Normalize apostrophes
    text = " ".join(text.split())  # Remove multiple spaces
    return text

# Collect English text
data = []
for filename in os.listdir(english_folder):
    if filename.endswith(".txt"):
        en_file = os.path.join(english_folder, filename)

        with open(en_file, "r", encoding="utf-8") as en_f:
            en_sentences = en_f.readlines()

            for en in en_sentences:
                en = clean_text(en)
                if en and len(en) > 20:  # Avoid very short texts
                    data.append({"text": en})

# Save dataset to CSV
df = pd.DataFrame(data)
df.to_csv(output_csv, index=False, encoding="utf-8")
print(f"✅ English dataset saved to {output_csv}")

# Save dataset to JSONL (better format for training)
with open(output_jsonl, "w", encoding="utf-8") as jsonl_f:
    for entry in data:
        jsonl_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
print(f"✅ English dataset saved to {output_jsonl}")

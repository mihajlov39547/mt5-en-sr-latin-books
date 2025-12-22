import os
import json

# Paths
prepared_folder = "prepared"
translated_folder = "translated"
output_jsonl_eng_to_sr = "eng_to_sr.jsonl"
output_jsonl_sr_to_eng = "sr_to_eng.jsonl"

# Collect parallel sentences
data_eng_to_sr = []
data_sr_to_eng = []

for filename in os.listdir(prepared_folder):
    if filename.endswith(".txt"):
        en_file = os.path.join(prepared_folder, filename)
        sr_file = os.path.join(translated_folder, filename)

        if os.path.exists(sr_file):  # Ensure both English & Serbian files exist
            with open(en_file, "r", encoding="utf-8") as en_f, open(sr_file, "r", encoding="utf-8") as sr_f:
                en_sentences = en_f.readlines()
                sr_sentences = sr_f.readlines()

                for en, sr in zip(en_sentences, sr_sentences):
                    en = en.strip()
                    sr = sr.strip()
                    if en and sr:
                        data_eng_to_sr.append({"source": en, "target": sr})
                        data_sr_to_eng.append({"source": sr, "target": en})

# Save English to Serbian dataset as JSONL
with open(output_jsonl_eng_to_sr, "w", encoding="utf-8") as jsonl_f:
    for entry in data_eng_to_sr:
        jsonl_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
print(f"✅ English to Serbian dataset saved to {output_jsonl_eng_to_sr}")

# Save Serbian to English dataset as JSONL
with open(output_jsonl_sr_to_eng, "w", encoding="utf-8") as jsonl_f:
    for entry in data_sr_to_eng:
        jsonl_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
print(f"✅ Serbian to English dataset saved to {output_jsonl_sr_to_eng}")

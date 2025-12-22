import os
import pandas as pd

# Paths
prepared_folder = "prepared"
translated_folder = "translated"
output_dir = "data"
output_csv_eng_to_sr = os.path.join(output_dir, "eng_to_sr.csv")
output_csv_sr_to_eng = os.path.join(output_dir, "sr_to_eng.csv")

os.makedirs(output_dir, exist_ok=True)

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

# Save English to Serbian dataset
df_eng_to_sr = pd.DataFrame(data_eng_to_sr)
df_eng_to_sr.to_csv(output_csv_eng_to_sr, index=False, encoding="utf-8")
print(f"✅ English to Serbian dataset saved to {output_csv_eng_to_sr}")

# Save Serbian to English dataset
df_sr_to_eng = pd.DataFrame(data_sr_to_eng)
df_sr_to_eng.to_csv(output_csv_sr_to_eng, index=False, encoding="utf-8")
print(f"✅ Serbian to English dataset saved to {output_csv_sr_to_eng}")

# mt5-en-sr-latin-books

## What is this repo?
This repository contains a **baseline** fine-tuning setup for `google/mt5-small` (mT5) on an **English → Serbian (sr-Latn)** parallel dataset derived from books.

It also contains helper scripts to:
- clean/source books into sentence-per-line text (`source/` → `prepared/`)
- translate prepared English into Serbian Latin (`translated/`)
- build parallel datasets (`eng_to_sr.csv`, `sr_to_eng.csv`, and JSONL variants)
- validate checkpoints and produce a machine-readable evaluation report

The longer-term goal is to implement additional training strategies (A–D) while keeping comparisons fair and reproducible.

---

## Dataset

### Parallel data (supervised translation)
The supervised translation datasets are:
- `eng_to_sr.csv` (English source, Serbian target)
- `sr_to_eng.csv` (Serbian source, English target)

Format (both files):
- CSV with columns:
  - `source` (input text)
  - `target` (expected output text)

How the parallel pairs are created:
- `prepared/*.txt` contains English sentences, one per line
- `translated/*.txt` contains Serbian Latin sentences, one per line
- Files are aligned by **line index** (the script uses `zip(en_lines, sr_lines)`)

Scripts:
- `create_dataset.py` → writes `eng_to_sr.csv` + `sr_to_eng.csv`
- `create_dataset_jsonl.py` → writes `eng_to_sr.jsonl` + `sr_to_eng.jsonl`

Important implications of the alignment method:
- Parallel quality depends on **sentence segmentation consistency**.
- If a translation step merges/splits lines, alignment will drift and harm training.

### Monolingual corpora (for continued pretraining / denoising)
For CPT-style experiments, you can build monolingual corpora:
- `serbian_corpus.csv` / `serbian_corpus.jsonl` from `translated/*.txt` via `create_serbian_corpus.py`
- `english_corpus.csv` / `english_corpus.jsonl` from `prepared/*.txt` via `create_english_corpus.py`

Format:
- CSV/JSONL with one field: `text`

### Book metadata
`book_titles.json` is a metadata file describing the book files in `prepared/` (and, if present, their Serbian counterparts in `translated/`). Each entry includes:
- `filename`, `stem`
- `title`, `author`
- detection `confidence` and `method`
- `opening_line_en` and `opening_line_sr`

---

## Baseline (current)

**Baseline definition**
- Train direction: **EN → SR**
- Data: `eng_to_sr.csv` only
- Model: `google/mt5-small`
- Trainer: `Seq2SeqTrainer`
- Early stopping: enabled, based on `eval_loss`
- Split: deterministic **70/20/10** (train/valid/test) via `seed=42`

### Why mT5
Serbian Latin requires diacritics (`č ć š ž đ`). The scripts include explicit sanity checks to ensure:
- the dataset contains diacritics
- the tokenizer can encode/decode diacritics
- tokenization does not “lose” diacritics

### Tokenizer details
This project uses the tokenizer that ships with the chosen base model (by default `google/mt5-small`). Concretely:

- **Tokenizer type:** SentencePiece (you’ll see `spiece.model` in model folders).
- **Loading behavior:** `AutoTokenizer.from_pretrained(..., use_fast=False)`.
  - mT5/T5 tokenizers are SentencePiece-based; using `use_fast=False` avoids noisy slow→fast conversion warnings and keeps behavior consistent across environments.
- **Task prefixes are part of the input:** training prepends a direction prefix to every `source` string, e.g. `translate English to Serbian: ...`.

Sanity checks implemented in `colab_train_t5.py`:
- **Diacritics round-trip:** verifies the tokenizer can encode+decode `čćšžđČĆŠŽĐ` without losing characters.
- **Dataset diacritics presence:** checks the loaded CSV `target` column actually contains diacritics (helps catch wrong file/cached data).
- **Tokenized label check:** decodes tokenized labels and confirms diacritics still appear after tokenization.

Padding/labels:
- Labels use `-100` for padding via `DataCollatorForSeq2Seq(label_pad_token_id=-100)` so padding tokens don’t contribute to the loss.

Checkpoint robustness:
- The training script includes a callback that **saves the tokenizer into each `checkpoint-*` directory**. This matters because evaluation/inference scripts (like `colab_test_t5.py`) prefer checkpoints that contain `spiece.model`.

---

## How training works (Colab-first)

The core training script is `colab_train_t5.py`.

### Expected Drive layout
The training script expects a Drive project directory:

```
/content/drive/MyDrive/T5/
  data/
    eng_to_sr.csv
    sr_to_eng.csv
  models/
  cache/
```

You can change this via the `CONFIG` dict at the top of `colab_train_t5.py`.

### Install / environment
In a fresh Colab runtime the script auto-installs:
- `transformers==4.49.0`
- `datasets==3.3.2`
- `evaluate==0.4.6`
- `sacrebleu==2.5.1`
- `sentencepiece==0.2.0`
- `accelerate==1.4.0`
- `protobuf>=4.25.0`

### Run training
In Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then:

```bash
python /content/drive/MyDrive/T5/colab_train_t5.py
```

Key configuration knobs (edit in `CONFIG`):
- `translation_direction`: `eng_to_sr` or `sr_to_eng`
- `learning_rate`, `num_train_epochs`
- batch sizes + `gradient_accumulation_steps`
- `max_input_length`, `max_target_length`
- checkpoint/eval cadence: `save_steps`, `eval_steps`
- early stopping: `early_stopping_patience`

### Output naming
Outputs go under Drive `models/` with a deterministic name that includes:
- model family (`mt5` or `t5`)
- direction (`eng_to_sr` or `sr_to_eng`)
- model slug (e.g. `google-mt5-small`)
- split signature (e.g. `s70v20t10_seed42`)

Example:

```
models/
  mt5_translation_model__eng_to_sr__google-mt5-small__s70v20t10_seed42/
    checkpoint-1000/
    checkpoint-2000/
    results.json
    dataset_fingerprint.txt
    config.json
    spiece.model
    model.safetensors (or pytorch_model*.bin)
```

### Tokenization cache
Tokenized datasets are cached to Drive under:

```
cache/tokenized_datasets/
```

The cache path includes:
- dataset fingerprint (size + mtime + SHA256 prefix)
- split signature
- max input/target lengths

This prevents “accidental training on an old tokenized dataset” when your CSV changes.

### What gets logged/saved
At the end of training:
- final model + tokenizer are saved to the run folder
- the script writes `results.json` with:
  - best checkpoint path
  - best `eval_loss`
  - decoding params
  - test metrics (if test split exists)

---

## Validation / benchmarking

### Full evaluation on the held-out test split
Use `colab_validate_t5.py` to evaluate a chosen checkpoint or the final model directory.

It:
- recreates the exact same 70/20/10 split (same seed)
- runs generation on test
- computes:
  - sacreBLEU (BLEU)
  - chrF++
  - diacritics precision/recall/F1 + “any diacritics rate”
- writes `validate_results.json`

Example:

```bash
python /content/drive/MyDrive/T5/colab_validate_t5.py \
  --base_dir "/content/drive/MyDrive/T5/models/mt5_translation_model__eng_to_sr__google-mt5-small__s70v20t10_seed42" \
  --direction eng_to_sr \
  --csv_path "/content/drive/MyDrive/T5/data/eng_to_sr.csv" \
  --which best \
  --max_input_length 256 \
  --max_target_length 256 \
  --num_beams 4
```

### Quick smoke tests (sanity prompts)
`colab_test_t5.py` loads the newest usable checkpoint (or final model) and runs short prompts including “station variants”:
- “Where is the train station?”
- “Where is the station?”
- “Where is the bus station?”

This is useful for spotting:
- “ASCII Serbian” failures (missing diacritics)
- weird memorization or mappings
- domain gaps (see notes below)

---

## Data pipeline (how the repo was built)

The intended flow is:

1) Put raw English books in `source/` (one `.txt` per book)
2) Convert to sentence-per-line cleaned English in `prepared/`
   - `process_books.py` does sentence splitting and aggressive cleanup
3) Translate `prepared/` → `translated/` (Serbian Latin)
   - `translate_prepared.py` uses Azure Translator batch document translation
4) Build parallel datasets:
   - `create_dataset.py` (CSV)
   - `create_dataset_jsonl.py` (JSONL)

---

## Additional Strategies (TODO roadmap)

The baseline is intentionally simple. Below are the planned strategies and what “done” should look like.

### Strategy A — Bidirectional supervised fine-tune (single model, two directions)
Train one model on both directions, mixed together, using task prefixes:

- `translate English to Serbian: {en}` → `{sr}`
- `translate Serbian to English: {sr}` → `{en}`

Hypothesis:
- Regularizes the model and improves semantic consistency.
- Encourages a shared bilingual latent space.

Implementation notes:
- Build a combined dataset from `eng_to_sr.csv` and `sr_to_eng.csv`.
- Add a `direction` field and generate the prefix accordingly.
- Experiment with mixing ratios (50/50, proportional to dataset sizes, etc.).
- Keep hyperparameters identical to baseline for fair comparisons.

What to report:
- EN→SR BLEU / chrF++
- SR→EN BLEU / chrF++
- Diacritics metrics for SR outputs

### Strategy B — Serbian-only continued pretraining (CPT) → then translation fine-tune
Two-stage training:

1) CPT on Serbian monolingual text with T5 denoising (span corruption)
2) Fine-tune on EN→SR translation (baseline setup) starting from CPT checkpoint

Hypothesis:
- Improves Serbian fluency/style and morphology for literary domain.

Implementation notes (high-level):
- Use `serbian_corpus.jsonl` or directly `translated/*.txt` as monolingual text.
- Create a denoising dataset (T5 span corruption):
  - input: corrupted Serbian sentence
  - target: removed spans (T5 format)
- Train for a fixed compute budget (steps), then switch to supervised translation.

What to report:
- Translation metrics (BLEU/chrF++ + diacritics)
- A fluency proxy on held-out Serbian (optional) and/or small human evaluation

### Strategy C — Mixed-language CPT (Serbian + English) → bidirectional fine-tune
Two-stage training:

1) CPT on mixed monolingual corpora (e.g., 50% SR, 50% EN)
2) Bidirectional supervised fine-tune (Strategy A) from that checkpoint

Hypothesis:
- Avoids over-adapting to Serbian-only distribution.
- Often improves bidirectional performance.

Implementation notes:
- Same denoising objective as Strategy B.
- Sample from EN and SR corpora with a fixed ratio.

What to report:
- Both directions metrics
- Compare against Strategy A (no CPT) and Strategy B (SR-only CPT)

### Strategy D — Back-translation augmentation (monolingual → synthetic parallel)
Use monolingual Serbian to create synthetic parallel data:

1) Train a preliminary SR→EN model (baseline but reversed, or a shorter run)
2) Translate large Serbian monolingual corpus → synthetic English
3) Train EN→SR on (synthetic EN → real SR), optionally mixed with real pairs

Hypothesis:
- Improves coverage when real parallel data is domain-limited.

Implementation notes:
- Generate X synthetic pairs (e.g. 100k–500k), then mix with real parallel.
- Keep a clean held-out test set that is never synthesized from.
- Add an ablation: real-only vs real+synthetic.

What to report:
- In-domain metrics (held-out literary sentences)
- Out-of-domain probes (see below)

---

## Evaluation design (important for fair comparisons)

Dataset is heavily literary. Everyday concepts (e.g., “train station / bus station”) may be rare or absent. So strategies often won’t fix out-of-domain prompts unless you add data that contains those concepts.

Recommended: maintain two test sets
1) **In-domain**: held-out sentences from the book corpus (current test split)
2) **Out-of-domain**: small curated set of everyday/travel prompts (your “station variants” and similar)

Report both clearly as: in-domain vs out-of-domain.

---

## Repo structure

Top-level important files:
- `colab_train_t5.py` — baseline fine-tuning script (mT5, caching, early stopping)
- `colab_validate_t5.py` — evaluation runner producing `validate_results.json`
- `colab_test_t5.py` — quick inference smoke tests
- `create_dataset.py` / `create_dataset_jsonl.py` — builds parallel datasets
- `create_serbian_corpus.py` / `create_english_corpus.py` — monolingual corpora for CPT

Data folders:
- `source/` — raw English books
- `prepared/` — cleaned English, sentence-per-line
- `translated/` — Serbian Latin translations, sentence-per-line

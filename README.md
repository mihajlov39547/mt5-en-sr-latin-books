# mt5-en-sr-latin-books

Baseline fine-tuning + evaluation setup for English ↔ Serbian (sr-Latn) translation using `google/mt5-small`, with datasets derived from book text.

This repository includes:
- Book text preprocessing (raw → sentence-per-line)
- Parallel dataset builders (CSV/JSONL)
- Monolingual corpus builders (CSV/JSONL)
- Colab-first training + evaluation scripts with caching and diacritics sanity checks

---

## 📁 Repository structure

```
.
├─ colab_train_t5.py
├─ colab_train_t5_strategy_a.py
├─ colab_pretrain_t5_strategy_b.py
├─ colab_train_t5_strategy_b.py
├─ colab_pretrain_t5_strategy_c.py
├─ colab_train_t5_strategy_c.py
├─ colab_train_t5_strategy_d.py
├─ colab_validate_t5.py
├─ colab_test_t5.py
├─ process_books.py
├─ create_dataset.py
├─ create_dataset_jsonl.py
├─ create_english_corpus.py
├─ create_serbian_corpus.py
├─ book_titles.json
├─ data/
├─ source/
├─ prepared/
└─ translated/
```

---

## 📦 Datasets

### Parallel translation (supervised)

Files (stored in `data/`):
- `data/eng_to_sr.csv` (English source → Serbian target)
- `data/sr_to_eng.csv` (Serbian source → English target)

Format:
- CSV with columns: `source`, `target`

How pairs are created:
- `prepared/*.txt` contains English sentences, one per line
- `translated/*.txt` contains Serbian Latin sentences, one per line
- Files are aligned by **line index** (`zip(en_lines, sr_lines)`).

Important implications of line-alignment:
- Dataset quality depends on consistent sentence segmentation.
- If translation merges/splits lines, alignment drifts and harms training.

Scripts:
- `create_dataset.py` → writes `data/eng_to_sr.csv` + `data/sr_to_eng.csv`
- `create_dataset_jsonl.py` → writes `data/eng_to_sr.jsonl` + `data/sr_to_eng.jsonl`

### Monolingual corpora (for CPT / denoising)

Files (stored in `data/`):
- `data/serbian_corpus.csv` / `data/serbian_corpus.jsonl` from `translated/*.txt` via `create_serbian_corpus.py`
- `data/english_corpus.csv` / `data/english_corpus.jsonl` from `prepared/*.txt` via `create_english_corpus.py`

Format:
- CSV/JSONL with one field: `text`

### Book metadata

`book_titles.json` contains metadata for the book files (title/author, matching stems, and opening lines).

---

## 🧪 Baseline (current)

Baseline definition:
- Train direction: **EN → SR**
- Data: `data/eng_to_sr.csv`
- Model: `google/mt5-small`
- Trainer: `Seq2SeqTrainer`
- Early stopping: enabled on `eval_loss`
- Split: deterministic **70/20/10** (train/valid/test) with `seed=42`

### Why mT5

Serbian Latin requires diacritics (`č ć š ž đ`). The Colab scripts include sanity checks to ensure:
- the dataset contains diacritics
- the tokenizer can encode/decode diacritics
- tokenization and generation do not “lose” diacritics

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

## 🧰 Data pipeline

Intended flow:

1) Put raw English books in `source/` (one `.txt` per book)
2) Convert to cleaned, sentence-per-line English in `prepared/` via:

```bash
python process_books.py
```

3) Translate `prepared/` → `translated/` (Serbian Latin), keeping one sentence per line
   - Note: translation tooling is intentionally external to this repo; you can use any provider as long as you preserve line alignment.
4) Build datasets into `data/`:

```bash
python create_dataset.py
python create_dataset_jsonl.py
python create_english_corpus.py
python create_serbian_corpus.py
```

---

## 🚀 Training (Colab-first)

The core training script is `colab_train_t5.py`.

### Expected Drive layout

```
/content/drive/MyDrive/T5/
  data/
    eng_to_sr.csv
    sr_to_eng.csv
  models/
  cache/
```

You can change these via the `CONFIG` dict in `colab_train_t5.py`.

### Install / environment

In a fresh Colab runtime:

```bash
!pip install -q transformers datasets evaluate sacrebleu sentencepiece accelerate protobuf
```

### Run training

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
python /content/drive/MyDrive/T5/colab_train_t5.py
```

Key configuration knobs (edit in `CONFIG`):
- `translation_direction`: `eng_to_sr` or `sr_to_eng`
- `learning_rate`, `num_train_epochs`
- batch sizes and `gradient_accumulation_steps`
- `max_input_length`, `max_target_length`
- checkpoint cadence: `save_steps`
- eval cadence: `eval_steps`
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

## ✅ Validation / benchmarking

### Full evaluation on the held-out test split

`colab_validate_t5.py`:
- recreates the same 70/20/10 split (seed=42)
- runs generation on the test split
- computes sacreBLEU, chrF++, and diacritics precision/recall/F1
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

`colab_test_t5.py` runs inference checks (including “station variants”) to quickly spot:
- “ASCII Serbian” failures (missing diacritics)
- brittle mappings or memorization
- out-of-domain gaps

---

## 🧪 Strategies (experimental)

All strategies are designed to be **as baseline-identical as possible** (same Trainer family, caching, checkpointing, resume guard, and metrics), so that differences are attributable to the strategy itself.

For each strategy below, the main thing we’re testing is the stated **hypothesis**.

### At-a-glance comparison

| Setup | Training objective | Data used | Stages | Default seed | Scripts | Metrics focus |
|---|---|---|---:|---:|---|---|
| **Baseline** | Supervised translation (single direction) | `eng_to_sr.csv` | 1 | 42 | `colab_train_t5.py` | BLEU, chrF++, diacritics (SR) |
| **A** | Supervised translation (bidirectional multitask) | `eng_to_sr.csv` + `sr_to_eng.csv` | 1 | 52 | `colab_train_t5_strategy_a.py` | BLEU/chrF++ both directions; diacritics on SR outputs |
| **B** | CPT denoising (SR) → supervised EN→SR | `serbian_corpus.csv` → `eng_to_sr.csv` | 2 | 62 | `colab_pretrain_t5_strategy_b.py` + `colab_train_t5_strategy_b.py` | EN→SR BLEU/chrF++; diacritics; plus CPT eval_loss |
| **C** | CPT denoising (SR+EN) → supervised bidirectional | `serbian_corpus.csv` + `english_corpus.csv` → (`eng_to_sr.csv` + `sr_to_eng.csv`) | 2 | 72 | `colab_pretrain_t5_strategy_c.py` + `colab_train_t5_strategy_c.py` | Both directions BLEU/chrF++; diacritics; plus CPT eval_loss |
| **D** | Single-stage mix: SR denoising + supervised EN→SR | `serbian_corpus.csv` + `eng_to_sr.csv` | 1 | 82 | `colab_train_t5_strategy_d.py` | EN→SR BLEU/chrF++; diacritics (translation-only eval) |

Notes:
- “Default seed” here refers to each script’s `CONFIG["split_seed"]` (and `sample_seed` where applicable).
- Regardless of strategy, use `colab_validate_t5.py` / `colab_test_t5.py` for consistent evaluation and quick probes.

---

## 🔁 Strategy A (implemented): Bidirectional supervised fine-tune

Strategy A trains **one** seq2seq model on **both** directions at once, using task prefixes:

- `translate English to Serbian: {en}` → `{sr}`
- `translate Serbian to English: {sr}` → `{en}`

### Hypothesis to test

- **Regularization:** training both directions together regularizes the model vs single-direction fine-tuning.
- **Semantic consistency:** encourages a shared bilingual latent space, improving meaning preservation.
- **Quality:** can improve overall translation quality (BLEU/chrF++), while keeping hyperparameters identical for fair comparison.
- **Serbian diacritics:** should not degrade diacritics behavior for SR outputs (and may improve it).

### Script

Use: `colab_train_t5_strategy_a.py`

It is designed as a drop-in sibling of `colab_train_t5.py`: it keeps the same training/eval/caching/resume machinery so comparisons remain fair.

### What is identical to baseline

- Model: `google/mt5-small`
- Hyperparameters / Trainer args (epochs, LR, batch sizes, lengths, checkpoint cadence, early stopping)
- Deterministic split: **70/20/10** with its strategy seed (default `split_seed=52`)
- Tokenization cache on Drive
- Diacritics sanity checks
- Checkpoint robustness: tokenizer is saved into every `checkpoint-*` directory
- Resume behavior: auto-resume from latest checkpoint **only if** dataset fingerprint matches
- Output summary: writes `results.json` in the same schema as baseline

### Core implementation details

1) **In-memory dataset mixing (no new CSV on disk)**

The script loads both CSVs from Drive:

- `data/eng_to_sr.csv`
- `data/sr_to_eng.csv`

Then it optionally subsamples each direction (fractions are configurable), concatenates, and shuffles once.

Config knobs (in `CONFIG["mix"]`):

- `eng_to_sr_fraction` (default `1.0`)
- `sr_to_eng_fraction` (default `1.0`)
- `sample_mode`: `"random"` (shuffle+take K) or `"deterministic_head"` (take first K)
- `sample_seed`: seed used for subsampling/shuffling when `sample_mode="random"`

2) **Direction column + per-example prefixes**

After loading, each direction gets a `direction` column (`"eng_to_sr"` or `"sr_to_eng"`).
During preprocessing, the input is built per-example using:

```
prefix_map = {
  "eng_to_sr": "translate English to Serbian: ",
  "sr_to_eng": "translate Serbian to English: ",
}
```

3) **Split is done after mixing**

The combined dataset is split into train/validation/test (70/20/10) *after* mixing so the splits contain a similar distribution of both directions.

4) **Caching + fingerprinting are extended to cover both CSVs and mix config**

- Tokenization cache path includes fingerprints for *both* CSVs + the mix config.
- `dataset_fingerprint.txt` is also based on both CSV fingerprints + mix config + split sig.

This prevents accidentally resuming/tokenizing on mismatched data.

### What to report (recommended)

The training script computes overall metrics on the held-out test split. For papers/tables, Strategy A is typically reported as:

- **EN→SR** BLEU / chrF++
- **SR→EN** BLEU / chrF++
- Diacritics metrics specifically for Serbian outputs

Note: the current script keeps the metric computation identical to baseline for comparability. If you want per-direction test metrics, we can add them under a `strategy_a` key in `results.json` without changing the baseline schema.

## 🔧 Strategy B (implemented): Serbian-only CPT → EN→SR fine-tune
Two-stage training:

1) CPT on Serbian monolingual text with T5 denoising (span corruption)
2) Fine-tune on EN→SR translation (baseline setup) starting from CPT checkpoint

### Hypothesis to test

- **Domain fluency:** CPT on `serbian_corpus.csv` improves Serbian fluency/style in the literary domain.
- **Morphology + agreement:** better handling of Serbian inflection (case/number/gender) and local syntactic agreement.
- **Diacritics robustness:** fewer “ASCII Serbian” regressions, especially in longer generations.
- **Downstream translation gains:** EN→SR improves more than SR→EN (CPT is SR-only), often reflected more strongly in chrF++ than BLEU.
- **Efficiency:** supervised fine-tuning converges faster (lower `eval_loss` earlier) when starting from the CPT checkpoint.

### Hypothesis to test

- **Domain fluency:** CPT on `serbian_corpus.csv` reduces unnatural phrasing and improves literary Serbian style.
- **Morphology + agreement:** better handling of Serbian inflection (case/number/gender) and local syntactic agreement.
- **Diacritics robustness:** SR outputs keep diacritics more consistently (fewer “ASCII Serbian” regressions), especially in longer generations.
- **Downstream translation gains:** after fine-tuning, EN→SR improves more than SR→EN (because the CPT stage is SR-only), with measurable gains in chrF++ and diacritics F1.
- **Compute efficiency:** translation fine-tuning should converge faster (lower eval_loss earlier) when starting from the CPT checkpoint.

Implementation notes (high-level):
- Use `data/serbian_corpus.csv` as monolingual text.
- Create a denoising dataset (T5 span corruption):
  - input: corrupted Serbian sentence
  - target: removed spans (T5 format)
- Train for a fixed compute budget (steps), then switch to supervised translation.

Script (CPT / pretrainer):
- `colab_pretrain_t5_strategy_b.py`

Script (translation fine-tune from CPT):
- `colab_train_t5_strategy_b.py`

### What to report

- EN→SR BLEU / chrF++
- Diacritics precision/recall/F1 (+ any-rate)
- Learning curves / best `eval_loss` checkpoint (optional)

How it connects to translation:
- After CPT finishes, `colab_train_t5_strategy_b.py` loads the saved CPT folder via `CONFIG["cpt_checkpoint_dir"]`.
- Set `cpt_checkpoint_dir` to the CPT output directory path, or leave as `"auto"` to use the default pretrainer output naming.
- The translation fine-tune script is baseline-identical except for starting from the CPT checkpoint (fair comparison).

What to report:
- Translation metrics (BLEU/chrF++ + diacritics)
- A fluency proxy on held-out Serbian (optional) and/or small human evaluation

## 🌍 Strategy C (implemented): Mixed-language CPT (SR+EN) → bidirectional fine-tune
Two-stage training:

1) CPT on mixed monolingual corpora (e.g., 50% SR, 50% EN)
2) Bidirectional supervised fine-tune (Strategy A) from that checkpoint (implemented)

Hypothesis:
- Avoids over-adapting to Serbian-only distribution.
- Often improves bidirectional performance.

Implementation notes:
- Same denoising objective as Strategy B.
- Sample from EN and SR corpora with a fixed ratio.

Script (CPT / pretrainer):
- `colab_pretrain_t5_strategy_c.py`

Script (translation fine-tune from CPT):
- `colab_train_t5_strategy_c.py`

How it connects to translation:
- After Strategy C CPT finishes, `colab_train_t5_strategy_c.py` loads the saved CPT folder via `CONFIG["cpt_checkpoint_dir"]`.
- Set `cpt_checkpoint_dir` to the Strategy C CPT output directory path, or leave as `"auto"` to use the default pretrainer output naming.
- The fine-tune stage is Strategy A-like (bidirectional supervised mixing) and baseline-identical aside from starting from the CPT checkpoint.

Default configuration:
- Uses `data/serbian_corpus.csv` + `data/english_corpus.csv`
- Mix ratio: 50% SR / 50% EN (configurable in `CONFIG["mix"]`)
- Uses full corpora by default; will downsample the larger corpus so the mix is balanced
  (set `CONFIG["mix"]["balance_mode"] = "keep_all"` to keep all rows)

What to report:
- Both directions metrics
- Compare against Strategy A (no CPT, implemented) and Strategy B (SR-only CPT)

## 🧩 Strategy D (implemented): Single-stage mixed training (SR denoising + EN→SR)

Strategy D is a **one-stage** training run that mixes:

1) **Serbian-only denoising / language adaptation** (T5 span corruption) on `data/serbian_corpus.csv`
2) **Supervised EN→SR translation** on `data/eng_to_sr.csv`

This keeps the “single-stage” spirit of Strategy A, but instead of mixing two
translation directions, it mixes **(denoise SR)** + **(translate EN→SR)**.

### Why this makes sense

- mT5 already has strong English representations.
- Your translation target is Serbian, so improving Serbian generation (via
  denoising) can help fluency, morphology, and diacritics.
- Mixing both objectives in one run avoids the explicit two-stage CPT→fine-tune
  pipeline.

### What script to run

- `colab_train_t5_strategy_d.py`

It uses the same baseline-style plumbing as the other scripts:

- deterministic splits
- tokenization caching on Drive
- checkpoint auto-resume (with dataset fingerprint guard)
- BLEU/chrF++ + diacritics metrics (computed on translation-only validation/test)
- `results.json` saved into the output model folder

### Hypothesis to test

- **Serbian generation quality:** mixing denoising improves Serbian fluency and morphology while learning EN→SR translation.
- **Diacritics:** denoising increases diacritics correctness and reduces ASCII fallbacks.
- **One-stage simplicity:** achieves some of Strategy B’s benefits without a separate CPT stage.

### Notes / knobs

In `colab_train_t5_strategy_d.py`, the main controls are:

- `CONFIG["mix"]["sr_denoise_fraction"]` and `CONFIG["mix"]["en_to_sr_fraction"]`
- corruption params: `noise_density`, `mean_noise_span_length`
- Model: `google/mt5-small`
- Trainer: `Seq2SeqTrainer`
- Early stopping: enabled on `eval_loss`
- Split: deterministic **70/20/10** (train/valid/test) with `seed=82` (Strategy D)

Prefix behavior (intentional):
- Translation uses the standard task prefix: `translate English to Serbian: `
- Denoising uses **no prefix** (matches Strategy B CPT): raw corrupted text with `<extra_id_*>` sentinels

## 📊 Evaluation design (important for fair comparisons)

Dataset is heavily literary. Everyday concepts (e.g., “train station / bus station”) may be rare or absent. So strategies often won’t fix out-of-domain prompts unless you add data that contains those concepts.

Recommended: maintain two test sets
1) **In-domain**: held-out sentences from the book corpus (current test split)
2) **Out-of-domain**: small curated set of everyday/travel prompts (your “station variants” and similar)

Report both clearly as: in-domain vs out-of-domain.

---

## 📖 Citation

If you use this dataset/repository, please cite:

`mihajlovic2025mihajlovic_mt5_en_sr_latin_books`

BibTeX:

```bibtex
@software{mihajlovic2025mihajlovic_mt5_en_sr_latin_books,
  author       = {Mihajlovic, Marko},
  title        = {mt5-en-sr-latin-books},
  year         = {2025},
  publisher    = {GitHub},
  url          = {https://github.com/mihajlov39547/mt5-en-sr-latin-books},
  note         = {Singidunum University, Belgrade, Serbia},
}
```

APA:

Mihajlovic, M. (2025). *mt5-en-sr-latin-books* [Computer software]. GitHub. https://github.com/mihajlov39547/mt5-en-sr-latin-books

This repository also includes `CITATION.cff` for GitHub’s “Cite this repository” feature.

## 📄 License

MIT License. See the [LICENSE](LICENSE) file.

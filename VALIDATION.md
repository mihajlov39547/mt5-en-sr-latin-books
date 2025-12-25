# Validation and benchmarking

This repo provides two complementary ways to evaluate models:

- **`colab_validate_t5.py`**: reproducible, command-line evaluation on the held-out test split.
- **`colab_test_t5.py`**: quick, qualitative smoke tests (“station variants”, etc.).

This document explains how to use **`colab_validate_t5.py`** for the **Baseline** and for each **Strategy A/B/C/D**, including the special-case evaluation logic for **Strategy D**.

---

## What `colab_validate_t5.py` does

Given a training run output directory (`--base_dir`), the validator will:

1. **Select which model folder to evaluate** (`--which`):
   - `best` (default): uses `results.json` (preferred) or `trainer_state.json`
   - `latest`: picks the latest `checkpoint-*`
   - `final`: evaluates the run folder itself (final saved model)

2. **Verify required files exist**:
   - `spiece.model` (SentencePiece tokenizer)
   - `config.json`
   - model weights (`model.safetensors` or `pytorch_model*.bin`)

3. **Build an evaluation test set**:
   - Baseline / A / B / C: recreates the 70/20/10 split using the same split procedure as `colab_train_t5.py`.
   - Strategy D: can reproduce **Strategy D’s keyed split** (stable by example content) and evaluates **translation-only** rows.

4. **Run generation on the test split** and compute:
   - sacreBLEU (BLEU)
   - chrF++
   - diacritics precision/recall/F1 + any-rate

5. **Write results** to `validate_results.json` in the run folder (or `--output_json`).

---

## Required flags (overview)

- `--base_dir`: training output directory (the folder that contains checkpoints / results.json)
- `--which`: `best` | `latest` | `final`
- `--strategy`: `baseline` | `a` | `b` | `c` | `d`

Optional but important:

- `--seed`: split seed. If omitted, defaults based on `--strategy`:
  - baseline=42, A=52, B=62, C=72, D=82
- `--direction`: `eng_to_sr` or `sr_to_eng` (Baseline/A/B/C only)
- `--csv_path`: CSV to split/evaluate (Baseline/A/B/C) or EN→SR CSV for Strategy D
- `--project_root`: the folder containing `data/`, `models/`, `cache/` (helps the validator find defaults)

---

## Baseline validation

### When to use

Use this to evaluate a model trained by `colab_train_t5.py`.

### Typical usage

- `--strategy baseline`
- `--direction` matches how you trained (`eng_to_sr` by default)
- `--seed 42` (default for baseline)

Example (Colab):

```bash
python /content/drive/MyDrive/T5/colab_validate_t5.py \
  --strategy baseline \
  --base_dir "/content/drive/MyDrive/T5/models/mt5_translation_model__eng_to_sr__google-mt5-small__s70v20t10_seed42" \
  --direction eng_to_sr \
  --which best \
  --num_beams 4
```

Notes:
- If you trained with different split ratios or seed, pass them via `--validation_split`, `--test_split`, and `--seed`.

---

## Strategy A validation (bidirectional supervised)

Strategy A trains one model on both directions, but the validator evaluates **one direction at a time**.

### Validate EN→SR

```bash
python /content/drive/MyDrive/T5/colab_validate_t5.py \
  --strategy a \
  --base_dir "<your_strategy_a_run_dir>" \
  --direction eng_to_sr \
  --csv_path "/content/drive/MyDrive/T5/data/eng_to_sr.csv" \
  --which best
```

### Validate SR→EN

```bash
python /content/drive/MyDrive/T5/colab_validate_t5.py \
  --strategy a \
  --base_dir "<your_strategy_a_run_dir>" \
  --direction sr_to_eng \
  --csv_path "/content/drive/MyDrive/T5/data/sr_to_eng.csv" \
  --which best
```

Important:
- For strict comparability with Strategy A training splits, keep `--seed 52` (default for `--strategy a`).

---

## Strategy B validation

Strategy B has two stages:

1. CPT pretraining: `colab_pretrain_t5_strategy_b.py`
2. Translation fine-tune: `colab_train_t5_strategy_b.py`

### Validate the translation fine-tune (recommended for MT comparison)

```bash
python /content/drive/MyDrive/T5/colab_validate_t5.py \
  --strategy b \
  --base_dir "<your_strategy_b_ft_run_dir>" \
  --direction eng_to_sr \
  --csv_path "/content/drive/MyDrive/T5/data/eng_to_sr.csv" \
  --which best
```

### Validate the CPT stage (not translation)

`colab_validate_t5.py` is translation-focused, so it’s not ideal for evaluating pure denoising CPT checkpoints.
Instead:
- compare CPT runs by `eval_loss` during CPT, and/or
- run a small Serbian-only qualitative probe, or
- add a dedicated CPT validation script if you want metrics for denoising.

---

## Strategy C validation

Strategy C also has two stages:

1. Mixed-language CPT: `colab_pretrain_t5_strategy_c.py`
2. Bidirectional fine-tune: `colab_train_t5_strategy_c.py`

Validation is the same as Strategy A (one direction at a time), but with Strategy C’s default seed.

### Validate EN→SR

```bash
python /content/drive/MyDrive/T5/colab_validate_t5.py \
  --strategy c \
  --base_dir "<your_strategy_c_ft_run_dir>" \
  --direction eng_to_sr \
  --csv_path "/content/drive/MyDrive/T5/data/eng_to_sr.csv" \
  --which best
```

### Validate SR→EN

```bash
python /content/drive/MyDrive/T5/colab_validate_t5.py \
  --strategy c \
  --base_dir "<your_strategy_c_ft_run_dir>" \
  --direction sr_to_eng \
  --csv_path "/content/drive/MyDrive/T5/data/sr_to_eng.csv" \
  --which best
```

---

## Strategy D validation (keyed split + translation-only)

Strategy D mixes Serbian denoising + EN→SR translation in one run.
To compare fairly, we validate:

- using Strategy D’s **keyed split** (stable by content)
- on **translation-only** test rows

### Required data

The validator will look for both files under your project root:

- `data/eng_to_sr.csv`
- `data/serbian_corpus.csv`

### Typical usage

```bash
python /content/drive/MyDrive/T5/colab_validate_t5.py \
  --strategy d \
  --base_dir "<your_strategy_d_run_dir>" \
  --which best
```

Optional override if the validator can’t infer `project_root` from `base_dir`:

```bash
python /content/drive/MyDrive/T5/colab_validate_t5.py \
  --strategy d \
  --base_dir "<your_strategy_d_run_dir>" \
  --project_root "/content/drive/MyDrive/T5" \
  --which best
```

Notes:
- `--direction` is ignored for Strategy D validation (it always evaluates EN→SR).
- `--csv_path` (if provided) is treated as the EN→SR CSV path.

---

## Optional: use the tokenized cache (Baseline/A/B/C)

If you used the baseline training scripts (or strategy siblings) that save tokenized datasets to Drive, you can ask the validator to reuse the cached, pre-split tokenized dataset:

```bash
python /content/drive/MyDrive/T5/colab_validate_t5.py \
  --strategy baseline \
  --base_dir "<run_dir>" \
  --direction eng_to_sr \
  --which best \
  --use_tokenized_cache
```

This is helpful when:
- you want the exact same test rows as training (guaranteed), and
- you want to skip tokenization overhead.

For Strategy D, tokenized-cache reuse is intentionally disabled (Strategy D’s tokenization cache is keyed differently).

---

## Output: `validate_results.json`

The validator writes a JSON file containing:

- which model folder was evaluated (`best`/`latest`/`final`)
- split parameters (seed + ratios)
- decoding parameters
- test metrics (BLEU, chrF++, diacritics, length ratio)

This file is designed to be easy to parse for tables/plots.

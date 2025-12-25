# Model note: `google/mt5-small`

This project uses **`google/mt5-small`** as the default backbone for Baseline and all strategies.

The goal of this document is to explain, at a high level:

- what mT5 is and how it differs from (English-only) T5
- what “small” means in practice
- how the model was pre-trained and what objective it learned
- why it’s a good fit for English ↔ Serbian (sr-Latn) translation and diacritics

---

## What is mT5?

**mT5** (“multilingual T5”) is a multilingual variant of Google’s **T5** (Text-To-Text Transfer Transformer) model family.

Key idea:

- Everything is framed as **text-to-text**.
- One model can handle many tasks by conditioning on the input text (often with a task prefix, e.g., `translate English to Serbian: ...`).

Unlike English-only T5 checkpoints, mT5 is designed to handle **many languages** and diverse scripts/characters.

---

## Architecture (what the model looks like)

mT5 uses the standard **Transformer encoder–decoder** architecture:

- **Encoder:** reads the input sequence and builds contextual representations.
- **Decoder:** generates the output sequence autoregressively.

This is the classic seq2seq layout used for translation.

### Why encoder–decoder is a good fit for translation

Translation requires a model to:

- ingest a full source sentence (encoder)
- generate a target sentence with different word order / morphology (decoder)

Encoder–decoder Transformers are a strong default choice for MT because they model this asymmetry directly.

---

## Tokenizer: SentencePiece (important for Serbian Latin)

mT5 uses a **SentencePiece** subword tokenizer (you’ll typically see `spiece.model` in saved checkpoints).

Why this matters here:

- Serbian Latin includes diacritics: `č ć š ž đ`.
- A multilingual SentencePiece vocabulary is much more likely to represent these characters cleanly than an English-only model.

In this repo we load the tokenizer with:

- `AutoTokenizer.from_pretrained(..., use_fast=False)`

This is mostly a stability choice: it avoids slow→fast conversion quirks and keeps behavior consistent in Colab.

---

## Pretraining: what mT5 learned before fine-tuning

mT5 is pretrained on a large multilingual text corpus to learn:

- general language modeling capabilities
- cross-lingual representations
- basic reasoning / summarization-style sequence transformations

### Objective: span corruption (“denoising”)

Like T5, mT5 is pretrained with a **span corruption** objective:

- Random contiguous spans of tokens are replaced with sentinel tokens: `<extra_id_0>`, `<extra_id_1>`, ...
- The model is trained to reconstruct the removed spans in order.

This is the same denoising format used in this repo’s CPT scripts (Strategies **B/C**) and in the denoising portion of Strategy **D**.

Why this objective is useful:

- It teaches the model to generate fluent text and to model longer-range dependencies.
- It naturally supports “fill in the blank” style generation, which transfers well to translation fine-tuning.

---

## What does “small” mean?

“Small” indicates the smaller parameter configuration in the mT5 family.

Practical implications:

- **Pros:** trains and runs faster, fits on common GPUs (e.g., Colab T4) with moderate sequence lengths.
- **Cons:** lower ceiling than larger checkpoints (e.g., `mt5-base`, `mt5-large`) on very challenging or highly domain-specific tasks.

For this project, `mt5-small` is a good trade-off because you want to compare strategies under limited compute and keep iteration cycles short.

---

## Why `google/mt5-small` fits this project

### 1) Multilingual coverage (English + Serbian)

Because mT5 is trained multilingual, it is much better positioned than English-only T5 to:

- model Serbian morphology
- generate Serbian word forms naturally
- handle diacritics and multilingual character distributions

### 2) Text-to-text format matches our scripts

All scripts in this repo use the text-to-text pattern:

- input: *prefixed* instruction + source text (for translation)
- output: target text

mT5 is explicitly designed for this.

### 3) Denoising compatibility

Strategies **B/C/D** rely on a T5-style denoising objective with sentinel tokens.

mT5 supports sentinel tokens out-of-the-box, so the corruption objective is native to the model family.

### 4) Practicality in Colab

`mt5-small` typically fits on common Colab GPUs with:

- max lengths around 256/256
- gradient accumulation
- optional gradient checkpointing

That makes it a solid default when comparing multiple experimental training pipelines.

---

## Known limitations / what to watch for

Even with mT5, you can still see:

- **ASCII Serbian fallback:** output drops diacritics; addressed in this repo by diacritics sanity checks + diacritics metrics.
- **Literal translations:** especially when the training corpus is narrow and literary.
- **Out-of-domain weakness:** “station variants” and everyday phrases may not exist in a literature-only corpus.

This is why we:

- keep an in-domain held-out test set
- also run out-of-domain qualitative probes
- track chrF++ and diacritics metrics, not only BLEU

---

## When to consider a different base model

You might consider upgrading if:

- you need higher absolute quality (try `mt5-base`)
- you need stronger English decoding or instruction following (different families may help)

But changing the backbone changes the experiment: for strategy comparisons, it’s best to keep the backbone fixed.

---

## Why we keep the backbone fixed (for strategy comparisons)

When comparing Strategies A/B/C/D against the Baseline, we want the differences to come from the **training protocol** (mixing, CPT, denoising, etc.), not from using a fundamentally stronger or weaker model.

Keeping `google/mt5-small` fixed helps ensure:

- **Controlled experiment:** one major variable changes at a time.
- **Fairness:** improvements aren’t just because a larger model was used.
- **Reproducibility:** fewer moving parts across runs.
- **Compute comparability:** strategies can be compared under similar hardware/time budgets.

Once you’ve identified the best *strategy*, a good next step is to re-run that strategy on a larger checkpoint (e.g., `mt5-base`) to see how quality scales.

---

## Where this repo uses mT5

- Baseline training: `colab_train_t5.py`
- Strategy A: `colab_train_t5_strategy_a.py`
- Strategy B: `colab_pretrain_t5_strategy_b.py` → `colab_train_t5_strategy_b.py`
- Strategy C: `colab_pretrain_t5_strategy_c.py` → `colab_train_t5_strategy_c.py`
- Strategy D: `colab_train_t5_strategy_d.py`

All scripts default to `CONFIG["model_name"] = "google/mt5-small"`.

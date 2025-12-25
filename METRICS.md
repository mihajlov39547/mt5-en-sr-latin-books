# Metrics and model selection

This project compares **Baseline** vs **Strategies A/B/C/D** using the same evaluation scripts and (as much as possible) the same training plumbing.

The purpose of this document is to explain:

- which metrics we compute
- what each metric captures (and what it misses)
- how to interpret metric trade-offs across strategies
- what “best model / best strategy” should look like for this dataset

---

## What we measure

### 1) BLEU (sacreBLEU)

**What it is:**
BLEU measures *n-gram overlap* between model output and the reference translation. This repo uses **sacreBLEU** (via `evaluate`) so scores are comparable across runs.

**Intuition:**
- BLEU rewards outputs that share many short phrases with the reference.
- It’s useful when your references are consistent and you want a standardized, well-known metric.

**Why we include it:**
- It’s a common baseline metric in MT literature.
- It provides continuity with other translation work.

**Where it helps:**
- Catching gross semantic failures (wrong content) often lowers BLEU.

**Limitations / pitfalls:**
- BLEU can be harsh when multiple valid translations exist (literary Serbian often has many valid paraphrases).
- BLEU can under-reward improvements in *fluency* when the wording diverges from the single reference.
- BLEU is not directly “diacritics-aware” (a missing `č` vs `c` can reduce overlap, but it’s not targeted).

**How to interpret:**
- Focus on *relative* comparisons across strategies using the same split.
- Look at BLEU together with chrF++ and diacritics metrics.

---

### 2) chrF++ (chrF with word-order=2)

**What it is:**
chrF is a character n-gram F-score metric. chrF++ extends chrF by including word n-grams; in this repo we use `sacrebleu.metrics.CHRF(word_order=2)`.

**Intuition:**
- Character-level matching is often more forgiving for morphologically rich languages (Serbian) and for small orthographic differences.
- It tends to correlate well when surface forms vary but are still “close”.

**Why we include it:**
- Serbian morphology (cases, agreements, inflections) often changes word endings. Character-level scoring is more sensitive to *near-misses* than BLEU.
- It’s typically a strong companion metric for Slavic languages.

**Where it helps:**
- Captures partial correctness (e.g., wrong case ending but correct stem).
- More sensitive to diacritics than BLEU because a diacritic changes characters.

**Limitations / pitfalls:**
- Still reference-based: a paraphrase can score lower even if it’s perfectly good.
- A high chrF++ doesn’t guarantee full semantic adequacy.

---

### 3) Serbian diacritics metrics (precision / recall / F1 / any-rate)

Serbian Latin requires these diacritics:

- `č ć š ž đ` (and uppercase variants)

Many translation failures show up as **ASCII Serbian** (e.g., `c` instead of `č`). This can make outputs look unnatural or even change meaning.

This repo computes targeted diacritics metrics (on Serbian outputs) to detect and quantify this failure mode.

#### 3.1 Diacritics precision

**Meaning:** of all diacritic characters the model produced, how many were “supported” by the reference (roughly: not hallucinated).

- Higher precision = the model doesn’t randomly insert diacritics in the wrong places.

#### 3.2 Diacritics recall

**Meaning:** of all diacritic characters present in the reference, how many appear in the model output.

- Higher recall = the model doesn’t lose diacritics (fewer ASCII fallbacks).

#### 3.3 Diacritics F1

The harmonic mean of precision and recall:

$$
F1 = \frac{2PR}{P+R}
$$

This is a compact summary of diacritics correctness.

#### 3.4 Diacritics any-rate

**Meaning:** fraction of generated outputs that contain *at least one* Serbian diacritic.

- Useful as a “sanity” signal: if this is near 0, your model is likely generating mostly ASCII Serbian.

**Why diacritics metrics matter here:**
- Literature-derived Serbian text uses diacritics frequently.
- A strategy that improves BLEU but collapses diacritics is usually not a “better” strategy for SR-Latn quality.

**Limitations / pitfalls:**
- These metrics are reference-dependent and are not a full spelling checker.
- They don’t verify *positional correctness* (we count character occurrences). They are intended as a robust, lightweight signal.

---

### 4) `eval_loss` and “best checkpoint”

Training scripts use early stopping and `load_best_model_at_end=True` (best by `eval_loss`).

**What it captures:**
- A general proxy for how well the model fits the validation objective.

**Why it matters:**
- It prevents over-training and reduces variance across runs.

**But:**
- Lower `eval_loss` does **not** always mean better translation quality.
- Always verify with translation metrics on held-out test data.

---

## How to compare strategies fairly

1) **Use the same evaluation split**
   - Baseline uses `split_seed=42`.
   - Each strategy uses its own default seed by design (A=52, B=62, C=72, D=82).
   - For a paper-quality comparison, consider running each strategy with multiple seeds, or standardizing the test set across strategies.

2) **Compare translation metrics on translation data**
   - Strategy D intentionally evaluates metrics on translation-only examples so denoising doesn’t pollute BLEU/chrF++.

3) **Report in-domain and out-of-domain separately**
   - The dataset is literary (in-domain).
   - The “station variants” are out-of-domain.

---

## What the “best model” looks like

A strong model/strategy should show:

### Translation adequacy
- Preserves meaning (entities, tense, negation, core relations)
- Doesn’t omit or invent key content

### Serbian fluency and style
- Natural Serbian word order
- Correct agreement and case
- Reads like Serbian, not English word order “translated literally”

### Diacritics robustness
- High diacritics recall and F1
- Low rate of ASCII fallback
- Stable diacritics even for longer generations

### Consistency
- Similar inputs yield similar translations (no random drift)
- Proper nouns and recurring phrases are handled consistently

### Generalization (not just memorization)
- Performs well on held-out test lines from the same books
- Doesn’t “copy” English into Serbian output
- Handles paraphrased or slightly modified inputs reasonably

### Efficiency / practicality
- Converges reliably without divergence (no NaN loss)
- Doesn’t require excessive compute to beat baseline

---

## What the “best strategy” looks like

The best strategy is the one that improves the *target outcomes* with minimal extra complexity.

Typical decision rules:

- If a strategy improves **chrF++ + diacritics F1** without hurting BLEU, it’s usually a clear win for SR-Latn quality.
- If BLEU improves but diacritics collapse (ASCII Serbian), it’s usually not acceptable for this dataset.
- If a two-stage strategy (B/C) improves quality but costs much more compute, Strategy D may be preferable if it achieves most of the gain in one stage.

Recommended reporting bundle per run:

- Test BLEU
- Test chrF++
- Diacritics precision/recall/F1 + any-rate
- Best checkpoint + `eval_loss`
- A short qualitative sample (5–20 sentences) showing typical error patterns

---

## Suggested qualitative checks (complements metrics)

Metrics won’t catch everything, especially for literature.

Suggested manual checks:

- Named entities (people/place names) preserved
- Dialogue punctuation and quotes
- Long sentence stability (no truncation, no repetition loops)
- “Station variants” style prompts (out-of-domain behavior)

---

## Where the metrics are implemented

- Training scripts (`colab_train_t5*.py`) compute BLEU/chrF++/diacritics during evaluation.
- `colab_validate_t5.py` computes full evaluation on the held-out test split.
- `colab_test_t5.py` provides quick smoke tests for common failure modes.

If you change evaluation logic, update this document so the paper/report stays aligned with the code.

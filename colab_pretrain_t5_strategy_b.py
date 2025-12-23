"""\
Strategy B: Serbian-only continued pretraining (CPT) with T5 denoising (span corruption)

Goal
----
Adapt the base model (default: google/mt5-small) to the *literary Serbian Latin* domain
before supervised translation fine-tuning.

This script is intentionally styled very similarly to `colab_train_t5.py`:
- Colab-first (Drive mounted, cache + models on Drive)
- Deterministic 70/20/10 split
- Tokenization caching to Drive
- Checkpoints with auto-resume
- Dataset fingerprint guard to prevent accidental resume on changed data
- Writes `results.json` in a baseline-like schema

Input data
----------
Uses the monolingual Serbian corpus CSV:
  /content/drive/MyDrive/T5/data/serbian_corpus.csv

CSV format:
  - column: text

Run in Colab
------------
1) Mount Drive:
   from google.colab import drive
   drive.mount('/content/drive')

2) Run:
   !python /content/drive/MyDrive/T5/colab_pretrain_t5_strategy_b.py

Notes
-----
- This implements the standard T5 denoising objective ("span corruption"):
  - Input: text with random spans replaced by sentinels <extra_id_0>, <extra_id_1>, ...
  - Target: concatenation of the removed spans, each preceded by its sentinel

- Safe defaults:
  - noise_density=0.15
  - mean_noise_span_length=3.0

- After CPT finishes, Strategy B translation fine-tune can start from the CPT checkpoint
  by setting baseline `CONFIG["model_name"]` to the CPT output directory.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
import hashlib
import inspect
import math
import json
import random


# Helps reduce CUDA memory fragmentation on long runs (safe no-op if unsupported).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")


# ---------------------------
# USER CONFIG
# ---------------------------
CONFIG = {
    # Drive locations
    "drive_project_dir": "/content/drive/MyDrive/T5",
    "data_dir": "data",
    "models_dir": "models",
    "cache_dir": "cache",

    # Model
    "model_name": "google/mt5-small",

    # For naming consistency with baseline results.json
    "training_task": "cpt_denoising_serbian",

    # Training
    # CPT typically trains by steps. We keep an epoch-based upper bound like baseline,
    # and rely on early stopping on validation loss.
    "num_train_epochs": 10,
    "learning_rate": 1e-4,
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "max_input_length": 256,
    "max_target_length": 256,

    # Dataset split ratios (70/20/10) with requested seed=62
    "validation_split": 0.2,
    "test_split": 0.1,
    "split_seed": 62,

    # Denoising / corruption params (safe defaults)
    # Following common T5 settings: noise_density ~0.15, mean span length ~3.
    # Sentinels are <extra_id_0>, <extra_id_1>, ... and exist in T5/mT5 vocabs.
    "noise_density": 0.15,
    "mean_noise_span_length": 3.0,

    # Memory optimizations
    "optim": "adamw_torch",
    "gradient_checkpointing": True,

    # Stability
    "max_grad_norm": 1.0,
    "warmup_ratio": 0.03,

    # Checkpointing
    "save_strategy": "steps",
    "save_steps": 1000,

    # Safety toggles
    "force_fresh_train": False,
    "force_retokenize": False,

    # Evaluation scheduling
    "eval_strategy": "steps",
    "eval_steps": 1000,

    # Eval speed
    "eval_subset_size": 500,

    # Early stopping
    "early_stopping": True,
    "early_stopping_patience": 8,
    "early_stopping_threshold": 0.0,

    # Mixed precision
    "use_fp16": False,
    "use_bf16": False,

    # GPU auto-tuning
    "auto_tune_gpu": True,
}


# ---------------------------
# Helpers (copied style from baseline)
# ---------------------------

def detect_and_tune_for_gpu() -> dict[str, any]:
    """Return CONFIG overrides based on detected GPU."""
    try:
        import torch
    except ImportError:
        return {}

    if not torch.cuda.is_available():
        return {"use_fp16": False, "use_bf16": False}

    gpu_name = torch.cuda.get_device_name(0).lower()
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    bf16_supported = torch.cuda.is_bf16_supported()

    print(f"\n[GPU Auto-Tune] Detected: {torch.cuda.get_device_name(0)}")
    print(f"[GPU Auto-Tune] Memory: {gpu_mem_gb:.1f} GB, bf16 supported: {bf16_supported}")

    overrides: dict[str, any] = {}

    if "t4" in gpu_name:
        overrides = {
            "train_batch_size": 8,
            "eval_batch_size": 8,
            "gradient_accumulation_steps": 2,
            "use_fp16": False,
            "use_bf16": False,
            "gradient_checkpointing": True,
            "eval_subset_size": 500,
        }
        print("[GPU Auto-Tune] Profile: T4 (conservative, ~11GB usage)")

    elif "a100" in gpu_name:
        if gpu_mem_gb >= 70:
            overrides = {
                "train_batch_size": 64,
                "eval_batch_size": 64,
                "gradient_accumulation_steps": 1,
                "use_fp16": False,
                "use_bf16": bf16_supported,
                "gradient_checkpointing": False,
                "eval_subset_size": 2000,
            }
            print("[GPU Auto-Tune] Profile: A100-80GB (high throughput)")
        else:
            overrides = {
                "train_batch_size": 32,
                "eval_batch_size": 32,
                "gradient_accumulation_steps": 1,
                "use_fp16": False,
                "use_bf16": bf16_supported,
                "gradient_checkpointing": False,
                "eval_subset_size": 1000,
            }
            print("[GPU Auto-Tune] Profile: A100-40GB (balanced)")

    elif "h100" in gpu_name:
        overrides = {
            "train_batch_size": 64,
            "eval_batch_size": 64,
            "gradient_accumulation_steps": 1,
            "use_fp16": False,
            "use_bf16": bf16_supported,
            "gradient_checkpointing": False,
            "eval_subset_size": 2000,
        }
        print("[GPU Auto-Tune] Profile: H100 (maximum throughput)")

    elif gpu_mem_gb >= 30:
        overrides = {
            "train_batch_size": 32,
            "eval_batch_size": 32,
            "gradient_accumulation_steps": 1,
            "use_fp16": not bf16_supported,
            "use_bf16": bf16_supported,
            "gradient_checkpointing": False,
            "eval_subset_size": 1000,
        }
        print(f"[GPU Auto-Tune] Profile: High-memory GPU ({gpu_mem_gb:.0f}GB)")

    elif gpu_mem_gb >= 20:
        overrides = {
            "train_batch_size": 16,
            "eval_batch_size": 16,
            "gradient_accumulation_steps": 1,
            "use_fp16": not bf16_supported,
            "use_bf16": bf16_supported,
            "gradient_checkpointing": True,
            "eval_subset_size": 500,
        }
        print(f"[GPU Auto-Tune] Profile: Medium GPU ({gpu_mem_gb:.0f}GB)")

    else:
        overrides = {
            "train_batch_size": 8,
            "eval_batch_size": 8,
            "gradient_accumulation_steps": 2,
            "use_fp16": False,
            "use_bf16": False,
            "gradient_checkpointing": True,
            "eval_subset_size": 500,
        }
        print(f"[GPU Auto-Tune] Profile: Small/Unknown GPU ({gpu_mem_gb:.0f}GB, conservative)")

    if overrides:
        print("[GPU Auto-Tune] Overrides:", {k: v for k, v in overrides.items() if v is not None})

    return overrides


def is_colab() -> bool:
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_BACKEND_VERSION"):
        return True
    if os.environ.get("COLAB_GPU") is not None:
        return True
    if Path("/content").exists():
        return True
    return "google.colab" in sys.modules


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_slug(text: str) -> str:
    return (
        str(text)
        .replace("/", "-")
        .replace("\\", "-")
        .replace(":", "-")
        .replace(" ", "-")
    )


def project_root() -> Path:
    return Path(CONFIG["drive_project_dir"]).expanduser()


def data_path(filename: str) -> Path:
    return project_root() / CONFIG["data_dir"] / filename


def file_fingerprint(path: Path) -> str:
    """Stable-enough fingerprint without hashing multi-GB files."""
    fingerprint = "unknown"
    try:
        st = path.stat()
        size = int(st.st_size)
        mtime = int(st.st_mtime)
        h = hashlib.sha256()
        with path.open("rb") as f:
            h.update(f.read(1024 * 1024))
        fingerprint = f"sz{size}__mt{mtime}__h{h.hexdigest()[:12]}"
    except Exception:
        pass
    return fingerprint


def corpus_csv_path() -> Path:
    return data_path("serbian_corpus.csv")


def output_dir() -> Path:
    model_slug = safe_slug(CONFIG["model_name"])
    family = "mt5" if "mt5" in str(CONFIG["model_name"]).lower() else "t5"

    valid_ratio = float(CONFIG.get("validation_split", 0.2))
    test_ratio = float(CONFIG.get("test_split", 0.1))
    train_ratio = 1.0 - valid_ratio - test_ratio
    split_seed = int(CONFIG.get("split_seed", 62))

    noise_density = float(CONFIG.get("noise_density", 0.15))
    mean_span = float(CONFIG.get("mean_noise_span_length", 3.0))

    def _pct(x: float) -> int:
        return int(round(float(x) * 100.0))

    split_sig = f"s{_pct(train_ratio)}v{_pct(valid_ratio)}t{_pct(test_ratio)}_seed{split_seed}"
    corrupt_sig = f"nd{noise_density:.3f}_ms{mean_span:.2f}"

    return (
        project_root()
        / CONFIG["models_dir"]
        / f"{family}_cpt_model__sr_denoise__{model_slug}__{corrupt_sig}__{split_sig}"
    )


def tokenized_cache_path() -> Path:
    model_slug = safe_slug(CONFIG["model_name"])

    csv_fp = corpus_csv_path()
    fingerprint = file_fingerprint(csv_fp)

    valid_ratio = float(CONFIG.get("validation_split", 0.2))
    test_ratio = float(CONFIG.get("test_split", 0.1))
    train_ratio = 1.0 - valid_ratio - test_ratio
    split_seed = int(CONFIG.get("split_seed", 62))

    noise_density = float(CONFIG.get("noise_density", 0.15))
    mean_span = float(CONFIG.get("mean_noise_span_length", 3.0))

    def _pct(x: float) -> int:
        return int(round(float(x) * 100.0))

    split_sig = f"s{_pct(train_ratio)}v{_pct(valid_ratio)}t{_pct(test_ratio)}_seed{split_seed}"
    corrupt_sig = f"nd{noise_density:.3f}_ms{mean_span:.2f}"

    return (
        project_root()
        / CONFIG["cache_dir"]
        / "tokenized_datasets"
        / f"sr_denoise__{model_slug}__{fingerprint}__{corrupt_sig}__{split_sig}__in{CONFIG['max_input_length']}__out{CONFIG['max_target_length']}"
    )


def find_latest_checkpoint(out_dir: Path) -> str | None:
    if not out_dir.exists():
        return None

    best_step = -1
    best_path: Path | None = None
    for child in out_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith("checkpoint-"):
            continue
        try:
            step = int(name.split("-", 1)[1])
        except Exception:
            continue
        if step > best_step:
            best_step = step
            best_path = child

    return str(best_path) if best_path else None


def require_drive_mounted() -> None:
    root = project_root()
    if not root.exists():
        raise RuntimeError(
            "Drive path not found. Did you mount Google Drive?\n"
            "Run in a Colab cell:\n"
            "  from google.colab import drive\n"
            "  drive.mount('/content/drive')\n"
            f"Expected folder: {root}"
        )


def print_paths() -> None:
    print("\nDrive paths")
    print("- project:", project_root())
    print("- serbian_corpus:", corpus_csv_path())
    print("- output_dir:", output_dir())
    print("- tokenized_cache:", tokenized_cache_path())


def ensure_packages() -> None:
    """Optional: install deps if missing (handy for fresh Colab runtimes)."""

    def _try_import() -> bool:
        try:
            import transformers  # noqa: F401
            import datasets  # noqa: F401
            import sentencepiece  # noqa: F401
            import accelerate  # noqa: F401
            import google.protobuf  # noqa: F401
            return True
        except Exception:
            return False

    if _try_import():
        return

    print("\nInstalling required packages (one-time for this Colab runtime)...")
    pkgs = [
        "transformers==4.49.0",
        "datasets==3.3.2",
        "sentencepiece==0.2.0",
        "accelerate==1.4.0",
        "protobuf>=4.25.0",
    ]

    import subprocess

    cmd = [sys.executable, "-m", "pip", "install", "-q"] + pkgs
    subprocess.check_call(cmd)


# ---------------------------
# Span corruption implementation
# ---------------------------

def _sample_span_length(rng: random.Random, mean_noise_span_length: float) -> int:
    """Sample a positive span length with approximately the desired mean.

    We use Poisson(lambda=mean-1) + 1 so the minimum span length is 1.
    This is simple, stable, and close enough for our purposes.
    """
    m = float(mean_noise_span_length)
    if m <= 1.0:
        return 1

    # Knuth Poisson sampler (numpy-free).
    lam = max(1e-9, m - 1.0)
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return max(1, k)


def _make_sentinels(tokenizer, n: int) -> list[str]:
    """Return [<extra_id_0>, <extra_id_1>, ...] as strings."""
    # These tokens exist for T5/mT5. If someone swaps in a model without them,
    # we fail fast with a clear message.
    sent = [f"<extra_id_{i}>" for i in range(int(n))]
    # validate a small subset
    for s in sent[: min(3, len(sent))]:
        tid = tokenizer.convert_tokens_to_ids(s)
        if tid is None or tid == tokenizer.unk_token_id:
            raise ValueError(
                f"Tokenizer does not support sentinel token {s}. "
                "Strategy B denoising requires a T5/mT5-compatible tokenizer."
            )
    return sent


def span_corrupt_text(
    text: str,
    *,
    tokenizer,
    noise_density: float,
    mean_noise_span_length: float,
    rng: random.Random,
) -> tuple[str, str]:
    """Create a (corrupted_input, target) pair in T5 span-corruption format.

    This is word-based corruption (splitting on whitespace) for simplicity and speed.
    It works well enough for CPT and keeps the implementation dependency-free.

    Edge cases:
    - Very short texts may yield zero noise tokens; we then force at least one token.
    - If `text` is empty/whitespace, returns empty pairs.
    """
    if not isinstance(text, str):
        return "", ""
    text = text.strip()
    if not text:
        return "", ""

    tokens = text.split()
    n = len(tokens)
    if n == 0:
        return "", ""

    # how many tokens we want to mask
    noise_density = float(noise_density)
    noise_tokens = int(round(n * noise_density))
    noise_tokens = max(1, min(n, noise_tokens))

    # sample spans until we reach noise_tokens
    spans: list[tuple[int, int]] = []
    covered = 0
    attempts = 0
    max_attempts = 5 * n

    # We'll pick spans by choosing random start indices and span lengths,
    # ensuring they don't overlap.
    taken = [False] * n
    while covered < noise_tokens and attempts < max_attempts:
        attempts += 1
        start = rng.randrange(0, n)
        if taken[start]:
            continue
        span_len = _sample_span_length(rng, mean_noise_span_length)
        end = min(n, start + span_len)

        # shrink if overlaps
        while end > start and any(taken[i] for i in range(start, end)):
            end -= 1
        if end <= start:
            continue

        # mark
        for i in range(start, end):
            taken[i] = True
        spans.append((start, end))
        covered += (end - start)

    if not spans:
        # fallback: mask a single token
        start = rng.randrange(0, n)
        spans = [(start, min(n, start + 1))]

    spans.sort(key=lambda x: x[0])

    # If we overshot slightly, that's fine. If we masked almost everything,
    # keep at least 1 non-masked token if possible.
    if n > 1:
        masked = sum(e - s for s, e in spans)
        if masked >= n:
            spans = spans[:1]

    sentinels = _make_sentinels(tokenizer, len(spans) + 1)

    # Build corrupted input and target
    corrupted_parts: list[str] = []
    target_parts: list[str] = []

    cur = 0
    for i, (s, e) in enumerate(spans):
        # non-masked prefix
        if s > cur:
            corrupted_parts.extend(tokens[cur:s])

        sentinel = sentinels[i]
        corrupted_parts.append(sentinel)

        removed = " ".join(tokens[s:e])
        target_parts.append(sentinel)
        if removed:
            target_parts.append(removed)

        cur = e

    # trailing non-masked suffix
    if cur < n:
        corrupted_parts.extend(tokens[cur:])

    # end sentinel
    target_parts.append(sentinels[len(spans)])

    return " ".join(corrupted_parts).strip(), " ".join(target_parts).strip()


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    start = time.time()

    if is_colab():
        ensure_packages()

    require_drive_mounted()

    from datasets import DatasetDict, load_dataset, load_from_disk
    import numpy as np
    import torch
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrainerCallback,
    )

    print_paths()

    # Pick device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"\nCUDA: {torch.cuda.get_device_name(0)}")

        if bool(CONFIG.get("auto_tune_gpu", True)):
            gpu_overrides = detect_and_tune_for_gpu()
            for key, value in gpu_overrides.items():
                CONFIG[key] = value
    else:
        device = "cpu"
        print("\nNo CUDA detected; using CPU")

    if device != "cuda":
        CONFIG["use_fp16"] = False
        CONFIG["use_bf16"] = False

    # Load and split corpus
    csv_path = corpus_csv_path()
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing corpus: {csv_path}\n"
            "Create it with create_serbian_corpus.py and put it in Drive at the path shown above."
        )

    print(f"\nLoading corpus: {csv_path}")
    dataset = load_dataset("csv", data_files=str(csv_path))["train"]

    if "text" not in dataset.column_names:
        raise ValueError(
            f"serbian_corpus.csv must have a 'text' column. Found columns: {dataset.column_names}"
        )

    # Quick diacritics presence sanity check on raw text
    def _print_text_diacritics_sanity(ds, sample_size: int = 5000) -> None:
        diacritics = "čćšžđČĆŠŽĐ"
        try:
            n = len(ds)
            if n <= 0:
                return
            k = min(int(sample_size), n)
            subset = ds.select(range(k)) if k < n else ds
            counts = {ch: 0 for ch in diacritics}
            any_lines = 0
            for t in subset["text"]:
                if not isinstance(t, str):
                    continue
                any_lines += 1
                for ch in diacritics:
                    if ch in t:
                        counts[ch] += 1
            any_diacritics = any(v > 0 for v in counts.values())
            print("\n[Data sanity] Diacritics in loaded Serbian corpus")
            print("- checked_rows:", any_lines)
            print("- counts_per_char:", counts)
            print("- any_diacritics:", any_diacritics)
            if not any_diacritics:
                print("[WARN] No diacritics found in corpus sample. Check that this is Serbian Latin text.")
        except Exception as exc:
            print(f"\n[WARN] Corpus diacritics sanity check failed: {type(exc).__name__}: {exc}")

    _print_text_diacritics_sanity(dataset)

    # ---------------------------
    # Fixed split: train/valid/test (70/20/10)
    # ---------------------------
    valid_ratio = float(CONFIG.get("validation_split", 0.2))
    test_ratio = float(CONFIG.get("test_split", 0.1))
    if valid_ratio < 0 or test_ratio < 0 or (valid_ratio + test_ratio) >= 1.0:
        raise ValueError(
            f"Invalid split ratios: validation_split={valid_ratio}, test_split={test_ratio}. "
            "Need validation_split + test_split < 1.0"
        )

    split_seed = int(CONFIG.get("split_seed", 62))
    split_test = dataset.train_test_split(test_size=test_ratio, seed=split_seed)
    train_valid = split_test["train"]
    test_ds = split_test["test"]

    valid_ratio_rel = valid_ratio / (1.0 - test_ratio)
    split_valid = train_valid.train_test_split(test_size=valid_ratio_rel, seed=split_seed)
    train_ds = split_valid["train"]
    valid_ds = split_valid["test"]

    dataset_dict = DatasetDict({"train": train_ds, "validation": valid_ds, "test": test_ds})
    print("- train:", len(dataset_dict["train"]))
    print("- valid:", len(dataset_dict["validation"]))
    print("- test :", len(dataset_dict["test"]))

    # Load model/tokenizer
    print(f"\nLoading model: {CONFIG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"], use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["model_name"])

    if bool(CONFIG.get("gradient_checkpointing")):
        try:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
            print("\n[Info] Enabled gradient checkpointing (use_cache=False)")
        except Exception as exc:
            print(f"\n[WARN] Could not enable gradient checkpointing: {type(exc).__name__}: {exc}")

    # Confirm tokenizer can represent Serbian diacritics at all (baseline helper logic)
    def _print_tokenizer_diacritics_support(tok) -> None:
        diacritics = "čćšžđČĆŠŽĐ"
        try:
            print("\n[Tokenizer sanity] Diacritics round-trip")
            bad: list[str] = []
            for ch in diacritics:
                ids = tok.encode(ch, add_special_tokens=False)
                decoded = tok.decode(ids, skip_special_tokens=True)
                ok = (decoded == ch)
                print(f"- {ch!r} -> ids={ids} -> {decoded!r} ok={ok}")
                if not ok:
                    bad.append(ch)
            if bad:
                print("[WARN] Tokenizer cannot round-trip these diacritics:", "".join(bad))
                print("[Hint] Use a T5/mT5 model family with SentencePiece that supports these chars.")
        except Exception as exc:
            print(f"\n[WARN] Tokenizer diacritics check failed: {type(exc).__name__}: {exc}")

    _print_tokenizer_diacritics_support(tokenizer)

    # Tokenize (cached)
    cache_path = tokenized_cache_path()
    ensure_dir(cache_path.parent)

    noise_density = float(CONFIG.get("noise_density", 0.15))
    mean_span = float(CONFIG.get("mean_noise_span_length", 3.0))

    def preprocess(examples: dict) -> dict:
        # IMPORTANT: Seed corruption per example deterministically so tokenized cache is stable.
        # We derive a local RNG from (split_seed + row_index) so reruns are reproducible.
        # datasets.map provides examples in batch; we use a batch-level seed and per-row offset.
        texts = examples["text"]
        out_inputs: list[str] = []
        out_targets: list[str] = []

        batch_seed = int(CONFIG.get("split_seed", 62))
        for i, t in enumerate(texts):
            rng = random.Random(batch_seed + i)
            corrupted, target = span_corrupt_text(
                t,
                tokenizer=tokenizer,
                noise_density=noise_density,
                mean_noise_span_length=mean_span,
                rng=rng,
            )
            out_inputs.append(corrupted)
            out_targets.append(target)

        model_inputs = tokenizer(
            out_inputs,
            max_length=CONFIG["max_input_length"],
            truncation=True,
            padding=False,
        )
        target_tokens = tokenizer(
            out_targets,
            max_length=CONFIG["max_target_length"],
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = target_tokens["input_ids"]
        return model_inputs

    if cache_path.exists() and not bool(CONFIG.get("force_retokenize")):
        print(f"\nLoading tokenized cache: {cache_path}")
        tokenized = load_from_disk(str(cache_path))
    else:
        print("\nTokenizing (first run for this config)...")
        tokenized = dataset_dict.map(
            preprocess,
            batched=True,
            remove_columns=dataset_dict["train"].column_names,
        )
        tokenized.save_to_disk(str(cache_path))
        print(f"\nSaved tokenized cache: {cache_path}")

    # Collator (dynamic padding)
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
        label_pad_token_id=-100,
    )

    # Eval subset for speed
    eval_dataset = tokenized["validation"]
    subset = CONFIG.get("eval_subset_size")
    if isinstance(subset, int) and 0 < subset < len(eval_dataset):
        print(f"\nEval subset: {subset} / {len(eval_dataset)}")
        eval_dataset = eval_dataset.select(range(subset))

    out_dir = output_dir()
    ensure_dir(out_dir)

    # Prevent accidental resume from a checkpoint trained on a different corpus version.
    valid_ratio = float(CONFIG.get("validation_split", 0.2))
    test_ratio = float(CONFIG.get("test_split", 0.1))
    train_ratio = 1.0 - valid_ratio - test_ratio
    split_seed = int(CONFIG.get("split_seed", 62))

    def _pct(x: float) -> int:
        return int(round(float(x) * 100.0))

    split_sig = f"s{_pct(train_ratio)}v{_pct(valid_ratio)}t{_pct(test_ratio)}_seed{split_seed}"

    current_fp = (
        f"{file_fingerprint(csv_path)}__{split_sig}"
        f"__nd{noise_density:.3f}__ms{mean_span:.2f}"
        f"__in{int(CONFIG['max_input_length'])}__out{int(CONFIG['max_target_length'])}"
    )
    fp_file = out_dir / "dataset_fingerprint.txt"
    previous_fp = None
    try:
        if fp_file.exists():
            previous_fp = fp_file.read_text(encoding="utf-8").strip() or None
        fp_file.write_text(current_fp, encoding="utf-8")
    except Exception:
        pass

    args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        eval_strategy=CONFIG.get("eval_strategy", CONFIG.get("save_strategy", "steps")),
        eval_steps=int(CONFIG.get("eval_steps", CONFIG.get("save_steps", 1000))),
        save_strategy=CONFIG.get("save_strategy", "steps"),
        save_steps=int(CONFIG.get("save_steps", 1000)),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        learning_rate=CONFIG["learning_rate"],
        num_train_epochs=CONFIG["num_train_epochs"],
        weight_decay=0.01,
        optim=str(CONFIG.get("optim", "adamw_torch")),
        max_grad_norm=float(CONFIG.get("max_grad_norm", 1.0)),
        warmup_ratio=float(CONFIG.get("warmup_ratio", 0.0)),
        per_device_train_batch_size=CONFIG["train_batch_size"],
        per_device_eval_batch_size=CONFIG["eval_batch_size"],
        gradient_accumulation_steps=CONFIG.get("gradient_accumulation_steps", 1),
        fp16=bool(CONFIG.get("use_fp16", False)),
        bf16=bool(CONFIG.get("use_bf16", False)),
        logging_steps=100,
        report_to="none",
        predict_with_generate=False,
    )

    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    class NanGuardCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return control
            loss = logs.get("loss")
            grad_norm = logs.get("grad_norm")

            def _bad_number(x) -> bool:
                try:
                    return not math.isfinite(float(x))
                except Exception:
                    return False

            if loss is not None and _bad_number(loss):
                print("\n[FATAL] Non-finite loss detected; stopping training to prevent wasting time.")
                control.should_training_stop = True
            if grad_norm is not None and _bad_number(grad_norm):
                print("\n[FATAL] Non-finite grad_norm detected; stopping training to prevent wasting time.")
                control.should_training_stop = True
            return control

    class SaveTokenizerToCheckpointCallback(TrainerCallback):
        def __init__(self, tokenizer_to_save):
            self._tokenizer = tokenizer_to_save

        def on_save(self, args, state, control, **kwargs):
            try:
                ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                if ckpt_dir.exists():
                    self._tokenizer.save_pretrained(str(ckpt_dir))
            except Exception as exc:
                print(f"\n[WARN] Could not save tokenizer into checkpoint: {type(exc).__name__}: {exc}")
            return control

    trainer_sig = inspect.signature(Seq2SeqTrainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Seq2SeqTrainer(**trainer_kwargs)
    trainer.add_callback(NanGuardCallback())
    trainer.add_callback(SaveTokenizerToCheckpointCallback(tokenizer))

    if bool(CONFIG.get("early_stopping")):
        trainer.add_callback(
            EarlyStoppingCallback(
                early_stopping_patience=int(CONFIG.get("early_stopping_patience", 5)),
                early_stopping_threshold=float(CONFIG.get("early_stopping_threshold", 0.0)),
            )
        )

    resume = None
    if not bool(CONFIG.get("force_fresh_train")):
        resume = find_latest_checkpoint(out_dir)

    if resume and previous_fp and previous_fp != current_fp:
        print("\n[Info] Serbian corpus changed since last run; not resuming from old checkpoints.")
        print("- previous:", previous_fp)
        print("- current :", current_fp)
        resume = None

    print("\nTraining (CPT denoising)...")
    train_started_at = time.time()
    if resume:
        print("Resuming from:", resume)
        trainer.train(resume_from_checkpoint=resume)
    else:
        trainer.train()

    train_elapsed_sec = float(time.time() - train_started_at)

    # Save final model
    print("\nSaving final model:", out_dir)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # ---------------------------
    # Final evaluation + results.json
    # ---------------------------
    # For CPT, we mostly care about eval_loss/perplexity-like signals.
    test_dataset = tokenized.get("test")
    test_metrics: dict[str, float] = {}
    if test_dataset is not None and len(test_dataset) > 0:
        print("\nEvaluating on test split...")
        pred = trainer.predict(test_dataset=test_dataset)
        test_metrics = {k: float(v) for k, v in pred.metrics.items() if isinstance(v, (int, float))}

    state = trainer.state
    best_eval_loss = None
    try:
        if isinstance(state.best_metric, (int, float)):
            best_eval_loss = float(state.best_metric)
    except Exception:
        pass

    results = {
        "model_name": str(CONFIG["model_name"]),
        "translation_direction": None,
        "training_task": str(CONFIG.get("training_task", "cpt_denoising")),
        "output_dir": str(out_dir),
        "dataset_csv": str(csv_path),
        "dataset_fingerprint": str(current_fp),
        "resume_from": str(resume) if resume else None,
        "split": {
            "train": 1.0 - float(CONFIG.get("validation_split", 0.2)) - float(CONFIG.get("test_split", 0.1)),
            "validation": float(CONFIG.get("validation_split", 0.2)),
            "test": float(CONFIG.get("test_split", 0.1)),
        },
        "corruption": {
            "noise_density": float(noise_density),
            "mean_noise_span_length": float(mean_span),
        },
        "training": {
            "global_step": int(getattr(state, "global_step", 0) or 0),
            "epoch": float(getattr(state, "epoch", 0.0) or 0.0),
            "train_runtime_sec": train_elapsed_sec,
            "best_eval_loss": best_eval_loss,
            "best_model_checkpoint": str(getattr(state, "best_model_checkpoint", None)),
        },
        "test_metrics": test_metrics,
    }

    try:
        results_path = out_dir / "results.json"
        results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\nWrote results:", results_path)
    except Exception as exc:
        print(f"\n[WARN] Could not write results.json: {type(exc).__name__}: {exc}")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/3600:.2f} hours")


if __name__ == "__main__":
    main()

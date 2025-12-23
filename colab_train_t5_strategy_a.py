"""\
Strategy A: Bidirectional supervised fine-tuning (single model, 2 directions)

This script is a drop-in sibling of `colab_train_t5.py`.

What changes vs baseline:
- Builds an in-memory *combined* dataset by mixing rows from:
    - data/eng_to_sr.csv  (English -> Serbian)
    - data/sr_to_eng.csv  (Serbian -> English)
  without creating a new CSV on disk.
- Adds a per-example `direction` column and applies a direction-specific task prefix.

Everything else intentionally stays identical:
- Same deterministic 70/20/10 split with seed
- Same tokenization caching (now keyed by *both* CSV fingerprints + mix config)
- Same Trainer arguments, early stopping, BLEU/chrF++ metrics

Usage in Google Colab:

1) Mount Drive:
   from google.colab import drive
   drive.mount('/content/drive')

2) Ensure these exist:
   /content/drive/MyDrive/T5/data/eng_to_sr.csv
   /content/drive/MyDrive/T5/data/sr_to_eng.csv

3) Run:
   !python /content/drive/MyDrive/T5/colab_train_t5_strategy_a.py

Notes:
- Mixing ratios are configurable via CONFIG["mix"].
- Default is 100% + 100% (dataset size ~ 2x baseline).
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

    # Direction (for consistency with baseline; Strategy A is always bidirectional)
    "translation_direction": "bidirectional",

    # Strategy A mixing config
    # Take a configurable fraction from each direction, then concatenate.
    # - 1.0/1.0 means 100% + 100% (default)
    # - 0.7/0.3 means keep 70% of eng->sr rows and 30% of sr->eng rows
    "mix": {
        "eng_to_sr_fraction": 1.0,
        "sr_to_eng_fraction": 1.0,
        # How to choose rows when fraction < 1.0:
        # - "deterministic_head": take first K rows (fastest, stable, but can bias if data is ordered)
        # - "random": shuffle with `sample_seed` and then take K (recommended for fair subsampling)
        "sample_mode": "random",
        # Seed used only when sample_mode == "random".
        # Keep aligned with baseline split_seed for reproducibility.
        "sample_seed": 42,
    },

    # Training
    "num_train_epochs": 10,
    "learning_rate": 1e-4,
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "max_input_length": 256,
    "max_target_length": 256,
    "validation_split": 0.2,
    "test_split": 0.1,
    # Split seed for deterministic 70/20/10 split (affects which exact rows end up in test).
    "split_seed": 42,

    # Memory optimizations / stability
    "optim": "adamw_torch",
    "gradient_checkpointing": True,
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
    "generation_num_beams": 1,

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
# Helpers (kept in-file to avoid refactors in baseline)
# ---------------------------

def detect_and_tune_for_gpu() -> dict[str, any]:
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
        print("[GPU Auto-Tune] Profile: T4 (conservative)")
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
                "generation_num_beams": 4,
            }
            print("[GPU Auto-Tune] Profile: A100-80GB")
        else:
            overrides = {
                "train_batch_size": 32,
                "eval_batch_size": 32,
                "gradient_accumulation_steps": 1,
                "use_fp16": False,
                "use_bf16": bf16_supported,
                "gradient_checkpointing": False,
                "eval_subset_size": 1000,
                "generation_num_beams": 2,
            }
            print("[GPU Auto-Tune] Profile: A100-40GB")
    elif "h100" in gpu_name:
        overrides = {
            "train_batch_size": 64,
            "eval_batch_size": 64,
            "gradient_accumulation_steps": 1,
            "use_fp16": False,
            "use_bf16": bf16_supported,
            "gradient_checkpointing": False,
            "eval_subset_size": 2000,
            "generation_num_beams": 4,
        }
        print("[GPU Auto-Tune] Profile: H100")
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
        print(f"[GPU Auto-Tune] Profile: Small/Unknown GPU ({gpu_mem_gb:.0f}GB)")

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


def print_diacritics_sanity(dataset, column: str = "target", sample_size: int = 5000) -> None:
    """Prints quick stats so you can confirm the loaded dataset actually contains diacritics."""
    diacritics = "čćšžđČĆŠŽĐ"
    try:
        n = len(dataset)
        if n <= 0:
            return
        k = min(int(sample_size), n)
        # deterministic subset for consistent debugging
        subset = dataset.select(range(k)) if k < n else dataset
        counts = {ch: 0 for ch in diacritics}
        any_lines = 0
        for t in subset[column]:
            if not isinstance(t, str):
                continue
            any_lines += 1
            for ch in diacritics:
                if ch in t:
                    counts[ch] += 1
        any_diacritics = any(v > 0 for v in counts.values())
        print("\n[Data sanity] Diacritics in loaded targets")
        print("- checked_rows:", any_lines)
        print("- counts_per_char:", counts)
        print("- any_diacritics:", any_diacritics)
        if not any_diacritics:
            print("[WARN] No Serbian diacritics found in loaded targets. Model will learn ASCII Serbian.")
    except Exception as exc:
        print(f"\n[WARN] Diacritics sanity check failed: {type(exc).__name__}: {exc}")


def print_tokenizer_diacritics_support(tokenizer) -> None:
    """Checks whether tokenizer can encode+decode Serbian diacritics."""
    diacritics = "čćšžđČĆŠŽĐ"
    try:
        print("\n[Tokenizer sanity] Diacritics round-trip")
        bad: list[str] = []
        for ch in diacritics:
            ids = tokenizer.encode(ch, add_special_tokens=False)
            decoded = tokenizer.decode(ids, skip_special_tokens=True)
            ok = (decoded == ch)
            print(f"- {ch!r} -> ids={ids} -> {decoded!r} ok={ok}")
            if not ok:
                bad.append(ch)
        if bad:
            print("[WARN] Tokenizer cannot round-trip these diacritics:", "".join(bad))
            print("[WARN] With this tokenizer/model family, outputs may stay ASCII.")
            print("[Hint] Consider switching base model to 'google/mt5-small' for better multilingual character coverage.")
    except Exception as exc:
        print(f"\n[WARN] Tokenizer diacritics check failed: {type(exc).__name__}: {exc}")


def print_tokenized_label_diacritics(tokenized, tokenizer, sample_size: int = 200) -> None:
    """Checks whether diacritics survive tokenization in labels."""
    diacritics = "čćšžđČĆŠŽĐ"
    try:
        ds = tokenized["train"]
        n = len(ds)
        if n <= 0:
            return
        k = min(int(sample_size), n)
        subset = ds.select(range(k)) if k < n else ds
        decoded = tokenizer.batch_decode(subset["labels"], skip_special_tokens=True)
        any_diacritics = any(any(ch in s for ch in diacritics) for s in decoded if isinstance(s, str))
        print("\n[Tokenization sanity] Diacritics in decoded tokenized labels")
        print("- checked_rows:", len(decoded))
        print("- any_diacritics:", any_diacritics)
        if not any_diacritics:
            print("[WARN] Diacritics are being lost during tokenization/decoding.")
            print("[WARN] If dataset had diacritics, this points to tokenizer/model limitations.")
        else:
            shown = 0
            for s in decoded:
                if not isinstance(s, str):
                    continue
                if any(ch in s for ch in diacritics):
                    print("- example:", s[:240])
                    shown += 1
                if shown >= 3:
                    break
    except Exception as exc:
        print(f"\n[WARN] Tokenized-label diacritics check failed: {type(exc).__name__}: {exc}")


def find_latest_checkpoint(out_dir: Path) -> str | None:
    """Baseline helper: find latest checkpoint-N folder in output directory."""
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


def output_dir() -> Path:
    model_slug = safe_slug(CONFIG["model_name"])
    family = "mt5" if "mt5" in str(CONFIG["model_name"]).lower() else "t5"

    valid_ratio = float(CONFIG.get("validation_split", 0.2))
    test_ratio = float(CONFIG.get("test_split", 0.1))
    train_ratio = 1.0 - valid_ratio - test_ratio
    split_seed = int(CONFIG.get("split_seed", 42))

    mix = CONFIG.get("mix") or {}
    e = float(mix.get("eng_to_sr_fraction", 1.0))
    s = float(mix.get("sr_to_eng_fraction", 1.0))
    sample_mode = str(mix.get("sample_mode", "random"))
    sample_seed = int(mix.get("sample_seed", 42))

    def _pct(x: float) -> int:
        return int(round(float(x) * 100.0))

    split_sig = f"s{_pct(train_ratio)}v{_pct(valid_ratio)}t{_pct(test_ratio)}_seed{split_seed}"
    mix_sig = f"mix_e{_pct(e)}_s{_pct(s)}__{safe_slug(sample_mode)}__seed{sample_seed}"

    return project_root() / CONFIG["models_dir"] / f"{family}_translation_model__bidirectional__{model_slug}__{mix_sig}__{split_sig}"


def tokenized_cache_path() -> Path:
    model_slug = safe_slug(CONFIG["model_name"])

    eng_fp = data_path("eng_to_sr.csv")
    sr_fp = data_path("sr_to_eng.csv")

    fingerprint = "unknown"
    try:
        fp_a = file_fingerprint(eng_fp)
        fp_b = file_fingerprint(sr_fp)
        mix = CONFIG.get("mix") or {}
        e = float(mix.get("eng_to_sr_fraction", 1.0))
        s = float(mix.get("sr_to_eng_fraction", 1.0))
        sample_mode = str(mix.get("sample_mode", "random"))
        sample_seed = int(mix.get("sample_seed", 42))
        fingerprint = f"eng__{fp_a}__sr__{fp_b}__e{e:.4f}__s{s:.4f}__{sample_mode}__seed{sample_seed}"
    except Exception:
        pass

    valid_ratio = float(CONFIG.get("validation_split", 0.2))
    test_ratio = float(CONFIG.get("test_split", 0.1))
    train_ratio = 1.0 - valid_ratio - test_ratio
    split_seed = int(CONFIG.get("split_seed", 42))

    def _pct(x: float) -> int:
        return int(round(float(x) * 100.0))

    split_sig = f"s{_pct(train_ratio)}v{_pct(valid_ratio)}t{_pct(test_ratio)}_seed{split_seed}"
    return (
        project_root()
        / CONFIG["cache_dir"]
        / "tokenized_datasets"
        / f"bidirectional__{model_slug}__{safe_slug(fingerprint)}__{split_sig}__in{CONFIG['max_input_length']}__out{CONFIG['max_target_length']}"
    )


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
    print("- eng_to_sr:", data_path("eng_to_sr.csv"))
    print("- sr_to_eng:", data_path("sr_to_eng.csv"))
    print("- output_dir:", output_dir())
    print("- tokenized_cache:", tokenized_cache_path())


def ensure_packages() -> None:
    def _try_import() -> bool:
        try:
            import transformers  # noqa: F401
            import datasets  # noqa: F401
            import evaluate  # noqa: F401
            import sacrebleu  # noqa: F401
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
        "evaluate==0.4.6",
        "sacrebleu==2.5.1",
        "sentencepiece==0.2.0",
        "accelerate==1.4.0",
        "protobuf>=4.25.0",
    ]

    import subprocess

    cmd = [sys.executable, "-m", "pip", "install", "-q"] + pkgs
    subprocess.check_call(cmd)


# ---------------------------
# Strategy A dataset building
# ---------------------------

def _validate_fraction(name: str, value: float) -> float:
    try:
        f = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a float in [0,1]. Got {value!r}") from exc
    if not (0.0 <= f <= 1.0):
        raise ValueError(f"{name} must be in [0,1]. Got {f}")
    return f


def _subsample(dataset, frac: float, *, mode: str, seed: int, label: str):
    """Return a subsampled view of `dataset` without materializing to disk."""
    if frac >= 1.0:
        return dataset
    n = len(dataset)
    k = int(math.floor(n * frac))
    k = max(0, min(n, k))
    if k == n:
        return dataset
    if k == 0:
        # keep an empty dataset with same schema
        return dataset.select([])

    mode = str(mode)
    if mode == "deterministic_head":
        return dataset.select(range(k))
    if mode == "random":
        # shuffle is deterministic given seed; select first k
        return dataset.shuffle(seed=int(seed)).select(range(k))

    raise ValueError(f"Unknown sample_mode={mode!r}. Use 'random' or 'deterministic_head'.")


def build_bidirectional_dataset(load_dataset_fn):
    """Load both CSVs, optionally subsample each direction, add `direction`, and concatenate."""
    from datasets import concatenate_datasets

    eng_path = data_path("eng_to_sr.csv")
    sr_path = data_path("sr_to_eng.csv")
    if not eng_path.exists():
        raise FileNotFoundError(f"Missing dataset: {eng_path}")
    if not sr_path.exists():
        raise FileNotFoundError(f"Missing dataset: {sr_path}")

    mix = CONFIG.get("mix") or {}
    eng_frac = _validate_fraction("mix.eng_to_sr_fraction", mix.get("eng_to_sr_fraction", 1.0))
    sr_frac = _validate_fraction("mix.sr_to_eng_fraction", mix.get("sr_to_eng_fraction", 1.0))
    sample_mode = str(mix.get("sample_mode", "random"))
    sample_seed = int(mix.get("sample_seed", 42))

    print(f"\n[Strategy A] Loading: {eng_path}")
    ds_eng = load_dataset_fn("csv", data_files=str(eng_path))["train"]
    print(f"[Strategy A] Loading: {sr_path}")
    ds_sr = load_dataset_fn("csv", data_files=str(sr_path))["train"]

    print(f"\n[Strategy A] Base sizes: eng_to_sr={len(ds_eng)}, sr_to_eng={len(ds_sr)}")

    ds_eng = _subsample(ds_eng, eng_frac, mode=sample_mode, seed=sample_seed, label="eng_to_sr")
    ds_sr = _subsample(ds_sr, sr_frac, mode=sample_mode, seed=sample_seed, label="sr_to_eng")

    ds_eng = ds_eng.add_column("direction", ["eng_to_sr"] * len(ds_eng))
    ds_sr = ds_sr.add_column("direction", ["sr_to_eng"] * len(ds_sr))

    combined = concatenate_datasets([ds_eng, ds_sr])

    # Shuffle once after concat so train/valid/test mixture is well distributed.
    combined = combined.shuffle(seed=sample_seed)

    print(f"[Strategy A] After mixing: combined={len(combined)} (eng_frac={eng_frac}, sr_frac={sr_frac}, mode={sample_mode})")
    return combined


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
    import evaluate
    import sacrebleu
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrainerCallback,
    )

    class SaveTokenizerToCheckpointCallback(TrainerCallback):
        """Save tokenizer into each checkpoint folder (baseline behavior)."""

        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def on_save(self, args, state, control, **kwargs):  # noqa: ANN001
            try:
                ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                if ckpt_dir.exists():
                    self.tokenizer.save_pretrained(str(ckpt_dir))
            except Exception as exc:
                print(f"\n[WARN] Could not save tokenizer into checkpoint: {type(exc).__name__}: {exc}")
            return control

    print_paths()

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

    # Load and build combined dataset
    dataset = build_bidirectional_dataset(load_dataset)

    # Baseline-style sanity checks (helpful to catch ASCII-only labels/tokenizer issues)
    try:
        print_diacritics_sanity(dataset, column="target")
    except Exception:
        pass

    # ---------------------------
    # Fixed split: train/valid/test (70/20/10 default)
    # ---------------------------
    valid_ratio = float(CONFIG.get("validation_split", 0.2))
    test_ratio = float(CONFIG.get("test_split", 0.1))
    if valid_ratio < 0 or test_ratio < 0 or (valid_ratio + test_ratio) >= 1.0:
        raise ValueError(
            f"Invalid split ratios: validation_split={valid_ratio}, test_split={test_ratio}. "
            "Need validation_split + test_split < 1.0"
        )

    split_seed = int(CONFIG.get("split_seed", 42))
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

    try:
        print_tokenizer_diacritics_support(tokenizer)
    except Exception:
        pass

    if bool(CONFIG.get("gradient_checkpointing")):
        try:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
            print("\n[Info] Enabled gradient checkpointing (use_cache=False)")
        except Exception as exc:
            print(f"\n[WARN] Could not enable gradient checkpointing: {type(exc).__name__}: {exc}")

    # Tokenize (cached)
    cache_path = tokenized_cache_path()
    ensure_dir(cache_path.parent)

    prefix_map = {
        "eng_to_sr": "translate English to Serbian: ",
        "sr_to_eng": "translate Serbian to English: ",
    }

    def preprocess(examples: dict) -> dict:
        directions = examples["direction"]
        inputs = [prefix_map[d] + t for d, t in zip(directions, examples["source"])]
        targets = examples["target"]

        model_inputs = tokenizer(
            inputs,
            max_length=CONFIG["max_input_length"],
            truncation=True,
            padding=False,
        )
        target_tokens = tokenizer(
            targets,
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

    try:
        print_tokenized_label_diacritics(tokenized, tokenizer)
    except Exception:
        pass

    # Metrics (identical to baseline)
    metric_bleu = evaluate.load("sacrebleu")
    chrfpp = sacrebleu.metrics.CHRF(word_order=2)

    diacritics = "čćšžđČĆŠŽĐ"

    def _diacritics_scores(preds: list[str], refs: list[str]) -> dict[str, float]:
        pred_counts = {ch: 0 for ch in diacritics}
        ref_counts = {ch: 0 for ch in diacritics}
        any_rate_n = 0
        for p in preds:
            if any(ch in p for ch in diacritics):
                any_rate_n += 1
            for ch in diacritics:
                pred_counts[ch] += p.count(ch)
        for r in refs:
            for ch in diacritics:
                ref_counts[ch] += r.count(ch)
        tp = sum(min(pred_counts[ch], ref_counts[ch]) for ch in diacritics)
        fp = sum(max(0, pred_counts[ch] - ref_counts[ch]) for ch in diacritics)
        fn = sum(max(0, ref_counts[ch] - pred_counts[ch]) for ch in diacritics)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        any_rate = any_rate_n / max(1, len(preds))
        return {
            "diacritics_precision": float(precision),
            "diacritics_recall": float(recall),
            "diacritics_f1": float(f1),
            "diacritics_any_rate": float(any_rate),
        }

    def compute_metrics(eval_preds):
        if hasattr(eval_preds, "predictions") and hasattr(eval_preds, "label_ids"):
            predictions, labels = eval_preds.predictions, eval_preds.label_ids
        else:
            predictions, labels = eval_preds

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        try:
            if getattr(predictions, "ndim", 0) == 3:
                predictions = predictions.argmax(-1)
        except Exception:
            pass

        preds = np.asarray(predictions)

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else pad_id
        vocab_size = getattr(tokenizer, "vocab_size", None)

        preds = np.where(preds < 0, pad_id, preds)
        if isinstance(vocab_size, int) and vocab_size > 0:
            preds = np.where(preds >= vocab_size, unk_id, preds)
        preds = preds.astype(np.int64, copy=False)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.asarray(labels)
        labels = np.where(labels != -100, labels, pad_id).astype(np.int64, copy=False)
        if isinstance(vocab_size, int) and vocab_size > 0:
            labels = np.where(labels >= vocab_size, unk_id, labels)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_labels = [[label] for label in decoded_labels]

        refs_flat = [r[0] for r in decoded_labels]

        metrics_out: dict[str, float] = {}
        try:
            result = metric_bleu.compute(predictions=decoded_preds, references=decoded_labels)
            metrics_out["bleu"] = float(result["score"])
        except Exception as exc:
            print(f"\n[WARN] BLEU computation failed: {type(exc).__name__}: {exc}")
            metrics_out["bleu"] = 0.0

        try:
            metrics_out["chrfpp"] = float(chrfpp.corpus_score(decoded_preds, [refs_flat]).score)
        except Exception as exc:
            print(f"\n[WARN] chrF++ computation failed: {type(exc).__name__}: {exc}")
            metrics_out["chrfpp"] = 0.0

        # NOTE: For bidirectional runs, diacritics metrics are meaningful mostly
        # for EN->SR examples, but we keep them identical for top-level comparability.
        try:
            metrics_out.update(_diacritics_scores(decoded_preds, refs_flat))
        except Exception as exc:
            print(f"\n[WARN] Diacritics metric computation failed: {type(exc).__name__}: {exc}")
            metrics_out.update({
                "diacritics_precision": 0.0,
                "diacritics_recall": 0.0,
                "diacritics_f1": 0.0,
                "diacritics_any_rate": 0.0,
            })

        return metrics_out

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
        label_pad_token_id=-100,
    )

    eval_dataset = tokenized["validation"]
    subset = CONFIG.get("eval_subset_size")
    if isinstance(subset, int) and 0 < subset < len(eval_dataset):
        print(f"\nEval subset: {subset} / {len(eval_dataset)}")
        eval_dataset = eval_dataset.select(range(subset))

    out_dir = output_dir()
    ensure_dir(out_dir)

    # Fingerprint includes both source CSVs + split + lengths + mix config
    valid_ratio = float(CONFIG.get("validation_split", 0.2))
    test_ratio = float(CONFIG.get("test_split", 0.1))
    train_ratio = 1.0 - valid_ratio - test_ratio
    split_seed = int(CONFIG.get("split_seed", 42))
    mix = CONFIG.get("mix") or {}
    mix_sig = json.dumps(mix, sort_keys=True)

    def _pct(x: float) -> int:
        return int(round(float(x) * 100.0))

    split_sig = f"s{_pct(train_ratio)}v{_pct(valid_ratio)}t{_pct(test_ratio)}_seed{split_seed}"
    current_fp = (
        f"eng__{file_fingerprint(data_path('eng_to_sr.csv'))}"
        f"__sr__{file_fingerprint(data_path('sr_to_eng.csv'))}"
        f"__mix__{hashlib.sha256(mix_sig.encode('utf-8')).hexdigest()[:12]}"
        f"__{split_sig}"
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
        predict_with_generate=True,
        generation_max_length=CONFIG["max_target_length"],
        generation_num_beams=CONFIG.get("generation_num_beams", 1),
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
                print("\n[FATAL] loss is NaN/Inf; stopping training.")
                control.should_training_stop = True
            if grad_norm is not None and _bad_number(grad_norm):
                print("\n[FATAL] grad_norm is NaN/Inf; stopping training.")
                control.should_training_stop = True
            return control

    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    callbacks = [NanGuardCallback(), SaveTokenizerToCheckpointCallback(tokenizer)]
    if bool(CONFIG.get("early_stopping", True)):
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=int(CONFIG.get("early_stopping_patience", 8)),
                early_stopping_threshold=float(CONFIG.get("early_stopping_threshold", 0.0)),
            )
        )

    # Newer Transformers deprecates `tokenizer=` in favor of `processing_class=`.
    trainer_sig = inspect.signature(Seq2SeqTrainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Seq2SeqTrainer(**trainer_kwargs)
    for cb in callbacks:
        trainer.add_callback(cb)

    resume = None
    if not bool(CONFIG.get("force_fresh_train")):
        resume = find_latest_checkpoint(out_dir)

    if resume and previous_fp and previous_fp != current_fp:
        print("\n[Info] Dataset CSVs/mix changed since last run; not resuming from old checkpoints.")
        print("- previous:", previous_fp)
        print("- current :", current_fp)
        resume = None

    print("\nTraining...")
    train_started_at = time.time()
    if resume:
        print("Resuming from:", resume)
        train_result = trainer.train(resume_from_checkpoint=resume)
    else:
        train_result = trainer.train()
    train_elapsed_sec = float(time.time() - train_started_at)
    # Save final model
    print("\nSaving final model:", out_dir)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # ---------------------------
    # Final test evaluation + results.json (baseline schema)
    # ---------------------------
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

    decoding = {
        "generation_max_length": int(CONFIG["max_target_length"]),
        "generation_num_beams": int(CONFIG.get("generation_num_beams", 1)),
    }

    results = {
        "model_name": str(CONFIG["model_name"]),
        "translation_direction": "bidirectional",
        "output_dir": str(out_dir),
        "dataset_csv": None,
        "dataset_fingerprint": str(current_fp),
        "resume_from": str(resume) if resume else None,
        "split": {
            "train": 1.0 - float(CONFIG.get("validation_split", 0.2)) - float(CONFIG.get("test_split", 0.1)),
            "validation": float(CONFIG.get("validation_split", 0.2)),
            "test": float(CONFIG.get("test_split", 0.1)),
        },
        "training": {
            "global_step": int(getattr(state, "global_step", 0) or 0),
            "epoch": float(getattr(state, "epoch", 0.0) or 0.0),
            "train_runtime_sec": train_elapsed_sec,
            "best_eval_loss": best_eval_loss,
            "best_model_checkpoint": str(getattr(state, "best_model_checkpoint", None)),
        },
        "decoding": decoding,
        "test_metrics": test_metrics,
        "strategy_a": {
            "mix": CONFIG.get("mix"),
            "tokenized_cache": str(cache_path),
        },
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

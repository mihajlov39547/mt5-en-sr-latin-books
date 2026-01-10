"""\
Colab T5 fine-tuning script (English ↔ Serbian)

How to use in Google Colab:

1) Mount Drive in a notebook cell:
   from google.colab import drive
   drive.mount('/content/drive')

0) In Colab, set: Runtime → Change runtime type → Hardware accelerator → GPU
    (This script is intended for NVIDIA GPUs like the T4.)

2) Put your CSV(s) in Google Drive:
   /content/drive/MyDrive/T5/data/eng_to_sr.csv
   /content/drive/MyDrive/T5/data/sr_to_eng.csv

   CSV format must have columns:
     - source
     - target

3) Run:
   !python /content/drive/MyDrive/T5/colab_train_t5.py

This script will:
- Cache tokenized datasets to Drive (so reruns skip tokenization)
- Save checkpoints + final model to Drive
- Auto-resume from the latest checkpoint

Notes:
- Disables TorchVision integration in Transformers to avoid optional vision import issues.
- Uses Seq2SeqTrainer + predict_with_generate for BLEU.
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

# Helps reduce CUDA memory fragmentation on long runs (safe no-op if unsupported).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Transformers may try to auto-detect TensorFlow if it's installed in the runtime.
# On Colab this often causes noisy TensorFlow/XLA CUDA plugin registration logs.
# This script is PyTorch-only, so explicitly disable TF integration early.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

# If TensorFlow still gets imported by something else, keep its native logs quieter.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

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
    # NOTE: mT5 is multilingual and handles Serbian diacritics much better than English-only T5.
    "model_name": "google/mt5-small",

    # Direction: "eng_to_sr" or "sr_to_eng"
    "translation_direction": "eng_to_sr",

    # Training
    # NOTE: Instead of relying on a fixed epoch count, we set a *high* upper bound
    # and enable early-stopping on validation loss.
    "num_train_epochs": 10,
    # If you see loss=0.0 and grad_norm=nan, the run has diverged; start with a safer LR.
    "learning_rate": 1e-4,
    # mT5-small on a Colab T4 (16GB) often OOMs at batch_size=16 with 256/256 lengths.
    # Keep the *effective* batch about the same via gradient accumulation.
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "max_input_length": 256,
    "max_target_length": 256,
    # Dataset split ratios (must sum to <= 1.0)
    # Requested default: 70/20/10 train/valid/test.
    "validation_split": 0.2,
    "test_split": 0.1,
    # Split seed for deterministic 70/20/10 split (affects which exact rows end up in test).
    "split_seed": 42,

    # Memory optimizations
    # - Gradient checkpointing reduces activation memory at the cost of some speed.
    # - AdamW is typically more stable than Adafactor; if you hit OOM, switch back to "adafactor".
    "optim": "adamw_torch",
    "gradient_checkpointing": True,

    # Stability
    "max_grad_norm": 1.0,
    "warmup_ratio": 0.03,

    # Checkpointing
    # IMPORTANT: Save checkpoints by steps so you can resume even if epoch-end BLEU eval crashes.
    # (Transformers' default order at epoch end is: evaluate → compute_metrics → save.)
    "save_strategy": "steps",   # "steps" (recommended) or "epoch"
    "save_steps": 1000,         # only used when save_strategy == "steps"

    # Safety toggles
    # If True: never resume from existing checkpoints (starts from base model each run)
    "force_fresh_train": False,
    # If True: ignore tokenized cache and re-tokenize dataset
    "force_retokenize": False,

    # Evaluation scheduling
    # IMPORTANT: If load_best_model_at_end=True, Transformers requires eval/save strategies to match.
    # When using step-based scheduling with load_best_model_at_end=True, Transformers also requires:
    #   save_steps % eval_steps == 0
    # Easiest safe default: eval_steps == save_steps.
    "eval_strategy": "steps",   # must match save_strategy when load_best_model_at_end=True
    "eval_steps": 1000,          # only used when eval_strategy == "steps"

    # Eval speed
    "eval_subset_size": 500,  # set None for full validation
    "generation_num_beams": 1,

    # Early stopping (runs on eval metric)
    # With metric_for_best_model="eval_loss", this stops when validation loss stops improving.
    "early_stopping": True,
    "early_stopping_patience": 4,      # number of evals with no improvement before stopping
    "early_stopping_threshold": 0.001,   # minimum improvement to be considered "better"

    # Mixed precision
    # fp16 can diverge depending on runtime/library versions; start stable.
    # bf16 is more stable on A100/H100 (Ampere+); auto-tuning will enable it if supported.
    "use_fp16": False,
    "use_bf16": False,

    # GPU auto-tuning
    # If True, automatically adjust batch sizes and precision based on detected GPU.
    # Set False to use the manual CONFIG values above.
    "auto_tune_gpu": True,
}


# ---------------------------
# Helpers
# ---------------------------

def detect_and_tune_for_gpu() -> dict[str, any]:
    """
    Detect GPU type and return optimized training parameters.
    Returns a dict of CONFIG overrides to apply.
    """
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

    # Default (conservative) - will be overridden below
    overrides: dict[str, any] = {}

    # -----------------------------
    # Tesla T4 (~15 GB, no bf16)
    # -----------------------------
    if "t4" in gpu_name:
        overrides = {
            "train_batch_size": 8,
            "eval_batch_size": 8,
            "gradient_accumulation_steps": 2,  # effective batch = 16
            "use_fp16": False,  # fp16 can be unstable on T4 for this model
            "use_bf16": False,
            "gradient_checkpointing": True,
            "eval_subset_size": 500,
        }
        print("[GPU Auto-Tune] Profile: T4 (conservative, ~11GB usage)")

    # -----------------------------
    # NVIDIA A100 (40 or 80 GB, bf16 supported)
    # -----------------------------
    elif "a100" in gpu_name:
        if gpu_mem_gb >= 70:  # 80GB variant
            overrides = {
                "train_batch_size": 64,
                "eval_batch_size": 64,
                "gradient_accumulation_steps": 1,  # effective batch = 64
                "use_fp16": False,
                "use_bf16": bf16_supported,
                "gradient_checkpointing": False,  # enough memory, skip for speed
                "eval_subset_size": 2000,
                "generation_num_beams": 4,
            }
            print("[GPU Auto-Tune] Profile: A100-80GB (high throughput)")
        else:  # 40GB variant
            overrides = {
                "train_batch_size": 32,
                "eval_batch_size": 32,
                "gradient_accumulation_steps": 1,  # effective batch = 32
                "use_fp16": False,
                "use_bf16": bf16_supported,
                "gradient_checkpointing": False,
                "eval_subset_size": 1000,
                "generation_num_beams": 2,
            }
            print("[GPU Auto-Tune] Profile: A100-40GB (balanced)")

    # -----------------------------
    # NVIDIA H100 (80 or 94 GB, bf16/fp8 supported)
    # -----------------------------
    elif "h100" in gpu_name:
        overrides = {
            "train_batch_size": 64,
            "eval_batch_size": 64,
            "gradient_accumulation_steps": 1,  # effective batch = 64
            "use_fp16": False,
            "use_bf16": bf16_supported,  # H100 excels at bf16
            "gradient_checkpointing": False,  # plenty of memory
            "eval_subset_size": 2000,
            "generation_num_beams": 4,
        }
        print("[GPU Auto-Tune] Profile: H100 (maximum throughput)")

    # -----------------------------
    # Other high-memory GPUs (V100, A10, L4, etc.)
    # -----------------------------
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
        # Small GPU / unknown - use conservative T4-like settings
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
    # When this script is executed via `!python ...` in Colab, it's a new process,
    # so `google.colab` may NOT be imported. Colab sets environment variables though.
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_BACKEND_VERSION"):
        return True
    if os.environ.get("COLAB_GPU") is not None:
        return True
    # Fallback: common filesystem marker
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


def output_dir() -> Path:
    model_slug = safe_slug(CONFIG["model_name"])
    direction = safe_slug(CONFIG["translation_direction"])
    # Keep T5 and mT5 runs in separate folders to avoid accidental checkpoint reuse.
    family = "mt5" if "mt5" in str(CONFIG["model_name"]).lower() else "t5"
    valid_ratio = float(CONFIG.get("validation_split", 0.2))
    test_ratio = float(CONFIG.get("test_split", 0.1))
    train_ratio = 1.0 - valid_ratio - test_ratio
    split_seed = int(CONFIG.get("split_seed", 42))

    def _pct(x: float) -> int:
        return int(round(float(x) * 100.0))

    split_sig = f"s{_pct(train_ratio)}v{_pct(valid_ratio)}t{_pct(test_ratio)}_seed{split_seed}"
    return project_root() / CONFIG["models_dir"] / f"{family}_translation_model__{direction}__{model_slug}__{split_sig}"


def tokenized_cache_path() -> Path:
    model_slug = safe_slug(CONFIG["model_name"])
    direction = safe_slug(CONFIG["translation_direction"])
    # IMPORTANT: cache must change when the input CSV changes.
    # Otherwise you can silently keep training on an old tokenized dataset.
    csv_name = "eng_to_sr.csv" if CONFIG["translation_direction"] == "eng_to_sr" else "sr_to_eng.csv"
    csv_fp = data_path(csv_name)

    fingerprint = "unknown"
    try:
        st = csv_fp.stat()
        size = int(st.st_size)
        mtime = int(st.st_mtime)
        h = hashlib.sha256()
        with csv_fp.open("rb") as f:
            h.update(f.read(1024 * 1024))
        fingerprint = f"sz{size}__mt{mtime}__h{h.hexdigest()[:12]}"
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
        / f"{direction}__{model_slug}__{fingerprint}__{split_sig}__in{CONFIG['max_input_length']}__out{CONFIG['max_target_length']}"
    )


def dataset_csv_path() -> Path:
    csv_name = "eng_to_sr.csv" if CONFIG["translation_direction"] == "eng_to_sr" else "sr_to_eng.csv"
    return data_path(csv_name)


def file_fingerprint(path: Path) -> str:
    """Stable-enough fingerprint to detect data changes without hashing multi-GB files."""
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
            # show a couple examples
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
    print("- eng_to_sr:", data_path("eng_to_sr.csv"))
    print("- sr_to_eng:", data_path("sr_to_eng.csv"))
    print("- output_dir:", output_dir())
    print("- tokenized_cache:", tokenized_cache_path())


def ensure_packages() -> None:
    """Optional: install deps if missing (handy for fresh Colab runtimes)."""

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
    # Keep this list small and stable.
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

    print_paths()

    # Pick device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"\nCUDA: {torch.cuda.get_device_name(0)}")

        # Auto-tune parameters based on detected GPU
        if bool(CONFIG.get("auto_tune_gpu", True)):
            gpu_overrides = detect_and_tune_for_gpu()
            for key, value in gpu_overrides.items():
                CONFIG[key] = value
    else:
        device = "cpu"
        print("\nNo CUDA detected; using CPU")

    # fp16/bf16 only makes sense on CUDA
    if device != "cuda":
        CONFIG["use_fp16"] = False
        CONFIG["use_bf16"] = False

    # Load and split dataset
    if CONFIG["translation_direction"] == "eng_to_sr":
        csv_path = dataset_csv_path()
        task_prefix = "translate English to Serbian: "
    else:
        csv_path = dataset_csv_path()
        task_prefix = "translate Serbian to English: "

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing dataset: {csv_path}\n"
            "Put your CSV in Drive at the path shown above."
        )

    print(f"\nLoading dataset: {csv_path}")
    dataset = load_dataset("csv", data_files=str(csv_path))["train"]

    # Sanity-check that the actually loaded CSV contains diacritics.
    # (If you previously trained on a cache/old file, this catches it early.)
    print_diacritics_sanity(dataset, column="target")

    # ---------------------------
    # Fixed split: train/valid/test
    # Default requested: 70/20/10
    # ---------------------------
    valid_ratio = float(CONFIG.get("validation_split", 0.2))
    test_ratio = float(CONFIG.get("test_split", 0.1))
    if valid_ratio < 0 or test_ratio < 0 or (valid_ratio + test_ratio) >= 1.0:
        raise ValueError(
            f"Invalid split ratios: validation_split={valid_ratio}, test_split={test_ratio}. "
            "Need validation_split + test_split < 1.0"
        )

    # First split off test, then split remaining into train/valid to preserve exact overall ratios.
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
    # Use the tokenizer that matches the model.
    # For SentencePiece-based models (T5/mT5), use_fast=False avoids "slow->fast" conversion warnings.
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"], use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["model_name"])

    # Optional memory saver for T5/mT5: checkpoint activations.
    if bool(CONFIG.get("gradient_checkpointing")):
        try:
            model.gradient_checkpointing_enable()
            # Required when using gradient checkpointing on (m)T5.
            model.config.use_cache = False
            print("\n[Info] Enabled gradient checkpointing (use_cache=False)")
        except Exception as exc:
            print(f"\n[WARN] Could not enable gradient checkpointing: {type(exc).__name__}: {exc}")

    # Confirm tokenizer can represent Serbian diacritics at all.
    print_tokenizer_diacritics_support(tokenizer)

    # Tokenize (cached)
    cache_path = tokenized_cache_path()
    ensure_dir(cache_path.parent)

    def preprocess(examples: dict) -> dict:
        inputs = [task_prefix + t for t in examples["source"]]
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

    # Confirm diacritics survive through tokenization into labels.
    print_tokenized_label_diacritics(tokenized, tokenizer)

    # Metrics
    metric_bleu = evaluate.load("sacrebleu")
    chrfpp = sacrebleu.metrics.CHRF(word_order=2)

    diacritics = "čćšžđČĆŠŽĐ"

    def _diacritics_scores(preds: list[str], refs: list[str]) -> dict[str, float]:
        # Bag-of-characters F1 over Serbian diacritics.
        # This is not a full alignment metric, but it directly captures the
        # "ASCII-only" failure mode.
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
        # `eval_preds` can be a tuple or an EvalPrediction-like object.
        if hasattr(eval_preds, "predictions") and hasattr(eval_preds, "label_ids"):
            predictions, labels = eval_preds.predictions, eval_preds.label_ids
        else:
            predictions, labels = eval_preds

        # Seq2SeqTrainer may return a tuple for predictions.
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # If predictions are logits (batch, seq, vocab), convert to token ids.
        try:
            if getattr(predictions, "ndim", 0) == 3:
                predictions = predictions.argmax(-1)
        except Exception:
            pass

        preds = np.asarray(predictions)

        # Guard against invalid token ids (e.g., -100 or out-of-range) which can crash SentencePiece.
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

        # Flatten references for metrics that want 1 ref per segment.
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

    # Prevent accidental resume from a checkpoint trained on a different dataset version.
    valid_ratio = float(CONFIG.get("validation_split", 0.2))
    test_ratio = float(CONFIG.get("test_split", 0.1))
    train_ratio = 1.0 - valid_ratio - test_ratio
    split_seed = int(CONFIG.get("split_seed", 42))

    def _pct(x: float) -> int:
        return int(round(float(x) * 100.0))

    split_sig = f"s{_pct(train_ratio)}v{_pct(valid_ratio)}t{_pct(test_ratio)}_seed{split_seed}"
    current_fp = (
        f"{file_fingerprint(csv_path)}__{split_sig}"
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
        # Train "until loss stops improving": pick best checkpoint by eval_loss
        # and allow early stopping to use the same metric.
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

    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
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
    
    # Newer Transformers deprecates `tokenizer=` in favor of `processing_class=`.
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
        print("\n[Info] Dataset CSV changed since last run; not resuming from old checkpoints.")
        print("- previous:", previous_fp)
        print("- current :", current_fp)
        resume = None
    print("\nTraining...")
    train_started_at = time.time()
    if resume:
        print("Resuming from:", resume)
        print(
            "[Info] If you see a warning about missing keys like "
            "'encoder.embed_tokens.weight'/'decoder.embed_tokens.weight', "
            "it's typically expected for (m)T5 because encoder/decoder embeddings are tied to 'shared.weight'."
        )
        trainer.train(resume_from_checkpoint=resume)
    else:
        trainer.train()

    train_elapsed_sec = float(time.time() - train_started_at)

    # Save final model
    print("\nSaving final model:", out_dir)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # ---------------------------
    # Final test evaluation + results.json
    # ---------------------------
    test_dataset = tokenized.get("test")
    test_metrics: dict[str, float] = {}
    if test_dataset is not None and len(test_dataset) > 0:
        print("\nEvaluating on test split...")
        pred = trainer.predict(test_dataset=test_dataset)
        # Transformers prefixes metrics with "test_" when called from predict.
        test_metrics = {k: float(v) for k, v in pred.metrics.items() if isinstance(v, (int, float))}

    # Collect run summary requested for your paper table
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
        "translation_direction": str(CONFIG["translation_direction"]),
        "output_dir": str(out_dir),
        "dataset_csv": str(csv_path),
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

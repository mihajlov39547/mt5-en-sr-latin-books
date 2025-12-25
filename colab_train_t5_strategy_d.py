"""\
Strategy D: Single-stage mixed training = Serbian denoising + EN→SR translation

Goal
----
Assumption: `google/mt5-small` already has strong English knowledge, but needs
additional Serbian capability.

Strategy D trains in ONE run by mixing two kinds of examples:

1) Serbian denoising (T5 span corruption) on monolingual Serbian text
   - source: `data/serbian_corpus.csv` (column: `text`)
    - task/prefix: (none; matches Strategy B CPT)

2) Supervised EN→SR translation
   - source: `data/eng_to_sr.csv` (columns: `source`, `target`)
   - task/prefix: "translate English to Serbian: "

The dataset is concatenated and shuffled so both objectives are learned together.
This is conceptually similar to Strategy A being “single-stage”, but instead of
mixing 2 translation directions, we mix (denoise SR) + (translate EN→SR).

Run in Colab
------------
1) Mount Drive:
   from google.colab import drive
   drive.mount('/content/drive')

2) Run:
   !python /content/drive/MyDrive/T5/colab_train_t5_strategy_d.py

Notes
-----
- Split (70/20/10) is determined from a stable *keyed split* so the supervised
  translation rows keep a deterministic test set independent of how many
  denoising-only rows you include.
- Metrics (BLEU/chrF++/diacritics) are computed only on the EN→SR translation
  subset of validation/test.
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

    # For naming/results
    "training_task": "strategy_d_mixed_denoise_sr_plus_translate_en_sr",

    # Training
    "num_train_epochs": 10,
    "learning_rate": 1e-4,
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "max_input_length": 256,
    "max_target_length": 256,

    # Dataset split ratios (70/20/10)
    "validation_split": 0.2,
    "test_split": 0.1,
    # Distinct seed for strategy D family
    "split_seed": 82,

    # Strategy D mixing
    "mix": {
        # Fraction of Serbian denoising examples to keep (1.0 = all)
        "sr_denoise_fraction": 1.0,
        # Fraction of EN→SR supervised examples to keep (1.0 = all)
        "en_to_sr_fraction": 1.0,
        # One of: "random" or "deterministic_head"
        "sample_mode": "random",
        "sample_seed": 82,
        # Optional caps (None disables)
        "sr_denoise_cap": None,
        "en_to_sr_cap": None,
    },

    # Denoising / corruption params
    "noise_density": 0.15,
    "mean_noise_span_length": 3.0,

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
# Helpers
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


def print_diacritics_sanity(dataset, column: str = "target", sample_size: int = 5000) -> None:
    diacritics = "čćšžđČĆŠŽĐ"
    try:
        n = len(dataset)
        if n <= 0:
            return
        k = min(int(sample_size), n)
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


# ---- Denoising helpers (adapted from Strategy B pretrainer) ----

def _random_segmentation(num_items: int, num_segments: int, *, rng: random.Random) -> list[int]:
    if num_segments <= 0:
        return [num_items]
    if num_segments >= num_items:
        return [1] * num_items

    cuts = sorted(rng.sample(range(1, num_items), num_segments - 1))
    seg_lengths: list[int] = []
    prev = 0
    for c in cuts:
        seg_lengths.append(c - prev)
        prev = c
    seg_lengths.append(num_items - prev)
    return seg_lengths


def span_corrupt_text(text: str, *, noise_density: float, mean_span_length: float, rng: random.Random) -> tuple[str, str]:
    """T5 span corruption.

    Returns (corrupted_input, target_output).
    """
    text = str(text)
    if not text:
        return "", ""

    tokens = text.split()
    n = len(tokens)
    if n == 0:
        return "", ""

    num_noise = int(round(n * float(noise_density)))
    num_noise = max(1, min(n - 1, num_noise)) if n > 1 else 1

    num_spans = max(1, int(round(num_noise / float(mean_span_length))))
    num_spans = max(1, min(num_noise, num_spans))

    noise_span_lengths = _random_segmentation(num_noise, num_spans, rng=rng)
    non_noise = n - num_noise
    non_noise_span_lengths = _random_segmentation(non_noise, num_spans + 1, rng=rng)

    input_tokens: list[str] = []
    output_tokens: list[str] = []

    token_idx = 0
    span_id = 0

    for non_len, noise_len in zip(non_noise_span_lengths, noise_span_lengths):
        if non_len > 0:
            input_tokens.extend(tokens[token_idx: token_idx + non_len])
            token_idx += non_len

        # sentinel token
        sentinel = f"<extra_id_{span_id}>"
        input_tokens.append(sentinel)
        output_tokens.append(sentinel)

        if noise_len > 0:
            output_tokens.extend(tokens[token_idx: token_idx + noise_len])
            token_idx += noise_len

        span_id += 1

    # tail non-noise
    tail_len = non_noise_span_lengths[-1]
    if tail_len > 0:
        input_tokens.extend(tokens[token_idx: token_idx + tail_len])
        token_idx += tail_len

    # EOS sentinel
    output_tokens.append(f"<extra_id_{span_id}>")

    return " ".join(input_tokens), " ".join(output_tokens)


def _validate_fraction(name: str, value: float) -> float:
    try:
        f = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a float in [0,1]. Got {value!r}") from exc
    if not (0.0 <= f <= 1.0):
        raise ValueError(f"{name} must be in [0,1]. Got {f}")
    return f


def _subsample(dataset, frac: float, *, mode: str, seed: int):
    if frac >= 1.0:
        return dataset
    n = len(dataset)
    k = int(math.floor(n * frac))
    k = max(0, min(n, k))
    if k == n:
        return dataset
    if k == 0:
        return dataset.select([])

    mode = str(mode)
    if mode == "deterministic_head":
        return dataset.select(range(k))
    if mode == "random":
        return dataset.shuffle(seed=int(seed)).select(range(k))

    raise ValueError(f"Unknown sample_mode={mode!r}. Use 'random' or 'deterministic_head'.")


def _cap_dataset(dataset, cap: int | None):
    if cap is None:
        return dataset
    try:
        cap_n = int(cap)
    except Exception:
        return dataset
    if cap_n <= 0:
        return dataset.select([])
    if cap_n >= len(dataset):
        return dataset
    return dataset.select(range(cap_n))


def output_dir() -> Path:
    family = "mt5" if "mt5" in str(CONFIG["model_name"]).lower() else "t5"
    model_slug = safe_slug(str(CONFIG["model_name"]))

    valid_ratio = float(CONFIG.get("validation_split", 0.2))
    test_ratio = float(CONFIG.get("test_split", 0.1))
    train_ratio = 1.0 - valid_ratio - test_ratio
    split_seed = int(CONFIG.get("split_seed", 82))

    mix = CONFIG.get("mix") or {}
    sr_frac = float(mix.get("sr_denoise_fraction", 1.0))
    en2sr_frac = float(mix.get("en_to_sr_fraction", 1.0))
    sample_mode = str(mix.get("sample_mode", "random"))
    sample_seed = int(mix.get("sample_seed", 82))

    noise_density = float(CONFIG.get("noise_density", 0.15))
    mean_span = float(CONFIG.get("mean_noise_span_length", 3.0))

    def _pct(x: float) -> int:
        return int(round(float(x) * 100.0))

    split_sig = f"s{_pct(train_ratio)}v{_pct(valid_ratio)}t{_pct(test_ratio)}_seed{split_seed}"
    mix_sig = f"mix_srden{sr_frac:.2f}_en2sr{en2sr_frac:.2f}__{safe_slug(sample_mode)}__seed{sample_seed}"
    corrupt_sig = f"nd{noise_density:.3f}_ms{mean_span:.2f}"

    return project_root() / CONFIG["models_dir"] / f"{family}_strategy_d__{model_slug}__{mix_sig}__{corrupt_sig}__{split_sig}"


def tokenized_cache_path() -> Path:
    model_slug = safe_slug(str(CONFIG["model_name"]))

    sr_path = data_path("serbian_corpus.csv")
    en2sr_path = data_path("eng_to_sr.csv")

    fingerprint = "unknown"
    try:
        fp_sr = file_fingerprint(sr_path)
        fp_en2sr = file_fingerprint(en2sr_path)
        mix = CONFIG.get("mix") or {}
        sr_frac = float(mix.get("sr_denoise_fraction", 1.0))
        en2sr_frac = float(mix.get("en_to_sr_fraction", 1.0))
        sample_mode = str(mix.get("sample_mode", "random"))
        sample_seed = int(mix.get("sample_seed", 82))
        fingerprint = (
            f"sr__{fp_sr}__en2sr__{fp_en2sr}"
            f"__srden{sr_frac:.4f}__en2sr{en2sr_frac:.4f}__{sample_mode}__seed{sample_seed}"
        )
    except Exception:
        pass

    valid_ratio = float(CONFIG.get("validation_split", 0.2))
    test_ratio = float(CONFIG.get("test_split", 0.1))
    train_ratio = 1.0 - valid_ratio - test_ratio
    split_seed = int(CONFIG.get("split_seed", 82))

    def _pct(x: float) -> int:
        return int(round(float(x) * 100.0))

    split_sig = f"s{_pct(train_ratio)}v{_pct(valid_ratio)}t{_pct(test_ratio)}_seed{split_seed}"
    return (
        project_root()
        / CONFIG["cache_dir"]
        / "tokenized_datasets"
        / (
            f"strategy_d__{model_slug}__{safe_slug(fingerprint)}__{split_sig}"
            f"__in{int(CONFIG['max_input_length'])}__out{int(CONFIG['max_target_length'])}"
        )
    )


def find_latest_checkpoint(out_dir: Path) -> str | None:
    if not out_dir.exists():
        return None

    best_step = -1
    best_path: Path | None = None
    for child in out_dir.iterdir():
        if not child.is_dir():
            continue
        if not child.name.startswith("checkpoint-"):
            continue
        try:
            step = int(child.name.split("-", 1)[1])
        except Exception:
            continue
        if step > best_step:
            best_step = step
            best_path = child

    return str(best_path) if best_path else None


def print_paths() -> None:
    print("\nDrive paths")
    print("- project:", project_root())
    print("- serbian_corpus:", data_path("serbian_corpus.csv"))
    print("- eng_to_sr:", data_path("eng_to_sr.csv"))
    print("- output_dir:", output_dir())
    print("- tokenized_cache:", tokenized_cache_path())


def stable_split_key(example: dict) -> str:
    """Stable key for deterministic split.

    For supervised EN→SR we can key on 'source' + 'target'.
    For denoising SR-only we key on the raw Serbian 'text'.
    """
    if example.get("task") == "translate_en_to_sr":
        return f"en2sr::{example.get('source','')}::{example.get('target','')}"
    return f"srden::{example.get('text','')}"


def assign_split(key: str, *, seed: int, valid_ratio: float, test_ratio: float) -> str:
    """Map a key to train/validation/test via deterministic hash."""
    h = hashlib.sha256((str(seed) + "::" + str(key)).encode("utf-8")).hexdigest()
    # use first 8 hex digits as int
    r = int(h[:8], 16) / float(0xFFFFFFFF)
    if r < float(test_ratio):
        return "test"
    if r < float(test_ratio) + float(valid_ratio):
        return "validation"
    return "train"


def main() -> None:
    start = time.time()

    if is_colab():
        ensure_packages()

    require_drive_mounted()

    from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
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

    # ---------------------------
    # Load datasets
    # ---------------------------
    sr_path = data_path("serbian_corpus.csv")
    en2sr_path = data_path("eng_to_sr.csv")
    if not sr_path.exists():
        raise FileNotFoundError(f"Missing dataset: {sr_path}")
    if not en2sr_path.exists():
        raise FileNotFoundError(f"Missing dataset: {en2sr_path}")

    ds_sr = load_dataset("csv", data_files=str(sr_path))["train"]
    ds_en2sr = load_dataset("csv", data_files=str(en2sr_path))["train"]

    mix = CONFIG.get("mix") or {}
    sr_frac = _validate_fraction("mix.sr_denoise_fraction", mix.get("sr_denoise_fraction", 1.0))
    en2sr_frac = _validate_fraction("mix.en_to_sr_fraction", mix.get("en_to_sr_fraction", 1.0))
    sample_mode = str(mix.get("sample_mode", "random"))
    sample_seed = int(mix.get("sample_seed", 82))

    ds_sr = _subsample(ds_sr, sr_frac, mode=sample_mode, seed=sample_seed)
    ds_en2sr = _subsample(ds_en2sr, en2sr_frac, mode=sample_mode, seed=sample_seed)

    ds_sr = _cap_dataset(ds_sr, mix.get("sr_denoise_cap"))
    ds_en2sr = _cap_dataset(ds_en2sr, mix.get("en_to_sr_cap"))

    print(f"\n[Strategy D] Base sizes: sr_corpus={len(ds_sr)}, eng_to_sr={len(ds_en2sr)}")

    # Tag tasks
    ds_sr = ds_sr.add_column("task", ["sr_denoise"] * len(ds_sr))
    ds_en2sr = ds_en2sr.add_column("task", ["translate_en_to_sr"] * len(ds_en2sr))

    # Convert schemas to a union: source_text + target_text
    def _sr_to_union(ex):
        return {
            "task": "sr_denoise",
            "source": ex.get("text"),
            "target": ex.get("text"),
            "text": ex.get("text"),
        }

    def _en2sr_to_union(ex):
        return {
            "task": "translate_en_to_sr",
            "source": ex.get("source"),
            "target": ex.get("target"),
            "text": None,
        }

    ds_sr_u = ds_sr.map(_sr_to_union, remove_columns=ds_sr.column_names)
    ds_en2sr_u = ds_en2sr.map(_en2sr_to_union, remove_columns=ds_en2sr.column_names)

    combined = concatenate_datasets([ds_sr_u, ds_en2sr_u]).shuffle(seed=sample_seed)

    # Deterministic keyed split
    valid_ratio = float(CONFIG.get("validation_split", 0.2))
    test_ratio = float(CONFIG.get("test_split", 0.1))
    split_seed = int(CONFIG.get("split_seed", 82))

    if valid_ratio < 0 or test_ratio < 0 or (valid_ratio + test_ratio) >= 1.0:
        raise ValueError(
            f"Invalid split ratios: validation_split={valid_ratio}, test_split={test_ratio}. "
            "Need validation_split + test_split < 1.0"
        )

    def _with_split(ex):
        key = stable_split_key(ex)
        return {"split": assign_split(key, seed=split_seed, valid_ratio=valid_ratio, test_ratio=test_ratio)}

    combined = combined.map(_with_split)

    train_ds = combined.filter(lambda x: x["split"] == "train")
    valid_ds = combined.filter(lambda x: x["split"] == "validation")
    test_ds = combined.filter(lambda x: x["split"] == "test")

    dataset_dict = DatasetDict({"train": train_ds, "validation": valid_ds, "test": test_ds})
    print("\n[Strategy D] Split sizes (mixed objective)")
    print("- train:", len(dataset_dict["train"]))
    print("- valid:", len(dataset_dict["validation"]))
    print("- test :", len(dataset_dict["test"]))

    # For metrics, only evaluate on translation subset
    valid_translation = valid_ds.filter(lambda x: x["task"] == "translate_en_to_sr")
    test_translation = test_ds.filter(lambda x: x["task"] == "translate_en_to_sr")

    # Diacritics sanity on translation targets
    print_diacritics_sanity(valid_translation, column="target")

    # Model + tokenizer
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

    print_tokenizer_diacritics_support(tokenizer)

    # Tokenize (cached)
    cache_path = tokenized_cache_path()
    ensure_dir(cache_path.parent)

    prefix_translate = "translate English to Serbian: "

    def preprocess(examples: dict) -> dict:
        tasks = examples["task"]
        sources = examples["source"]
        targets = examples["target"]

        inputs: list[str] = []
        out_targets: list[str] = []

        noise_density = float(CONFIG.get("noise_density", 0.15))
        mean_span = float(CONFIG.get("mean_noise_span_length", 3.0))

        # Match Strategy B CPT determinism style: seed corruption from split_seed
        # and the within-batch index (stable for a given dataset/map invocation).
        batch_seed = int(CONFIG.get("split_seed", 82))

        for i, task in enumerate(tasks):
            if task == "translate_en_to_sr":
                inputs.append(prefix_translate + str(sources[i]))
                out_targets.append(str(targets[i]))
            else:
                rng = random.Random(batch_seed + int(i))
                corrupted, target_text = span_corrupt_text(
                    str(sources[i]),
                    noise_density=noise_density,
                    mean_span_length=mean_span,
                    rng=rng,
                )
                # Match Strategy B CPT behavior: no explicit denoising prefix.
                inputs.append(corrupted)
                out_targets.append(target_text)

        model_inputs = tokenizer(
            inputs,
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

    # Metrics (translation-only)
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

    # For evaluation, use only translation tokenized subset
    # We re-tokenize eval subsets via map on the raw eval dataset to avoid needing
    # to carry task labels through the tokenized dataset.
    # (Kept simple and baseline-like.)

    out_dir = output_dir()
    ensure_dir(out_dir)

    # Fingerprint for resume guard
    sr_fp = file_fingerprint(sr_path)
    en2sr_fp = file_fingerprint(en2sr_path)
    mix_sig = json.dumps(CONFIG.get("mix") or {}, sort_keys=True)

    valid_ratio = float(CONFIG.get("validation_split", 0.2))
    test_ratio = float(CONFIG.get("test_split", 0.1))
    train_ratio = 1.0 - valid_ratio - test_ratio
    split_seed = int(CONFIG.get("split_seed", 82))

    def _pct(x: float) -> int:
        return int(round(float(x) * 100.0))

    split_sig = f"s{_pct(train_ratio)}v{_pct(valid_ratio)}t{_pct(test_ratio)}_seed{split_seed}"

    current_fp = (
        f"sr__{sr_fp}__en2sr__{en2sr_fp}"
        f"__mix__{hashlib.sha256(mix_sig.encode('utf-8')).hexdigest()[:12]}"
        f"__{split_sig}__in{int(CONFIG['max_input_length'])}__out{int(CONFIG['max_target_length'])}"
    )

    fp_file = out_dir / "dataset_fingerprint.txt"
    previous_fp = None
    try:
        if fp_file.exists():
            previous_fp = fp_file.read_text(encoding="utf-8").strip() or None
        fp_file.write_text(current_fp, encoding="utf-8")
    except Exception:
        pass

    # build eval tokenized datasets from translation-only subsets
    def _tokenize_translation_subset(raw_ds):
        ds_dict = DatasetDict({"eval": raw_ds})
        tok = ds_dict.map(
            preprocess,
            batched=True,
            remove_columns=raw_ds.column_names,
        )
        return tok["eval"]

    eval_dataset = valid_translation
    subset = CONFIG.get("eval_subset_size")
    if isinstance(subset, int) and 0 < subset < len(eval_dataset):
        print(f"\nEval subset (translation-only): {subset} / {len(eval_dataset)}")
        eval_dataset = eval_dataset.select(range(subset))

    eval_tokenized = _tokenize_translation_subset(eval_dataset)

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

    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=eval_tokenized,
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
                print("\n[FATAL] Non-finite loss detected; stopping training.")
                control.should_training_stop = True
            if grad_norm is not None and _bad_number(grad_norm):
                print("\n[FATAL] Non-finite grad_norm detected; stopping training.")
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
                early_stopping_patience=int(CONFIG.get("early_stopping_patience", 8)),
                early_stopping_threshold=float(CONFIG.get("early_stopping_threshold", 0.0)),
            )
        )

    resume = None
    if not bool(CONFIG.get("force_fresh_train")):
        resume = find_latest_checkpoint(out_dir)

    if resume and previous_fp and previous_fp != current_fp:
        print("\n[Info] Dataset changed since last run; not resuming from old checkpoints.")
        print("- previous:", previous_fp)
        print("- current :", current_fp)
        resume = None

    print("\nTraining (Strategy D mixed objective)...")
    train_started_at = time.time()
    if resume:
        print("Resuming from:", resume)
        train_result = trainer.train(resume_from_checkpoint=resume)
    else:
        train_result = trainer.train()

    train_elapsed_sec = float(time.time() - train_started_at)

    print("\nSaving final model:", out_dir)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # Final test evaluation on translation subset
    test_metrics: dict[str, float] = {}
    if test_translation is not None and len(test_translation) > 0:
        print("\nEvaluating on test split (translation-only)...")
        test_tokenized = _tokenize_translation_subset(test_translation)
        pred = trainer.predict(test_dataset=test_tokenized)
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
        "training_task": str(CONFIG.get("training_task")),
        "output_dir": str(out_dir),
        "dataset_csv": {
            "serbian_corpus": str(sr_path),
            "eng_to_sr": str(en2sr_path),
        },
        "dataset_fingerprint": str(current_fp),
        "resume_from": str(resume) if resume else None,
        "split": {
            "train": 1.0 - float(CONFIG.get("validation_split", 0.2)) - float(CONFIG.get("test_split", 0.1)),
            "validation": float(CONFIG.get("validation_split", 0.2)),
            "test": float(CONFIG.get("test_split", 0.1)),
        },
        "decoding": decoding,
        "training": {
            "global_step": int(getattr(state, "global_step", 0) or 0),
            "epoch": float(getattr(state, "epoch", 0.0) or 0.0),
            "train_runtime_sec": train_elapsed_sec,
            "best_eval_loss": best_eval_loss,
            "best_model_checkpoint": str(getattr(state, "best_model_checkpoint", None)),
        },
        "test_metrics": test_metrics,
        "strategy_d": {
            "mix": CONFIG.get("mix"),
            "tokenized_cache": str(cache_path),
            "noise_density": float(CONFIG.get("noise_density", 0.15)),
            "mean_noise_span_length": float(CONFIG.get("mean_noise_span_length", 3.0)),
            "translation_direction": "eng_to_sr",
            "eval_is_translation_only": True,
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

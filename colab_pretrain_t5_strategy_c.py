"""\
Strategy C: Mixed-language continued pretraining (CPT) with T5 denoising (span corruption)

Goal
----
Continue pretraining (CPT) on an in-domain *mixed* monolingual corpus:
- Serbian (sr-Latn) from `data/serbian_corpus.csv`
- English from `data/english_corpus.csv`

Default mix is 50% Serbian / 50% English (configurable).

This script is intentionally styled very similarly to:
- `colab_train_t5.py` (baseline plumbing: Drive/cache, deterministic split, checkpointing)
- `colab_pretrain_t5_strategy_b.py` (denoising objective + fingerprinting + results.json)

Key behaviors
-------------
- Loads BOTH corpora fully (unless you configure caps)
- Builds a mixed dataset by sampling with a specified ratio
- Deterministic 70/20/10 split with a configurable seed
- Tokenization caching to Drive (cache key includes BOTH corpus fingerprints + mix config)
- Auto-resume from latest checkpoint, guarded by a dataset fingerprint

Input data
----------
Expected files under Drive:
  /content/drive/MyDrive/T5/data/serbian_corpus.csv   (column: text)
  /content/drive/MyDrive/T5/data/english_corpus.csv   (column: text)

Run in Colab
------------
1) Mount Drive:
   from google.colab import drive
   drive.mount('/content/drive')

2) Run:
   !python /content/drive/MyDrive/T5/colab_pretrain_t5_strategy_c.py

Notes
-----
- Uses standard T5 span-corruption objective:
  - Input: text with random spans replaced by sentinels <extra_id_0>, <extra_id_1>, ...
  - Target: concatenated removed spans, each preceded by its sentinel

- Default corruption params are the same as Strategy B:
  - noise_density=0.15
  - mean_noise_span_length=3.0

- By default, we downsample the larger corpus so both languages contribute equally.
  If you want to keep all rows and just control the sampling ratio, set
  CONFIG["mix"]["balance_mode"] = "keep_all".
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
    "training_task": "cpt_denoising_mixed_sr_en",

    # Training
    "num_train_epochs": 10,
    "learning_rate": 1e-4,
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "max_input_length": 256,
    "max_target_length": 256,

    # Dataset split ratios (70/20/10)
    # Using seed=72 for Strategy C family (distinct from Strategy B's 62 and baseline's 42).
    "validation_split": 0.2,
    "test_split": 0.1,
    "split_seed": 72,

    # Denoising / corruption params (safe defaults; same as Strategy B)
    "noise_density": 0.15,
    "mean_noise_span_length": 3.0,

    # Mixed corpus config
    "mix": {
        # Target fractions in the *mixed dataset*.
        # With balance_mode="downsample_to_smallest" (default), 0.5/0.5 means equal counts.
        "sr_fraction": 0.5,
        "en_fraction": 0.5,

        # How to make the mix when corpora sizes differ:
        # - "downsample_to_smallest" (default): choose N=min(len(sr), len(en)) then sample
        #   sr_fraction*N SR rows and en_fraction*N EN rows.
        # - "keep_all": keep all rows from both corpora, but still shuffle; fractions are only
        #   used for naming/fingerprint (not for dropping).
        "balance_mode": "downsample_to_smallest",

        # Sampling method:
        # - "random": shuffle then take K (deterministic with seed)
        # - "deterministic_head": take first K rows (useful for debugging)
        "sample_mode": "random",

        # Optional hard caps (None = no cap)
        "max_sr_rows": None,
        "max_en_rows": None,
    },

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
# Helpers (copied style from baseline / Strategy B)
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


def sr_corpus_csv_path() -> Path:
    return data_path("serbian_corpus.csv")


def en_corpus_csv_path() -> Path:
    return data_path("english_corpus.csv")


def output_dir() -> Path:
    model_slug = safe_slug(CONFIG["model_name"])
    family = "mt5" if "mt5" in str(CONFIG["model_name"]).lower() else "t5"

    valid_ratio = float(CONFIG.get("validation_split", 0.2))
    test_ratio = float(CONFIG.get("test_split", 0.1))
    train_ratio = 1.0 - valid_ratio - test_ratio
    split_seed = int(CONFIG.get("split_seed", 62))

    noise_density = float(CONFIG.get("noise_density", 0.15))
    mean_span = float(CONFIG.get("mean_noise_span_length", 3.0))

    mix = CONFIG.get("mix", {})
    sr_frac = float(mix.get("sr_fraction", 0.5))
    en_frac = float(mix.get("en_fraction", 0.5))
    balance_mode = str(mix.get("balance_mode", "downsample_to_smallest"))

    def _pct(x: float) -> int:
        return int(round(float(x) * 100.0))

    split_sig = f"s{_pct(train_ratio)}v{_pct(valid_ratio)}t{_pct(test_ratio)}_seed{split_seed}"
    corrupt_sig = f"nd{noise_density:.3f}_ms{mean_span:.2f}"
    mix_sig = f"mix_sr{sr_frac:.2f}_en{en_frac:.2f}_{safe_slug(balance_mode)}"

    return project_root() / CONFIG["models_dir"] / f"{family}_cpt_model__mix_sr_en__{model_slug}__{corrupt_sig}__{mix_sig}__{split_sig}"


def tokenized_cache_path(sr_fp: str, en_fp: str) -> Path:
    model_slug = safe_slug(CONFIG["model_name"])

    valid_ratio = float(CONFIG.get("validation_split", 0.2))
    test_ratio = float(CONFIG.get("test_split", 0.1))
    train_ratio = 1.0 - valid_ratio - test_ratio
    split_seed = int(CONFIG.get("split_seed", 62))

    noise_density = float(CONFIG.get("noise_density", 0.15))
    mean_span = float(CONFIG.get("mean_noise_span_length", 3.0))

    mix = CONFIG.get("mix", {})
    sr_frac = float(mix.get("sr_fraction", 0.5))
    en_frac = float(mix.get("en_fraction", 0.5))
    balance_mode = str(mix.get("balance_mode", "downsample_to_smallest"))
    sample_mode = str(mix.get("sample_mode", "random"))
    max_sr_rows = mix.get("max_sr_rows")
    max_en_rows = mix.get("max_en_rows")

    def _pct(x: float) -> int:
        return int(round(float(x) * 100.0))

    split_sig = f"s{_pct(train_ratio)}v{_pct(valid_ratio)}t{_pct(test_ratio)}_seed{split_seed}"
    corrupt_sig = f"nd{noise_density:.3f}_ms{mean_span:.2f}"
    mix_sig = (
        f"mix_sr{sr_frac:.2f}_en{en_frac:.2f}_{safe_slug(balance_mode)}__{safe_slug(sample_mode)}"
        f"__cap_sr{max_sr_rows}_cap_en{max_en_rows}"
    )

    fingerprints = f"sr[{sr_fp}]__en[{en_fp}]"

    return (
        project_root()
        / CONFIG["cache_dir"]
        / "tokenized_datasets"
        / f"mix_sr_en_denoise__{model_slug}__{fingerprints}__{corrupt_sig}__{mix_sig}__{split_sig}__in{CONFIG['max_input_length']}__out{CONFIG['max_target_length']}"
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


def print_paths(sr_fp: str, en_fp: str) -> None:
    print("\nDrive paths")
    print("- project:", project_root())
    print("- serbian_corpus:", sr_corpus_csv_path())
    print("- english_corpus:", en_corpus_csv_path())
    print("- output_dir:", output_dir())
    print("- tokenized_cache:", tokenized_cache_path(sr_fp, en_fp))


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
# Span corruption implementation (same as Strategy B)
# ---------------------------

def _sample_span_length(rng: random.Random, mean_noise_span_length: float) -> int:
    m = float(mean_noise_span_length)
    if m <= 1.0:
        return 1

    lam = max(1e-9, m - 1.0)
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return max(1, k)


def _make_sentinels(tokenizer, n: int) -> list[str]:
    sent = [f"<extra_id_{i}>" for i in range(int(n))]
    for s in sent[: min(3, len(sent))]:
        tid = tokenizer.convert_tokens_to_ids(s)
        if tid is None or tid == tokenizer.unk_token_id:
            raise ValueError(
                f"Tokenizer does not support sentinel token {s}. "
                "Strategy C denoising requires a T5/mT5-compatible tokenizer."
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
    if not isinstance(text, str):
        return "", ""
    text = text.strip()
    if not text:
        return "", ""

    tokens = text.split()
    n = len(tokens)
    if n == 0:
        return "", ""

    noise_density = float(noise_density)
    noise_tokens = int(round(n * noise_density))
    noise_tokens = max(1, min(n, noise_tokens))

    spans: list[tuple[int, int]] = []
    covered = 0
    attempts = 0
    max_attempts = 5 * n

    taken = [False] * n
    while covered < noise_tokens and attempts < max_attempts:
        attempts += 1
        start = rng.randrange(0, n)
        if taken[start]:
            continue
        span_len = _sample_span_length(rng, mean_noise_span_length)
        end = min(n, start + span_len)

        while end > start and any(taken[i] for i in range(start, end)):
            end -= 1
        if end <= start:
            continue

        for i in range(start, end):
            taken[i] = True
        spans.append((start, end))
        covered += (end - start)

    if not spans:
        start = rng.randrange(0, n)
        spans = [(start, min(n, start + 1))]

    spans.sort(key=lambda x: x[0])

    if n > 1:
        masked = sum(e - s for s, e in spans)
        if masked >= n:
            spans = spans[:1]

    sentinels = _make_sentinels(tokenizer, len(spans) + 1)

    corrupted_parts: list[str] = []
    target_parts: list[str] = []

    cur = 0
    for i, (s, e) in enumerate(spans):
        if s > cur:
            corrupted_parts.extend(tokens[cur:s])

        sentinel = sentinels[i]
        corrupted_parts.append(sentinel)

        removed = " ".join(tokens[s:e])
        target_parts.append(sentinel)
        if removed:
            target_parts.append(removed)

        cur = e

    if cur < n:
        corrupted_parts.extend(tokens[cur:])

    target_parts.append(sentinels[len(spans)])

    return " ".join(corrupted_parts).strip(), " ".join(target_parts).strip()


# ---------------------------
# Mixed dataset building
# ---------------------------

def _validate_mix_config() -> None:
    mix = CONFIG.get("mix", {})
    sr_frac = float(mix.get("sr_fraction", 0.5))
    en_frac = float(mix.get("en_fraction", 0.5))
    if sr_frac < 0 or en_frac < 0:
        raise ValueError("mix.sr_fraction and mix.en_fraction must be >= 0")
    if (sr_frac + en_frac) <= 0:
        raise ValueError("mix fractions must sum to > 0")


def _cap_dataset(ds, cap):
    if cap is None:
        return ds
    try:
        cap_i = int(cap)
    except Exception:
        return ds
    if cap_i <= 0:
        return ds
    return ds.select(range(min(cap_i, len(ds))))


def _sample_k(ds, k: int, *, seed: int, mode: str):
    k = int(k)
    if k <= 0:
        return ds.select([])
    if k >= len(ds):
        return ds

    mode = str(mode)
    if mode == "deterministic_head":
        return ds.select(range(k))

    # shuffle then take head (deterministic)
    return ds.shuffle(seed=int(seed)).select(range(k))


def build_mixed_corpus(sr_ds, en_ds):
    """Return a mixed Dataset with a single column: text."""
    _validate_mix_config()

    mix = CONFIG.get("mix", {})
    sr_frac = float(mix.get("sr_fraction", 0.5))
    en_frac = float(mix.get("en_fraction", 0.5))
    balance_mode = str(mix.get("balance_mode", "downsample_to_smallest"))
    sample_mode = str(mix.get("sample_mode", "random"))

    max_sr_rows = mix.get("max_sr_rows")
    max_en_rows = mix.get("max_en_rows")

    sr_ds = _cap_dataset(sr_ds, max_sr_rows)
    en_ds = _cap_dataset(en_ds, max_en_rows)

    seed = int(CONFIG.get("split_seed", 62))

    # normalize fractions
    total = sr_frac + en_frac
    sr_frac_n = sr_frac / total
    en_frac_n = en_frac / total

    if balance_mode == "keep_all":
        mixed = sr_ds.concatenate(en_ds)
        mixed = mixed.shuffle(seed=seed)
        return mixed

    if balance_mode != "downsample_to_smallest":
        raise ValueError(f"Unknown mix.balance_mode: {balance_mode}")

    # Downsample to smallest corpus size for better balancing.
    n_base = min(len(sr_ds), len(en_ds))

    sr_k = int(round(n_base * sr_frac_n))
    en_k = int(round(n_base * en_frac_n))

    # rounding can drift; keep total == n_base when possible
    drift = n_base - (sr_k + en_k)
    if drift != 0:
        # assign drift to the larger fraction to keep ratio close
        if sr_frac_n >= en_frac_n:
            sr_k = max(0, sr_k + drift)
        else:
            en_k = max(0, en_k + drift)

    sr_sample = _sample_k(sr_ds, sr_k, seed=seed + 11, mode=sample_mode)
    en_sample = _sample_k(en_ds, en_k, seed=seed + 23, mode=sample_mode)

    mixed = sr_sample.concatenate(en_sample)
    mixed = mixed.shuffle(seed=seed)
    return mixed


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    start = time.time()

    if is_colab():
        ensure_packages()

    require_drive_mounted()

    from datasets import DatasetDict, load_dataset, load_from_disk
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

    # Load corpora
    sr_path = sr_corpus_csv_path()
    en_path = en_corpus_csv_path()

    if not sr_path.exists():
        raise FileNotFoundError(
            f"Missing corpus: {sr_path}\n"
            "Create it with create_serbian_corpus.py and put it in Drive at the path shown above."
        )
    if not en_path.exists():
        raise FileNotFoundError(
            f"Missing corpus: {en_path}\n"
            "Create it with create_english_corpus.py and put it in Drive at the path shown above."
        )

    sr_fp = file_fingerprint(sr_path)
    en_fp = file_fingerprint(en_path)

    print_paths(sr_fp, en_fp)

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

    print(f"\nLoading Serbian corpus: {sr_path}")
    sr_ds = load_dataset("csv", data_files=str(sr_path))["train"]
    print(f"Loading English corpus: {en_path}")
    en_ds = load_dataset("csv", data_files=str(en_path))["train"]

    for name, ds in [("serbian", sr_ds), ("english", en_ds)]:
        if "text" not in ds.column_names:
            raise ValueError(f"{name}_corpus.csv must have a 'text' column. Found columns: {ds.column_names}")

    # Light sanity check: Serbian diacritics presence in SR corpus
    def _print_sr_diacritics_sanity(ds, sample_size: int = 5000) -> None:
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
            print("\n[Data sanity] Diacritics in Serbian corpus")
            print("- checked_rows:", any_lines)
            print("- counts_per_char:", counts)
            print("- any_diacritics:", any_diacritics)
            if not any_diacritics:
                print("[WARN] No diacritics found in Serbian corpus sample. Check that this is Serbian Latin text.")
        except Exception as exc:
            print(f"\n[WARN] Serbian diacritics sanity check failed: {type(exc).__name__}: {exc}")

    _print_sr_diacritics_sanity(sr_ds)

    # Build mixed dataset
    mixed = build_mixed_corpus(sr_ds, en_ds)
    print("\nMixed corpus")
    print("- total:", len(mixed))

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
    split_test = mixed.train_test_split(test_size=test_ratio, seed=split_seed)
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

    # Validate sentinel tokens exist
    _make_sentinels(tokenizer, 3)

    # Confirm tokenizer can represent Serbian diacritics (baseline helper logic)
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
    cache_path = tokenized_cache_path(sr_fp, en_fp)
    ensure_dir(cache_path.parent)

    noise_density = float(CONFIG.get("noise_density", 0.15))
    mean_span = float(CONFIG.get("mean_noise_span_length", 3.0))

    def preprocess(examples: dict) -> dict:
        # Deterministic corruption per batch (stable cache).
        texts = examples["text"]
        out_inputs: list[str] = []
        out_targets: list[str] = []

        batch_seed = int(CONFIG.get("split_seed", 62))
        # Mix in a hint from the batch content size to reduce identical corruption patterns.
        # (Still deterministic for the same dataset + map order.)
        for i, t in enumerate(texts):
            rng = random.Random((batch_seed * 1315423911) ^ (i + 1))
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

    # Dataset fingerprint guard
    valid_ratio = float(CONFIG.get("validation_split", 0.2))
    test_ratio = float(CONFIG.get("test_split", 0.1))
    train_ratio = 1.0 - valid_ratio - test_ratio
    split_seed = int(CONFIG.get("split_seed", 62))

    mix = CONFIG.get("mix", {})
    sr_frac = float(mix.get("sr_fraction", 0.5))
    en_frac = float(mix.get("en_fraction", 0.5))
    balance_mode = str(mix.get("balance_mode", "downsample_to_smallest"))
    sample_mode = str(mix.get("sample_mode", "random"))
    max_sr_rows = mix.get("max_sr_rows")
    max_en_rows = mix.get("max_en_rows")

    def _pct(x: float) -> int:
        return int(round(float(x) * 100.0))

    split_sig = f"s{_pct(train_ratio)}v{_pct(valid_ratio)}t{_pct(test_ratio)}_seed{split_seed}"

    current_fp = (
        f"sr[{sr_fp}]__en[{en_fp}]__{split_sig}"
        f"__mix_sr{sr_frac:.3f}_en{en_frac:.3f}__{balance_mode}__{sample_mode}__cap_sr{max_sr_rows}__cap_en{max_en_rows}"
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
        print("\n[Info] Corpora/mix config changed since last run; not resuming from old checkpoints.")
        print("- previous:", previous_fp)
        print("- current :", current_fp)
        resume = None

    print("\nTraining (CPT denoising, mixed corpora)...")
    train_started_at = time.time()
    if resume:
        print("Resuming from:", resume)
        trainer.train(resume_from_checkpoint=resume)
    else:
        trainer.train()

    train_elapsed_sec = float(time.time() - train_started_at)

    print("\nSaving final model:", out_dir)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

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
        "training_task": str(CONFIG.get("training_task", "cpt_denoising_mixed")),
        "output_dir": str(out_dir),
        "dataset_csv": [str(sr_path), str(en_path)],
        "dataset_fingerprint": str(current_fp),
        "resume_from": str(resume) if resume else None,
        "split": {
            "train": 1.0 - float(CONFIG.get("validation_split", 0.2)) - float(CONFIG.get("test_split", 0.1)),
            "validation": float(CONFIG.get("validation_split", 0.2)),
            "test": float(CONFIG.get("test_split", 0.1)),
        },
        "mix": {
            "sr_fraction": float(CONFIG.get("mix", {}).get("sr_fraction", 0.5)),
            "en_fraction": float(CONFIG.get("mix", {}).get("en_fraction", 0.5)),
            "balance_mode": str(CONFIG.get("mix", {}).get("balance_mode", "downsample_to_smallest")),
            "sample_mode": str(CONFIG.get("mix", {}).get("sample_mode", "random")),
            "max_sr_rows": CONFIG.get("mix", {}).get("max_sr_rows"),
            "max_en_rows": CONFIG.get("mix", {}).get("max_en_rows"),
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

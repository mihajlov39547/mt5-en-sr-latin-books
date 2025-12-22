"""\
Colab checkpoint/final-model validator for (m)T5 English↔Serbian translation.

What this script does
- Selects a model directory to validate:
  - best checkpoint (from results.json or trainer_state.json), OR
  - latest checkpoint, OR
  - base output folder (final save)
- Verifies the directory contains the required tokenizer + model files
- Recreates the SAME 70/20/10 split (seed=42) from your CSV
- Runs generation on the test split and computes:
  - sacreBLEU (BLEU)
  - chrF++
  - diacritics precision/recall/F1 + any-rate
- Writes a machine-readable report to validate_results.json

Example (Colab)
  from google.colab import drive
  drive.mount('/content/drive')

  !python /content/drive/MyDrive/T5/colab_validate_t5.py \
      --base_dir "/content/drive/MyDrive/T5/models/mt5_translation_model__eng_to_sr__google-mt5-small" \
      --direction eng_to_sr \
      --csv_path "/content/drive/MyDrive/T5/data/eng_to_sr.csv" \
      --which best \
      --max_input_length 256 \
      --max_target_length 256 \
      --num_beams 4

Notes
- This is intended for evaluation/benchmarking. It may be slow on CPU.
- It does not require any training checkpoints beyond the one you choose.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Avoid optional vision imports in Transformers
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")


_DIACRITICS = "čćšžđČĆŠŽĐ"


def safe_slug(text: str) -> str:
    # Keep consistent with colab_train_t5.py naming.
    out = []
    for ch in str(text):
        if ch.isalnum():
            out.append(ch.lower())
        else:
            out.append("-")
    s = "".join(out)
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip("-")


def file_fingerprint(path: Path) -> str:
    """Stable-enough fingerprint to detect data changes without hashing huge files."""
    import hashlib

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


def infer_project_root_from_run_dir(run_dir: Path) -> Path | None:
    # Expected: /.../T5/models/<run_name>
    if run_dir.parent.name == "models":
        return run_dir.parent.parent
    return None


def infer_model_slug_from_run_dir(run_dir: Path) -> str | None:
    # Expected run dir naming: mt5_translation_model__{direction}__{model_slug}
    # e.g. mt5_translation_model__eng_to_sr__google-mt5-small
    parts = run_dir.name.split("__")
    if len(parts) >= 3:
        return parts[-1]
    return None


def tokenized_cache_path(
    *,
    project_root: Path,
    direction: str,
    model_slug: str,
    csv_path: Path,
    split_seed: int,
    validation_split: float,
    test_split: float,
    max_input_length: int,
    max_target_length: int,
) -> Path:
    fp = file_fingerprint(csv_path)
    split_seed = int(split_seed)

    valid_ratio = float(validation_split)
    test_ratio = float(test_split)
    train_ratio = 1.0 - valid_ratio - test_ratio

    def _pct(x: float) -> int:
        return int(round(float(x) * 100.0))

    split_sig = f"s{_pct(train_ratio)}v{_pct(valid_ratio)}t{_pct(test_ratio)}_seed{split_seed}"
    return (
        project_root
        / "cache"
        / "tokenized_datasets"
        / f"{safe_slug(direction)}__{safe_slug(model_slug)}__{fp}__{split_sig}__in{int(max_input_length)}__out{int(max_target_length)}"
    )


def is_colab() -> bool:
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_BACKEND_VERSION"):
        return True
    if os.environ.get("COLAB_GPU") is not None:
        return True
    if Path("/content").exists():
        return True
    return "google.colab" in sys.modules


def ensure_packages() -> None:
    """Install deps if missing (useful in fresh Colab runtimes)."""

    def _ok() -> bool:
        try:
            import transformers  # noqa: F401
            import datasets  # noqa: F401
            import evaluate  # noqa: F401
            import sacrebleu  # noqa: F401
            import sentencepiece  # noqa: F401
            return True
        except Exception:
            return False

    if _ok():
        return

    print("Installing required packages (one-time for this runtime)...")
    import subprocess

    pkgs = [
        "transformers==4.49.0",
        "datasets==3.3.2",
        "evaluate==0.4.6",
        "sacrebleu==2.5.1",
        "sentencepiece==0.2.0",
        "accelerate==1.4.0",
        "protobuf>=4.25.0",
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + pkgs)


def _has_file(d: Path, name: str) -> bool:
    return (d / name).exists()


def _has_model_weights(d: Path) -> bool:
    # Transformers may save either safetensors or pytorch bin shards.
    if (d / "model.safetensors").exists():
        return True
    if any(d.glob("model-*.safetensors")):
        return True
    if (d / "pytorch_model.bin").exists():
        return True
    if any(d.glob("pytorch_model-*.bin")):
        return True
    return False


def validate_dir_contents(model_dir: Path) -> list[str]:
    """Return a list of missing required files."""
    missing: list[str] = []

    # Tokenizer
    if not _has_file(model_dir, "spiece.model"):
        missing.append("spiece.model")

    # Model
    if not _has_file(model_dir, "config.json"):
        missing.append("config.json")
    if not _has_model_weights(model_dir):
        missing.append("model weights (pytorch_model*.bin or model*.safetensors)")

    return missing


def find_latest_checkpoint(base_dir: Path) -> Path | None:
    if not base_dir.exists():
        return None
    best_step = -1
    best_path: Path | None = None
    for child in base_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("checkpoint-"):
            continue
        try:
            step = int(child.name.split("-", 1)[1])
        except Exception:
            continue
        if step > best_step:
            best_step = step
            best_path = child
    return best_path


def find_best_checkpoint(base_dir: Path) -> Path | None:
    """Try results.json then trainer_state.json to locate best checkpoint."""
    # Prefer results.json (written by updated colab_train_t5.py)
    results_path = base_dir / "results.json"
    if results_path.exists():
        try:
            data = json.loads(results_path.read_text(encoding="utf-8"))
            ckpt = data.get("training", {}).get("best_model_checkpoint")
            if isinstance(ckpt, str) and ckpt:
                p = Path(ckpt)
                if p.exists() and p.is_dir():
                    return p
        except Exception:
            pass

    # Fall back to trainer_state.json (common Transformers file)
    trainer_state = base_dir / "trainer_state.json"
    if trainer_state.exists():
        try:
            data = json.loads(trainer_state.read_text(encoding="utf-8"))
            ckpt = data.get("best_model_checkpoint")
            if isinstance(ckpt, str) and ckpt:
                p = Path(ckpt)
                if p.exists() and p.is_dir():
                    return p
        except Exception:
            pass

    return None


def pick_model_dir(base_dir: Path, which: str) -> Path:
    which = (which or "best").lower().strip()
    if which == "best":
        best = find_best_checkpoint(base_dir)
        if best is not None:
            return best
        latest = find_latest_checkpoint(base_dir)
        if latest is not None:
            return latest
        return base_dir

    if which == "latest":
        latest = find_latest_checkpoint(base_dir)
        return latest if latest is not None else base_dir

    if which == "final":
        return base_dir

    raise ValueError("--which must be one of: best, latest, final")


def _diacritics_scores(preds: list[str], refs: list[str]) -> dict[str, float]:
    pred_counts = {ch: 0 for ch in _DIACRITICS}
    ref_counts = {ch: 0 for ch in _DIACRITICS}
    any_rate_n = 0

    for p in preds:
        if any(ch in p for ch in _DIACRITICS):
            any_rate_n += 1
        for ch in _DIACRITICS:
            pred_counts[ch] += p.count(ch)

    for r in refs:
        for ch in _DIACRITICS:
            ref_counts[ch] += r.count(ch)

    tp = sum(min(pred_counts[ch], ref_counts[ch]) for ch in _DIACRITICS)
    fp = sum(max(0, pred_counts[ch] - ref_counts[ch]) for ch in _DIACRITICS)
    fn = sum(max(0, ref_counts[ch] - pred_counts[ch]) for ch in _DIACRITICS)

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base_dir",
        required=True,
        help="Training output directory that contains checkpoint-* folders and/or final model",
    )
    ap.add_argument(
        "--which",
        default="best",
        choices=["best", "latest", "final"],
        help="Which folder to validate: best checkpoint, latest checkpoint, or final base_dir",
    )
    ap.add_argument(
        "--direction",
        default="eng_to_sr",
        choices=["eng_to_sr", "sr_to_eng"],
        help="Translation direction (controls the task prefix)",
    )
    ap.add_argument(
        "--csv_path",
        default="",
        help="Path to the CSV with columns source,target. If omitted, uses a default under /content/drive/MyDrive/T5/data",
    )
    ap.add_argument(
        "--use_tokenized_cache",
        action="store_true",
        help=(
            "If set, tries to load the tokenized DatasetDict saved by colab_train_t5.py (includes train/validation/test split). "
            "This guarantees identical test rows and avoids re-tokenization when the cache exists."
        ),
    )
    ap.add_argument(
        "--tokenized_cache_dir",
        default="",
        help=(
            "Optional explicit path to a tokenized dataset directory created by datasets.save_to_disk(). "
            "If provided with --use_tokenized_cache, this path is used directly."
        ),
    )
    ap.add_argument(
        "--project_root",
        default="",
        help=(
            "Optional path to the T5 project root (the folder containing cache/ and models/). "
            "If omitted, it is inferred from --base_dir when possible."
        ),
    )
    ap.add_argument("--seed", type=int, default=42, help="Split seed (must match training for comparable results)")
    ap.add_argument("--validation_split", type=float, default=0.2, help="Validation split ratio")
    ap.add_argument("--test_split", type=float, default=0.1, help="Test split ratio")
    ap.add_argument("--max_input_length", type=int, default=256)
    ap.add_argument("--max_target_length", type=int, default=256)
    ap.add_argument("--max_new_tokens", type=int, default=256, help="Generation cap (use <= max_target_length for parity)")
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument(
        "--limit_test",
        type=int,
        default=0,
        help="If >0, evaluate only the first N test rows for a faster smoke test",
    )
    ap.add_argument(
        "--output_json",
        default="",
        help="Where to write validate_results.json (default: base_dir/validate_results.json)",
    )
    args = ap.parse_args()

    if is_colab():
        ensure_packages()

    from datasets import DatasetDict, load_dataset, load_from_disk
    import numpy as np
    import torch
    import evaluate
    import sacrebleu
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    base_dir = Path(args.base_dir)
    model_dir = pick_model_dir(base_dir, args.which)

    print("\n=== Selection ===")
    print("base_dir:", base_dir)
    print("which   :", args.which)
    print("model_dir:", model_dir)

    missing = validate_dir_contents(model_dir)
    if missing:
        print("\n[FAIL] Selected directory is missing required files:")
        for m in missing:
            print("-", m)
        # Still write a report for reproducibility
        out_path = Path(args.output_json) if args.output_json else (base_dir / "validate_results.json")
        report = {
            "ok": False,
            "base_dir": str(base_dir),
            "which": args.which,
            "model_dir": str(model_dir),
            "missing": missing,
        }
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Wrote:", out_path)
        raise SystemExit(2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # Load model/tokenizer from selected folder
    tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir)).to(device)
    model.eval()

    # Load dataset (raw CSV) OR reuse tokenized cache from training.
    if args.csv_path.strip():
        csv_path = Path(args.csv_path)
    else:
        # Default Drive location
        csv_name = "eng_to_sr.csv" if args.direction == "eng_to_sr" else "sr_to_eng.csv"
        csv_path = Path("/content/drive/MyDrive/T5/data") / csv_name

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print("\n=== Data ===")
    print("csv_path:", csv_path)

    valid_ratio = float(args.validation_split)
    test_ratio = float(args.test_split)
    if valid_ratio < 0 or test_ratio < 0 or (valid_ratio + test_ratio) >= 1.0:
        raise ValueError("validation_split + test_split must be < 1.0")

    used_tokenized_cache = False
    tokenized: DatasetDict | None = None
    dataset_dict: DatasetDict | None = None

    if bool(args.use_tokenized_cache):
        cache_dir: Path | None = None

        if args.tokenized_cache_dir.strip():
            cache_dir = Path(args.tokenized_cache_dir)
        else:
            project_root = Path(args.project_root) if args.project_root.strip() else infer_project_root_from_run_dir(base_dir)
            model_slug = infer_model_slug_from_run_dir(base_dir)
            if project_root is not None and model_slug is not None:
                cache_dir = tokenized_cache_path(
                    project_root=project_root,
                    direction=str(args.direction),
                    model_slug=model_slug,
                    csv_path=csv_path,
                    split_seed=int(args.seed),
                    validation_split=float(args.validation_split),
                    test_split=float(args.test_split),
                    max_input_length=int(args.max_input_length),
                    max_target_length=int(args.max_target_length),
                )

        if cache_dir is not None:
            print("tokenized_cache_dir (candidate):", cache_dir)
            if cache_dir.exists():
                try:
                    tokenized = load_from_disk(str(cache_dir))
                    if not isinstance(tokenized, DatasetDict) or "test" not in tokenized:
                        raise TypeError("Loaded cache is not a DatasetDict with a 'test' split")
                    used_tokenized_cache = True
                    print("[OK] Loaded tokenized DatasetDict from cache.")
                except Exception as exc:
                    print(f"[WARN] Failed to load tokenized cache ({type(exc).__name__}: {exc}); falling back to CSV split.")
            else:
                print("[WARN] Tokenized cache directory does not exist; falling back to CSV split.")
        else:
            print("[WARN] Could not infer tokenized cache directory; falling back to CSV split.")

    if not used_tokenized_cache:
        dataset = load_dataset("csv", data_files=str(csv_path))["train"]

        split_test = dataset.train_test_split(test_size=test_ratio, seed=int(args.seed))
        train_valid = split_test["train"]
        test_ds = split_test["test"]

        valid_ratio_rel = valid_ratio / (1.0 - test_ratio)
        split_valid = train_valid.train_test_split(test_size=valid_ratio_rel, seed=int(args.seed))

        dataset_dict = DatasetDict({
            "train": split_valid["train"],
            "validation": split_valid["test"],
            "test": test_ds,
        })

        print("rows train/valid/test:", len(dataset_dict["train"]), len(dataset_dict["validation"]), len(dataset_dict["test"]))

    test_eval = tokenized["test"] if used_tokenized_cache and tokenized is not None else dataset_dict["test"]
    if args.limit_test and int(args.limit_test) > 0:
        n = min(int(args.limit_test), len(test_eval))
        test_eval = test_eval.select(range(n))
        print("limit_test:", n)

    prefix = "translate English to Serbian: " if args.direction == "eng_to_sr" else "translate Serbian to English: "

    # Generate predictions
    preds: list[str] = []
    refs: list[str] = []

    t0 = time.time()
    batch_size = 8 if device == "cuda" else 2

    pad_id = tok.pad_token_id
    if pad_id is None:
        # SentencePiece models usually have pad=0; keep safe.
        pad_id = 0

    for i in range(0, len(test_eval), batch_size):
        batch = test_eval[i : i + batch_size]

        if used_tokenized_cache:
            # Cache contains already-prefixed inputs + labels.
            input_features = [{"input_ids": ids, "attention_mask": am} for ids, am in zip(batch["input_ids"], batch["attention_mask"]) ]
            inputs = tok.pad(input_features, return_tensors="pt").to(device)

            label_rows = batch["labels"]
            # Defensive: replace -100 with pad for decoding.
            cleaned_labels = []
            for row in label_rows:
                cleaned_labels.append([pad_id if int(x) == -100 else int(x) for x in row])
            targets = tok.batch_decode(cleaned_labels, skip_special_tokens=True)
        else:
            sources = [prefix + str(s) for s in batch["source"]]
            targets = [str(t) for t in batch["target"]]
            inputs = tok(
                sources,
                return_tensors="pt",
                truncation=True,
                max_length=int(args.max_input_length),
                padding=True,
            ).to(device)

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=int(args.max_new_tokens),
                num_beams=int(args.num_beams),
            )

        decoded = tok.batch_decode(out, skip_special_tokens=True)
        preds.extend(decoded)
        refs.extend(targets)

    elapsed = float(time.time() - t0)

    # Metrics
    metric_bleu = evaluate.load("sacrebleu")
    chrfpp = sacrebleu.metrics.CHRF(word_order=2)

    bleu = metric_bleu.compute(predictions=preds, references=[[r] for r in refs])["score"]
    chrf = chrfpp.corpus_score(preds, [refs]).score
    diac = _diacritics_scores(preds, refs)

    # Length ratio is useful to detect truncation/over-short outputs
    pred_len = float(np.mean([len(p) for p in preds])) if preds else 0.0
    ref_len = float(np.mean([len(r) for r in refs])) if refs else 0.0
    length_ratio = float(pred_len / ref_len) if ref_len > 0 else 0.0

    print("\n=== Metrics (test split) ===")
    print("BLEU  :", float(bleu))
    print("chrF++:", float(chrf))
    print("len_ratio(pred/ref):", length_ratio)
    for k, v in diac.items():
        print(k + ":", v)

    report = {
        "ok": True,
        "base_dir": str(base_dir),
        "which": args.which,
        "model_dir": str(model_dir),
        "direction": args.direction,
        "csv_path": str(csv_path),
        "used_tokenized_cache": bool(used_tokenized_cache),
        "split": {
            "train": 1.0 - valid_ratio - test_ratio,
            "validation": valid_ratio,
            "test": test_ratio,
            "seed": int(args.seed),
        },
        "device": device,
        "decoding": {
            "max_input_length": int(args.max_input_length),
            "max_target_length": int(args.max_target_length),
            "max_new_tokens": int(args.max_new_tokens),
            "num_beams": int(args.num_beams),
        },
        "test": {
            "num_rows": int(len(test_eval)),
            "runtime_sec": elapsed,
            "bleu": float(bleu),
            "chrfpp": float(chrf),
            "length_ratio": float(length_ratio),
            **diac,
        },
    }

    out_path = Path(args.output_json) if args.output_json else (base_dir / "validate_results.json")
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nWrote:", out_path)


if __name__ == "__main__":
    main()

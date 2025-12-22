"""\
Colab inference sanity checks for the fine-tuned (m)T5 translation model.

This is designed to be robust to Colab disconnects:
- It auto-selects the newest checkpoint-* that contains tokenizer files (spiece.model).
- It prints clear, labeled diagnostics so you can quickly verify:
    - You are loading the right folder
    - The tokenizer can represent Serbian diacritics
    - Your dataset CSV contains diacritics (optional)
    - The model can *generate* diacritics and produces reasonable translations

How to run (Colab):
    from google.colab import drive
    drive.mount('/content/drive')

    !python /content/drive/MyDrive/T5/colab_test_t5.py \
            --base_dir "/content/drive/MyDrive/T5/models/mt5_translation_model__eng_to_sr__google-mt5-small" \
            --csv_path "/content/drive/MyDrive/T5/data/eng_to_sr.csv" \
            --check_csv \
            --max_length 256 \
            --max_new_tokens 128 \
            --num_beams 4

Tip:
- If your base_dir has a final model save (not just checkpoints), this script will also
    accept that. Otherwise it will load from the newest usable checkpoint.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


_DIACRITICS = "čćšžđČĆŠŽĐ"


def _has_sentencepiece_model(folder: str | os.PathLike) -> bool:
    p = Path(folder)
    return (p / "spiece.model").exists()


def _step_from_checkpoint_name(path: str) -> int | None:
    m = re.search(r"checkpoint-(\d+)$", str(path))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def pick_model_dir(base_dir: str) -> str:
    """Pick newest checkpoint-* with tokenizer; else fall back to base_dir."""
    ckpts = glob.glob(str(Path(base_dir) / "checkpoint-*"))
    ckpts = [p for p in ckpts if Path(p).is_dir()]

    def _step(p: str) -> int:
        step = _step_from_checkpoint_name(p)
        return step if isinstance(step, int) else -1

    ckpts = sorted(ckpts, key=_step)

    # Prefer checkpoints that include tokenizer files (spiece.model).
    for p in reversed(ckpts):
        if _has_sentencepiece_model(p):
            return p

    # Fall back to base_dir if it has a tokenizer.
    if _has_sentencepiece_model(base_dir):
        return base_dir

    # As a last resort, return the newest checkpoint even if incomplete,
    # so the user can see what's missing.
    if ckpts:
        return ckpts[-1]

    return base_dir


def _print_header(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def _non_ascii_chars(s: str) -> list[str]:
    return sorted({c for c in s if ord(c) > 127})


def _contains_any_diacritics(s: str) -> bool:
    return any(ch in s for ch in _DIACRITICS)


def tokenizer_diacritics_roundtrip(tokenizer) -> None:
    _print_header("Tokenizer sanity: diacritics round-trip")
    bad: list[str] = []
    for ch in _DIACRITICS:
        ids = tokenizer.encode(ch, add_special_tokens=False)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        ok = decoded == ch
        print(f"- {ch!r} -> ids={ids} -> {decoded!r} ok={ok}")
        if not ok:
            bad.append(ch)
    if bad:
        print("\n[WARN] Tokenizer cannot round-trip these diacritics:", "".join(bad))
        print("[WARN] With this tokenizer/model family, output may remain ASCII.")
    else:
        print("\n[OK] Tokenizer round-trips Serbian diacritics.")


def dataset_csv_diacritics_check(csv_path: str, sample_size: int = 5000) -> None:
    _print_header("Dataset sanity: diacritics present in CSV targets")
    p = Path(csv_path)
    if not p.exists():
        print("[SKIP] CSV not found:", csv_path)
        return

    try:
        import pandas as pd
    except Exception as exc:
        print("[SKIP] pandas not available:", f"{type(exc).__name__}: {exc}")
        return

    try:
        df = pd.read_csv(str(p))
    except UnicodeDecodeError:
        df = pd.read_csv(str(p), encoding="utf-8", errors="replace")

    if "target" not in df.columns:
        print("[FAIL] Missing 'target' column.")
        print("Columns:", list(df.columns))
        return

    col = df["target"].dropna().astype(str)
    if len(col) == 0:
        print("[FAIL] No non-empty rows in target column.")
        return

    n = min(int(sample_size), len(col))
    sample = col.sample(n, random_state=0)
    counts = {ch: int(sample.str.contains(ch, regex=False).sum()) for ch in _DIACRITICS}
    print("rows_checked:", n)
    print("counts_per_char:", counts)
    print("any_diacritics:", any(v > 0 for v in counts.values()))


def run_generation_tests(
    *,
    tokenizer,
    model,
    device: str,
    max_length: int,
    max_new_tokens: int,
    num_beams: int,
) -> None:
    _print_header("Generation sanity: translation smoke tests")

    def run_one(prompt: str) -> None:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=int(max_length),
        ).to(device)

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                num_beams=int(num_beams),
                # Conservative decoding constraints; helps avoid obvious repeats.
                no_repeat_ngram_size=3,
            )

        pred = tokenizer.decode(out[0], skip_special_tokens=True)
        print("PROMPT:", prompt)
        print("OUTPUT:", pred)
        print("REPR  :", repr(pred))
        print("NON-ASCII:", _non_ascii_chars(pred))
        print("HAS_SR_DIACRITICS:", _contains_any_diacritics(pred))
        print("-" * 88)

    print(
        "[Info] What this checks:\n"
        "- Outputs are non-empty\n"
        "- NON-ASCII includes diacritics when appropriate\n"
        "- Prompts around the same concept translate consistently\n"
    )

    # Required test set (per your request)
    tests = [
        "translate English to Serbian: Where is the train station?",
        "translate English to Serbian: Where is the station?",
        "translate English to Serbian: Where is the bus station?",
    ]

    print("\n[Section] Short prompts: station variants")
    for t in tests:
        run_one(t)

    print("\n[Section] Short prompts: everyday sentences")
    for t in [
        "translate English to Serbian: I like to learn new languages.",
        "translate English to Serbian: I would like to pay by card, please.",
        "translate English to Serbian: Please speak more slowly; I'm still learning Serbian.",
    ]:
        run_one(t)

    print("\n[Section] Longer prompts: book-like sentences")
    for t in [
        "translate English to Serbian: We live in a censorious age, and an author cannot take too much precaution to anticipate prejudice, misapprehension, and the temerity of malice, ignorance, and presumption.",
        "translate English to Serbian: Sometimes, dazzled by the tinsel of a character which he has no opportunity to investigate, he pours forth the homage of admiration upon some false benefactor, and afterwards is ashamed of his own praise.",
    ]:
        run_one(t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base_dir",
        default="/content/drive/MyDrive/T5/models/mt5_translation_model__eng_to_sr__google-mt5-small",
        help="Model output dir that contains checkpoint-* folders",
    )
    ap.add_argument(
        "--csv_path",
        default="",
        help="Optional path to eng_to_sr.csv (or sr_to_eng.csv) to sanity-check diacritics",
    )
    ap.add_argument(
        "--check_csv",
        action="store_true",
        help="If set, reads CSV and checks that Serbian diacritics exist in the 'target' column",
    )
    ap.add_argument("--max_length", type=int, default=256, help="Tokenizer truncation max_length")
    ap.add_argument("--max_new_tokens", type=int, default=128, help="Generation max_new_tokens")
    ap.add_argument("--num_beams", type=int, default=4, help="Beam size")
    args = ap.parse_args()

    model_dir = pick_model_dir(args.base_dir)
    _print_header("Load sanity: model folder selection")
    print("base_dir :", args.base_dir)
    print("model_dir:", model_dir)
    step = _step_from_checkpoint_name(model_dir)
    if isinstance(step, int):
        print("checkpoint_step:", step)

    # Helpful debug if tokenizer is missing
    if not _has_sentencepiece_model(model_dir):
        p = Path(model_dir)
        print("\n[WARN] spiece.model not found in selected directory.")
        if p.exists() and p.is_dir():
            print("Contents:", sorted([c.name for c in p.iterdir()])[:200])
        print(
            "\nIf this is a checkpoint from an older run, it may not include tokenizer files. "
            "Re-run training with the updated colab_train_t5.py (it now saves tokenizer into each checkpoint)."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    model.eval()

    # 1) Tokenizer sanity
    tokenizer_diacritics_roundtrip(tok)

    # 2) Optional dataset sanity (CSV)
    if bool(args.check_csv):
        csv_path = args.csv_path.strip() or "/content/drive/MyDrive/T5/data/eng_to_sr.csv"
        dataset_csv_diacritics_check(csv_path)

    # 3) Generation sanity (multiple prompts)
    run_generation_tests(
        tokenizer=tok,
        model=model,
        device=device,
        max_length=int(args.max_length),
        max_new_tokens=int(args.max_new_tokens),
        num_beams=int(args.num_beams),
    )


if __name__ == "__main__":
    main()

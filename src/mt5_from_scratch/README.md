# mT5-from-scratch (educational, PyTorch-only)

This folder contains a small, **educational** implementation of a T5/mT5-style encoder–decoder Transformer in **pure PyTorch**.

It’s **inspired by** the architecture used by `google/mt5-small`, but it’s **not** a drop-in replacement for Hugging Face Transformers and it does **not** ship with pretrained weights.

If your goal is to understand how mT5/T5 works internally (RMSNorm, relative position bias, gated FFN, encoder–decoder attention, KV caching), this code is meant to be readable and hackable.

## What’s implemented

Core components:

- **Encoder–decoder Transformer** with tied input/output embeddings
- **RMSNorm** (T5-style)
- **Relative position bias** with T5-style bucketing (`RelativePositionBias`)
- **Multi-head attention** (self-attn + cross-attn)
- **Gated feed-forward** block (SiLU-gated)
- **KV caching** for efficient autoregressive decoding
- Minimal greedy `generate()` for demonstration

## Folder contents

- `config.py` – `MT5SmallConfig` hyperparameters (mT5-small-like defaults)
- `layers.py` – RMSNorm / Relative position bias / attention / blocks / `KVCache`
- `model.py` – encoder/decoder stacks + `MT5ForConditionalGeneration` + greedy generation
- `train_smoke.py` – tiny forward+backward sanity check

## Quick start

### 1) Smoke test (forward + backward)

This confirms:
- forward pass works
- loss computes
- `loss.backward()` works (autograd)
- one optimizer step works

```powershell
python -c "import sys; sys.path.insert(0, 'src'); from mt5_from_scratch.train_smoke import main; main()"
```

### 2) Tiny greedy generation (uses KV cache)

```powershell
python -c "import sys; sys.path.insert(0, 'src'); import torch; from mt5_from_scratch import MT5ForConditionalGeneration, MT5SmallConfig; cfg=MT5SmallConfig(vocab_size=256,d_model=64,d_kv=16,d_ff=128,num_layers=2,num_heads=2,dropout_rate=0.0); m=MT5ForConditionalGeneration(cfg).eval(); x=torch.randint(0,256,(1,8)); y=m.generate(x,max_length=12); print(y)"
```

Note: with random weights the output is usually uninteresting (often predicts token `0`).

## API sketch

### `KVCache`

Located in `layers.py`.

- Stores cached key/value tensors for a single attention module:
  - `key`: `(batch, heads, seq_len, d_kv)`
  - `value`: `(batch, heads, seq_len, d_kv)`
- `update()` appends new key/value on the sequence dimension.

### `MT5ForConditionalGeneration`

Located in `model.py`.

- `forward(..., use_cache=True)` returns `Seq2SeqLMOutput(..., past_key_values=...)`
- `generate()` performs greedy decoding and uses the cache to avoid recomputing past K/V.

The `past_key_values` format is:

- For the decoder stack: a list of length `num_layers`
- Each element is a tuple:
  - `(self_attn_cache: KVCache, cross_attn_cache: KVCache)`

## Design notes (what’s “mT5-like” vs what’s simplified)

### mT5/T5-like pieces

- Encoder–decoder layout
- Relative position bias bucketing
- RMSNorm
- Gated feed-forward (SiLU)
- Decoder self-attention is causal
- Decoder start token is `pad_token_id` (T5-style shift-right)

### KV caching (implemented)

- **Decoder self-attention** cache grows by 1 token each decode step
- **Decoder cross-attention** caches encoder K/V once and reuses it
- Relative position bias for cached decoding is computed using explicit absolute positions

### Simplified attention masking (important)

This implementation uses a simple **additive attention mask** convention and deliberately avoids many production optimizations.

- Padding masks are created as additive masks of shape `(b, 1, 1, seq)` with `0` for keep and `-1e9` for masked.
- Causal masking is applied by adding a (triangular) `-1e9` mask inside the attention module.

What’s simplified compared to production implementations:

- No fused attention kernels (e.g. SDPA/FlashAttention paths)
- No precomputed / cached causal masks
- Mask handling aims to be correct and readable, not maximally fast

## TODO / Improvements

To make this implementation more faithful and more performant:

1. **Use `torch.nn.functional.scaled_dot_product_attention`** (SDPA) when available for faster attention on GPU.
2. **Precompute/cached causal masks** (or rely on SDPA’s `is_causal=True`) to avoid building masks every step.
3. **More faithful relative position bias** plumbing:
   - Ensure bias handling matches the exact T5 behavior in every mode (teacher forcing vs incremental decoding).
4. **More efficient KV cache storage**:
   - Current implementation appends via `torch.cat` each step (simple but not optimal).
   - Consider preallocation or chunked growth strategies.
5. **Better generation**:
   - Add sampling (temperature/top-k/top-p)
   - Add beam search
   - Add early stopping / length penalty options
6. **Weight compatibility**:
   - Add a mapping/utilities to load weights from Hugging Face `google/mt5-small` into this model.
   - Add checksum tests on a few layers after loading.
7. **Training utilities**:
   - Implement a span-corruption data collator and a minimal training script.
   - Add tiny overfit test to confirm learning.

## License / attribution

This repository (including this folder) is released under the **MIT License** (see the root `LICENSE`).

If you use this code in academic or research work:

- Please cite this repository using the root **`CITATION.cff`**.
- If you discuss results or architecture details, also cite the original **T5** / **mT5** papers and Google’s released checkpoints.

This folder does not include any pretrained weights.

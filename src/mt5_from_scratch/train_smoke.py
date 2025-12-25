from __future__ import annotations

import torch

from mt5_from_scratch import MT5ForConditionalGeneration, MT5SmallConfig


def main() -> None:
    # Keep this intentionally tiny: it's a smoke test that forward+backward works.
    torch.manual_seed(0)

    cfg = MT5SmallConfig(
        # Reduce vocab for local smoke test speed.
        vocab_size=4096,
        d_model=128,
        d_kv=32,
        d_ff=256,
        num_layers=2,
        num_heads=4,
        dropout_rate=0.0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MT5ForConditionalGeneration(cfg).to(device)
    model.train()

    bsz, src_len, tgt_len = 2, 16, 12
    input_ids = torch.randint(0, cfg.vocab_size, (bsz, src_len), device=device)

    # Make some pads
    input_ids[0, -2:] = cfg.pad_token_id

    labels = torch.randint(0, cfg.vocab_size, (bsz, tgt_len), device=device)
    labels[0, -3:] = cfg.pad_token_id

    out = model(input_ids=input_ids, labels=labels)
    assert out.loss is not None

    out.loss.backward()

    # One optimizer step
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt.step()
    opt.zero_grad(set_to_none=True)

    print(
        {
            "device": str(device),
            "loss": float(out.loss.detach().cpu()),
            "params": sum(p.numel() for p in model.parameters()),
        }
    )


if __name__ == "__main__":
    # Allow running as: python -m mt5_from_scratch.train_smoke
    # or: python src/mt5_from_scratch/train_smoke.py (if src is on PYTHONPATH)
    main()

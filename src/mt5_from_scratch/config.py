from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MT5SmallConfig:
    """Config approximating mT5-small (T5.1.1-style) for educational use.

    Notes:
    - This is *not* guaranteed to be exactly Google's mt5-small config.
    - It is close enough structurally (dims, layer counts, relative bias, etc.)
      to be useful for understanding + small-scale pretraining.

    If you want a smaller model for quick experimentation, shrink `d_model`,
    `num_layers`, etc.
    """

    # Vocabulary
    vocab_size: int = 250_112  # common for mt5; you can override for smaller toy vocabs

    # Model dims
    d_model: int = 512
    d_kv: int = 64
    d_ff: int = 1024  # T5 uses 2048 for 512? Many variants use 1024/2048; adjust as needed.

    # Layers/heads
    num_layers: int = 8
    num_decoder_layers: int | None = None  # if None, use num_layers
    num_heads: int = 8

    # Dropout
    dropout_rate: float = 0.1

    # Relative attention bias
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128

    # Special tokens
    pad_token_id: int = 0
    eos_token_id: int = 1

    # T5 specifics
    layer_norm_epsilon: float = 1e-6
    tie_word_embeddings: bool = True

    # Generation defaults
    max_length: int = 256

    def decoder_layers(self) -> int:
        return int(self.num_layers if self.num_decoder_layers is None else self.num_decoder_layers)

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """T5-style RMSNorm (no mean subtraction, scale only)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


class RelativePositionBias(nn.Module):
    """T5 relative position bias: bucketed relative distances.

    Produces an attention bias of shape (1, num_heads, q_len, k_len).
    """

    def __init__(
        self,
        *,
        num_buckets: int,
        max_distance: int,
        num_heads: int,
    ):
        super().__init__()
        self.num_buckets = int(num_buckets)
        self.max_distance = int(max_distance)
        self.num_heads = int(num_heads)
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.num_heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position: torch.Tensor,
        *,
        num_buckets: int,
        max_distance: int,
        bidirectional: bool,
    ) -> torch.Tensor:
        """Adapted from the T5 paper / common implementations."""

        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            sign = (n < 0).to(torch.long)
            n = n.abs()
        else:
            n = torch.clamp(n, min=0)
            sign = 0

        # now n is non-negative
        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (
            (torch.log(n.float() / max_exact + 1e-6) / math.log(max_distance / max_exact))
            * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.clamp(val_if_large, max=num_buckets - 1)
        bucket = torch.where(is_small, n.to(torch.long), val_if_large)
        if bidirectional:
            bucket = bucket + sign * num_buckets
        return bucket

    def forward(self, q_len: int, k_len: int, *, bidirectional: bool) -> torch.Tensor:
        device = self.relative_attention_bias.weight.device
        q_pos = torch.arange(q_len, dtype=torch.long, device=device)[:, None]
        k_pos = torch.arange(k_len, dtype=torch.long, device=device)[None, :]
        rel = k_pos - q_pos  # (q, k)

        buckets = self._relative_position_bucket(
            rel,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
            bidirectional=bidirectional,
        )
        # (q,k,heads)
        values = self.relative_attention_bias(buckets)
        # -> (1, heads, q, k)
        return values.permute(2, 0, 1).unsqueeze(0)

    def forward_with_positions(
        self,
        q_pos: torch.Tensor,
        k_pos: torch.Tensor,
        *,
        bidirectional: bool,
    ) -> torch.Tensor:
        """Compute bias for explicit absolute position indices.

        Args:
            q_pos: (q_len,) absolute positions.
            k_pos: (k_len,) absolute positions.

        Returns:
            (1, num_heads, q_len, k_len) bias.
        """
        device = self.relative_attention_bias.weight.device
        q_pos = q_pos.to(device=device, dtype=torch.long)[:, None]
        k_pos = k_pos.to(device=device, dtype=torch.long)[None, :]
        rel = k_pos - q_pos

        buckets = self._relative_position_bucket(
            rel,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
            bidirectional=bidirectional,
        )
        values = self.relative_attention_bias(buckets)
        return values.permute(2, 0, 1).unsqueeze(0)


class KVCache:
    """Simple container for cached keys and values."""

    def __init__(self, key: torch.Tensor, value: torch.Tensor):
        self.key = key    # (b, h, cached_len, d)
        self.value = value  # (b, h, cached_len, d)

    def update(self, new_key: torch.Tensor, new_value: torch.Tensor) -> "KVCache":
        """Append new K/V to the cache and return updated cache."""
        self.key = torch.cat([self.key, new_key], dim=2)
        self.value = torch.cat([self.value, new_value], dim=2)
        return self

    @property
    def seq_len(self) -> int:
        return self.key.size(2)


class MultiHeadAttention(nn.Module):
    """T5-style MHA with optional relative position bias, attention masks, and KV caching."""

    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        d_kv: int,
        dropout: float,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.d_kv = int(d_kv)
        self.dropout = float(dropout)

        inner_dim = self.num_heads * self.d_kv
        self.q = nn.Linear(self.d_model, inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, inner_dim, bias=False)
        self.o = nn.Linear(inner_dim, self.d_model, bias=False)

    def _shape(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        # (b, s, h*d) -> (b, h, s, d)
        return x.view(x.size(0), seq_len, self.num_heads, self.d_kv).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        key_value_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
        causal: bool = False,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, KVCache | None]:
        """Returns: (attn_output, position_bias_used, present_key_value).

        Args:
            hidden_states: Query input (b, q_len, d_model).
            key_value_states: If provided, used for K/V (cross-attention). Otherwise self-attention.
            attention_mask: Additive mask broadcastable to (b, 1, q, k).
            position_bias: Relative position bias (1, h, q, k).
            causal: If True, apply causal mask for autoregressive decoding.
            past_key_value: Cached K/V from previous steps (for incremental decoding).
            use_cache: If True, return updated KVCache.

        Returns:
            Tuple of (output, position_bias, present_key_value).
        """
        bsz, q_len, _ = hidden_states.shape
        is_cross_attention = key_value_states is not None

        # Compute Q
        q = self._shape(self.q(hidden_states), q_len)

        # Compute K/V or use cache
        if is_cross_attention:
            # Cross-attention: K/V come from encoder
            if past_key_value is not None:
                # Reuse cached encoder K/V (doesn't change during decoding)
                k = past_key_value.key
                v = past_key_value.value
                present_key_value = past_key_value if use_cache else None
            else:
                k = self._shape(self.k(key_value_states), key_value_states.size(1))
                v = self._shape(self.v(key_value_states), key_value_states.size(1))
                present_key_value = KVCache(k, v) if use_cache else None
        else:
            # Self-attention
            k_new = self._shape(self.k(hidden_states), q_len)
            v_new = self._shape(self.v(hidden_states), q_len)

            if past_key_value is not None:
                # Append new K/V to cache
                past_key_value.update(k_new, v_new)
                k = past_key_value.key
                v = past_key_value.value
                present_key_value = past_key_value if use_cache else None
            else:
                k = k_new
                v = v_new
                present_key_value = KVCache(k.clone(), v.clone()) if use_cache else None

        k_len = k.size(2)

        # Attention scores: (b, h, q, k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_kv)

        # Position bias
        if position_bias is None:
            position_bias = torch.zeros((1, self.num_heads, q_len, k_len), device=scores.device, dtype=scores.dtype)
        else:
            # During incremental decoding, we may need to slice position_bias
            # to match current q_len (last token) attending to full k_len
            if position_bias.size(2) != q_len or position_bias.size(3) != k_len:
                # Slice: take last q_len rows, all k_len columns
                position_bias = position_bias[:, :, -q_len:, :k_len]

        scores = scores + position_bias

        # Attention mask
        if attention_mask is not None:
            # Expect broadcastable to (b, 1, q_len, k_len).
            # In cached decoding, callers might pass a (b,1,1,1) mask for the last token;
            # we expand it to cover k_len.
            if attention_mask.dim() != 4:
                raise ValueError(f"attention_mask must be 4D additive mask, got shape {tuple(attention_mask.shape)}")

            if attention_mask.size(-1) == 1 and k_len > 1:
                attention_mask = attention_mask.expand(-1, -1, -1, k_len)
            elif attention_mask.size(-1) != k_len:
                raise ValueError(
                    f"attention_mask last dim must match k_len={k_len} (or be 1 for broadcast), got {attention_mask.size(-1)}"
                )

            if attention_mask.size(2) == 1 and q_len > 1:
                attention_mask = attention_mask.expand(-1, -1, q_len, -1)
            elif attention_mask.size(2) != q_len:
                raise ValueError(
                    f"attention_mask q dim must match q_len={q_len} (or be 1 for broadcast), got {attention_mask.size(2)}"
                )

            scores = scores + attention_mask

        # Causal mask for decoder self-attention
        if causal:
            # For incremental decoding: q_len=1, need mask of shape (1, k_len)
            # For full sequence: need upper triangular mask
            if q_len == 1:
                # Single token attending to all previous + itself: no masking needed
                pass
            else:
                causal_mask = torch.full((q_len, k_len), fill_value=-1e9, device=scores.device, dtype=scores.dtype)
                # Account for cached positions: offset the diagonal
                offset = k_len - q_len
                causal_mask = torch.triu(causal_mask, diagonal=1 + offset)
                scores = scores + causal_mask.view(1, 1, q_len, k_len)

        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.matmul(attn, v)  # (b, h, q, d)
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.num_heads * self.d_kv)
        out = self.o(out)

        return out, position_bias, present_key_value


class GatedFeedForward(nn.Module):
    """T5-style gated feed-forward: w_i0, w_i1 with activation on one branch."""

    def __init__(self, *, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.wi_0 = nn.Linear(d_model, d_ff, bias=False)
        self.wi_1 = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_gated = F.silu(self.wi_0(x)) * self.wi_1(x)
        x_gated = F.dropout(x_gated, p=self.dropout, training=self.training)
        return self.wo(x_gated)


@dataclass
class BlockOutput:
    hidden_states: torch.Tensor
    position_bias: torch.Tensor | None = None
    present_key_value: KVCache | None = None


@dataclass
class DecoderBlockOutput:
    hidden_states: torch.Tensor
    position_bias: torch.Tensor | None = None
    encoder_decoder_position_bias: torch.Tensor | None = None
    self_attn_cache: KVCache | None = None
    cross_attn_cache: KVCache | None = None


class EncoderBlock(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        d_kv: int,
        d_ff: int,
        num_heads: int,
        dropout: float,
        layer_norm_epsilon: float,
        has_relative_attention_bias: bool,
        relpos: RelativePositionBias | None,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            d_kv=d_kv,
            dropout=dropout,
        )
        self.self_attn_layer_norm = RMSNorm(d_model, eps=layer_norm_epsilon)
        self.ff_layer_norm = RMSNorm(d_model, eps=layer_norm_epsilon)
        self.ff = GatedFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.dropout = float(dropout)

        self.has_relative_attention_bias = bool(has_relative_attention_bias)
        self.relpos = relpos

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        position_bias: torch.Tensor | None,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> BlockOutput:
        # Self-attn
        normed = self.self_attn_layer_norm(hidden_states)
        if self.has_relative_attention_bias and position_bias is None:
            if self.relpos is None:
                raise RuntimeError("relpos module missing")
            position_bias = self.relpos(normed.size(1), normed.size(1), bidirectional=True)

        attn_out, position_bias, present_key_value = self.self_attn(
            normed,
            attention_mask=attention_mask,
            position_bias=position_bias,
            causal=False,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + F.dropout(attn_out, p=self.dropout, training=self.training)

        # FF
        normed = self.ff_layer_norm(hidden_states)
        ff = self.ff(normed)
        hidden_states = hidden_states + F.dropout(ff, p=self.dropout, training=self.training)

        return BlockOutput(
            hidden_states=hidden_states,
            position_bias=position_bias,
            present_key_value=present_key_value,
        )


class DecoderBlock(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        d_kv: int,
        d_ff: int,
        num_heads: int,
        dropout: float,
        layer_norm_epsilon: float,
        has_relative_attention_bias: bool,
        relpos: RelativePositionBias | None,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            d_kv=d_kv,
            dropout=dropout,
        )
        self.cross_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            d_kv=d_kv,
            dropout=dropout,
        )
        self.self_attn_layer_norm = RMSNorm(d_model, eps=layer_norm_epsilon)
        self.cross_attn_layer_norm = RMSNorm(d_model, eps=layer_norm_epsilon)
        self.ff_layer_norm = RMSNorm(d_model, eps=layer_norm_epsilon)
        self.ff = GatedFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.dropout = float(dropout)

        self.has_relative_attention_bias = bool(has_relative_attention_bias)
        self.relpos = relpos

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
        position_bias: torch.Tensor | None,
        encoder_decoder_position_bias: torch.Tensor | None,
        past_self_attn_key_value: KVCache | None = None,
        past_cross_attn_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> DecoderBlockOutput:
        # Decoder self-attn (causal)
        normed = self.self_attn_layer_norm(hidden_states)
        if self.has_relative_attention_bias and position_bias is None:
            if self.relpos is None:
                raise RuntimeError("relpos module missing")
            # Correct handling for cached decoding:
            # q positions are offset by past length; k positions cover the full cache.
            past_len = past_self_attn_key_value.seq_len if past_self_attn_key_value is not None else 0
            q_len = normed.size(1)
            k_len = past_len + q_len
            device = normed.device
            q_pos = torch.arange(past_len, past_len + q_len, device=device)
            k_pos = torch.arange(0, k_len, device=device)
            position_bias = self.relpos.forward_with_positions(q_pos, k_pos, bidirectional=False)

        sa_out, position_bias, present_self_attn_kv = self.self_attn(
            normed,
            attention_mask=attention_mask,
            position_bias=position_bias,
            causal=True,
            past_key_value=past_self_attn_key_value,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + F.dropout(sa_out, p=self.dropout, training=self.training)

        # Cross-attn (bidirectional relative bias typically not used here; we allow caller-provided bias)
        normed = self.cross_attn_layer_norm(hidden_states)
        ca_out, encoder_decoder_position_bias, present_cross_attn_kv = self.cross_attn(
            normed,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            position_bias=encoder_decoder_position_bias,
            causal=False,
            past_key_value=past_cross_attn_key_value,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + F.dropout(ca_out, p=self.dropout, training=self.training)

        # FF
        normed = self.ff_layer_norm(hidden_states)
        ff = self.ff(normed)
        hidden_states = hidden_states + F.dropout(ff, p=self.dropout, training=self.training)

        return DecoderBlockOutput(
            hidden_states=hidden_states,
            position_bias=position_bias,
            encoder_decoder_position_bias=encoder_decoder_position_bias,
            self_attn_cache=present_self_attn_kv,
            cross_attn_cache=present_cross_attn_kv,
        )

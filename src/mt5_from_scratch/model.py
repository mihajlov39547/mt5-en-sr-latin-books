from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MT5SmallConfig
from .layers import DecoderBlock, EncoderBlock, KVCache, RelativePositionBias, RMSNorm


@dataclass
class Seq2SeqLMOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor
    past_key_values: list | None = None  # For KV caching during generation


def _make_padding_attention_mask(attention_mask: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    """Convert (b, s) mask with 1=keep, 0=pad into additive mask (b,1,1,s)."""
    # additive mask: 0 for keep, -1e9 for mask.
    mask = (1.0 - attention_mask.float()) * -1e9
    return mask[:, None, None, :].to(dtype=dtype)


class MT5Stack(nn.Module):
    def __init__(self, config: MT5SmallConfig, *, is_decoder: bool):
        super().__init__()
        self.config = config
        self.is_decoder = bool(is_decoder)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)

        # T5 shares a single relative position bias across blocks; first block uses it.
        self.relpos = RelativePositionBias(
            num_buckets=config.relative_attention_num_buckets,
            max_distance=config.relative_attention_max_distance,
            num_heads=config.num_heads,
        )

        blocks = []
        for i in range(config.num_layers):
            has_rel_bias = i == 0
            if not is_decoder:
                blocks.append(
                    EncoderBlock(
                        d_model=config.d_model,
                        d_kv=config.d_kv,
                        d_ff=config.d_ff,
                        num_heads=config.num_heads,
                        dropout=config.dropout_rate,
                        layer_norm_epsilon=config.layer_norm_epsilon,
                        has_relative_attention_bias=has_rel_bias,
                        relpos=self.relpos,
                    )
                )
            else:
                blocks.append(
                    DecoderBlock(
                        d_model=config.d_model,
                        d_kv=config.d_kv,
                        d_ff=config.d_ff,
                        num_heads=config.num_heads,
                        dropout=config.dropout_rate,
                        layer_norm_epsilon=config.layer_norm_epsilon,
                        has_relative_attention_bias=has_rel_bias,
                        relpos=self.relpos,
                    )
                )
        self.block = nn.ModuleList(blocks)
        self.final_layer_norm = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_values: list | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, list | None]:
        """Forward pass with optional KV caching.

        Args:
            input_ids: Input token IDs.
            attention_mask: Padding mask (1=keep, 0=pad).
            encoder_hidden_states: For decoder, the encoder outputs.
            encoder_attention_mask: For decoder cross-attention, encoder padding mask.
            past_key_values: List of cached KV per layer. For decoder, each element is
                (self_attn_cache, cross_attn_cache).
            use_cache: If True, return updated past_key_values.

        Returns:
            (hidden_states, present_key_values) where present_key_values is None if use_cache=False.
        """
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).to(torch.long)

        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.dropout(hidden_states)

        # masks
        attn_mask = _make_padding_attention_mask(attention_mask, dtype=hidden_states.dtype)

        if self.is_decoder:
            if encoder_hidden_states is None:
                raise ValueError("decoder requires encoder_hidden_states")
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_states.size(0),
                    encoder_hidden_states.size(1),
                    device=encoder_hidden_states.device,
                    dtype=torch.long,
                )
            enc_attn_mask = _make_padding_attention_mask(encoder_attention_mask, dtype=hidden_states.dtype)
        else:
            enc_attn_mask = None

        # If we're doing incremental decoding with cache, the decoder attention mask must cover
        # the full cached key length, even if input_ids contains only the last token.
        if self.is_decoder and past_key_values is not None and len(past_key_values) > 0:
            first_layer_past = past_key_values[0]
            # past is (self_attn_cache, cross_attn_cache)
            past_self = first_layer_past[0] if first_layer_past is not None else None
            if past_self is not None:
                k_len = past_self.seq_len + input_ids.size(1)
                # Build an all-ones (no padding) mask for cached prefix, and append current attention_mask.
                prefix = torch.ones(
                    (attention_mask.size(0), past_self.seq_len),
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                )
                attention_mask = torch.cat([prefix, attention_mask], dim=1)
                attn_mask = _make_padding_attention_mask(attention_mask, dtype=hidden_states.dtype)

        position_bias = None
        encoder_decoder_position_bias = None
        present_key_values = [] if use_cache else None

        for i, layer in enumerate(self.block):
            layer_past = past_key_values[i] if past_key_values is not None else None

            if not self.is_decoder:
                out = layer(
                    hidden_states,
                    attention_mask=attn_mask,
                    position_bias=position_bias,
                    past_key_value=layer_past,
                    use_cache=use_cache,
                )
                hidden_states = out.hidden_states
                position_bias = out.position_bias
                if use_cache:
                    present_key_values.append(out.present_key_value)
            else:
                past_self_attn = layer_past[0] if layer_past is not None else None
                past_cross_attn = layer_past[1] if layer_past is not None else None

                out = layer(
                    hidden_states,
                    attention_mask=attn_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=enc_attn_mask,
                    position_bias=position_bias,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    past_self_attn_key_value=past_self_attn,
                    past_cross_attn_key_value=past_cross_attn,
                    use_cache=use_cache,
                )
                hidden_states = out.hidden_states
                position_bias = out.position_bias
                encoder_decoder_position_bias = out.encoder_decoder_position_bias
                if use_cache:
                    present_key_values.append((out.self_attn_cache, out.cross_attn_cache))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states, present_key_values


class MT5ForConditionalGeneration(nn.Module):
    """Educational, minimal mT5/T5-like encoder-decoder with tied embeddings."""

    def __init__(self, config: MT5SmallConfig):
        super().__init__()
        self.config = config
        self.encoder = MT5Stack(config, is_decoder=False)
        self.decoder = MT5Stack(config, is_decoder=True)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            # Tie both stacks' input embeddings and output head.
            self.decoder.embed_tokens.weight = self.encoder.embed_tokens.weight
            self.lm_head.weight = self.encoder.embed_tokens.weight

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        past_key_values: list | None = None,
        use_cache: bool = False,
        encoder_outputs: torch.Tensor | None = None,
    ) -> Seq2SeqLMOutput:
        """Forward pass with optional KV caching for efficient generation.

        Args:
            input_ids: Encoder input token IDs.
            attention_mask: Encoder padding mask.
            decoder_input_ids: Decoder input token IDs.
            decoder_attention_mask: Decoder padding mask.
            labels: Target labels for loss computation.
            past_key_values: Cached KV from previous decoding steps.
            use_cache: If True, return updated past_key_values.
            encoder_outputs: Pre-computed encoder hidden states (avoids re-encoding).

        Returns:
            Seq2SeqLMOutput with loss, logits, and optionally past_key_values.
        """
        if decoder_input_ids is None:
            if labels is None:
                raise ValueError("Provide decoder_input_ids or labels")
            decoder_input_ids = self._shift_right(labels)

        # Encode (or reuse cached encoder outputs)
        if encoder_outputs is None:
            enc, _ = self.encoder(input_ids, attention_mask=attention_mask, use_cache=False)
        else:
            enc = encoder_outputs

        # Decode
        dec, present_key_values = self.decoder(
            decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=enc,
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        logits = self.lm_head(dec)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=present_key_values if use_cache else None,
        )

    def _shift_right(self, labels: torch.Tensor) -> torch.Tensor:
        """T5-style shift right: start token is pad_token_id."""
        pad = self.config.pad_token_id
        # replace -100 (if user used it) with pad before shifting
        labels = labels.clone()
        labels[labels == -100] = pad

        shifted = labels.new_zeros(labels.shape)
        shifted[:, 0] = pad
        shifted[:, 1:] = labels[:, :-1]
        return shifted

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        max_length: int | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Greedy autoregressive generation with KV caching.

        Args:
            input_ids: Encoder input token IDs (b, src_len).
            attention_mask: Encoder padding mask.
            max_length: Maximum number of tokens to generate.
            eos_token_id: Stop when this token is generated.

        Returns:
            Generated token IDs (b, generated_len) including the start token.
        """
        self.eval()
        max_length = max_length or self.config.max_length
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        pad_token_id = self.config.pad_token_id

        bsz = input_ids.size(0)
        device = input_ids.device

        # Encode once
        encoder_outputs, _ = self.encoder(input_ids, attention_mask=attention_mask, use_cache=False)

        # Start with pad token (T5's decoder start token)
        decoder_input_ids = torch.full((bsz, 1), pad_token_id, dtype=torch.long, device=device)

        # Track which sequences have finished
        finished = torch.zeros(bsz, dtype=torch.bool, device=device)
        past_key_values = None

        for _ in range(max_length - 1):
            out = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids[:, -1:] if past_key_values else decoder_input_ids,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                use_cache=True,
            )

            # Greedy: take argmax of last position
            next_token_logits = out.logits[:, -1, :]
            next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)  # (b, 1)

            # Replace finished sequences' tokens with pad
            next_tokens = torch.where(finished.unsqueeze(-1), pad_token_id, next_tokens)

            decoder_input_ids = torch.cat([decoder_input_ids, next_tokens], dim=1)
            past_key_values = out.past_key_values

            # Check for EOS
            finished = finished | (next_tokens.squeeze(-1) == eos_token_id)
            if finished.all():
                break

        return decoder_input_ids

"""Minimal, educational mT5/T5-style encoder-decoder Transformer (PyTorch).

This is NOT a reimplementation of Google's training stack, and it does not ship
pretrained weights. It aims to match the T5/mT5 architecture closely enough to:
- understand how it works
- run forward/backward
- pretrain on your own data (span corruption)

See README.md in this folder.
"""

from .config import MT5SmallConfig
from .layers import KVCache
from .model import MT5ForConditionalGeneration

__all__ = [
	"MT5SmallConfig",
	"MT5ForConditionalGeneration",
	"KVCache",
]

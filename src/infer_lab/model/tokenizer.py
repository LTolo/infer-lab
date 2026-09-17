"""A deterministic byte-level tokenizer.

Real tokenizers are orthogonal to inference-engine work, and shipping a
pretrained vocabulary would make the repo non-self-contained.  A byte-level
vocabulary keeps round-tripping exact and the vocab size fixed at 256 + specials.
"""

from __future__ import annotations

BOS_ID = 256
EOS_ID = 257
PAD_ID = 258
NUM_SPECIAL = 3


class ByteTokenizer:
    vocab_size = 256 + NUM_SPECIAL

    def encode(self, text: str, *, add_bos: bool = True) -> list[int]:
        ids = list(text.encode("utf-8"))
        return [BOS_ID, *ids] if add_bos else ids

    def decode(self, ids: list[int]) -> str:
        payload = bytes(i for i in ids if i < 256)
        return payload.decode("utf-8", errors="replace")

    def is_eos(self, token_id: int) -> bool:
        return token_id == EOS_ID

"""Speculative decoding (draft + target verification).

Decode is memory-bound: generating one token costs a full pass over the weights
regardless of how many tokens you actually produce.  Speculative decoding
exploits that slack -- a cheap draft model proposes ``k`` tokens, and the target
model verifies all ``k`` in a *single* forward pass.  If the draft is right, you
got k tokens for the price of one; if it is wrong, you fall back to exactly the
token the target would have produced anyway.

Crucially this is **lossless** for greedy decoding: the accepted sequence is
token-for-token identical to plain autoregressive decoding with the target
model.  ``tests/test_speculative.py`` asserts exactly that.

KV rollback
-----------
Rejected draft tokens have already written KV.  With a paged cache the rollback
is free: truncating ``SequenceKV.length`` makes those slots logically invisible
and the next write overwrites them in place.  No memcpy, no reallocation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from infer_lab.engine.request import SamplingParams
from infer_lab.engine.sampling import sample_token, token_probs
from infer_lab.kv.paged_cache import PagedKVCache, SequenceKV
from infer_lab.model.numpy_model import NumpyTransformer


@dataclass
class SpeculationStats:
    rounds: int = 0
    proposed: int = 0
    accepted: int = 0
    bonus_tokens: int = 0
    target_forwards: int = 0
    draft_forwards: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def speedup_estimate(self) -> float:
        """Tokens emitted per target forward pass -- the metric that actually matters."""
        emitted = self.accepted + self.bonus_tokens
        return emitted / self.target_forwards if self.target_forwards else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "rounds": self.rounds,
            "proposed": self.proposed,
            "accepted": self.accepted,
            "bonus_tokens": self.bonus_tokens,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "target_forwards": self.target_forwards,
            "draft_forwards": self.draft_forwards,
            "tokens_per_target_forward": round(self.speedup_estimate, 4),
        }


@dataclass
class SpeculativeDecoder:
    target: NumpyTransformer
    draft: NumpyTransformer
    target_cache: PagedKVCache
    draft_cache: PagedKVCache
    num_draft_tokens: int = 4
    stats: SpeculationStats = field(default_factory=SpeculationStats)

    # ------------------------------------------------------------------ proposal
    def _propose(self, draft_kv: SequenceKV, last_token: int, position: int,
                 params: SamplingParams, rng: np.random.Generator) -> list[int]:
        """Autoregressively extend by ``num_draft_tokens`` using the cheap model."""
        proposals: list[int] = []
        token, pos = last_token, position
        for _ in range(self.num_draft_tokens):
            logits = self.draft.forward_prefill(np.array([token]), draft_kv, start_pos=pos)
            self.stats.draft_forwards += 1
            token = sample_token(logits, params, rng)
            proposals.append(token)
            pos += 1
        return proposals

    # -------------------------------------------------------------- verification
    def _verify(self, target_kv: SequenceKV, last_token: int, position: int,
                proposals: list[int], params: SamplingParams,
                rng: np.random.Generator) -> tuple[list[int], int]:
        """One target forward over [last_token, *proposals[:-1]].

        Returns ``(accepted_tokens, num_accepted_proposals)``.  The returned list
        always contains at least one token: either a corrected token at the first
        mismatch, or the bonus token when every proposal was accepted.
        """
        window = [last_token, *proposals[:-1]]
        logits = self.target.forward_prefill(
            np.array(window), target_kv, start_pos=position, return_all_logits=True
        )
        self.stats.target_forwards += 1

        accepted: list[int] = []
        for i, proposed in enumerate(proposals):
            if params.temperature == 0.0:
                target_token = int(np.argmax(logits[i]))
                if target_token != proposed:
                    accepted.append(target_token)
                    return accepted, i
            else:
                # Modified rejection sampling: accept with p_target/p_draft, else
                # resample from the residual so the output distribution is exact.
                p_target = token_probs(logits[i], params)
                if rng.random() > float(p_target[proposed]):
                    accepted.append(int(rng.choice(p_target.shape[-1], p=p_target)))
                    return accepted, i
            accepted.append(proposed)

        bonus = sample_token(logits[-1], params, rng)
        accepted.append(bonus)
        self.stats.bonus_tokens += 1
        return accepted, len(proposals)

    # ---------------------------------------------------------------- public API
    def generate(self, prompt_token_ids: list[int], params: SamplingParams,
                 *, eos_id: int | None = None) -> list[int]:
        rng = np.random.default_rng(params.seed if params.seed is not None else 0)
        target_kv = self.target_cache.new_sequence()
        draft_kv = self.draft_cache.new_sequence()

        prompt = np.asarray(prompt_token_ids, dtype=np.int64)
        logits = self.target.forward_prefill(prompt, target_kv)
        self.stats.target_forwards += 1
        self.draft.forward_prefill(prompt, draft_kv)
        self.stats.draft_forwards += 1

        output = [sample_token(logits, params, rng)]
        position = len(prompt_token_ids)

        while len(output) < params.max_tokens:
            if eos_id is not None and not params.ignore_eos and output[-1] == eos_id:
                break

            draft_kv.length = position          # roll back any rejected draft KV
            proposals = self._propose(draft_kv, output[-1], position, params, rng)
            self.stats.rounds += 1
            self.stats.proposed += len(proposals)

            target_kv.length = position
            accepted, num_accepted = self._verify(
                target_kv, output[-1], position, proposals, params, rng
            )
            self.stats.accepted += num_accepted

            for token in accepted:
                if len(output) >= params.max_tokens:
                    break
                output.append(token)
                position += 1

            # discard KV written for rejected proposals
            target_kv.length = position
            draft_kv.length = min(draft_kv.length, position)

        target_kv.release()
        draft_kv.release()
        return output[:params.max_tokens]

    def reset_stats(self) -> None:
        self.stats = SpeculationStats()

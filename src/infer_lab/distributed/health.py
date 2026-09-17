"""Distributed failure detection for the collective ring.

A collective is a barrier: if one rank dies, every other rank blocks forever on
``recv``.  In production that shows up as a *silent* throughput cliff, not a
crash, which is the single most expensive failure mode in distributed inference.

The monitor here turns that into a diagnosable event: every rank heartbeats, a
peer that misses ``failure_threshold`` intervals is declared DEAD, and the ring
is torn down with a message naming the culprit instead of hanging.
"""

from __future__ import annotations

import enum
import threading
import time
from dataclasses import dataclass, field


class PeerState(enum.Enum):
    HEALTHY = "healthy"
    SUSPECT = "suspect"     # missed at least one interval
    DEAD = "dead"           # missed failure_threshold intervals
    UNKNOWN = "unknown"     # never heartbeated


@dataclass
class PeerHealth:
    rank: int
    last_heartbeat: float | None = None
    state: PeerState = PeerState.UNKNOWN
    missed_intervals: int = 0
    observed_step: int = -1

    def age_s(self, now: float | None = None) -> float:
        if self.last_heartbeat is None:
            return float("inf")
        return (now or time.monotonic()) - self.last_heartbeat


@dataclass
class HealthMonitor:
    """Heartbeat-based phi-lite failure detector for a fixed-size ring."""

    world_size: int
    interval_s: float = 0.5
    failure_threshold: int = 3
    peers: dict[int, PeerHealth] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self.peers = {r: PeerHealth(rank=r) for r in range(self.world_size)}

    # -------------------------------------------------------------------- report
    def heartbeat(self, rank: int, step: int = -1) -> None:
        with self._lock:
            peer = self.peers[rank]
            peer.last_heartbeat = time.monotonic()
            peer.missed_intervals = 0
            peer.state = PeerState.HEALTHY
            if step >= 0:
                peer.observed_step = step

    def evaluate(self, now: float | None = None) -> dict[int, PeerState]:
        now = now or time.monotonic()
        with self._lock:
            for peer in self.peers.values():
                if peer.last_heartbeat is None:
                    peer.state = PeerState.UNKNOWN
                    continue
                missed = int(peer.age_s(now) // self.interval_s)
                peer.missed_intervals = missed
                if missed == 0:
                    peer.state = PeerState.HEALTHY
                elif missed < self.failure_threshold:
                    peer.state = PeerState.SUSPECT
                else:
                    peer.state = PeerState.DEAD
            return {r: p.state for r, p in self.peers.items()}

    # ------------------------------------------------------------------ queries
    def dead_peers(self, now: float | None = None) -> list[int]:
        return [r for r, s in self.evaluate(now).items() if s is PeerState.DEAD]

    def healthy_count(self, now: float | None = None) -> int:
        return sum(1 for s in self.evaluate(now).values() if s is PeerState.HEALTHY)

    def is_quorate(self, now: float | None = None) -> bool:
        """A collective needs *every* rank -- there is no partial all-reduce."""
        return self.healthy_count(now) == self.world_size

    def stragglers(self, now: float | None = None) -> list[int]:
        """Ranks lagging behind the leading step -- the cause of bubble growth."""
        with self._lock:
            steps = [p.observed_step for p in self.peers.values() if p.observed_step >= 0]
            if not steps:
                return []
            leader = max(steps)
            return sorted(r for r, p in self.peers.items()
                          if 0 <= p.observed_step < leader)

    def snapshot(self, now: float | None = None) -> dict[str, object]:
        states = self.evaluate(now)
        return {
            "world_size": self.world_size,
            "healthy": self.healthy_count(now),
            "dead": [r for r, s in states.items() if s is PeerState.DEAD],
            "suspect": [r for r, s in states.items() if s is PeerState.SUSPECT],
            "stragglers": self.stragglers(now),
            "quorate": self.is_quorate(now),
        }

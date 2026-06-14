"""Live node-speed tracking — the latency signal behind speed-aware routing (DESIGN §5).

The provider pool learns how fast each node answers from REAL traffic (passive, no wasted calls) and
folds it into a per-``(node, model)`` estimate the router uses to break ties toward the fastest
healthy node. A rare idle probe tops up quiet nodes (see the brain's idle-latency task). This module
is the pure, deterministic core: the normalizer (so a 3-word reply and a 500-word reply are
comparable) and the EWMA accumulator. No I/O, no threads — the pool owns those — so it is
unit-testable with a fake clock.

Unit: **seconds per ~256-token turn** — the SAME unit the benchmark writes to the catalogue's
``return_time``, so a live estimate can seed from, and compare against, the qualification snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass

# Report seconds per ~256-token turn so the number is verbosity-independent — a terse reply and a
# long one on the same node land near the same value. Shared with the benchmark so live and
# qualified latencies are one unit. ~4 chars/token keeps it tokenizer-free (no core dependency).
LATENCY_NORM_TOKENS: int = 256
LATENCY_MIN_TOKENS: int = 32
_CHARS_PER_TOKEN: int = 4


def normalize_latency(elapsed_s: float, output: str) -> float:
    """Wall-clock seconds for one generation → seconds per ~256-token turn (verbosity-independent).

    The floor (``LATENCY_MIN_TOKENS``) stops a terse/refusing reply from dividing by a tiny token
    count into a nonsense-fast number.
    """
    approx_tokens = max(LATENCY_MIN_TOKENS, len(output) // _CHARS_PER_TOKEN)
    return round(elapsed_s / approx_tokens * LATENCY_NORM_TOKENS, 3)


@dataclass
class LatencyStat:
    """An exponentially-weighted estimate of one ``(node, model)``'s seconds-per-turn.

    ``value`` is the current estimate (``None`` until known). Real samples drive it; a *seed* (the
    benchmark snapshot) only fills the gap before any real sample exists, so live traffic always
    wins over a frozen qualification number. ``samples`` counts only real observations (seeds don't
    count), so a consumer can tell a measured node from a merely-seeded one.
    """

    value: float | None = None
    samples: int = 0
    last_ts: float = 0.0

    def observe(self, sample_s: float, *, alpha: float, now: float) -> None:
        """Fold a real measurement in. The first real sample replaces any seed outright; after that
        it's an EWMA (``alpha`` weights the newest sample) so the estimate tracks current load."""
        if self.samples == 0 or self.value is None:
            self.value = sample_s
        else:
            self.value = round(alpha * sample_s + (1 - alpha) * self.value, 3)
        self.samples += 1
        self.last_ts = now

    def seed(self, value_s: float) -> None:
        """Prime the estimate from the catalogue's qualification latency — only while no real sample
        has landed (``samples == 0``), so seeding never overwrites lived experience."""
        if self.samples == 0:
            self.value = value_s

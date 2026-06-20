"""Storage-pipeline throughput probe -- does the SQLite single-writer keep up on weak hardware?

The one piece of Mimir's performance story that's *unverified* until you run it on the real target:
on a Raspberry Pi or an old laptop (slow CPU, SD-card / spinning disk), can the storage layer absorb
a turn's writes without becoming the thing the user waits on? Run this ON THAT MACHINE before
deploying there:

    python bench/storage_throughput.py                 # defaults: 2000 writes, 200 turns, dim 256
    python bench/storage_throughput.py --turns 500 --dim 768 --db /tmp/probe.db

It drives the REAL `StorageGateway` (the async single-writer + WAL the live system uses) the way a
turn does -- a burst of writes per "turn" (a memory + its embedding, a triple, the exchange log, the
conversation log, access bookkeeping) -- and reports:

  - sustained write throughput (memories/sec),
  - per-turn write-burst DRAIN latency (p50 / p95 / max) -- the number that matters,
  - read latency (recall) under concurrent write load.

The bar: a turn's write burst should drain in well under the time a local model takes to generate a
reply (hundreds of ms to seconds). If the p95 drain is a meaningful fraction of a second on your
hardware, the SD card / disk is your bottleneck -- move the DB to faster storage (USB SSD, tmpfs for
the WAL) before blaming the model.

This is a probe, not a test -- not part of the pytest suite (it writes thousands of rows and is
wall-clock bound). Zero extra dependencies; uses only what the core already ships.
"""

from __future__ import annotations

import argparse
import math
import tempfile
import threading
import time
from pathlib import Path

from mimir.cognition.working_memory import record_exchange
from mimir.storage.gateway import StorageGateway
from mimir.storage.models import Memory, MemoryKind, Triple
from mimir.storage.repo import (
    list_memories,
    record_access,
    record_conversation_turn,
    save_memory,
    save_triple,
)


def _embedding(dim: int, seed: int) -> list[float]:
    """A deterministic vector (no numpy, no RNG dependency) -- only the byte cost matters here."""
    return [math.sin(seed * 0.1 + i) for i in range(dim)]


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1))))
    return ordered[k]


def _write_throughput(gw: StorageGateway, n: int, dim: int) -> float:
    """Sustained memory-write rate: save_memory blocks until committed -> a real drain rate."""
    start = time.perf_counter()
    for i in range(n):
        save_memory(gw, Memory(text=f"throughput probe row {i}", embedding=_embedding(dim, i),
                               user="probe"))
    elapsed = time.perf_counter() - start
    return n / elapsed if elapsed else float("inf")


def _turn_burst(gw: StorageGateway, turn: int, dim: int) -> float:
    """Simulate one turn's write burst and return its drain latency (ms). The shape a live turn
    commits: a fact + embedding, a graph triple, the two recency logs, and an access touch."""
    start = time.perf_counter()
    mid = save_memory(gw, Memory(text=f"turn {turn} fact", embedding=_embedding(dim, turn),
                                 user="probe"))
    save_triple(gw, Triple(subject=f"entity{turn}", relation="relates to",
                           object=f"entity{turn + 1}", user="probe"))
    record_exchange(gw, user="probe", user_text=f"q{turn}", reply=f"a{turn}")
    record_conversation_turn(gw, user="probe", user_text=f"q{turn}", reply=f"a{turn}",
                             session_id="probe")
    record_access(gw, [mid])
    gw.flush()  # wait for the whole burst to land -- what the next turn's reads would see
    return (time.perf_counter() - start) * 1000.0


def _read_latency_under_load(gw: StorageGateway, samples: int, dim: int) -> tuple[list[float], int]:
    """Measure recall (list_memories) latency while a background thread writes -- contended case."""
    stop = threading.Event()
    writes = [0]

    def writer() -> None:
        i = 0
        while not stop.is_set():
            save_memory(gw, Memory(text=f"bg {i}", embedding=_embedding(dim, i), user="bg"))
            i += 1
            time.sleep(0.002)  # throttle to a realistic write cadence (~hundreds/sec, not a flood)
        writes[0] = i

    t = threading.Thread(target=writer, daemon=True)
    t.start()
    latencies: list[float] = []
    for _ in range(samples):
        start = time.perf_counter()
        list_memories(gw, kind=MemoryKind.MEMORY)
        latencies.append((time.perf_counter() - start) * 1000.0)
        time.sleep(0.005)
    stop.set()
    t.join(timeout=2.0)
    return latencies, writes[0]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=2000, help="memories for the throughput phase")
    ap.add_argument("--turns", type=int, default=200, help="simulated turns for the burst phase")
    ap.add_argument("--dim", type=int, default=256, help="embedding dimension (byte cost per row)")
    ap.add_argument("--reads", type=int, default=200, help="recall samples under write load")
    ap.add_argument("--db", default=None, help="db path (default: a temp file, deleted after)")
    args = ap.parse_args()

    tmp = None
    if args.db:
        db_path = args.db
    else:
        tmp = tempfile.TemporaryDirectory()
        db_path = str(Path(tmp.name) / "throughput.db")

    print(f"Storage throughput probe -> {db_path}  (dim={args.dim})\n")
    gw = StorageGateway(db_path)
    try:
        rate = _write_throughput(gw, args.n, args.dim)
        print(f"  writes:        {rate:8.0f} memories/sec  ({args.n} rows, {args.dim}-d vectors)")

        drains = [_turn_burst(gw, i, args.dim) for i in range(args.turns)]
        print(f"  turn burst:    p50 {_pct(drains, 50):6.1f} ms   p95 {_pct(drains, 95):6.1f} ms   "
              f"max {max(drains):6.1f} ms   ({args.turns} turns)")

        reads, bg = _read_latency_under_load(gw, args.reads, args.dim)
        print(f"  recall@load:   p50 {_pct(reads, 50):6.1f} ms   p95 {_pct(reads, 95):6.1f} ms   "
              f"max {max(reads):6.1f} ms   (while {bg} bg writes landed)")

        p95 = _pct(drains, 95)
        verdict = ("[ok] storage keeps up -- a turn's writes drain far faster than a model reply"
                   if p95 < 100 else
                   "[ok] ok -- drains comfortably under a typical reply" if p95 < 500 else
                   "[!] slow storage -- the per-turn drain is a meaningful fraction of a second; "
                   "move the DB to faster storage (USB SSD / tmpfs WAL) before deploying here")
        stats = gw.get_stats()
        print(f"\n  {verdict}")
        print(f"  gateway: {stats['written']} writes, {stats['coalesced']} coalesced, "
              f"{stats['retries']} retries, {stats['errors']} errors")
    finally:
        gw.close()
        if tmp:
            tmp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

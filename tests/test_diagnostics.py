"""Self-observability (DESIGN §10): the error-capture ring, the live context block, the nightly
digest. Errors logged anywhere under the `mimir` logger become visible to the system itself."""

from __future__ import annotations

import logging

from mimir.brain import Mimir
from mimir.diagnostics import ErrorRecord, RingErrorHandler, render_errors


def test_ring_captures_warning_and_above_not_info() -> None:
    h = RingErrorHandler(capacity=10)
    lg = logging.getLogger("mimir.test.iso1")
    lg.addHandler(h)
    lg.setLevel(logging.DEBUG)
    try:
        lg.info("ignored-info")
        lg.warning("w1")
        try:
            raise ValueError("boom")
        except Exception:
            lg.error("e1", exc_info=True)
    finally:
        lg.removeHandler(h)
    msgs = [r.message for r in h.recent(limit=99)]
    assert not any("ignored-info" in m for m in msgs)         # below WARNING → not captured
    assert any("w1" in m for m in msgs)
    assert any("e1" in m and "ValueError: boom" in m for m in msgs)  # exc summary appended
    assert h.counts() == {"WARNING": 1, "ERROR": 1}


def test_ring_is_capped_and_filters_by_level() -> None:
    h = RingErrorHandler(capacity=3)
    lg = logging.getLogger("mimir.test.iso2")
    lg.addHandler(h)
    lg.setLevel(logging.DEBUG)
    try:
        for i in range(5):
            lg.warning(f"w{i}")
        lg.error("the-error")
    finally:
        lg.removeHandler(h)
    assert len(h.recent(limit=99)) == 3                       # oldest dropped (ring of 3)
    assert [r.message for r in h.recent(limit=99, min_level="ERROR")] == ["the-error"]


def test_within_window_filters_by_recency() -> None:
    h = RingErrorHandler()
    lg = logging.getLogger("mimir.test.iso3")
    lg.addHandler(h)
    lg.setLevel(logging.DEBUG)
    try:
        lg.warning("recent")
    finally:
        lg.removeHandler(h)
    now = h.recent()[-1].ts
    assert h.within(3600, now)                                # within an hour → present
    assert h.within(60, now + 10_000) == []                  # far in the future → aged out


def test_render_errors_is_compact() -> None:
    out = render_errors([
        ErrorRecord(ts=0, level="ERROR", logger="mimir.sentinel", message="sentinel failed — X"),
        ErrorRecord(ts=0, level="WARNING", logger="mimir.fleet", message="node down"),
    ])
    assert out == "- [error] sentinel: sentinel failed — X\n- [warning] fleet: node down"


def test_brain_surfaces_recent_errors_in_context(brain: Mimir) -> None:
    logging.getLogger("mimir.fleet").warning("node .50 unreachable ZZMARKER")
    block = brain._error_context()
    assert block and "ZZMARKER" in block
    brain.config.surface_errors = False                      # the off-switch
    assert brain._error_context() is None


def test_brain_health_digest_records_and_reads_back(brain: Mimir) -> None:
    try:
        raise RuntimeError("nightly boom")
    except Exception:
        logging.getLogger("mimir.sleep").error("consolidation failed", exc_info=True)
    digest = brain.digest_errors()
    assert digest["total"] >= 1 and digest["counts"].get("ERROR", 0) >= 1
    assert brain.health_digest()["total"] == digest["total"]  # persisted to kv, reads back


def test_build_context_emits_system_health_section() -> None:
    from mimir.context.build import build_context
    from mimir.embed.base import EmbeddingMode
    bundle = build_context(
        query="hi", user=None, identity="x", retrieved=[], sentinel_note=None,
        embed_mode=EmbeddingMode.BOOTSTRAP, budget_tokens=4096,
        system_health="- [error] sentinel: boom",
    )
    assert any(s.name == "system_health" for s in bundle.sections)


def test_backend_health_single_provider_is_quiet(brain: Mimir) -> None:
    # Single local provider → no fleet health line, but pool_health still works.
    assert brain._backend_health_line() is None
    assert brain.pool_health()["nodes"] >= 1


def test_backend_degraded_surfaces_in_context(brain: Mimir) -> None:
    from mimir.config import RoleSpec
    from mimir.model.gateway import ModelGateway
    from mimir.model.pool import ProviderPool
    from tests.test_model_pool import DownProvider, FleetProvider

    a = FleetProvider("A", ["m"])
    b = DownProvider("B", ["m"])
    pool = ProviderPool([("A", a), ("B", b)], sleep=lambda _: None)
    pool.refresh()
    brain._model = ModelGateway(pool, {"chat": RoleSpec(model="m")})
    line, degraded = brain._backend_health_line()
    assert degraded and "1/2 nodes up" in line and "down: B" in line
    ctx = brain._error_context()                      # degraded backend shows even with no errors
    assert ctx and "Backend:" in ctx and "down: B" in ctx

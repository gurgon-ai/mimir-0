"""Executable spec for the second pass of 2026-10-31: the capability ledger, tools on the streaming
path, the unkept-action follow-up, the chat window, keep_alive, system-generated provenance kept
out of knowledge, and names snapped at the door.
"""

from __future__ import annotations

import time

from mimir.brain import Mimir
from mimir.cognition.burst import BurstResult, BurstWorker, ResponseContext
from mimir.cognition.ledger import CapabilityLedger
from mimir.cognition.names import Roster, phonetic_key, snap
from mimir.cognition.tools import ToolCall
from mimir.config import Config
from mimir.storage.models import EvidenceTier, Memory, MemoryKind, is_system_generated
from mimir.storage.repo import save_memory

# -- ledger -------------------------------------------------------------------------------


def test_ledger_registers_touches_reviews_and_persists(brain: Mimir) -> None:
    t = [1_000_000.0]
    led = CapabilityLedger(brain._storage, clock=lambda: t[0])
    led.register("tool", "draw")
    led.register("tool", "notebook")
    led.touch("tool", "notebook", ok=True)
    assert led.review() == []  # inside the grace period: quiet
    t[0] += 8 * 86400
    found = {(f.kind, f.name, f.status) for f in led.review()}
    assert ("tool", "draw", "never") in found  # registered, never fired
    assert ("tool", "notebook", "stale") not in found  # 8 d silent is under the 14 d bar
    t[0] += 8 * 86400
    found = {(f.kind, f.name, f.status) for f in led.review()}
    assert ("tool", "notebook", "stale") in found
    for _ in range(3):
        led.touch("tool", "draw", ok=False)
    assert ("tool", "draw", "failing") in {(f.kind, f.name, f.status) for f in led.review()}
    # survives a reload
    again = CapabilityLedger(brain._storage, clock=lambda: t[0])
    assert again.snapshot()["tool:notebook"]["count"] == 1
    assert "gone quiet" in led.render(led.review())


def test_the_brain_registers_its_capabilities_and_the_burst_touches_them(brain: Mimir) -> None:
    snap_ = brain._ledger.snapshot()
    assert "tool:notebook" in snap_ and "burst:sentinel" in snap_ and "loop:inner_life" in snap_
    brain.turn("hello there", user="g")
    brain._burst.wait_idle()
    assert brain._ledger.snapshot()["burst:sentinel"]["count"] >= 1
    rep = brain.capability_report()
    assert "entries" in rep and "findings" in rep


def test_burst_on_ran_reports_failures(monkeypatch) -> None:
    seen: list[tuple[str, bool]] = []
    w = BurstWorker(on_ran=lambda n, ok: seen.append((n, ok)))

    def boom() -> BurstResult:
        raise RuntimeError("x")

    w.register("good", lambda ctx: lambda: BurstResult(), base_priority=1.0)
    w.register("bad", lambda ctx: boom, base_priority=2.0)
    w.signal(ResponseContext(user_text="a", reply="b"))
    w.drain_once()
    assert ("good", True) in seen and ("bad", False) in seen
    assert set(w.task_names()) == {"good", "bad"}


# -- tier three: an unkept action is surfaced, a kept one is not ------------------------------


def test_a_failed_tool_call_is_put_in_front_of_the_next_turn(brain: Mimir) -> None:
    failed = ToolCall(tool="notebook", args={})
    failed.status, failed.result = "error", "tool 'notebook' failed: disk"
    ok = ToolCall(tool="record_milestone", args={})
    ok.status, ok.result = "ok", "recorded"
    task = brain._actions_followup_task(
        ResponseContext(user_text="x", reply="y", actions=[failed, ok])
    )
    assert task is not None
    out = task()
    assert (
        "did not run" in out.surface
        and "notebook" in out.surface
        and "record_milestone" not in out.surface
    )
    assert (
        brain._actions_followup_task(ResponseContext(user_text="x", reply="y", actions=[ok]))
        is None
    )


# -- streaming path runs a tool round --------------------------------------------------------


def test_tools_run_on_the_streaming_path(brain: Mimir, monkeypatch) -> None:
    from mimir.cognition import notebook as nb

    calls: list[list] = []

    def fake_stream(role: str, messages: list, **k: object):
        calls.append(messages)
        has_results = any("Tool results:" in str(m.get("content")) for m in messages)
        if not has_results:
            yield ('<TOOL name="notebook" '
                   'args={"op": "write", "title": "ideas", "body": "stream idea"}>')
            return
        yield from ("Noted", " in", " the", " notebook.")

    monkeypatch.setattr(brain._model, "chat_stream", fake_stream)
    gen = brain.turn_stream("jot this in my notebook: stream idea", user="alex")
    tokens: list[str] = []
    try:
        while True:
            tokens.append(next(gen))
    except StopIteration as stop:
        intro = stop.value
    assert "".join(tokens) == "Noted in the notebook."  # the marker was never streamed
    assert nb.read(brain._storage, "ideas") == "stream idea"  # the tool actually ran
    assert intro["actions"][0]["tool"] == "notebook" and intro["actions"][0]["status"] == "ok"
    assert len(calls) == 2
    assert brain._ledger.snapshot()["tool:notebook"]["count"] >= 1


# -- a turn arriving with queued burst work settles it, it does not wait out the timeout -------


def test_a_turn_with_pending_burst_work_settles_it_instead_of_timing_out(brain: Mimir) -> None:
    """Regression (2026-11-01): declaring the foreground active BEFORE waiting on the burst made
    the burst yield to the very turn that was waiting for it; the turn then sat out wait_idle's
    30 s. Seen only on a slow CI runner, where turn two arrived before turn one's window ran."""
    import threading

    brain._burst.stop()  # park the loop so the queued window cannot run yet
    brain._burst.signal(ResponseContext(user_text="earlier", reply="noted", user="g"))
    assert brain._burst.pending() >= 1
    seen_active: list[bool] = []
    done = threading.Event()

    def _turn() -> None:
        brain.turn("hello again", user="g")
        done.set()

    threading.Thread(target=_turn, daemon=True).start()
    time.sleep(0.3)
    seen_active.append(brain._turn_active)
    assert done.is_set() is False  # the turn is waiting on the burst, as designed
    brain._burst.start()  # the loop comes back; with the fix it drains and the turn proceeds
    assert done.wait(timeout=8.0), "turn waited out the burst timeout instead of settling it"
    assert seen_active == [False]  # the foreground was not declared while it waited


# -- chat window ---------------------------------------------------------------------------


def test_in_conversation_covers_the_window_after_a_turn(brain: Mimir) -> None:
    brain._turn_active = False
    brain._last_turn_at = 0.0
    assert brain.in_conversation() is False
    brain._last_turn_at = time.time() - 10
    assert brain.in_conversation() is True
    brain._last_turn_at = time.time() - brain.config.chat_window_s - 1
    assert brain.in_conversation() is False
    brain._turn_active = True
    assert brain.in_conversation() is True
    brain._turn_active = False


# -- keep_alive is a top-level field -----------------------------------------------------------


def test_keep_alive_goes_top_level_not_into_options(monkeypatch) -> None:
    from mimir.model.providers.ollama import OllamaProvider

    prov = OllamaProvider("http://x")
    seen: list[dict] = []

    def fake_post(path, payload, timeout=None):
        seen.append(payload)
        return {"message": {"content": "hi there"}, "done_reason": "stop"}

    monkeypatch.setattr(prov, "_post", fake_post)
    prov.chat("m", [{"role": "user", "content": "q"}], {"keep_alive": -1, "num_ctx": 8192})
    assert seen[0]["keep_alive"] == -1 and "keep_alive" not in seen[0]["options"]
    seen.clear()
    prov.chat("m", [{"role": "user", "content": "q"}], {"num_ctx": 8192})
    assert "keep_alive" not in seen[0]


# -- the system's own notes are not knowledge -------------------------------------------------


def test_system_generated_provenance_is_kept_out_of_facts_and_self_model(brain: Mimir) -> None:
    assert is_system_generated("inner life") and is_system_generated("inner council")
    assert not is_system_generated("stated by alex") and not is_system_generated("conversation")
    from mimir.cognition.narratives import _gather_daily_material
    from mimir.cognition.self_model import build_brief, gather_signals

    now = time.time()
    for prov in ("stated by alex", "inner life"):
        save_memory(
            brain._storage,
            Memory(
                text=f"A memory with provenance {prov} and some length.",
                kind=MemoryKind.MEMORY,
                evidence_tier=EvidenceTier.INFERRED,
                confidence=0.5,
                salience=1.0,
                embedding=None,
                provenance=prov,
                user=None,
            ),
        )
    material, _ = _gather_daily_material(brain._storage, now - 60)
    assert "stated by alex" in material and "provenance inner life" not in material
    sig = gather_signals(brain._storage)
    assert sig.own_notes >= 1 and "of my own thinking" in build_brief(sig)


# -- names at the door -------------------------------------------------------------------------


def test_phonetic_keys_and_the_two_snap_tiers() -> None:
    assert phonetic_key("Katelyn") == phonetic_key("Caitlin") == phonetic_key("Kaitlyn")
    assert phonetic_key("Kohl") == phonetic_key("Cole")
    r = Roster()
    r.add("Caitlin", ["Katelyn", "Kaitlyn"])
    r.add("Cole", ["Kohl"])
    r.add_from_text("Sam, and the horses Pepper and Rex")
    assert "pepper" in r.names and "sam" in r.names
    assert (
        snap("katelyn seems insistent. Draw Kohl and KATELYN together", r)[0]
        == "Caitlin seems insistent. Draw Cole and CAITLIN together"
    )
    assert snap("Draw Katelin by the gate please", r)[1] == [("Katelin", "Caitlin")]  # phonetic tier
    assert snap("Code is what we write. The code compiles.", r)[1] == []  # sentence start
    assert snap("", r)[1] == []


def test_names_are_snapped_at_the_turn_from_config(
    db_path: str, mock_config: Config, monkeypatch
) -> None:
    mock_config.known_names = ["Caitlin"]
    mock_config.name_variants = {"Caitlin": ["Katelyn"]}
    m = Mimir(mock_config)
    try:
        seen: list[str] = []

        def fake_chat(role: str, messages: list, **k: object) -> str:
            seen.append(messages[-1]["content"])
            return "Noted."

        monkeypatch.setattr(m._model, "chat", fake_chat)
        m.turn("Katelyn is asleep on the couch", user="g")
        assert any("Caitlin is asleep" in s for s in seen) and not any("Katelyn" in s for s in seen)
    finally:
        m.close()

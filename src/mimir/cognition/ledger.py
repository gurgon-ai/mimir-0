"""ledger.py — the capability ledger: when did each part of the system last do anything?

Silent feature death was the dominant bug class in the production system this core came from: a
tool refused by a gate for months behind an intercept that caught most requests; a watcher that
trusted an HTTP probe instead of its process handle; a post-turn task that skipped itself. Nothing
anywhere was different for any of it, because nothing recorded that a capability had stopped
firing. A system that cannot feel a part of itself stop cannot fix it, report it, or route around it.

This is the sense. Every capability the brain exposes — tools, burst tasks, sleep phases, the idle
loops — is *registered* here at boot and *touched* whenever it runs. The ledger persists in the kv
table (one JSON document, written through the storage gateway's single writer), so it survives a
restart. :meth:`CapabilityLedger.review` turns it into findings: registered and never fired after a
grace period, or fired before but not for longer than ``stale_after_s``. The brain puts those into
the system-health prompt section and the nightly digest; the operator sees them in the UI.

Touching is cheap (a dict write and one small kv write) and never raises — a ledger fault must not
break the capability it was recording (DESIGN §10).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from ..storage.gateway import StorageGateway
from ..storage.repo import kv_get, kv_set

log = logging.getLogger("mimir.ledger")

KV_KEY = "capability_ledger"
DEFAULT_GRACE_S = 7 * 86400          # a registered capability may be quiet this long after first seen
DEFAULT_STALE_AFTER_S = 14 * 86400   # a capability that HAS fired is stale after this much silence


@dataclass(slots=True)
class Entry:
    kind: str
    name: str
    registered_at: float
    last_fired: float | None = None
    last_ok: float | None = None
    count: int = 0
    failures: int = 0

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.name}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "name": self.name, "registered_at": self.registered_at,
            "last_fired": self.last_fired, "last_ok": self.last_ok, "count": self.count,
            "failures": self.failures,
        }


@dataclass(slots=True)
class Finding:
    kind: str
    name: str
    status: str          # "never" | "stale" | "failing"
    detail: str
    age_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "name": self.name, "status": self.status,
                "detail": self.detail, "age_s": self.age_s}


@dataclass
class CapabilityLedger:
    storage: StorageGateway
    clock: Any = time.time
    _entries: dict[str, Entry] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self._load()

    # -- persistence ---------------------------------------------------------------------

    def _load(self) -> None:
        try:
            raw = kv_get(self.storage, KV_KEY)
            if not raw:
                return
            data = json.loads(raw)
            for key, d in (data or {}).items():
                self._entries[key] = Entry(
                    kind=str(d.get("kind", "")), name=str(d.get("name", "")),
                    registered_at=float(d.get("registered_at") or 0.0),
                    last_fired=d.get("last_fired"), last_ok=d.get("last_ok"),
                    count=int(d.get("count") or 0), failures=int(d.get("failures") or 0),
                )
        except Exception as exc:  # a corrupt ledger must not take the brain down; start fresh
            log.warning("ledger: could not load (%s); starting empty", exc)
            self._entries = {}

    def _save(self) -> None:
        try:
            doc = {k: e.as_dict() for k, e in self._entries.items()}
            kv_set(self.storage, KV_KEY, json.dumps(doc))
        except Exception as exc:  # never raise out of a touch
            log.warning("ledger: could not persist: %s", exc)

    # -- the two verbs -------------------------------------------------------------------

    def register(self, kind: str, name: str) -> None:
        """Declare a capability exists. Idempotent; keeps an existing entry's history."""
        key = f"{kind}:{name}"
        with self._lock:
            if key not in self._entries:
                self._entries[key] = Entry(kind=kind, name=name, registered_at=float(self.clock()))
                self._save()

    def touch(self, kind: str, name: str, *, ok: bool = True) -> None:
        """Record that a capability ran (and whether it succeeded). Registers it if unknown."""
        key = f"{kind}:{name}"
        now = float(self.clock())
        with self._lock:
            e = self._entries.get(key)
            if e is None:
                e = Entry(kind=kind, name=name, registered_at=now)
                self._entries[key] = e
            e.last_fired = now
            e.count += 1
            if ok:
                e.last_ok = now
            else:
                e.failures += 1
            self._save()

    # -- reading it back -----------------------------------------------------------------

    def entries(self) -> list[Entry]:
        with self._lock:
            return sorted(self._entries.values(), key=lambda e: (e.kind, e.name))

    def review(
        self, *, grace_s: float = DEFAULT_GRACE_S, stale_after_s: float = DEFAULT_STALE_AFTER_S,
        now: float | None = None,
    ) -> list[Finding]:
        """What has gone quiet. ``never``: registered longer than ``grace_s`` ago and never fired.
        ``stale``: fired before, silent longer than ``stale_after_s``. ``failing``: its last three
        or more runs all failed (fires, but never succeeds — the draw tool's exact shape)."""
        t = float(self.clock() if now is None else now)
        out: list[Finding] = []
        for e in self.entries():
            if e.last_fired is None:
                age = t - e.registered_at
                if age >= grace_s:
                    out.append(Finding(e.kind, e.name, "never",
                                       f"registered {age / 86400:.0f} d ago, never fired", age))
                continue
            silent = t - e.last_fired
            if silent >= stale_after_s:
                out.append(Finding(e.kind, e.name, "stale",
                                   f"last fired {silent / 86400:.0f} d ago", silent))
            elif e.failures >= 3 and (e.last_ok is None or e.last_ok < e.last_fired) \
                    and e.count >= 3 and e.failures >= e.count:
                out.append(Finding(e.kind, e.name, "failing",
                                   f"{e.failures} run(s), none succeeded", silent))
        return out

    def render(self, findings: list[Finding]) -> str:
        """A compact line for the system-health prompt section; empty when nothing is wrong."""
        if not findings:
            return ""
        parts = [f"{f.kind} {f.name!r} ({f.status}: {f.detail})" for f in findings[:8]]
        more = f" (+{len(findings) - 8} more)" if len(findings) > 8 else ""
        return "Capabilities gone quiet: " + "; ".join(parts) + more + "."

    def snapshot(self) -> dict[str, Any]:
        return {e.key: e.as_dict() for e in self.entries()}

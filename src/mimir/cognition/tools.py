"""The motor port — tools the model can invoke, run through one guarded dispatcher (Phase 2,
docs/EXTENSIBILITY.md). The brain's "hands."

A ``Tool`` is a named capability with a light param schema and a handler that returns a string and
**never raises**. The ``dispatch`` function is the single guarded choke point — it validates the
call, **trust-gates state-changing actions** (a peer AI / non-trusted speaker is barred from
actuating, the same policy that bars it from trusted memory), and runs the handler — so safety can't
be bypassed no matter how a call arrives.

**Tool-calling only:** the model invokes a tool *deliberately* by emitting an in-band call
(generalizing the Library ``<FETCH id=N>`` marker), not by the brain acting on incidental prose.
Core ships the slot + trivial read-only reference tools; real hands are connectors users register.

Pure + dependency-light: the brain owns the registry and drives the invocation loop; these functions
are deterministic so tests drive them directly.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("mimir.tools")

# A tool call is one line: <TOOL name="x" args={"k": "v"}>. DOTALL so multi-line args work.
_TOOL_CALL_RE = re.compile(r"<TOOL\s+name=[\"']?([\w.\-]+)[\"']?\s+args=(\{.*?\})\s*/?>", re.DOTALL)
_MAX_SELECTED = 8  # cap tools offered per turn — small models degrade when handed too many


@dataclass(slots=True)
class ActionContext:
    """Who is driving this turn — the dispatcher's trust input. ``trusted`` decides whether a
    state-changing tool may fire (the brain sets it from the speaker policy)."""

    user: str | None = None
    speaker_kind: str = "human"
    trusted: bool = False


@dataclass(slots=True)
class Tool:
    """A registered capability the model can invoke. ``handler(args, ctx) -> str`` does the work
    and returns a string; it must never raise (a fault becomes an error string the model sees)."""

    name: str
    description: str
    handler: Callable[[dict[str, Any], ActionContext], str]
    schema: dict[str, dict[str, Any]] = field(default_factory=dict)  # {param: {"required": bool}}
    state_changing: bool = False   # actuates the world → trust-gated (vs a read-only lookup)
    keywords: tuple[str, ...] = ()  # cheap pre-selection; empty = only offered if `always`
    always: bool = False            # offered every turn regardless of keywords


@dataclass(slots=True)
class ToolCall:
    """One invocation + its outcome — surfaced to the caller via ``TurnResult.actions``."""

    tool: str
    args: dict[str, Any]
    result: str = ""
    status: str = "ok"  # ok | unknown | error | blocked

    def as_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "args": self.args, "result": self.result, "status": self.status}


class ToolRegistry:
    """The registered tools, keyed by name (register replaces by name — hot-safe)."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def select(self, query: str, *, limit: int = _MAX_SELECTED) -> list[Tool]:
        """Keyword / always-on pre-selection so the model isn't handed every tool. Always-on tools
        first, then any whose keyword appears in the query, capped at ``limit``."""
        ql = query.lower()
        chosen = [t for t in self._tools.values() if t.always]
        for t in self._tools.values():
            if t not in chosen and t.keywords and any(k in ql for k in t.keywords):
                chosen.append(t)
        return chosen[:limit]


def validate_args(tool: Tool, args: dict[str, Any]) -> str | None:
    """A light schema check — required keys present. Returns an error message, or ``None`` if OK."""
    missing = [p for p, spec in tool.schema.items() if spec.get("required") and p not in args]
    return f"missing required arg(s): {', '.join(missing)}" if missing else None


def dispatch(registry: ToolRegistry, call: ToolCall, ctx: ActionContext) -> ToolCall:
    """The single guarded choke point. Resolve → validate → trust-gate (state-changing) → run.
    Mutates and returns ``call`` with its result + status. A handler fault becomes an error string,
    never a raise — every actuation path is safe by construction."""
    tool = registry.get(call.tool)
    if tool is None:
        call.status, call.result = "unknown", f"no such tool: {call.tool!r}"
        return call
    err = validate_args(tool, call.args)
    if err is not None:
        call.status, call.result = "error", err
        return call
    if tool.state_changing and not ctx.trusted:
        # Trust in code: only a trusted human speaker may actuate. A peer AI / guest is barred —
        # the same policy that keeps it out of trusted memory keeps it off the hands (§3b).
        call.status = "blocked"
        call.result = f"refused: {tool.name!r} changes state; speaker not trusted to actuate"
        log.warning("tools: blocked state-changing %r for untrusted speaker (kind=%s, user=%s)",
                    tool.name, ctx.speaker_kind, ctx.user)
        return call
    try:
        call.result = tool.handler(call.args, ctx) or ""
        call.status = "ok"
    except Exception as exc:  # a tool fault is data for the model, not a crash (§10)
        call.status, call.result = "error", f"tool {tool.name!r} failed: {exc}"
        log.error("tools: handler %r raised: %s", tool.name, exc, exc_info=True)
    return call


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Extract any ``<TOOL name=... args={...}>`` invocations the model emitted. Tolerant: malformed
    args parse to ``{}`` (the dispatcher's schema check then reports what's missing)."""
    calls: list[ToolCall] = []
    for match in _TOOL_CALL_RE.finditer(text or ""):
        try:
            args = json.loads(match.group(2))
        except (ValueError, TypeError):
            args = {}
        calls.append(ToolCall(tool=match.group(1), args=args if isinstance(args, dict) else {}))
    return calls


def tools_hint(tools: list[Tool]) -> str:
    """The prompt block telling the model which tools it may call and the exact call syntax."""
    if not tools:
        return ""
    catalog = "\n".join(
        f"- {t.name}({', '.join(t.schema)}): {t.description}" for t in tools
    )
    return (
        "Tools you may call when it genuinely helps (otherwise just answer). To call one, reply "
        'EXACTLY one line and nothing else:\n<TOOL name="toolname" args={"key": "value"}>\n'
        f"Available tools:\n{catalog}"
    )

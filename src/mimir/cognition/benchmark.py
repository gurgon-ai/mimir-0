"""Fleet benchmarking — score model→role fitness (DESIGN §4), Phase 2 of the fleet.

This fills the catalogue's empty ``return_time`` + ``quality`` (and the capability breakdown) by
running each model through a short, cheapest-first battery:

- a deterministic **capability "IQ test"** — *talk* (instruction following), *tools* (emit a valid
  tool call), *code* (write parseable code). Zero judge cost; just checkable constraints.
- a **coherence** pass scored by a panel of *other* models (the council-as-judge idea), guarded by
  a **canary pair**: the judges must rank a known-good answer above a deliberately garbled one, or
  the *qualifier itself* is untrusted and coherence is skipped (DESIGN §4 — never a silent pass).

Quality is the aggregate of whatever scores were obtained; speed is the average call time. An
approved-family allowlist is the floor (the README recommends families); everything else still
gets benchmarked, the scores just speak for themselves.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass

from ..model.gateway import ModelGateway
from ..model.provider import Message
from ..storage.gateway import StorageGateway
from ..storage.repo import list_catalogue, update_catalogue_scores

log = logging.getLogger("mimir.benchmark")

ChatFn = Callable[[list[Message]], str]

# Families known to follow instructions well — the recommended floor (README curates this).
APPROVED_FAMILIES = ("llama", "qwen", "gemma", "mistral", "phi", "command-r", "deepseek")

_NUMBERED_RE = re.compile(r"^\s*\d+[.)]")
_SCORE_RE = re.compile(r"(\d*\.?\d+)")


def is_approved(family: str) -> bool:
    fam = family.lower()
    return any(pattern in fam for pattern in APPROVED_FAMILIES)


# -- output parsing helpers -----------------------------------------------------------


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines)
    return t


def _extract_json(text: str) -> dict[str, object] | None:
    t = _strip_fences(text)
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(t[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _defines_function(code: str, name: str) -> bool:
    try:
        tree = ast.parse(_strip_fences(code))
    except SyntaxError:
        return False
    return any(isinstance(n, ast.FunctionDef) and n.name == name for n in ast.walk(tree))


def _parse_score(text: str) -> float | None:
    match = _SCORE_RE.search(text)
    if not match:
        return None
    try:
        return max(0.0, min(1.0, float(match.group(1))))
    except ValueError:
        return None


# -- the capability battery (deterministic) -------------------------------------------


def _check_pong(out: str) -> bool:
    return out.strip().rstrip(".!").upper().startswith("PONG")


def _check_json_ok(out: str) -> bool:
    data = _extract_json(out)
    return data is not None and data.get("ok") is True


def _check_three_numbered(out: str) -> bool:
    return sum(1 for line in out.splitlines() if _NUMBERED_RE.match(line)) >= 3


def _check_tool_call(out: str) -> bool:
    data = _extract_json(out)
    if data is None:
        return False
    return isinstance(data.get("tool"), str) and isinstance(data.get("args"), dict)


def _check_add(out: str) -> bool:
    return _defines_function(out, "add")


CAPABILITY_TESTS: dict[str, list[tuple[str, Callable[[str], bool]]]] = {
    "talk": [
        ("Reply with exactly this and nothing else: PONG", _check_pong),
        ('Return only this JSON object and nothing else: {"ok": true}', _check_json_ok),
        (
            "List exactly three fruits as a numbered list (1., 2., 3.). Nothing else.",
            _check_three_numbered,
        ),
    ],
    "tools": [
        (
            "You can call tools. To get the weather in Paris, respond with ONLY a JSON object of "
            'the form {"tool": "<name>", "args": {...}} and nothing else.',
            _check_tool_call,
        ),
    ],
    "code": [
        (
            "Write a Python function named add that takes a and b and returns their sum. "
            "Respond with only the code.",
            _check_add,
        ),
    ],
}


def score_capability(chat_fn: ChatFn, capability: str) -> float:
    """Fraction of a capability's checkable cases that the model passes."""
    cases = CAPABILITY_TESTS[capability]
    passed = 0
    for prompt, check in cases:
        try:
            out = chat_fn([{"role": "user", "content": prompt}])
        except Exception as exc:  # a failed call scores 0 for that case
            log.warning("benchmark: capability call failed: %s", exc)
            out = ""
        if check(out):
            passed += 1
    return passed / len(cases)


# -- coherence (judged, with a canary) ------------------------------------------------

# Invented facts, so a model must use the *context*, not its training knowledge.
_CTX = "Fact: the Ariko river flows north through the village of Temb and freezes solid in winter."
_Q = "Which direction does the Ariko river flow, and what happens to it in winter?"
_GOOD = "The Ariko river flows north, and in winter it freezes solid."
_GARBLED = "The Ariko river flows south into a warm sea and stays tropical and ice-free all year."


def judge_coherence(model: ModelGateway, answer: str, *, max_judges: int = 3) -> float | None:
    """Panel of other models rate an answer's faithfulness to the context (None if unscorable)."""
    judges = [m for m in model.available_models() if "embed" not in m.lower()][:max_judges]
    if not judges:
        return None
    prompt = (
        f"Context:\n{_CTX}\n\nQuestion: {_Q}\n\nAnswer to grade:\n{answer}\n\n"
        "Rate how faithful the answer is to the context and free of invented details. "
        "Respond with ONLY a number from 0.0 to 1.0."
    )
    scores: list[float] = []
    for judge in judges:
        try:
            out = model.chat_with_model(judge, [{"role": "user", "content": prompt}])
        except Exception:
            continue
        val = _parse_score(out)
        if val is not None:
            scores.append(val)
    return sum(scores) / len(scores) if scores else None


def judges_trustworthy(model: ModelGateway) -> bool:
    """Canary: the panel must rank a known-good answer above a garbled one (DESIGN §4)."""
    good = judge_coherence(model, _GOOD)
    bad = judge_coherence(model, _GARBLED)
    if good is None or bad is None:
        return False
    ok = good > bad
    if not ok:
        log.error(
            "benchmark: CANARY INVERTED — judges scored garbled (%.2f) >= good (%.2f); "
            "coherence scoring is untrusted and skipped",
            bad,
            good,
        )
    return ok


# -- per-model + fleet ----------------------------------------------------------------


@dataclass(slots=True)
class ModelBenchmark:
    model: str
    talk: float
    tools: float
    code: float
    coherence: float | None
    return_time: float
    quality: float


@dataclass(slots=True)
class FleetBenchmarkResult:
    benchmarked: int
    judges_ok: bool
    results: list[ModelBenchmark]


def benchmark_model(model: ModelGateway, model_name: str, *, judge: bool = True) -> ModelBenchmark:
    """Run the battery against one model. ``judge=False`` skips the coherence pass."""
    def chat_fn(messages: list[Message]) -> str:
        return model.chat_with_model(model_name, messages)

    started = time.monotonic()
    talk = score_capability(chat_fn, "talk")
    tools = score_capability(chat_fn, "tools")
    code = score_capability(chat_fn, "code")
    n_calls = sum(len(CAPABILITY_TESTS[c]) for c in ("talk", "tools", "code"))
    return_time = (time.monotonic() - started) / max(1, n_calls)

    coherence: float | None = None
    if judge:
        try:
            answer = chat_fn([{"role": "user", "content": f"Context:\n{_CTX}\n\n{_Q}"}])
            coherence = judge_coherence(model, answer)
        except Exception as exc:
            log.warning("benchmark: coherence pass failed for %s: %s", model_name, exc)

    scores = [talk, tools, code] + ([coherence] if coherence is not None else [])
    quality = sum(scores) / len(scores)
    return ModelBenchmark(
        model=model_name,
        talk=talk,
        tools=tools,
        code=code,
        coherence=coherence,
        return_time=round(return_time, 3),
        quality=round(quality, 3),
    )


def benchmark_fleet(
    model: ModelGateway,
    storage: StorageGateway,
    *,
    only_approved: bool = True,
    limit: int = 8,
    max_params_b: float = 30.0,
    judge: bool = True,
) -> FleetBenchmarkResult:
    """Benchmark the distinct models in the catalogue and write their scores back.

    Quality is node-independent, so each model is benchmarked once and the scores written to all of
    its catalogue rows. Models are tried **smallest-first** and anything over ``max_params_b`` is
    skipped, so a giant model can't hang the run before the practical ones are scored (raise the cap
    to benchmark the big ones explicitly). Embedding models are skipped (they aren't chat models).
    """
    sizes: dict[str, float] = {}
    for entry in list_catalogue(storage):
        if "embed" in entry.model.lower():
            continue
        if only_approved and not is_approved(entry.family):
            continue
        if max_params_b and entry.params_b and entry.params_b > max_params_b:
            continue
        sizes.setdefault(entry.model, entry.params_b)
    models = sorted(sizes, key=lambda m: sizes[m])[:limit]  # smallest (fastest) first

    judges_ok = judges_trustworthy(model) if judge else False
    results: list[ModelBenchmark] = []
    for model_name in models:
        try:
            bench = benchmark_model(model, model_name, judge=judges_ok)
        except Exception as exc:
            log.warning("benchmark: %s failed: %s", model_name, exc)
            continue
        update_catalogue_scores(
            storage,
            model_name,
            return_time=bench.return_time,
            quality=bench.quality,
            talk=bench.talk,
            tools=bench.tools,
            code=bench.code,
            coherence=bench.coherence,
        )
        results.append(bench)
    log.info("benchmark: scored %d model(s); judges_ok=%s", len(results), judges_ok)
    return FleetBenchmarkResult(benchmarked=len(results), judges_ok=judges_ok, results=results)

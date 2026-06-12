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
from ..model.providers.ollama import OllamaProvider
from ..storage.gateway import StorageGateway
from ..storage.repo import list_catalogue, update_catalogue_scores, update_catalogue_speed
from .epistemics import score_epistemic_competence

log = logging.getLogger("mimir.benchmark")

ChatFn = Callable[[list[Message]], str]

# Per-node speed probes are direct, single-attempt, short-timeout calls — a model that doesn't
# answer a trivial prompt within this is recorded as that-slow (the timeout IS the signal).
SPEED_TIMEOUT_S = 20.0
_SPEED_PROMPT: list[Message] = [{"role": "user", "content": "Reply with the single word: ok"}]

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


def _check_no_brackets(out: str) -> bool:
    """A disciplined model emits NO square brackets when told not to — the tag-leak test.

    The prompts that use this checker prime the model with ``[tier=...; source=...]`` scaffolding
    and forbid reproducing it. A model that mimics the format (the failure that forced the output
    sanitizer, DESIGN §10) leaks a ``[`` or ``]`` and fails.
    """
    return bool(out.strip()) and "[" not in out and "]" not in out


def _check_no_dog_or_cat(out: str) -> bool:
    """Honors a negative lexical constraint ('do not use the words dog or cat') in a short reply."""
    low = out.lower()
    return bool(low.strip()) and "dog" not in low and "cat" not in low and len(low.split()) <= 4


# The production-faithful discipline probe: a tag-saturated recall block under the REAL soft
# instruction. Validated to reproduce the leak on gemma3:4b while gemma4:e2b/e4b stay clean.
_DISCIPLINE_RECALL = (
    "What you know that's relevant — each fact is attributed. Use these facts naturally in your "
    "reply and attribute in plain words when it matters, but do NOT copy the bracketed "
    "[tier=...; source=...] tags into your response.\n"
    "<RECALL>\n"
    "- Greg's home server runs a full backup every Sunday [tier=document; source=handbook.md]\n"
    "- Greg has a dog named Kodi [tier=stated_by_primary_user; source=Greg]\n"
    "- The Ariko river freezes solid in winter [tier=conversation; source=chat]\n"
    "- Mona's favorite tea is genmaicha [tier=stated_by_trusted; source=Mona]\n"
    "</RECALL>\n\n"
    "Greg asks: \"Remind me when the full backup runs and what my dog's name is.\" "
    "Answer in two short sentences."
)


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
    # Discipline = honoring prohibitions, above all NOT reproducing the internal
    # [tier=...; source=...] scaffolding the model is shown. This is the capability that separates
    # an identity-safe chat/reasoning model from one that leaks the prompt's tags (DESIGN §4, §10).
    #
    # The case below replicates the PRODUCTION condition that actually triggers the leak: a full
    # recall block saturated with tags, under the real (soft) "do not copy the tags" instruction —
    # NOT an artificially strong "use no brackets at all." A weak single-tag prompt is too easy
    # (gemma3:4b passes it yet leaks in real chat); this one reproduces the failure (gemma3:4b
    # leaks ~3/3, gemma4:e2b/e4b stay clean). Leakage is probabilistic, so the prompt is sampled
    # 3x and discipline is the fraction of bracket-free samples — a consistent leaker scores ~0 and
    # falls far below the floor; an occasional slipper still clears it. (DESIGN §4: consistency
    # across K runs is a real score.) One negative-lexical case rounds out the dimension.
    "discipline": [
        (_DISCIPLINE_RECALL, _check_no_brackets),
        (_DISCIPLINE_RECALL, _check_no_brackets),
        (_DISCIPLINE_RECALL, _check_no_brackets),
        (
            "Name one common household pet that is not a dog and not a cat. Reply with a single "
            "word, and do not use the words 'dog' or 'cat'.",
            _check_no_dog_or_cat,
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
    discipline: float
    epistemics: float
    coherence: float | None
    return_time: float
    quality: float


@dataclass(slots=True)
class FleetBenchmarkResult:
    benchmarked: int
    judges_ok: bool
    results: list[ModelBenchmark]
    eligible: int = 0          # approved, non-embedding models in the catalogue (any size)
    skipped_too_big: int = 0   # eligible models skipped because they exceed max_params_b
    skipped_too_slow: int = 0  # eligible models skipped because a trivial call exceeded the budget


def benchmark_model(model: ModelGateway, model_name: str, *, judge: bool = True) -> ModelBenchmark:
    """Run the battery against one model. ``judge=False`` skips the coherence pass."""
    def chat_fn(messages: list[Message]) -> str:
        return model.chat_with_model(model_name, messages)

    started = time.monotonic()
    talk = score_capability(chat_fn, "talk")
    tools = score_capability(chat_fn, "tools")
    code = score_capability(chat_fn, "code")
    discipline = score_capability(chat_fn, "discipline")
    # Epistemics: does the model exploit Mimir's tiered/provenance/gated context (DESIGN §3)?
    # The structured-arm competence — the qualification signal for the identity-bearing roles.
    epistemics = score_epistemic_competence(chat_fn, samples=2)
    n_calls = sum(len(CAPABILITY_TESTS[c]) for c in ("talk", "tools", "code", "discipline")) + 6
    return_time = (time.monotonic() - started) / max(1, n_calls)

    coherence: float | None = None
    if judge:
        try:
            answer = chat_fn([{"role": "user", "content": f"Context:\n{_CTX}\n\n{_Q}"}])
            coherence = judge_coherence(model, answer)
        except Exception as exc:
            log.warning("benchmark: coherence pass failed for %s: %s", model_name, exc)

    scores = [talk, tools, code, discipline, epistemics] + (
        [coherence] if coherence is not None else []
    )
    quality = sum(scores) / len(scores)
    return ModelBenchmark(
        model=model_name,
        talk=talk,
        tools=tools,
        code=code,
        discipline=discipline,
        epistemics=round(epistemics, 3),
        coherence=coherence,
        return_time=round(return_time, 3),
        quality=round(quality, 3),
    )


def _measure_node_speed(
    node: str, model_name: str, *, timeout_s: float = SPEED_TIMEOUT_S
) -> float | None:
    """Time a trivial call to a *specific* node directly (no pool/retry); None for non-URL nodes.

    Returns elapsed seconds — even on timeout/failure, the elapsed time is the 'too slow' signal.
    """
    if not node.startswith("http"):
        return None  # mock/non-URL node — nothing real to time
    provider = OllamaProvider(node, timeout=timeout_s)
    started = time.monotonic()
    try:
        provider.chat(model_name, _SPEED_PROMPT, {})
    except Exception:
        return round(time.monotonic() - started, 3)  # the (timed-out) duration ranks it slow
    return round(time.monotonic() - started, 3)


# When the user hasn't set a latency target, still skip a model whose trivial-prompt call takes
# longer than this — it's not viable for interactive use, and the full battery would stall the run.
_DEFAULT_SKIP_S: float = 30.0


def benchmark_fleet(
    model: ModelGateway,
    storage: StorageGateway,
    *,
    only_approved: bool = True,
    limit: int = 8,
    max_params_b: float = 30.0,
    judge: bool = True,
    latency_budget_s: float = 0.0,
    progress: Callable[[int, int, str], None] | None = None,
) -> FleetBenchmarkResult:
    """Benchmark the distinct models in the catalogue and write their scores back.

    Quality is node-independent, so each model is benchmarked once and the scores written to all of
    its catalogue rows. Models are tried **smallest-first** and anything over ``max_params_b`` is
    skipped, so a giant model can't hang the run before the practical ones are scored (raise the cap
    to benchmark the big ones explicitly). Embedding models are skipped (they aren't chat models).

    ``progress(index, total, model_name)`` is called before each model — the benchmark is
    multi-minute and otherwise silent, so this (and the per-model log line) is how a UI or a log
    reader can tell it's alive (DESIGN §10 — stay observable).
    """
    sizes: dict[str, float] = {}
    nodes_with: dict[str, list[str]] = {}
    eligible: set[str] = set()   # approved, non-embedding (any size) — for coverage reporting
    too_big: set[str] = set()    # eligible but over the size cap
    for entry in list_catalogue(storage):
        nodes_with.setdefault(entry.model, []).append(entry.node)
        if "embed" in entry.model.lower():
            continue
        if only_approved and not is_approved(entry.family):
            continue
        eligible.add(entry.model)
        if max_params_b and entry.params_b and entry.params_b > max_params_b:
            too_big.add(entry.model)
            continue
        sizes.setdefault(entry.model, entry.params_b)
    models = sorted(sizes, key=lambda m: sizes[m])[:limit]  # smallest (fastest) first

    total = len(models)
    skip_budget = latency_budget_s if latency_budget_s > 0 else _DEFAULT_SKIP_S
    too_slow: set[str] = set()
    log.info("benchmark: starting — %d model(s) to score (judge=%s, skip > %.0fs)",
             total, judge, skip_budget)
    judges_ok = judges_trustworthy(model) if judge else False
    results: list[ModelBenchmark] = []
    for i, model_name in enumerate(models, start=1):
        log.info("benchmark: [%d/%d] %s …", i, total, model_name)
        if progress is not None:
            progress(i, total, model_name)
        # Latency pre-gate: a single trivial-prompt probe (timeout = the budget). A model that can't
        # answer "ok" within budget isn't viable for interactive use — skip it before the expensive
        # full battery stalls the run. The probe also seeds this node's speed.
        probe_node = next((n for n in nodes_with.get(model_name, []) if n.startswith("http")), None)
        if probe_node is not None:
            speed = _measure_node_speed(probe_node, model_name, timeout_s=skip_budget)
            if speed is not None and speed >= skip_budget:
                log.warning("benchmark: [%d/%d] %s SKIPPED — %.1fs >= %.0fs latency budget",
                            i, total, model_name, speed, skip_budget)
                too_slow.add(model_name)
                continue
            if speed is not None:
                update_catalogue_speed(storage, probe_node, model_name, speed)
        try:
            bench = benchmark_model(model, model_name, judge=judges_ok)
        except Exception as exc:
            log.warning("benchmark: [%d/%d] %s FAILED: %s", i, total, model_name, exc)
            continue
        log.info("benchmark: [%d/%d] %s done — quality=%.2f in %.1fs",
                 i, total, model_name, bench.quality, bench.return_time)
        update_catalogue_scores(
            storage,
            model_name,
            return_time=bench.return_time,
            quality=bench.quality,
            talk=bench.talk,
            tools=bench.tools,
            code=bench.code,
            coherence=bench.coherence,
            discipline=bench.discipline,
            epistemics=bench.epistemics,
        )
        # Per-node speed for the OTHER nodes (the probe node was already measured above).
        for node in nodes_with.get(model_name, []):
            if node == probe_node:
                continue
            elapsed = _measure_node_speed(node, model_name, timeout_s=skip_budget)
            if elapsed is not None:
                update_catalogue_speed(storage, node, model_name, elapsed)
        results.append(bench)
    log.info(
        "benchmark: scored %d of %d eligible; %d too big, %d too slow; judges_ok=%s",
        len(results), len(eligible), len(too_big), len(too_slow), judges_ok,
    )
    return FleetBenchmarkResult(
        benchmarked=len(results), judges_ok=judges_ok, results=results,
        eligible=len(eligible), skipped_too_big=len(too_big), skipped_too_slow=len(too_slow),
    )

"""Fleet benchmarking — score model→role fitness (DESIGN §4), Phase 2 of the fleet.

This fills the catalogue's empty ``return_time`` + ``quality`` (and the capability breakdown) by
running each model through a short, cheapest-first battery:

- a deterministic **capability "IQ test"** — *talk* (instruction following), *tools* (emit a valid
  tool call), *code* (parseable code), *reasoning*, *discipline*, *epistemics*, *vision*. Every
  dimension is a checkable constraint (regex/exact/probe) that genuinely separates models — no judge
  pass (the old peer-judged *coherence* was dropped: it duplicated *epistemics* and, on a vague 0–1
  scale, every model landed mid-range, so it discriminated nothing).

Quality is the aggregate of whatever scores were obtained; speed is the average call time. An
approved-family allowlist is the floor (the README recommends families); everything else still
gets benchmarked, the scores just speak for themselves.
"""

from __future__ import annotations

import ast
import base64
import json
import logging
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

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

# A prompt that forces a real-length generation, so the timed call reflects throughput, not just
# round-trip on a 3-token reply. We normalize the result to seconds-per-256-token turn.
_LATENCY_PROMPT: list[Message] = [
    {"role": "user",
     "content": "Explain in three short paragraphs how a city treats its drinking water."},
]
_LATENCY_NORM_TOKENS: int = 256   # report seconds per ~256-token turn (verbosity-independent)
_LATENCY_MIN_TOKENS: int = 32     # floor so a terse/refusing model can't divide-by-tiny to nonsense
_GATE_PREDICT: int = 128          # tokens the pre-gate generates: enough to clear the decode
#                                   warmup ramp (a 64-token probe under-reads a fast MoE ~20%)


def _throughput_seconds(eval_count: int, eval_duration_ns: int) -> float | None:
    """Seconds per ~256-token turn from Ollama's own generation metrics: ``eval_count`` tokens
    generated in ``eval_duration`` ns of *pure generation* (decode). Model-load / VRAM-swap and
    prompt-eval are excluded, so this is true per-token throughput (TPS = ``256 / result``) — and
    it's identical whether the sample was 64 tokens or 600, which is what makes the pre-gate probe
    and the full battery finally agree. Returns None if Ollama didn't report usable metrics (the
    caller falls back to a wall-clock estimate)."""
    if eval_count <= 0 or eval_duration_ns <= 0:
        return None
    return round((eval_duration_ns / 1e9) / eval_count * _LATENCY_NORM_TOKENS, 3)


def _wallclock_latency(out: str, elapsed: float) -> float:
    """Fallback per-256-token estimate from wall-clock + an output-length token guess. Used ONLY
    when generation metrics are unavailable (mock / non-Ollama provider). Contaminated by load and
    prompt-eval time, so it's a last resort — real Ollama paths use :func:`_throughput_seconds`."""
    approx_tokens = max(_LATENCY_MIN_TOKENS, len(out) // 4)  # ~4 chars/token; no tokenizer dep
    return round(elapsed / approx_tokens * _LATENCY_NORM_TOKENS, 3)


def _measure_turn_latency(
    chat_fn: Callable[[list[Message]], str],
    timed_fn: Callable[[list[Message]], tuple[str, int, int]] | None = None,
) -> float | None:
    """Time one real generation and normalize to seconds per ~256-token turn (TPS = 256 / result).

    With ``timed_fn`` (the real Ollama path) latency comes from Ollama's own
    ``eval_count``/``eval_duration`` — *pure decode* time, immune to the cold model-load / VRAM-swap
    that made a fast MoE (gemma4:26b, 4B active, ~210 TPS) record a fake ~38s/turn on a contended
    node and lose speed-weighted roles to a genuinely slower dense model. eval_duration is already
    an average over every generated token, so a single sample is stable (decode TPS barely varies
    run-to-run); we don't need to average many turns. Without ``timed_fn`` (mock/gateway) we fall
    back to a wall-clock estimate over the output length.

    Returns **None** if the probe fails (timeout/transport error) — NEVER 0.0. A failed probe is the
    opposite of instant: recording 0.0 made a timing-out model sort as the *fastest* and pass any
    latency cap. None means 'unmeasured'; every consumer treats it as not-fast / not-viable
    (``return_time or 1e9``) and the matrix re-times it.
    """
    started = time.monotonic()
    try:
        if timed_fn is not None:
            out, eval_count, eval_dur = timed_fn(_LATENCY_PROMPT)
        else:
            out, eval_count, eval_dur = chat_fn(_LATENCY_PROMPT), 0, 0
    except Exception as exc:
        log.warning("benchmark: latency probe failed after %.1fs (recorded as unmeasured): %s",
                    time.monotonic() - started, exc)
        return None
    tps = _throughput_seconds(eval_count, eval_dur)
    return tps if tps is not None else _wallclock_latency(out, time.monotonic() - started)

# Families known to follow instructions well — the recommended floor (README curates this).
APPROVED_FAMILIES = ("llama", "qwen", "gemma", "mistral", "phi", "command-r", "deepseek")

_NUMBERED_RE = re.compile(r"^\s*\d+[.)]")


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


_ANSWER_INT_RE = re.compile(r"-?\d+")


def _last_int(out: str) -> int | None:
    """The last integer in the reply — models put the final answer last ('… the answer is 242')."""
    nums = _ANSWER_INT_RE.findall(out.replace(",", ""))
    return int(nums[-1]) if nums else None


def _expect_int(target: int) -> Callable[[str], bool]:
    """A reasoning checker that passes iff the model's final integer equals ``target``."""
    return lambda out: _last_int(out) == target


def _check_reverse_python(out: str) -> bool:
    """'PYTHON' reversed + lowercased is 'nohtyp' — an instruction-following transform."""
    return "nohtyp" in out.lower()


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
    # Reasoning = can it actually SOLVE a problem with one verifiable answer, not just comply with a
    # format? The rest of the battery (PONG, a weather JSON, def add) is passed by any competent
    # model, so quality saturates and can't separate a capable model from a merely fluent one. These
    # are deterministic regex/exact checks — no code execution, no judge — spanning arithmetic, char
    # counting (a classic small-model failure), pattern completion, a code-trace, proportional
    # reasoning, and an instruction transform. This is the dimension that stops quality from
    # rubber-stamping a model that 'can't do the job' (DESIGN §4).
    "reasoning": [
        ("A tank holds 240 liters. It drains at 8 liters per minute for 6 minutes, then 50 liters "
         "are added. How many liters are in the tank now? Reply with only the final number.",
         _expect_int(242)),
        ("How many times does the letter 'r' appear in the word 'strawberry'? Reply with only the "
         "number.", _expect_int(3)),
        ("What is the next number in this sequence: 2, 6, 12, 20, 30, ? Reply with only the "
         "number.", _expect_int(42)),
        ("What does this Python print: print(len(set([1, 2, 2, 3, 3, 3]))) — reply with only the "
         "number.", _expect_int(3)),
        ("If 3 pens cost 6 dollars, how much do 5 pens cost in dollars? Reply with only the "
         "number.", _expect_int(10)),
        ("Take the word PYTHON, reverse it, and write it in lowercase. Reply with only the result.",
         _check_reverse_python),
    ],
}


# -- vision (empirical: the probe IS the determination, not advertised metadata) --------

# The fixed probe image (the word GLYPHON + three red circles on white), from the repo root.
# Absent → vision is 'not tested' (None), never a false zero. See assets/vision_probe.README.md.
_VISION_PROBE = Path(__file__).resolve().parents[3] / "assets" / "vision_probe.png"
# (prompt, check, weight). Reading the made-up word GLYPHON is the strongest vision signal (it can't
# be guessed) and carries most weight; counting the shapes is lighter (a text model could guess it).
# Passing EITHER case counts as having vision — the board bands it 🟡 (partial) and the vision role
# admits it (see `_VISION_FLOOR`); both cases → ✅. A text model that sees nothing scores 0 → ❌.
_VISION_CASES = [
    ("What single word is written in this image? Reply with just the word.",
     lambda out: "glyphon" in out.lower(), 0.6),
    ("How many red circles are in this image? Reply with only the number.",
     lambda out: bool(re.search(r"\b(3|three)\b", out.lower())), 0.4),
]


def _probe_image_b64() -> str | None:
    try:
        return base64.b64encode(_VISION_PROBE.read_bytes()).decode("ascii")
    except OSError:
        return None


def score_vision(chat_fn: ChatFn) -> float | None:
    """Empirically determine vision capability: send the fixed probe image with each question and
    score reading the word (OCR) + counting the shapes. A text-only model can't read GLYPHON or see
    the circles, so it scores ~0 — that failure IS the determination. ``None`` if the probe image is
    missing (not tested) rather than a misleading zero."""
    b64 = _probe_image_b64()
    if b64 is None:
        log.warning("benchmark: vision probe image missing at %s — skipping vision", _VISION_PROBE)
        return None
    score = 0.0
    for prompt, check, weight in _VISION_CASES:
        try:
            out = chat_fn([{"role": "user", "content": prompt, "images": [b64]}])
        except Exception as exc:  # a non-vision model may error on an image — a 0 for that case
            log.warning("benchmark: vision call failed: %s", exc)
            out = ""
        if check(out):
            score += weight
    return score


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


# -- per-model + fleet ----------------------------------------------------------------


@dataclass(slots=True)
class ModelBenchmark:
    model: str
    talk: float
    tools: float
    code: float
    discipline: float
    epistemics: float
    reasoning: float
    vision: float | None        # empirical image-probe score; None = not tested (no probe image)
    return_time: float | None   # None = probe failed/unmeasured (NOT fast — see measurement)
    quality: float


@dataclass(slots=True)
class FleetBenchmarkResult:
    benchmarked: int
    results: list[ModelBenchmark]
    eligible: int = 0          # approved, non-embedding models in the catalogue (any size)
    skipped_too_big: int = 0   # eligible models skipped because they exceed max_params_b
    skipped_too_small: int = 0  # eligible models skipped because they're under min_params_b
    skipped_too_slow: int = 0  # eligible models skipped because a trivial call exceeded the budget


# A node can pass the quick speed probe yet hang the real battery (intermittent, or it loads for a
# 1-token warm but stalls on actual generation), grinding every call into a per-call timeout — ~12
# calls × 60s ≈ a "stuck on one model for 20 minutes" run that scores a fast model a false ~0.
# _NodeUnreliable aborts that battery so the caller can FAIL OVER to another node the model is on
# (DESIGN §6 — capability is never failed on speed; a bad node is not a bad model). It subclasses
# BaseException on purpose: the scorers catch `Exception` (one bad answer → 0 for that case), so a
# plain exception would be swallowed — this sails through them straight to the failover.
class _NodeUnreliable(BaseException):
    pass


_NODE_FAIL_THRESHOLD = 3   # transport failures on one node → abandon it and fail over to another


# Per-node cross-node vision probe budget (model load + a couple of short reads). BOUNDED so a slow
# or cold node — or one whose Ollama times out on the image — can't turn the best-across-nodes pass
# into a 10-minute hang. A working vision model reads the tiny probe in seconds once warm.
_VISION_PROBE_TIMEOUT = 40.0


def _node_vision(node: str, model_name: str, num_ctx: int) -> float | None:
    """Empirically probe ONLY vision for a model on one node (direct provider). Vision is per-node:
    a byte-identical model file reads images fine under one Ollama version and mangles them under
    another (a runtime regression), so the same model can score 1.0 on one node and 0.0 on another.
    Bounded by ``_VISION_PROBE_TIMEOUT``: warm the model first (skip the node if it can't load in
    time), then run the short reads warm. Returns the vision score, or None if the node can't load
    it in budget / has no probe image — so the caller just moves on, never a 10-minute stall."""
    if not node.startswith("http"):
        return None
    provider = OllamaProvider(node, timeout=_VISION_PROBE_TIMEOUT)
    try:   # warm into VRAM first; a node that can't load it in budget contributes nothing → skip
        provider.chat(model_name, [{"role": "user", "content": "ok"}],
                      {"num_ctx": num_ctx, "max_tokens": 1})
    except Exception:
        return None

    def det(messages: list[Message]) -> str:
        return provider.chat(model_name, messages,
                             {"num_ctx": num_ctx, "temperature": 0.0, "max_tokens": 48})

    try:
        return score_vision(det)
    except Exception as exc:
        log.warning("benchmark: cross-node vision probe failed for %s on %s: %s",
                    model_name, node, exc)
        return None


def benchmark_model(
    model: ModelGateway, model_name: str, *, num_ctx: int = 8192,
    framework: bool = True, call_timeout_s: float = 60.0, node: str | None = None,
) -> ModelBenchmark:
    """Run the battery against one model.

    Every battery/epistemics/latency call routes through ``chat_fn``, which pins ``num_ctx`` so the
    layered epistemic prompts aren't truncated to Ollama's 2048 default (which would cut off the
    high-tier fact and silently break the tier-deference test), AND a tight ``call_timeout_s`` with
    **no pool retries**, so a slow/wedged model fails fast (scoring 0 for that case) instead of
    stalling the whole run on the production 120s ceiling × 3 retries.

    ``framework=False`` is the tournament's **triage** mode: it runs only the cheap capability
    dimensions (talk/tools/code/discipline/reasoning + latency) and SKIPS the expensive
    identity-qualification work — the multi-sample 8k-ctx epistemic gauntlet — so a model about to
    vetoed isn't dragged through the gauntlet. The survivors then get the full benchmark in a later
    round. ``quality`` is the mean of whatever dimensions actually ran, so a triage score is
    comparable only to other triage scores (not to a full score).

    The single-answer dims (talk/tools/code/reasoning/vision) are scored at **temperature 0**
    (greedy) so a near-tied model isn't flipped to a worse score by an unlucky high-temperature draw
    — they have ONE right answer, so greedy IS the capability. ``discipline``/``epistemics`` keep
    the role/default temperature on purpose: their signal is *consistency across sampled runs* (a
    probabilistic tag-leak / repeated gauntlet), which a greedy single shot would erase.
    """
    # ``node`` pins the WHOLE battery to one warm node (direct provider, no pool/retry) so the
    # scheduler can run different models on different nodes at once without the pool re-routing a
    # model's calls mid-battery and thrashing VRAM. Without it (mock / single-local), route via the
    # gateway. ``make_chat(temp)`` builds a call fn at a given temperature (None = role/default).
    if node and node.startswith("http"):
        scorer = OllamaProvider(node, timeout=call_timeout_s)
        warmer = OllamaProvider(node, timeout=WARMUP_TIMEOUT_S)

        def make_chat(temp: float | None) -> ChatFn:
            opts = ({"num_ctx": num_ctx} if temp is None
                    else {"num_ctx": num_ctx, "temperature": temp})
            return lambda messages: scorer.chat(model_name, messages, dict(opts))

        def timed_latency(messages: list[Message]) -> tuple[str, int, int]:
            # Greedy + Ollama's own decode metrics → load-immune TPS (DESIGN §4).
            return scorer.chat_timed(
                model_name, messages, {"num_ctx": num_ctx, "temperature": 0.0})

        def _warm() -> None:
            warmer.chat(model_name, [{"role": "user", "content": "ok"}],
                        {"num_ctx": num_ctx, "max_tokens": 1})
    else:
        timed_latency = None   # mock/gateway: no decode metrics → wall-clock fallback

        def make_chat(temp: float | None) -> ChatFn:
            base: dict[str, object] = {"num_ctx": num_ctx, "__timeout_s__": call_timeout_s}
            if temp is not None:
                base["temperature"] = temp
            return lambda messages: model.chat_with_model(
                model_name, messages, params=dict(base), max_retries=0)

        def _warm() -> None:
            model.chat_with_model(
                model_name, [{"role": "user", "content": "ok"}],
                params={"num_ctx": num_ctx, "max_tokens": 1}, max_retries=0,
            )

    # Warm the model into VRAM before timing. ONE token (max_tokens=1) so the load can't turn into a
    # multi-minute reason-fest on a thinking model; no retries. The real scoring calls follow warm.
    try:
        _warm()
    except Exception as exc:
        log.warning("benchmark: warmup failed for %s on %s: %s", model_name, node, exc)
        if node and node.startswith("http"):
            # Can't even load one token here → this node is unusable for the model; fail over.
            raise _NodeUnreliable(f"{model_name}: warmup failed on {node}") from exc

    # Fail-fast guard: count transport failures (timeout/refused) across the battery; once a node
    # trips the threshold, abandon it (_NodeUnreliable) so the caller fails over to another node,
    # rather than grinding every remaining call into a timeout (the "20 minutes on one model" bug).
    fail = {"n": 0}

    def _guard(fn: ChatFn) -> ChatFn:
        def wrapped(messages: list[Message]) -> str:
            if fail["n"] >= _NODE_FAIL_THRESHOLD:
                raise _NodeUnreliable(f"{model_name}: node {node} unreliable")
            try:
                return fn(messages)
            except Exception as exc:
                fail["n"] += 1
                if fail["n"] >= _NODE_FAIL_THRESHOLD:
                    raise _NodeUnreliable(
                        f"{model_name}: node {node} unreliable after {fail['n']} failed calls"
                    ) from exc
                raise   # let the scorer count this one case as 0 and continue
        return wrapped

    chat_fn = _guard(make_chat(None))  # role/default temp — variance IS the signal
    det = _guard(make_chat(0.0))       # greedy — verifiable single-answer dims, so luck can't flip

    talk = score_capability(det, "talk")
    tools = score_capability(det, "tools")
    code = score_capability(det, "code")
    discipline = score_capability(chat_fn, "discipline")  # sampled at temp: catches stochastic leak
    # Reasoning: can it actually solve problems with verifiable answers (not just follow a format)?
    # This is what keeps quality from saturating near 1.0 for any fluent model (DESIGN §4).
    reasoning = score_capability(det, "reasoning")
    # Representative latency from one real-length generation (NOT the battery average): the battery
    # calls emit only a few tokens, so their round-trip is dominated by overhead and can't tell a
    # slow remote 12B from a snappy local 3B — which lets a big model look 'instant' and sweep even
    # the speed-weighted roles it should lose. A real turn generates hundreds of tokens, throughput
    # is what the user feels (DESIGN §4: latency must reflect an actual turn).
    return_time = _measure_turn_latency(chat_fn, timed_latency)

    # The expensive identity-qualification dimensions — only in the FULL benchmark (not at triage).
    # Epistemics: does the model exploit Mimir's tiered/provenance/gated context (DESIGN §3)? The
    # structured-arm competence (layered gauntlet + grounding + long-context) — the chat qualifier.
    epistemics = (score_epistemic_competence(chat_fn, samples=2, num_ctx=num_ctx)
                  if framework else 0.0)

    # Vision: empirical image-probe capability (greedy — a fixed-answer probe). Informational — kept
    # OUT of quality so a text-only model scoring ~0 isn't penalized for a dim it never claimed.
    vision = score_vision(det)

    scores = [talk, tools, code, discipline, reasoning]
    if framework:
        scores.append(epistemics)
    quality = sum(scores) / len(scores)
    return ModelBenchmark(
        model=model_name,
        talk=talk,
        tools=tools,
        code=code,
        discipline=discipline,
        epistemics=round(epistemics, 3),
        reasoning=round(reasoning, 3),
        vision=round(vision, 3) if vision is not None else None,
        return_time=round(return_time, 3) if return_time is not None else None,
        quality=round(quality, 3),
    )


# Generous window to let even a large model load into VRAM during warmup before we time it.
WARMUP_TIMEOUT_S: float = 120.0


def _measure_node_speed(
    node: str, model_name: str, *, timeout_s: float = SPEED_TIMEOUT_S, warmup: bool = False,
    num_ctx: int = 8192,
) -> float | None:
    """Measure a node's **per-turn latency** for a model directly (no pool/retry); None for non-URL.

    Times a *representative* ~64-token generation and normalizes to **seconds per ~256-token turn**
    — the same units as the latency cap. A trivial "reply ok" can't tell a model that's snappy on
    one token from one that's 13s/turn on real generation, so it would let a slow model sail through
    the pre-gate and only reveal itself after the multi-call battery (the bug behind a 7s cap not
    skipping a 160s model). This measures the thing the cap actually means.

    With ``warmup``, the model is **loaded into VRAM first** with a single token (so a thinking
    model can't reason for minutes during the untimed load), so the timing reflects steady-state,
    not the one-time cold swap. A model that can't load within ``WARMUP_TIMEOUT_S`` is reported as
    that-slow (skipped). ``num_ctx`` matches the battery so the model loads once and stays warm.

    Returns normalized seconds/turn — even on timeout/failure, the (large) elapsed time is the 'too
    slow' signal that trips the gate.
    """
    if not node.startswith("http"):
        return None  # mock/non-URL node — nothing real to time
    opts = {"num_ctx": num_ctx}
    if warmup:
        try:
            OllamaProvider(node, timeout=WARMUP_TIMEOUT_S).chat(
                model_name, _SPEED_PROMPT, {**opts, "max_tokens": 1}
            )
        except Exception:
            return round(WARMUP_TIMEOUT_S, 3)  # couldn't load in time → unusably slow → skip
    # Bound the measurement generously (a model that can't emit 64 tokens in this window is doomed),
    # but always at least 30s so a near-the-cap model isn't cut by the measurement timeout itself.
    bound = max(30.0, timeout_s)
    provider = OllamaProvider(node, timeout=bound)
    started = time.monotonic()
    try:
        out, eval_count, eval_dur = provider.chat_timed(
            model_name, _LATENCY_PROMPT, {**opts, "max_tokens": _GATE_PREDICT})
    except Exception:
        # Failed generating → ranks it slow → skip. Use the elapsed but FLOOR it at the timeout
        # bound: a real timeout already ≈ bound, and this stops an *instant* transport failure
        # (elapsed ≈ 0) being recorded as the fastest model — a failed probe must never sort fast.
        return round(max(bound, time.monotonic() - started), 3)
    # True decode TPS from Ollama's metrics (load-immune) — identical units to the battery, so the
    # pre-gate probe and the full benchmark finally agree. Wall-clock only if metrics are absent.
    tps = _throughput_seconds(eval_count, eval_dur)
    return tps if tps is not None else _wallclock_latency(out, time.monotonic() - started)


# When the user hasn't set a latency target, still skip a model whose trivial-prompt call takes
# longer than this — it's not viable for interactive use, and the full battery would stall the run.
_DEFAULT_SKIP_S: float = 30.0


def _choose_test_node(
    candidates: list[str], probe: Callable[[str], float | None], test_budget: float,
) -> tuple[str | None, dict[str, float]]:
    """Pick which node to run a model's battery on, probing candidates in order. Returns the FIRST
    node fast enough to test on (``speed <= test_budget``) — stopping there so we don't probe the
    rest — else the FASTEST node that ran at all. Capability is established on the best available
    node and is **never failed for being slow** (DESIGN §6/§1); a node that can't run it (probe
    returns None) is skipped. Records every probed speed for the per-node placement matrix.
    """
    speeds: dict[str, float] = {}
    fastest: str | None = None
    for node in candidates:
        speed = probe(node)
        if speed is None:
            continue   # node couldn't run it at all (down / won't load) → try the next
        speeds[node] = speed
        if speed <= test_budget:
            return node, speeds   # fast enough — test here, don't bother probing the rest
        if fastest is None or speed < speeds[fastest]:
            fastest = node
    return fastest, speeds   # none under budget → the fastest that ran (None if none ran)


def _outside_in(by_size: list[str]) -> list[str]:
    """Reorder a smallest→largest list into big, small, big, small … so a running-average time
    estimate samples both extremes from the first two models and converges to the true mean fast."""
    out: list[str] = []
    lo, hi = 0, len(by_size) - 1
    take_big = True
    while lo <= hi:
        if take_big:
            out.append(by_size[hi])
            hi -= 1
        else:
            out.append(by_size[lo])
            lo += 1
        take_big = not take_big
    return out


def benchmark_fleet(
    model: ModelGateway,
    storage: StorageGateway,
    *,
    only_approved: bool = True,
    limit: int = 8,
    max_params_b: float = 30.0,
    min_params_b: float = 0.0,
    latency_budget_s: float = 0.0,
    num_ctx: int = 8192,
    only_models: set[str] | None = None,
    disabled_nodes: set[str] | None = None,
    framework: bool = True,
    persist: bool = True,
    progress: Callable[[int, int, str, float | None], None] | None = None,
    on_result: Callable[[ModelBenchmark, str], None] | None = None,
    on_done: Callable[[str], None] | None = None,
) -> FleetBenchmarkResult:
    """Benchmark the distinct models in the catalogue and write their scores back.

    Quality is node-independent, so each model is benchmarked once and the scores written to all of
    its catalogue rows. Models are tried **smallest-first** and anything over ``max_params_b`` is
    skipped, so a giant model can't hang the run before the practical ones are scored (raise the cap
    to benchmark the big ones explicitly). Embedding models are skipped (they aren't chat models).

    ``progress(index, total, model_name)`` is called before each model — the benchmark is
    multi-minute and otherwise silent, so this (and the per-model log line) is how a UI or a log
    reader can tell it's alive (DESIGN §10 — stay observable).

    Tournament knobs (default to the classic one-pass full benchmark):
    - ``only_models`` — restrict the run to this set (a later round re-testing the survivors).
    - ``framework`` — ``False`` runs the cheap **triage** dimensions only (no epistemic gauntlet,
      no judge), for a fast first-round narrowing.
    - ``persist`` — ``False`` makes the run **ephemeral**: results stream via ``on_result`` but are
      NOT written to the catalogue, so a triage/scouting round can't pollute the real scores.
    - ``on_done(model_name)`` — called once per model when it's finished (scored, failed over to
      exhaustion, or had no viable node), so a UI can clear it from an in-flight/progress view even
      when it never produced a result.
    """
    # An inverted size band (floor above ceiling) is always a transposed pair of fields — it would
    # otherwise silently qualify NOTHING (min ≤ size ≤ max is unsatisfiable). Swap it, loudly,
    # rather than dead-end with an empty, unexplained round (DESIGN §10 — no silent empty state).
    if min_params_b and max_params_b and min_params_b > max_params_b:
        log.warning("benchmark: inverted size band — min %.1fB > max %.1fB; swapping (it would "
                    "qualify nothing). Set min ≤ max to silence.", min_params_b, max_params_b)
        min_params_b, max_params_b = max_params_b, min_params_b
    sizes: dict[str, float] = {}
    nodes_with: dict[str, list[str]] = {}
    eligible: set[str] = set()   # approved, non-embedding (any size) — for coverage reporting
    too_big: set[str] = set()    # eligible but over the size cap
    too_small: set[str] = set()  # eligible but under the size floor (user has hardware for more)
    for entry in list_catalogue(storage):
        if disabled_nodes and entry.node in disabled_nodes:
            continue  # the user vetoed this node — qualify and time nothing on it (DESIGN §5)
        nodes_with.setdefault(entry.model, []).append(entry.node)
        if "embed" in entry.model.lower():
            continue
        if only_approved and not is_approved(entry.family):
            continue
        if only_models is not None and entry.model not in only_models:
            continue  # tournament: a later round only re-tests the survivors the user kept
        eligible.add(entry.model)
        if max_params_b and entry.params_b and entry.params_b > max_params_b:
            too_big.add(entry.model)
            continue
        # A size FLOOR (opt-in): on capable hardware a tiny model that scores 'high enough' and wins
        # on latency keeps beating a bigger, genuinely-better one the test can't separate. Excluding
        # it from scoring keeps it out of recommendations entirely (DESIGN §4 — the rig is the one
        # fact only the user knows). The any-reachable fallback still uses it if it's all there is.
        if min_params_b and entry.params_b and entry.params_b < min_params_b:
            too_small.add(entry.model)
            continue
        sizes.setdefault(entry.model, entry.params_b)
    # Outside-in order (biggest, smallest, biggest, smallest …): the running-average ETA samples
    # both extremes immediately so it converges fast — unlike smallest-first, which back-loads all
    # the slow models and makes any early estimate wildly optimistic.
    by_size = sorted(sizes, key=lambda m: sizes[m])
    models = _outside_in(by_size)[:limit]

    total = len(models)
    # Capability is per-MODEL and NEVER failed on speed (DESIGN §6 / BENCHMARK_SCHEDULER): a model
    # slow on one node may be excellent on another. We pick the fastest node that can run it, fall
    # back through its other nodes if one is down/too-slow, and record per-node speed for routing.
    test_budget = max(_DEFAULT_SKIP_S, latency_budget_s)   # "fast enough to bother testing here"
    call_timeout = 60.0   # hang protection per scoring call (no retries); NOT the latency cap
    no_viable: set[str] = set()   # installed only on nodes that couldn't run it — not a quality cut
    log.info("benchmark: starting — %d model(s) to score (test budget %.0fs/turn)",
             total, test_budget)
    results: list[ModelBenchmark] = []

    # CONCURRENT: one worker per enabled http node (the worker is the node's VRAM lock — a node does
    # one model at a time). Each model's candidate nodes are rotated by index so different models
    # start on different nodes (spread → real parallelism). Mock/single-local (no http node) → the
    # gateway path, one worker, sequential (order-preserving, so the specs hold).
    http_nodes = sorted({n for m in models for n in nodes_with.get(m, []) if n.startswith("http")})
    node_locks = {n: threading.Lock() for n in http_nodes}
    state_lock = threading.Lock()
    loop_start = time.monotonic()
    done_count = 0

    def _qualify(item: tuple[int, str]) -> None:
        nonlocal done_count
        idx, model_name = item
        cands = [n for n in nodes_with.get(model_name, []) if n.startswith("http")]
        if cands:
            r = idx % len(cands)
            cands = cands[r:] + cands[:r]   # rotate so models spread across nodes

        def probe(n: str) -> float | None:
            with node_locks[n]:   # VRAM: hold the node while we warm + time it
                return _measure_node_speed(n, model_name, timeout_s=test_budget,
                                           warmup=True, num_ctx=num_ctx)

        chosen, speeds = _choose_test_node(cands, probe, test_budget) if cands else (None, {})
        if persist and speeds:
            with state_lock:
                for n, s in speeds.items():
                    update_catalogue_speed(storage, n, model_name, s)

        # Failover order: the best (chosen) node first, then the model's OTHER candidate nodes. A
        # node can pass the probe yet hang the real battery, and the model usually runs on several
        # nodes — so don't record a false ~0 on a bad node when a good one is a retry away (DESIGN
        # §6: a model is never failed on speed). No http candidates → gateway path (mock/single-
        # local). All candidates failed even the probe → no viable node (not a quality cut).
        if not cands:
            order: list[str | None] = [None]
        elif chosen is None:
            order = []
        else:
            order = [chosen] + [n for n in cands if n != chosen]

        if progress is not None:   # the model NOW entering scoring (not one that just finished)
            with state_lock:
                d = done_count
            eta = (time.monotonic() - loop_start) / d * (total - d) if d else None
            progress(d, total, model_name, eta)

        bench: ModelBenchmark | None = None
        used: str | None = None
        try:
            for node in order:
                lock = node_locks.get(node) if node else None
                if lock:
                    lock.acquire()
                try:
                    bench = benchmark_model(model, model_name, num_ctx=num_ctx, framework=framework,
                                            call_timeout_s=call_timeout, node=node)
                    used = node
                    break
                except _NodeUnreliable as exc:
                    log.warning("benchmark: %s unreliable on %s — failing over: %s",
                                model_name, node, exc)
                except Exception as exc:
                    log.warning("benchmark: %s failed on %s: %s", model_name, node, exc)
                finally:
                    if lock:
                        lock.release()

            # Best-across-nodes vision: vision is per-node (a runtime-version regression makes an
            # identical model file read images on one node, mangle them on another), so a multimodal
            # model that was scored on a node whose Ollama breaks its vision shouldn't read red when
            # it sees perfectly on another node. Only for a model that actually carries a vision
            # projector (reliable /api/show check, NOT the flaky tags caps) — so text models aren't
            # probed everywhere — and only when the test-node score is short of full.
            if (bench is not None and persist and used and used.startswith("http")
                    and bench.vision is not None and bench.vision < 1.0
                    and OllamaProvider(used).has_vision(model_name)):
                best_vis = bench.vision
                for node in cands:
                    if node == used or best_vis >= 1.0:
                        continue
                    lock = node_locks.get(node)
                    # Best-effort: if the node is busy scoring another model, skip it rather than
                    # block — vision is informational, and a later run settles it (DESIGN §5/§6).
                    if lock is not None and not lock.acquire(timeout=10):
                        continue
                    try:
                        v = _node_vision(node, model_name, num_ctx)
                    finally:
                        if lock is not None:
                            lock.release()
                    if v is not None and v > best_vis:
                        best_vis = v
                        log.info("benchmark: %s vision %.1f on %s (best across nodes)",
                                 model_name, v, node)
                bench.vision = round(best_vis, 3)

            with state_lock:
                done_count += 1
                if bench is None:
                    no_viable.add(model_name)
                    log.warning("benchmark: %s — no node could score it (%d candidate node(s))",
                                model_name, len(order))
                else:
                    speed_str = f"{bench.return_time:.1f}s/turn" if bench.return_time is not None \
                        else "speed unmeasured"
                    log.info("benchmark: [%d/%d] %s done — q=%.2f, %s%s",
                             done_count, total, model_name, bench.quality, speed_str,
                             f" on {used}" if used else "")
                    if persist:
                        # Node-independent scores model-wide; return_time is per-node (only the node
                        # that scored it — stamping every row would wreck the placement matrix).
                        update_catalogue_scores(
                            storage, model_name, quality=bench.quality,
                            talk=bench.talk, tools=bench.tools, code=bench.code,
                            discipline=bench.discipline, epistemics=bench.epistemics,
                            reasoning=bench.reasoning, vision=bench.vision,
                        )
                        if used and bench.return_time is not None:
                            update_catalogue_speed(storage, used, model_name, bench.return_time)
                    results.append(bench)
                    if on_result is not None:
                        on_result(bench, used or (nodes_with.get(model_name) or [""])[0])
        finally:
            # ALWAYS signal completion (scored, failed over to exhaustion, or no viable node), so a
            # UI clears it from the in-flight/progress view instead of leaving a ghost climbing.
            if on_done is not None:
                on_done(model_name)

    workers = max(1, len(http_nodes))   # ~one model per node; mock/single-local → 1 → sequential
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mimir-bench") as ex:
        list(ex.map(_qualify, enumerate(models)))

    log.info(
        "benchmark: scored %d of %d eligible; %d too big, %d too small, %d no viable node",
        len(results), len(eligible), len(too_big), len(too_small), len(no_viable),
    )
    return FleetBenchmarkResult(
        benchmarked=len(results), results=results,
        eligible=len(eligible), skipped_too_big=len(too_big),
        skipped_too_small=len(too_small), skipped_too_slow=len(no_viable),
    )


def complete_speed_matrix(
    storage: StorageGateway, *, min_quality: float = 0.5, num_ctx: int = 8192,
    timeout_s: float = 90.0, disabled_nodes: set[str] | None = None,
    only_models: set[str] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> int:
    """The **final time trial**: fill the per-node placement matrix. Qualification establishes
    capability (per-model) and times each model on whatever node it ran on; this pass speed-tests
    each **acceptable** model (``quality >= min_quality``) on every enabled http node it's installed
    on but **not yet timed on** — "which edge runs what, how fast" for the background pool.

    Reads the EXISTING catalogue (no rescan — that would wipe the quality scores). Slow results are
    RECORDED, never dropped: a slow (model, node) is still a real resource for capacity-bound
    work. A generous per-probe timeout captures a slow-but-real number rather than cutting it.
    Concurrent across nodes (one model at a time per node — VRAM). Returns the pairings probed.
    """
    disabled_nodes = disabled_nodes or set()
    pending: dict[str, list[str]] = {}   # node → models still to time on it
    for e in list_catalogue(storage):
        if not e.node.startswith("http") or e.node in disabled_nodes:
            continue
        if "embed" in e.model.lower() or e.return_time is not None:
            continue   # not a chat model, or already timed on this node
        if e.quality is None or e.quality < min_quality:
            continue   # only bother timing models good enough to place
        if only_models is not None and e.model not in only_models:
            continue
        pending.setdefault(e.node, []).append(e.model)
    total = sum(len(v) for v in pending.values())
    log.info("matrix: %d (model, node) pairings to time across %d node(s)", total, len(pending))
    if not total:
        return 0
    state_lock = threading.Lock()
    done = 0

    def _worker(item: tuple[str, list[str]]) -> None:
        nonlocal done
        node, models = item
        for m in models:
            if progress is not None:   # the pairing NOW being timed (not the one just finished)
                with state_lock:
                    d = done
                progress(d, total, f"{m} @ {node}")
            speed = _measure_node_speed(node, m, timeout_s=timeout_s, warmup=True, num_ctx=num_ctx)
            with state_lock:
                done += 1
                if speed is not None:
                    update_catalogue_speed(storage, node, m, speed)   # record even if slow

    with ThreadPoolExecutor(max_workers=max(1, len(pending)), thread_name_prefix="trial") as ex:
        list(ex.map(_worker, pending.items()))
    log.info("matrix: timed %d pairing(s)", total)
    return total

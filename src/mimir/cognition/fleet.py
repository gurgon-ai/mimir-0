"""Fleet scan + report — the persisted catalogue and its summary (DESIGN §5).

``scan_fleet`` inventories every node in the pool (model + family + weight + quant + capabilities)
and rebuilds the ``model_catalogue``. ``fleet_report`` reads it back into a per-node summary — the
human-facing "what's on my network" view. Phase 2 benchmarking fills the ``return_time`` and
``quality`` columns; Phase 3 turns the catalogue into per-role recommendations.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from ..model.gateway import ModelGateway
from ..model.provider import is_embedding_model
from ..storage.gateway import StorageGateway
from ..storage.models import CatalogueEntry
from ..storage.repo import list_catalogue, replace_catalogue
from .benchmark import is_approved
from .registry import is_recommended

log = logging.getLogger("mimir.fleet")


@dataclass(slots=True)
class FleetScanResult:
    nodes: int
    models: int


def scan_fleet(
    model: ModelGateway, storage: StorageGateway, *, now: float | None = None
) -> FleetScanResult:
    """Inventory every fleet node and rebuild the catalogue. Returns counts."""
    clock = time.time() if now is None else now
    entries: list[CatalogueEntry] = []
    for node, _label, infos in model.inventory_details():
        for info in infos:
            entries.append(
                CatalogueEntry(
                    node=node,
                    model=info.name,
                    family=info.family,
                    params_b=info.params_b,
                    quantization=info.quantization,
                    context_length=info.context_length,
                    capabilities=info.capabilities,
                    scanned_at=clock,
                )
            )
    replace_catalogue(storage, entries)
    nodes = len({e.node for e in entries})
    log.info("fleet: catalogued %d model(s) across %d node(s)", len(entries), nodes)
    return FleetScanResult(nodes=nodes, models=len(entries))


# Each role's required capability and whether it prefers speed or quality.
# chat/bake/reasoning are the live roles; tools/code are forward-looking (DESIGN §9 extension
# points) — recommended now so you know which model to use when you enable them.
# chat and reasoning are identity-bearing — they speak AS the system and synthesize its self-model.
# They gate on BOTH `discipline` (don't leak the prompt's [tier=...] tags) AND `epistemics` (do
# exploit the tiered/provenance/gated context — DESIGN §3). A model that mimics the scaffolding OR
# ignores evidence tiers can't clear the floor for these roles (§4). Each role lists every
# capability it requires; a candidate must clear the floor on all of them.
# The objective (every role): the best-scoring model FOR THIS SYSTEM that we're willing to wait for.
# Latency is a hard cap (max_latency_s excludes too-slow models before scoring), NOT a penalty — so
# WITHIN the cap, quality leads outright and speed only breaks ties. A dominant big model wins even
# when a tiny one is faster; paying up to the cap is the whole point.
ROLE_NEEDS: dict[str, tuple[tuple[str, ...], str]] = {
    "chat": (("discipline", "epistemics", "reasoning"), "quality"),
    "bake": (("talk",), "quality"),
    "reasoning": (("discipline", "epistemics", "reasoning"), "quality"),
    "tools": (("tools",), "quality"),
    "code": (("code", "reasoning"), "quality"),
    # The second lineup (DESIGN §5a): off-the-record cognition that never speaks AS the assistant,
    # so it is deliberately NOT discipline/epistemics-gated — a capable model that "leaks" the
    # identity is fine here. A reasoning-competence floor only. `background` staffs off-hot-path
    # reasoning (one best); `council` staffs adversarial deliberation (a diverse pool, see
    # `council_roster`). The brain harness queries these via `roster_for` to staff itself.
    "background": (("reasoning",), "quality"),
    "council": (("reasoning",), "quality"),
    # Vision (DESIGN §4 "Round 4"): EMPIRICALLY gated — only models that passed the image probe
    # qualify (vision >= floor); ranked by overall quality (the most capable model that can also
    # SEE). For the image/document-vision path; a non-vision fleet simply has no pick (None).
    "vision": (("vision",), "quality"),
}
# Roles whose value is a DIVERSE POOL, not a single best — staffed by `council_roster`, not by the
# single-best ranking. `roster_for` routes these to the diversity picker.
_POOL_ROLES = frozenset({"council"})
_CAPABILITY_FLOOR = 0.5


def _bar_reason(data: dict[str, Any], capabilities: tuple[str, ...]) -> str | None:
    """Why this model is barred from a role, or ``None`` if it clears every required floor.

    The SINGLE source of truth for the role gate — both ``recommend_roles`` (who wins) and the
    model-pool board (who's barred, and why) call it, so the leaderboard's explanation can never
    drift from the actual decision. Returns a human-readable reason naming the first failing
    capability and the floor it missed (e.g. ``"discipline 0.25 < 0.50"``), or ``"not benchmarked
    yet"`` when the model hasn't been scored. Never a silent drop (DESIGN §10).
    """
    if data.get("quality") is None:
        return "not benchmarked yet"
    for cap in capabilities:
        val = data.get(cap) or 0.0
        if val < _CAPABILITY_FLOOR:
            return f"{cap} {val:.2f} < {_CAPABILITY_FLOOR:.2f}"
    return None


def _collapse_by_model(
    storage: StorageGateway, disabled: set[str], disabled_nodes: set[str]
) -> dict[str, dict[str, Any]]:
    """One slot per model (capability is model-wide), tracking its fastest enabled node.

    Shared by every role query so they agree on the candidate set: ``disabled`` models and
    ``disabled_nodes`` are excluded, and speed (per-``(node, model)``) collapses to the model's
    fastest enabled node — the one a router would actually use.
    """
    by_model: dict[str, dict[str, Any]] = {}
    for entry in list_catalogue(storage):
        if entry.model in disabled or entry.node in disabled_nodes:
            continue
        slot = by_model.setdefault(
            entry.model,
            {
                "family": entry.family,
                "quality": entry.quality,
                "talk": entry.talk,
                "tools": entry.tools,
                "code": entry.code,
                "coherence": entry.coherence,
                "discipline": entry.discipline,
                "epistemics": entry.epistemics,
                "reasoning": entry.reasoning,
                "vision": entry.vision,
                "return_time": entry.return_time,
                "node": entry.node,  # the fastest node for this model (speed is per-node)
                "nodes": [],
            },
        )
        slot["nodes"].append(entry.node)
        # Track the fastest node: return_time is now measured per (node, model).
        if entry.return_time is not None and (
            slot["return_time"] is None or entry.return_time < slot["return_time"]
        ):
            slot["return_time"] = entry.return_time
            slot["node"] = entry.node
    return by_model


def _rank_for_role(
    by_model: dict[str, dict[str, Any]], role: str
) -> list[tuple[str, dict[str, Any]]]:
    """The role-eligible models, best first — the SINGLE ranking both single-best
    (``recommend_roles``) and pooled (``roster_for``) staffing read, so a role's "best" never
    disagrees with itself.

    Eligibility is the shared ``_bar_reason`` gate (never a silent drop); order follows the role's
    ``prefer``: ``fast`` → lowest latency; ``quality`` → highest quality, latency only breaking ties
    (under the cap you've already decided the wait is worth it — no soft speed penalty; §4).
    """
    capabilities, prefer = ROLE_NEEDS[role]
    candidates = [
        (name, data)
        for name, data in by_model.items()
        if _bar_reason(data, capabilities) is None
    ]
    if prefer == "fast":
        candidates.sort(key=lambda c: c[1]["return_time"] or 1e9)
    else:
        candidates.sort(key=lambda c: (-(c[1]["quality"] or 0.0), c[1]["return_time"] or 1e9))
    return candidates


def _as_pick(name: str, data: dict[str, Any], prefer: str) -> dict[str, Any]:
    """A ranked candidate as the public pick shape (``recommend_roles``/``roster_for`` output)."""
    return {
        "model": name,
        "family": data["family"],
        "quality": data["quality"],
        "return_time": data["return_time"],
        "node": data["node"],  # the fastest node holding this model
        "nodes": data["nodes"],
        "prefer": prefer,
    }


def recommend_roles(
    storage: StorageGateway, *, disabled: set[str] | None = None,
    disabled_nodes: set[str] | None = None,
) -> dict[str, dict[str, Any] | None]:
    """From the benchmarked catalogue, recommend the best model for each role (DESIGN §4).

    Recommend-only — it does not reassign roles. ``None`` for a role means nothing benchmarked
    clears the capability floor yet (run a benchmark first). ``disabled`` models and
    ``disabled_nodes`` (a user's enable/disable choices) are excluded — a model with no enabled node
    can't be a champion, and a disabled node never wins the fastest-node pick. The pool roles
    (``council``) appear here as their single strongest member; their diverse roster is
    ``council_roster`` / ``roster_for``.
    """
    by_model = _collapse_by_model(storage, disabled or set(), disabled_nodes or set())
    recommendations: dict[str, dict[str, Any] | None] = {}
    for role, (_caps, prefer) in ROLE_NEEDS.items():
        ranked = _rank_for_role(by_model, role)
        recommendations[role] = _as_pick(*ranked[0], prefer) if ranked else None
    return recommendations


def roster_for(
    storage: StorageGateway, role: str, *, n: int = 1,
    disabled: set[str] | None = None, disabled_nodes: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Staff a role from the qualified fleet — the brain harness's single query into the catalogue.

    The bridge from qualification → the harness (DESIGN §5a): the harness asks "give me N models for
    role R" instead of a human reading a view. Pool roles (``council``) return a diversity-first
    spread of up to ``n`` models (the second lineup — families before depth; ``council_roster``);
    every other role returns up to ``n`` role-eligible models, best by the role's preference. The
    loose roles (``background``, ``council``) are not discipline-gated. Returns ``[]`` when nothing
    qualifies yet (run a benchmark) — an empty roster, never a silent stub. Unknown role raises.
    """
    if role in _POOL_ROLES:
        return council_roster(
            storage, size=n, disabled=disabled, disabled_nodes=disabled_nodes
        )["roster"]
    if role not in ROLE_NEEDS:
        raise ValueError(f"unknown role {role!r}; known: {sorted(ROLE_NEEDS)}")
    by_model = _collapse_by_model(storage, disabled or set(), disabled_nodes or set())
    _caps, prefer = ROLE_NEEDS[role]
    return [_as_pick(name, data, prefer) for name, data in _rank_for_role(by_model, role)[:n]]


# Per-role ideal size (params_b) for the pre-benchmark heuristic — a defensible first guess only;
# measured scores override it the moment a benchmark runs (DESIGN §4, "future-proof"). chat and
# reasoning carry identity, so favour a mid-size capable model; bake (extraction) can run smaller.
_ROLE_IDEAL_SIZE_B: dict[str, float] = {"chat": 12.0, "reasoning": 12.0, "bake": 7.0}


def resolve_auto_model(
    storage: StorageGateway,
    role: str,
    *,
    available: set[str],
    disabled: set[str] | None = None,
) -> str | None:
    """Resolve a role's model for ``auto`` routing (DESIGN §4) — best model with no explicit pin.

    Hierarchy, each level vetoing disabled models and anything the pool can't currently reach:

    1. **measured-best** — the benchmarked, role-gated recommendation (quality + the discipline
       floor for identity roles). Measured scores always win, so the system future-proofs itself
       as new models appear and get benchmarked.
    2. **approved-family heuristic** — before any benchmark exists, prefer a curated-family model
       near the role's ideal size (approved models win the first round).
    3. **any reachable enabled model** — last resort, so ``auto`` always yields something runnable.

    Returns a concrete model name, or ``None`` if nothing is reachable yet.
    """
    disabled = disabled or set()

    def usable(name: str) -> bool:
        # Never route a chat-style role to an embedding model (and catch the ones without "embed" in
        # the name — all-minilm, bge, … — that a bare substring check would miss).
        return name in available and name not in disabled and not is_embedding_model(name)

    # 1. Measured-best (already gated + disabled-filtered by recommend_roles).
    rec = recommend_roles(storage, disabled=disabled).get(role)
    if rec and usable(rec["model"]):
        return str(rec["model"])

    # 2/3. Heuristic over the catalogue's discovery fields (present even before benchmarking).
    by_model: dict[str, CatalogueEntry] = {}
    for entry in list_catalogue(storage):
        if usable(entry.model):
            by_model.setdefault(entry.model, entry)
    pool = list(by_model.values())
    if pool:
        ideal = _ROLE_IDEAL_SIZE_B.get(role, 12.0)
        # Registry-recommended-for-this-role wins the first round (INFERENCE_ENGINE §4): a fresh
        # user with both gemma3:4b and gemma4:e4b installed gets gemma4:e4b for chat, not the
        # known-weak gemma3:4b — BEFORE any benchmark. Then approved-family, then any reachable.
        recommended = [e for e in pool if is_recommended(e.model, role)]
        approved = [e for e in pool if is_approved(e.family)]
        ranked = recommended or approved or pool
        # closest to the role's ideal size; tie-break toward the larger (more capable) model.
        best = min(ranked, key=lambda e: (abs((e.params_b or ideal) - ideal), -(e.params_b or 0.0)))
        return best.model

    # Catalogue not built yet (no scan) but the pool can reach models → pick deterministically.
    reachable = sorted(n for n in available if usable(n))
    return reachable[0] if reachable else None


def fleet_model_pool(
    storage: StorageGateway,
    *,
    disabled: set[str] | None = None,
    active_roles: dict[str, str] | None = None,
    auto_roles: set[str] | None = None,
) -> dict[str, Any]:
    """One row per distinct chat model: discovery + scores + a ``passed`` flag + state (DESIGN §4).

    The data behind the web UI's Model Pool tab. ``passed`` means the model was benchmarked and
    cleared the qualification floor; ``roles`` is which roles it currently serves; ``enabled``
    reflects the user's veto. Embedding models are omitted (they aren't routable chat models).
    """
    disabled = disabled or set()
    active_roles = active_roles or {}
    auto_roles = auto_roles or set()
    serving: dict[str, list[str]] = {}
    for role, model in active_roles.items():
        serving.setdefault(model, []).append(role)

    rows: dict[str, dict[str, Any]] = {}
    for e in list_catalogue(storage):
        if "embed" in e.model.lower():
            continue
        slot = rows.get(e.model)
        if slot is None:
            rows[e.model] = slot = {
                "model": e.model,
                "family": e.family,
                "params_b": e.params_b,
                "quality": e.quality,
                "talk": e.talk,
                "tools": e.tools,
                "code": e.code,
                "discipline": e.discipline,
                "epistemics": e.epistemics,
                "reasoning": e.reasoning,
                "vision": e.vision,
                "return_time": e.return_time,
                "node": e.node,  # the node giving the best (lowest) return_time — for the picker
                "approved": is_approved(e.family),
                "benchmarked": e.quality is not None,
                "nodes": [],
            }
        slot["nodes"].append(e.node)
        if e.return_time is not None and (
            slot["return_time"] is None or e.return_time < slot["return_time"]
        ):
            slot["return_time"] = e.return_time
            slot["node"] = e.node

    models: list[dict[str, Any]] = []
    for model, slot in rows.items():
        slot["enabled"] = model not in disabled
        slot["passed"] = slot["quality"] is not None and slot["quality"] >= _CAPABILITY_FLOOR
        slot["roles"] = sorted(serving.get(model, []))
        # Explain the verdict per role: which roles it clears the floor for (badge-able), and WHY
        # it's barred from the rest — never a silent drop (DESIGN §10). Same gate as recommend_roles
        # via the shared _bar_reason, so the board can't contradict the actual pick.
        eligible_roles: list[str] = []
        barred: dict[str, str] = {}
        for role, (capabilities, _prefer) in ROLE_NEEDS.items():
            reason = _bar_reason(slot, capabilities)
            if reason is None:
                eligible_roles.append(role)
            else:
                barred[role] = reason
        slot["eligible_roles"] = sorted(eligible_roles)
        slot["barred"] = barred
        models.append(slot)
    # Best first: enabled, then highest quality, then fastest.
    models.sort(key=lambda s: (not s["enabled"], -(s["quality"] or -1.0), s["return_time"] or 1e9))
    return {
        "models": models,
        "auto_roles": sorted(auto_roles),
        "active_roles": active_roles,
    }


def placement_matrix(
    storage: StorageGateway, *, disabled: set[str] | None = None,
    disabled_nodes: set[str] | None = None,
) -> dict[str, Any]:
    """The per-node worker roster — the DISPLAY side of the placement matrix the speed-test fills.

    Grouped by node, every model installed on that node appears with **this node's** measured speed,
    the (node-independent) capability scores, its per-role eligibility/bars, and two crowns: the
    node's **champion** (best quality, this-node speed breaking ties — the most capable model you
    can run here) and the **fastest** floor-clearing model on it. Unlike the results board (one row
    per model, on its single capability-test node), a model appears on EVERY node it runs on — so a
    strong multi-node worker like a mid-size model is finally visible as such (DESIGN §5/§5a).
    """
    disabled = disabled or set()
    disabled_nodes = disabled_nodes or set()
    by_node: dict[str, list[dict[str, Any]]] = {}
    for e in list_catalogue(storage):
        if "embed" in e.model.lower():
            continue
        slot: dict[str, Any] = {
            "model": e.model, "family": e.family, "params_b": e.params_b,
            "quality": e.quality, "talk": e.talk, "tools": e.tools, "code": e.code,
            "discipline": e.discipline, "epistemics": e.epistemics, "reasoning": e.reasoning,
            "coherence": e.coherence, "return_time": e.return_time,  # THIS node's per-turn latency
            "enabled": e.model not in disabled,
        }
        eligible: list[str] = []
        barred: dict[str, str] = {}
        for role, (caps, _prefer) in ROLE_NEEDS.items():
            reason = _bar_reason(slot, caps)
            if reason is None:
                eligible.append(role)
            else:
                barred[role] = reason
        slot["eligible_roles"] = sorted(eligible)
        slot["barred"] = barred
        by_node.setdefault(e.node, []).append(slot)

    for models in by_node.values():
        # Best first: quality leads, this node's speed breaks ties (a faster model wins when equal).
        models.sort(key=lambda m: (-(m["quality"] or -1.0),
                                   m["return_time"] if m["return_time"] is not None else 1e9))
        runnable = [m for m in models
                    if (m["quality"] or 0.0) >= _CAPABILITY_FLOOR and m["return_time"] is not None]
        if runnable:
            runnable[0]["champion"] = True   # already sorted → best quality, speed tiebreak
            min(runnable, key=lambda m: m["return_time"])["fastest"] = True

    # Include EVERY node (disabled ones too) so the view can show them greyed with a re-enable
    # toggle — hiding a disabled node would make it un-toggleable. The caller marks disabled ones.
    return {
        "nodes": len(by_node),
        "enabled_nodes": len([n for n in by_node if n not in disabled_nodes]),
        "by_node": by_node,
        "disabled_nodes": sorted(disabled_nodes),
    }


def council_roster(
    storage: StorageGateway, *, size: int = 5, min_quality: float = _CAPABILITY_FLOOR,
    disabled: set[str] | None = None, disabled_nodes: set[str] | None = None,
) -> dict[str, Any]:
    """A diverse adversarial-council roster — a SPREAD of model families, not the top-N ranking.

    This is the "second lineup": the non-user-facing pool for council / inner-dialogue / background
    reasoning. Its value is **diversity** — different model families fail in different ways, so a
    council of five qwen variants is worth far less than five different families (the parent runs a
    16-persona council across 7 families for exactly this). So we rank models *within* each family
    by quality, then **round-robin across families** — each family's best first, then seconds — so
    the roster pulls from as many distinct families as possible before doubling up.

    Capacity-bound, **not latency-gated** (DESIGN §5a, BENCHMARK_SCHEDULER §7): any model clearing a
    quality floor on an enabled node qualifies — the big-and-slow models a chat cap excludes are
    prime council members, so no size/latency cap applies here. (They must still be *graded* to
    appear: benchmark with the size cap off to pull the big pool in.) Returns the seated roster, the
    families represented, and the bench (qualified but not seated — the next-up).
    """
    disabled = disabled or set()
    disabled_nodes = disabled_nodes or set()
    # Collapse to one entry per model (capability is model-wide); track its fastest enabled node.
    by_model: dict[str, dict[str, Any]] = {}
    for e in list_catalogue(storage):
        if "embed" in e.model.lower() or e.model in disabled:
            continue
        if e.quality is None or e.quality < min_quality:
            continue
        # Membership = the shared `council` role gate (a reasoning floor, NOT discipline-gated), so
        # the seated roster can't disagree with the eligibility the board shows (DESIGN §10).
        gateable = {"quality": e.quality, "reasoning": e.reasoning}
        if _bar_reason(gateable, ROLE_NEEDS["council"][0]):
            continue
        slot = by_model.setdefault(e.model, {
            "model": e.model, "family": e.family or "?", "quality": e.quality,
            "reasoning": e.reasoning, "params_b": e.params_b,
            "node": None, "return_time": None, "nodes": [],
        })
        if e.node not in disabled_nodes:
            slot["nodes"].append(e.node)
            if e.return_time is not None and (
                slot["return_time"] is None or e.return_time < slot["return_time"]
            ):
                slot["return_time"] = e.return_time
                slot["node"] = e.node
    candidates = [s for s in by_model.values() if s["nodes"]]  # must run on ≥1 enabled node

    families: dict[str, list[dict[str, Any]]] = {}
    for s in candidates:
        families.setdefault(s["family"], []).append(s)
    for members in families.values():  # within a family: best quality first (reasoning breaks ties)
        members.sort(key=lambda s: (-(s["quality"] or 0.0), -(s["reasoning"] or 0.0)))
    fam_order = sorted(families, key=lambda f: -(families[f][0]["quality"] or 0.0))

    roster: list[dict[str, Any]] = []
    rank = 0
    while len(roster) < size and any(rank < len(families[f]) for f in fam_order):
        for fam in fam_order:
            if rank < len(families[fam]):
                roster.append(families[fam][rank])
                if len(roster) >= size:
                    break
        rank += 1

    seated = {s["model"] for s in roster}
    bench = sorted((s for s in candidates if s["model"] not in seated),
                   key=lambda s: -(s["quality"] or 0.0))
    return {
        "roster": roster,
        "families": sorted({s["family"] for s in roster}),
        "size": len(roster),
        "requested": size,
        "pool": len(candidates),
        "pool_families": len(families),
        "bench": bench,
    }


def fleet_report(storage: StorageGateway) -> dict[str, Any]:
    """The catalogue as a per-node summary (the 'report' the operator sees)."""
    entries = list_catalogue(storage)
    by_node: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        by_node.setdefault(entry.node, []).append(
            {
                "model": entry.model,
                "family": entry.family,
                "params_b": entry.params_b,
                "quantization": entry.quantization,
                "return_time": entry.return_time,
                "quality": entry.quality,
            }
        )
    return {
        "nodes": len(by_node),
        "models": len(entries),
        "by_node": by_node,
        "recommendations": recommend_roles(storage),
    }

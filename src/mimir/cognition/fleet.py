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
}
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


def recommend_roles(
    storage: StorageGateway, *, disabled: set[str] | None = None,
    disabled_nodes: set[str] | None = None,
) -> dict[str, dict[str, Any] | None]:
    """From the benchmarked catalogue, recommend the best model for each role (DESIGN §4).

    Recommend-only — it does not reassign roles. ``None`` for a role means nothing benchmarked
    clears the capability floor yet (run a benchmark first). ``disabled`` models and
    ``disabled_nodes`` (a user's enable/disable choices) are excluded — a model with no enabled node
    can't be a champion, and a disabled node never wins the fastest-node pick.
    """
    disabled = disabled or set()
    disabled_nodes = disabled_nodes or set()
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

    recommendations: dict[str, dict[str, Any] | None] = {}
    for role, (capabilities, prefer) in ROLE_NEEDS.items():
        candidates = [
            (name, data)
            for name, data in by_model.items()
            if _bar_reason(data, capabilities) is None
        ]
        if not candidates:
            recommendations[role] = None
            continue
        if prefer == "fast":
            name, data = min(candidates, key=lambda c: c[1]["return_time"] or 1e9)
        else:  # "quality" — the objective: the best-scoring model we're willing to WAIT for. The
            # latency cap already excluded anything too slow, so within it quality leads outright
            # and return_time only breaks ties (a faster model wins when quality is equal). No
            # soft speed penalty: under the cap, you've already decided the wait is worth it (§4).
            name, data = max(
                candidates, key=lambda c: (c[1]["quality"], -(c[1]["return_time"] or 1e9))
            )
        recommendations[role] = {
            "model": name,
            "family": data["family"],
            "quality": data["quality"],
            "return_time": data["return_time"],
            "node": data["node"],  # the fastest node holding this model
            "nodes": data["nodes"],
            "prefer": prefer,
        }
    return recommendations


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
        return name in available and name not in disabled and "embed" not in name.lower()

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
                "return_time": e.return_time,
                "approved": is_approved(e.family),
                "benchmarked": e.quality is not None,
                "nodes": [],
            }
        slot["nodes"].append(e.node)
        if e.return_time is not None and (
            slot["return_time"] is None or e.return_time < slot["return_time"]
        ):
            slot["return_time"] = e.return_time

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

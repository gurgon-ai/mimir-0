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


# Each role's required capability and whether it prefers speed, quality, or a balance.
# chat/bake/reasoning are the live roles; tools/code are forward-looking (DESIGN §9 extension
# points) — recommended now so you know which model to use when you enable them.
# chat and reasoning are identity-bearing — they speak AS the system and synthesize its self-model.
# So they gate on `discipline` (honoring prohibitions, not leaking the prompt's [tier=...] tags),
# not just `talk`: a model that mimics the scaffolding can't clear the floor for these roles (§4).
ROLE_NEEDS: dict[str, tuple[str, str]] = {
    "chat": ("discipline", "balanced"),
    "bake": ("talk", "quality"),
    "reasoning": ("discipline", "quality"),
    "tools": ("tools", "quality"),
    "code": ("code", "quality"),
}
_CAPABILITY_FLOOR = 0.5


def recommend_roles(storage: StorageGateway) -> dict[str, dict[str, Any] | None]:
    """From the benchmarked catalogue, recommend the best model for each role (DESIGN §4).

    Recommend-only — it does not reassign roles. ``None`` for a role means nothing benchmarked
    clears the capability floor yet (run a benchmark first).
    """
    by_model: dict[str, dict[str, Any]] = {}
    for entry in list_catalogue(storage):
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
    for role, (capability, prefer) in ROLE_NEEDS.items():
        candidates = [
            (name, data)
            for name, data in by_model.items()
            if data["quality"] is not None and (data.get(capability) or 0.0) >= _CAPABILITY_FLOOR
        ]
        if not candidates:
            recommendations[role] = None
            continue
        if prefer == "fast":
            name, data = min(candidates, key=lambda c: c[1]["return_time"] or 1e9)
        elif prefer == "quality":
            name, data = max(
                candidates, key=lambda c: (c[1]["quality"], -(c[1]["return_time"] or 1e9))
            )
        else:  # balanced: favour quality, lightly penalise slowness
            name, data = max(
                candidates, key=lambda c: c[1]["quality"] - 0.1 * (c[1]["return_time"] or 0.0)
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

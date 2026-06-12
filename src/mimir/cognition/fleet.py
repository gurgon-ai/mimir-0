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
    return {"nodes": len(by_node), "models": len(entries), "by_node": by_node}

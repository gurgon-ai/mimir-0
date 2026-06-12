"""The recommended-models registry (INFERENCE_ENGINE.md §4) — a curated, versioned default.

Pure data (``recommended_models.toml``), loaded read-only. It (1) seeds SAFE defaults before any
benchmark, so ``auto`` routing doesn't land on a known-weak model out of the box, and (2) names the
models trusted to *judge* unknown ones (cold-start trust). It is **not** a whitelist: any installed
model can still be measured and used — this only orders the pre-benchmark heuristic and, once
benchmarking runs, measured scores override it (DESIGN §4 / INFERENCE_ENGINE §3a).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from fnmatch import fnmatch
from functools import lru_cache
from importlib.resources import files

_REGISTRY_FILE = "recommended_models.toml"


@dataclass(frozen=True, slots=True)
class RecommendedModel:
    family: str
    tag_patterns: tuple[str, ...]
    roles: tuple[str, ...]
    judge_ok: bool
    min_params_b: float
    notes: str
    expected: dict[str, tuple[float, float]] = field(default_factory=dict)

    def matches(self, model: str) -> bool:
        low = model.lower()
        return any(fnmatch(low, pat.lower()) for pat in self.tag_patterns)


@lru_cache(maxsize=1)
def _load() -> tuple[int, tuple[RecommendedModel, ...]]:
    raw = files("mimir.cognition").joinpath(_REGISTRY_FILE).read_text(encoding="utf-8")
    data = tomllib.loads(raw)
    models = tuple(
        RecommendedModel(
            family=str(m["family"]),
            tag_patterns=tuple(str(p) for p in m.get("tag_patterns", [])),
            roles=tuple(str(r) for r in m.get("roles", [])),
            judge_ok=bool(m.get("judge_ok", False)),
            min_params_b=float(m.get("min_params_b", 0.0)),
            notes=str(m.get("notes", "")),
            expected={
                str(k): (float(v[0]), float(v[1])) for k, v in m.get("expected", {}).items()
            },
        )
        for m in data.get("model", [])
    )
    return int(data.get("version", 0)), models


def registry_version() -> int:
    """The registry's version — participates in staleness (INFERENCE_ENGINE §8)."""
    return _load()[0]


def recommended_models() -> tuple[RecommendedModel, ...]:
    return _load()[1]


def is_recommended(model: str, role: str | None = None) -> bool:
    """True if ``model`` matches a recommended entry (and, if given, one fit for ``role``)."""
    for entry in recommended_models():
        if role is not None and role not in entry.roles:
            continue
        if entry.matches(model):
            return True
    return False


def is_trusted_judge(model: str) -> bool:
    """True if ``model`` is a recommended family trusted to judge unknown models (cold start)."""
    return any(e.judge_ok and e.matches(model) for e in recommended_models())

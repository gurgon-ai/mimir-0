"""Vision in documents (#3): an ingested image is described + transcribed by the vision-role model
into recallable text. Needs a vision model — without one an image ingest fails loud (DESIGN §10)."""

from __future__ import annotations

import dataclasses

from mimir.brain import Mimir
from mimir.config import Config, RoleSpec
from mimir.storage.models import MemoryKind
from mimir.storage.repo import list_memories

# A 1×1 PNG (valid image bytes) — enough to exercise the path; the mock "vision" model is scripted.
_PNG_1x1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753"
    "de0000000c4944415408d763f8cfc0f01f0005000186a0d4f60000000049454e44ae426082"
)


def _vision_brain(mock_config: Config, tmp_path):
    cfg = dataclasses.replace(
        mock_config, documents_folder=str(tmp_path / "docs"),
        roles={**mock_config.roles, "vision": RoleSpec("mock")})  # a bound vision model
    return Mimir(cfg)


def test_image_is_described_into_recallable_text(mock_config: Config, tmp_path) -> None:
    brain = _vision_brain(mock_config, tmp_path)
    try:
        # Scripted vision model: returns a transcription only when an image is attached.
        brain._model.chat = lambda role, messages, **k: (
            "A safety sign. Text: DANGER HIGH VOLTAGE." if any(m.get("images") for m in messages)
            else "")
        folder = brain._docs_folder()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "sign.png").write_bytes(_PNG_1x1)
        report = brain.ingest_pending_documents()
        assert "sign.png" in report["ingested"] and not report["failed"]
        # The description is now document-tier, recallable text.
        chunks = [m for m in list_memories(brain._storage, kind=MemoryKind.MEMORY) if m.source]
        assert any("HIGH VOLTAGE" in m.text for m in chunks)
        # …and shown in the unified Library doc list with the description as its summary.
        doc = next(d for d in brain.library_overview()["documents"] if d["filename"] == "sign.png")
        assert "VOLTAGE" in (doc["summary"] or "")
    finally:
        brain.close()


def test_image_without_a_vision_model_fails_loud(mock_config: Config, tmp_path) -> None:
    # No `vision` role bound → an image can't be described; the scan reports it, never silent.
    cfg = dataclasses.replace(mock_config, documents_folder=str(tmp_path / "docs"))
    brain = Mimir(cfg)
    try:
        folder = brain._docs_folder()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "sign.png").write_bytes(_PNG_1x1)
        report = brain.ingest_pending_documents()
        assert any(f["name"] == "sign.png" and "vision" in f["error"].lower()
                   for f in report["failed"])
        assert "sign.png" not in report["ingested"]
    finally:
        brain.close()


def test_describe_image_raises_when_toggle_off(mock_config: Config, tmp_path) -> None:
    cfg = dataclasses.replace(
        mock_config, documents_folder=str(tmp_path / "docs"),
        roles={**mock_config.roles, "vision": RoleSpec("mock")}, vision_describe_images=False)
    brain = Mimir(cfg)
    try:
        folder = brain._docs_folder()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "sign.png").write_bytes(_PNG_1x1)
        report = brain.ingest_pending_documents()
        assert any(f["name"] == "sign.png" for f in report["failed"])   # off → reported, skipped
    finally:
        brain.close()

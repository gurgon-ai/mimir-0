"""Spec for OllamaProvider payload shaping — `think` placement (the latency knob)."""

from __future__ import annotations

from mimir.model.providers.ollama import _split_think, _to_options


def test_think_defaults_off_and_is_pulled_out_of_options() -> None:
    # Default: think off, and it must NOT leak into the options block (Ollama ignores it there).
    think, opts = _split_think({"temperature": 0.4, "num_ctx": 8192})
    assert think is False
    assert "think" not in _to_options(opts)
    assert _to_options(opts) == {"temperature": 0.4, "num_ctx": 8192}


def test_think_opt_in_per_role() -> None:
    # A role can opt in (bool or a level string), and it's removed from the options.
    think, opts = _split_think({"think": True, "temperature": 0.3})
    assert think is True and "think" not in opts
    think2, _ = _split_think({"think": "high"})
    assert think2 == "high"


def test_max_tokens_still_maps_to_num_predict() -> None:
    # Regression: pulling think out doesn't disturb the existing options translation.
    assert _to_options({"max_tokens": 256})["num_predict"] == 256

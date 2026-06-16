"""Trust policy: who-said-it → evidence tier (DESIGN §3b). The caller declares the speaker; the
server config decides how much that speaker is believed — so an exposed API can't fake trust."""

from __future__ import annotations

import pytest

from mimir.cognition.bake import _tier_and_provenance, normalize_speaker_kind
from mimir.storage.models import EvidenceTier


def _tier(user, primary=None, trusted=None):
    return _tier_and_provenance(user, primary, trusted)[0]


def test_zero_config_single_user_trusts_the_lone_speaker() -> None:
    # No policy at all → a build-your-own-UI that names its user gets full trust. "Just works."
    assert _tier("greg") is EvidenceTier.STATED_BY_PRIMARY_USER


def test_unattributed_calls_are_conversation_tier() -> None:
    assert _tier(None) is EvidenceTier.CONVERSATION
    assert _tier(None, primary="greg", trusted=["julien"]) is EvidenceTier.CONVERSATION


def test_policy_only_believes_configured_identities() -> None:
    primary, trusted = "greg", ["julien"]
    assert _tier("greg", primary, trusted) is EvidenceTier.STATED_BY_PRIMARY_USER
    assert _tier("julien", primary, trusted) is EvidenceTier.STATED_BY_TRUSTED
    # an unrecognized named speaker (open-API caller, peer AI, guest) is attributed, NOT believed
    assert _tier("mimir-parent", primary, trusted) is EvidenceTier.CONVERSATION


def test_unrecognized_speaker_is_still_attributed() -> None:
    _tier_, provenance = _tier_and_provenance("mimir-parent", "greg", ["julien"])
    assert provenance == "stated by mimir-parent"  # attribution kept, just at conversation tier


def test_trusted_list_without_primary_still_gates() -> None:
    # Setting any policy (even just trusted_users) leaves unlisted named speakers at conversation.
    assert _tier("julien", None, ["julien"]) is EvidenceTier.STATED_BY_TRUSTED
    assert _tier("random", None, ["julien"]) is EvidenceTier.CONVERSATION


def test_peer_kind_is_below_human_and_marked_ai_sourced() -> None:
    tier, provenance = _tier_and_provenance("mimir-home", "greg", [], is_peer=True)
    assert tier is EvidenceTier.STATED_BY_PEER
    assert tier.multiplier < EvidenceTier.CONVERSATION.multiplier  # below human conversation
    assert provenance == "stated by peer AI mimir-home"  # attributed AND marked as an AI


def test_peer_flag_wins_over_a_trusted_identity() -> None:
    # An agent can't reach a human tier by also being named primary/trusted — kind wins.
    assert _tier_and_provenance("greg", "greg", [], is_peer=True)[0] is EvidenceTier.STATED_BY_PEER


def test_normalize_speaker_kind() -> None:
    assert normalize_speaker_kind(None) == "human"        # absent → human (back-compat)
    assert normalize_speaker_kind("human") == "human"
    assert normalize_speaker_kind("user") == "human"
    assert normalize_speaker_kind("ai_peer") == "ai_peer"
    assert normalize_speaker_kind("AI") == "ai_peer"      # case-insensitive
    with pytest.raises(ValueError):                        # a typo fails loud, never elevates
        normalize_speaker_kind("robot")


def test_turn_bakes_ai_peer_input_below_human(brain) -> None:
    from mimir.storage.models import MemoryKind
    from mimir.storage.repo import list_memories

    # Same speaker, same statement — as a human it's believed (zero-config → primary); as an AI peer
    # it's marked AI-sourced and tiered down. The caller's declared kind makes the difference.
    brain.turn("The north fence is wooden.", user="mimir-home", speaker_kind="ai_peer")
    mems = list_memories(brain._storage, kind=MemoryKind.MEMORY)
    assert mems and all(m.evidence_tier is EvidenceTier.STATED_BY_PEER for m in mems)
    assert all((m.provenance or "").startswith("stated by peer AI") for m in mems)


def test_turn_rejects_an_unknown_speaker_kind(brain) -> None:
    with pytest.raises(ValueError):
        brain.turn("hello", user="x", speaker_kind="robot")

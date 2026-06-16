"""Trust policy: who-said-it → evidence tier (DESIGN §3b). The caller declares the speaker; the
server config decides how much that speaker is believed — so an exposed API can't fake trust."""

from __future__ import annotations

from mimir.cognition.bake import _tier_and_provenance
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

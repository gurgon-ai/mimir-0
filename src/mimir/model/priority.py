"""Request priority tiers for the model gateway (mirrors the parent system's scheme).

Lower integer = higher priority = runs first / preempts under load. These drive the provider
pool's admission rules: during backend saturation a CHAT_CRITICAL request is always attempted,
USER_ADJACENT is attempted with retries clamped, and background tiers fail fast so they defer
instead of piling onto a struggling backend (DESIGN §5).
"""

from __future__ import annotations

from enum import IntEnum


class Priority(IntEnum):
    CHAT_CRITICAL = 0  # the live user turn — must run now
    USER_ADJACENT = 1  # post-turn work the user is still waiting on (bake, sentinel, query embed)
    BACKGROUND = 2  # scheduled/background cognition
    IDLE = 3  # lowest-priority background


# Default priority per cognitive role. Callers may override per call.
DEFAULT_ROLE_PRIORITY: dict[str, Priority] = {
    "chat": Priority.CHAT_CRITICAL,
    "embed": Priority.USER_ADJACENT,
    "bake": Priority.USER_ADJACENT,
    "reasoning": Priority.USER_ADJACENT,
}

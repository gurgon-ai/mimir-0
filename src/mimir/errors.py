"""Typed exceptions for Mimir 0.

The disease this project guards against is *silence* (DESIGN §10). Every failure that
matters has a named type here so the core can fail loud, point at a cause, and never
collapse into a bare ``except`` that swallows the signal. There are no generic
``raise Exception(...)`` calls in core — always one of these.
"""

from __future__ import annotations


class MimirError(Exception):
    """Base class for every error Mimir raises on purpose."""


class ConfigError(MimirError):
    """A config file is missing, malformed, or describes an impossible setup.

    Raised loud at boot with an instruction for the user — never silently defaulted
    in a way that changes behavior without them knowing.
    """


class SchemaError(MimirError):
    """The on-disk database does not match the schema this code expects.

    The startup schema check raises this instead of silently migrating in an
    unexpected direction or — worse — falling back to a different store (DESIGN §10).
    The message must tell the user exactly what mismatched and what to do.
    """


class MigrationError(MimirError):
    """A migration step failed or the migration ladder is internally inconsistent."""


class StorageError(MimirError):
    """A write routed through the storage gateway failed.

    Carries the original exception as ``__cause__`` so the failure is never anonymous.
    """


class ProviderError(MimirError):
    """A model/embeddings provider call failed.

    ``transient=True`` marks a fault the caller may retry or defer (a busy backend),
    as opposed to a permanent misconfiguration. Background cognition checks this flag
    to back off instead of corrupting state against a busy backend (DESIGN §5).
    """

    def __init__(self, message: str, *, transient: bool = False, timeout: bool = False) -> None:
        super().__init__(message)
        self.transient = transient
        self.timeout = timeout  # the call hit its time limit — node is dead/molasses, not just busy


class ModelGatewayError(MimirError):
    """The model gateway could not route or fulfill a request (e.g. unknown role)."""


class ContextBudgetError(MimirError):
    """``build_context()`` could not honor a hard budget constraint.

    Truncating a *low*-tier section is a warning, not an error. This is reserved for
    the cases the design treats as faults — e.g. a required section that cannot fit at all.
    """


class SelfTestError(MimirError):
    """The §6 acceptance loop, run as a runtime self-test, did not pass.

    'No bake / no recall / no sentinel' is a *fault*, not a quiet state (DESIGN §10).
    """


class IngestError(MimirError):
    """A document could not be ingested — unsupported type, unreadable file, or a missing
    optional extractor (e.g. PDF support needs the ``[documents]`` extra). Raised loud with
    an instruction, never a silent skip.
    """


class NotebookError(MimirError):
    """A notebook operation could not proceed — e.g. editing a notebook that doesn't exist, or
    creating one past the self-grooming soft cap. Surfaced, never silently dropped (§10).
    """

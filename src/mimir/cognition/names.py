"""names.py — snap misheard or mistyped names to known ones before anything is stored.

Speech-to-text writes an unfamiliar name the way it sounds, and people mistype names. If both
spellings reach memory, the facts about one person or animal split across two entities and every
lookup by either name finds half of them. The fix has to happen where text enters, before bake,
graph and recall see it; correcting stored facts afterwards means finding every store they reached.
(Priming the speech model with the names instead makes it produce fluent text from audio it
couldn't actually resolve.)

Two tiers, both reported back as ``(heard, canonical)`` pairs:

1. **Confirmed variants** (``[identity] name_variants = { Caitlin = ["Katelyn"] }``) and user
   aliases are replaced anywhere, in any case.
2. **Phonetic match**: a capitalised word that isn't the first word of its sentence (recognisers
   and people capitalise proper nouns and sentence starts, nothing else), isn't itself a known
   name, and has the same consonant-skeleton key as exactly one known name, with character
   similarity of at least 0.5. "Coal" at the start of a sentence stays "Coal"; "Katelin"
   mid-sentence becomes "Caitlin".

The roster is the operator, the identity anchors, ``[identity] known_names``, and capitalised words
from the onboarding answers about who else is around. This module never touches memory: it rewrites
the turn text and returns what it changed.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

MIN_RATIO = 0.5
_PHONETIC_SUBS = (
    ("ph", "f"),
    ("ck", "k"),
    ("ch", "x"),
    ("sh", "x"),
    ("th", "0"),
    ("dg", "j"),
    ("gh", ""),
    ("c", "k"),
    ("q", "k"),
    ("x", "ks"),
    ("z", "s"),
)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*|[^A-Za-z]+")
_CAP_WORD_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")


def phonetic_key(word: str) -> str:
    """Consonant-skeleton key: Caitlin/Katelyn → 'ktln', Cole/Kohl → 'kl', Jon/John → 'jn'."""
    w = re.sub(r"[^a-z]", "", (word or "").lower())
    if not w:
        return ""
    for a, b in _PHONETIC_SUBS:
        w = w.replace(a, b)
    first, rest = w[0], re.sub(r"[aeiouhwy]", "", w[1:])
    return re.sub(r"(.)\1+", r"\1", first + rest)


@dataclass(slots=True)
class Roster:
    """Known names (canonical spelling) and confirmed variants (lower-case variant → canonical)."""

    names: dict[str, str] = field(default_factory=dict)  # lower → Canonical
    variants: dict[str, str] = field(default_factory=dict)  # lower variant → Canonical

    def add(self, name: str, variants: list[str] | None = None) -> None:
        name = (name or "").strip()
        if not name or not name.replace("-", "").replace("'", "").isalpha():
            return
        self.names[name.lower()] = name
        for v in variants or []:
            v = (v or "").strip()
            if v and v.lower() not in self.names:
                self.variants[v.lower()] = name

    def add_from_text(self, text: str) -> int:
        """Harvest capitalised words from free text ("Sam, and the horses Pepper and Rex")."""
        n = 0
        for m in _CAP_WORD_RE.finditer(text or ""):
            if m.group().lower() not in self.names:
                self.add(m.group())
                n += 1
        return n

    def __len__(self) -> int:
        return len(self.names)


def snap(text: str, roster: Roster) -> tuple[str, list[tuple[str, str]]]:
    """Return ``(snapped_text, [(heard, canonical), …])``. Never raises."""
    if not text or not isinstance(text, str) or not roster.names:
        return text, []
    keys: dict[str, list[str]] = {}
    for lower, name in roster.names.items():
        keys.setdefault(phonetic_key(lower), []).append(name)
    subs: list[tuple[str, str]] = []
    out: list[str] = []
    sentence_start = True
    for m in _WORD_RE.finditer(text):
        tok = m.group()
        if not tok[0].isalpha():
            out.append(tok)
            if re.search(r"[.!?\n]", tok):
                sentence_start = True
            continue
        low = tok.lower()
        rep: str | None = None
        if low in roster.variants and low not in roster.names:
            rep = roster.variants[low]
        elif (
            low not in roster.names
            and tok[0].isupper()
            and not sentence_start
            and len(low) >= 3
            and not tok.isupper()
        ):
            cands = keys.get(phonetic_key(low), [])
            if (
                len(cands) == 1
                and difflib.SequenceMatcher(None, low, cands[0].lower()).ratio() >= MIN_RATIO
            ):
                rep = cands[0]
        if rep:
            subs.append((tok, rep))
            out.append(rep.upper() if tok.isupper() and len(tok) > 1 else rep)
        else:
            out.append(tok)
        sentence_start = False
    if not subs:
        return text, []
    return "".join(out), subs

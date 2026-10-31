"""names.py — snap heard or mistyped names to the household's known names, once, at the door.

A name the speech recogniser has never seen comes out the way it sounds. In the production system
this core came from, a dog's name was transcribed "Katelyn" for weeks; memory extraction filed eight
facts and a graph node under that spelling beside nine under the right one, so every lookup by
either name got half the animal. The place to fix a heard name is where the text enters, before
bake, graph and recall ever see it. Priming the speech model with names was rejected: it fabricates
fluent text from hopeless audio.

Two tiers, both logged:

1. **Variants** the operator has confirmed (``[identity] name_variants = { Caitlin = ["Katelyn"] }``) and
   user aliases snap anywhere, any case.
2. **Phonetic match**: a capitalised word that is not the first word of its sentence (recognisers and
   people capitalise proper nouns and sentence starts, nothing else), is not itself a known name, and
   whose consonant-skeleton key equals exactly one roster name's key with character similarity of at
   least 0.5. "Code" at a sentence start stays "Code"; "Katelin" after "draw" becomes "Caitlin".

The roster is the operator and identity anchors, ``[identity] known_names``, and the capitalised
words of the onboarding "household" and "pets" answers. Nothing here touches memory; it rewrites the
turn text and reports what it changed.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

MIN_RATIO = 0.5
_PHONETIC_SUBS = (("ph", "f"), ("ck", "k"), ("ch", "x"), ("sh", "x"), ("th", "0"), ("dg", "j"),
                  ("gh", ""), ("c", "k"), ("q", "k"), ("x", "ks"), ("z", "s"))
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*|[^A-Za-z]+")
_CAP_WORD_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")


def phonetic_key(word: str) -> str:
    """A small consonant-skeleton key: Katelyn/Caitlin/Kaitlyn → 'mk', Kohl/Cole → 'kd', Jon/John → 'jn'."""
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

    names: dict[str, str] = field(default_factory=dict)      # lower → Canonical
    variants: dict[str, str] = field(default_factory=dict)   # lower variant → Canonical

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
        """Harvest capitalised words from a free-text answer ("Sam, and the horses Pepper and Rex")."""
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
        elif (low not in roster.names and tok[0].isupper() and not sentence_start
              and len(low) >= 3 and not tok.isupper()):
            cands = keys.get(phonetic_key(low), [])
            if len(cands) == 1 and difflib.SequenceMatcher(
                    None, low, cands[0].lower()).ratio() >= MIN_RATIO:
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

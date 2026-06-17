"""Citation guard (DESIGN §10): catch a reply that cites a source the system does not actually hold.

A model — a small one especially — will sometimes answer from training-data general knowledge and
dress it in a real-looking citation (invent ``[National Fire Code 2020]``, ``[OSHA 1910.120]``). For
an evidence-tiered, provenance-tracked system that is the worst failure: a hallucination wearing a
source, which reads as *more* trustworthy than an unsourced guess.

This is a deterministic, zero-model-cost post-check. Any bracketed citation whose named source
matches **nothing** the system holds (its ingested documents + library) is flagged — annotated,
never silently deleted (fail loud, DESIGN §10). It verifies the *source exists*, not that the cited
content is truly inside it (that needs a semantic check); paired with the prompt framing and a
capable model, it closes the invented-source gap. Conservative by design: it only flags brackets
that clearly look like a document citation and match no known source, so a real one isn't accused.
"""

from __future__ import annotations

import re

# A bracketed span that could be a citation — bounded so a stray "[" doesn't swallow the reply.
_CITE_RE = re.compile(r"\[([^\[\]]{3,200})\]")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_EXT_RE = re.compile(r"\.(?:docx|pdf|md|markdown|txt|text)\b", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-z0-9]+")


def _norm(text: str) -> str:
    """Lowercase, strip a file extension + punctuation, collapse runs — for fuzzy matching."""
    text = _EXT_RE.sub(" ", text.lower())
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(normalized: str) -> set[str]:
    return {w for w in _WORD_RE.findall(normalized) if len(w) >= 3}


def _citation_title(bracket: str) -> str:
    """The source-naming part of a citation: everything before the first locator separator (':' or
    ','). ``[Manual Nov 2024, COR 1.8]`` → ``Manual Nov 2024``; ``[file.docx:Sec]`` → ``file``."""
    cut = len(bracket)
    for sep in (":", ","):
        i = bracket.find(sep)
        if i != -1:
            cut = min(cut, i)
    return bracket[:cut].strip()


def _looks_like_citation(bracket: str, title_norm: str) -> bool:
    """A real document citation, not prose like ``[1]`` or ``[see above]``. Qualify on a strong
    document signal (a year or a file extension) or a multi-word title (≥3 meaningful tokens)."""
    if _YEAR_RE.search(bracket) or _EXT_RE.search(bracket):
        return True
    return len(_tokens(title_norm)) >= 3


def _matches_known(title_norm: str, known: list[tuple[str, set[str]]]) -> bool:
    """Does this citation title name a source the system holds? Substring either way, or ≥2 shared
    meaningful words (so 'Servus Group OHS Manual Nov 2024' still matches its file)."""
    if not title_norm:
        return True  # nothing to check — don't accuse
    tt = _tokens(title_norm)
    for src_norm, src_tokens in known:
        if not src_norm:
            continue
        if title_norm in src_norm or src_norm in title_norm:
            return True
        if len(tt & src_tokens) >= 2:
            return True
    return False


def unverified_citations(reply: str, known_sources: set[str]) -> list[str]:
    """Citation-like brackets in ``reply`` that name no source the system holds (deduped, in order).

    ``known_sources`` is the raw set of document filenames/titles the system actually has. With no
    known sources the guard stays silent (a fresh install citing nothing isn't suspicious)."""
    if not known_sources:
        return []
    known = [(_norm(s), _tokens(_norm(s))) for s in known_sources]
    bad: list[str] = []
    for bracket in _CITE_RE.findall(reply):
        title_norm = _norm(_citation_title(bracket))
        if not _looks_like_citation(bracket, title_norm):
            continue
        if not _matches_known(title_norm, known) and bracket not in bad:
            bad.append(bracket)
    return bad


def citation_warning(bad: list[str]) -> str:
    """The fail-loud note appended when a reply cited sources the system doesn't hold (or '')."""
    if not bad:
        return ""
    joined = "; ".join(f"[{b}]" for b in bad)
    return ("\n\n⚠ Unverified citation(s) — not matched to any document in the library; treat as "
            f"general knowledge, not a quote from your sources: {joined}")


def annotate_unverified_citations(reply: str, known_sources: set[str]) -> str:
    """Append the fail-loud warning to ``reply`` if any citation names an unknown source."""
    return reply + citation_warning(unverified_citations(reply, known_sources))

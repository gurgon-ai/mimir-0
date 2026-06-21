"""Offline encyclopedia via Kiwix/ZIM over HTTP — an optional live reference layer (DESIGN §9).

**Zero Python dependency.** Mimir queries a running ``kiwix-serve`` with the standard library
(HTTP + ``html.parser``), exactly as it talks to a model endpoint. The user downloads any ZIM
(Wikipedia nopic, a medical wiki, a top-50k slice, or something else), runs ``kiwix-serve`` over it,
and points config at it — no install, no compiled wheel, works on the same edge boxes that run the
models. Each turn's query is searched live and the top articles' lead text is injected as an
attributed reference section, so the knowledge layer is "populated" with a whole encyclopedia at no
ingest cost.

**Fail-open + bounded.** Every network call is wrapped and time-capped: a missing, slow, or broken
wiki yields no section and never raises into — or stalls — the turn (DESIGN §10).
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from collections.abc import Callable
from html.parser import HTMLParser

log = logging.getLogger("mimir.wiki")

_SKIP_TAGS = {"script", "style", "head", "nav", "sup", "table"}


class _TextExtractor(HTMLParser):
    """Collect visible text from article HTML — skipping scripts, styles, nav, and footnotes."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag in _SKIP_TAGS:
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if self._skip == 0:
            text = data.strip()
            if text:
                self._parts.append(text)

    def text(self) -> str:
        return " ".join(self._parts)


def _strip_html(html: str, limit: int) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:  # malformed markup must never break a turn
        pass
    text = " ".join(parser.text().split())
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + "…"
    return text


class WikiSource:
    """Live reference lookups against a ``kiwix-serve`` over a ZIM."""

    def __init__(
        self, *, url: str, book: str, max_articles: int = 2, max_chars: int = 800,
        timeout_s: float = 2.0, fetch: Callable[[str], bytes] | None = None,
    ) -> None:
        self._url = url.rstrip("/")
        self._book = book
        self._max = max(1, max_articles)
        self._chars = max(80, max_chars)
        self._timeout = timeout_s
        self._fetch = fetch or self._http_get  # injectable for tests

    def _http_get(self, path: str) -> bytes:
        with urllib.request.urlopen(self._url + path, timeout=self._timeout) as resp:
            data: bytes = resp.read()
        return data

    def search(self, query: str) -> list[dict[str, str]]:
        """Top articles for ``query`` as ``[{title, text}]`` — fail-open (``[]`` on any error).

        Uses kiwix-serve's ``/suggest`` (JSON titles+paths), then fetches each article and strips it
        to its lead text. The trailing "search for …" suggestion (no ``path``) is ignored.
        """
        query = (query or "").strip()
        if not query or not self._book:
            return []
        book = urllib.parse.quote(self._book)
        try:
            raw = self._fetch(f"/suggest?content={book}&term={urllib.parse.quote(query)}")
            items = json.loads(raw.decode("utf-8", "replace"))
        except Exception as exc:  # unreachable / bad json / timeout — log + yield nothing (§10)
            log.warning("wiki: suggest failed: %s", exc)
            return []
        out: list[dict[str, str]] = []
        for item in items if isinstance(items, list) else []:
            path = item.get("path")
            title = item.get("value") or item.get("label")
            if not path or not title:
                continue  # the fulltext "containing '…'" entry has no path — skip it
            try:
                html = self._fetch(
                    f"/content/{book}/{urllib.parse.quote(path, safe='/')}"
                ).decode("utf-8", "replace")
            except Exception as exc:
                log.warning("wiki: content %s failed: %s", path, exc)
                continue
            text = _strip_html(html, self._chars)
            if text:
                out.append({"title": str(title), "text": text})
            if len(out) >= self._max:
                break
        return out

    def status(self) -> dict[str, object]:
        """Reachability for the UI: ``{reachable, url, book, error?}`` — never raises."""
        info: dict[str, object] = {"url": self._url, "book": self._book}
        if not self._book:
            return {**info, "reachable": False, "error": "no book configured"}
        try:
            raw = self._fetch(f"/suggest?content={urllib.parse.quote(self._book)}&term=mimir")
            json.loads(raw.decode("utf-8", "replace"))
            return {**info, "reachable": True}
        except Exception as exc:
            return {**info, "reachable": False, "error": str(exc)}

    def context(self, query: str) -> str | None:
        """The reference-section body for ``query`` (one short block per article), or ``None``."""
        results = self.search(query)
        if not results:
            return None
        return "\n\n".join(f"{r['title']}: {r['text']}" for r in results)

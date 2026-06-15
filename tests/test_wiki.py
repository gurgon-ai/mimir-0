"""Executable spec for the optional offline-encyclopedia source (Kiwix/ZIM over HTTP, DESIGN §9).

Pure stdlib + an injected fetcher, so there's no real server (or dependency) in the test. Fail-open:
a missing/broken wiki yields nothing, never an error.
"""

from __future__ import annotations

import json

from mimir.brain import Mimir
from mimir.cognition.wiki import WikiSource, _strip_html


def _fetch(suggest: list[dict], articles: dict[str, str]):
    def fetch(path: str) -> bytes:
        if path.startswith("/suggest"):
            return json.dumps(suggest).encode("utf-8")
        for p, html in articles.items():
            if path.endswith(p):
                return html.encode("utf-8")
        raise KeyError(path)
    return fetch


def test_strip_html_drops_scripts_and_truncates() -> None:
    html = "<body><script>junk()</script><style>x{}</style><p>An apple is a pome fruit.</p></body>"
    text = _strip_html(html, 200)
    assert "junk" not in text and "x{}" not in text
    assert "An apple is a pome fruit." in text
    assert _strip_html("<p>" + "word " * 100 + "</p>", 40).endswith("…")


def test_search_parses_suggest_and_strips_articles() -> None:
    suggest = [
        {"value": "Penicillin", "path": "A/Penicillin", "kind": "path"},
        {"value": "containing 'penicillin'…", "kind": "pattern"},  # no path → skipped
    ]
    articles = {"A/Penicillin": "<body><script>x()</script><p>Penicillin is an antibiotic.</p>"}
    w = WikiSource(url="http://x", book="wiki", fetch=_fetch(suggest, articles))
    res = w.search("penicillin")
    assert len(res) == 1
    assert res[0]["title"] == "Penicillin"
    assert "antibiotic" in res[0]["text"] and "x()" not in res[0]["text"]


def test_search_caps_to_max_articles() -> None:
    suggest = [{"value": f"T{i}", "path": f"A/T{i}"} for i in range(5)]
    articles = {f"A/T{i}": f"<p>article {i}</p>" for i in range(5)}
    w = WikiSource(url="http://x", book="wiki", max_articles=2, fetch=_fetch(suggest, articles))
    assert len(w.search("t")) == 2


def test_fail_open_on_unreachable_server() -> None:
    def boom(path: str) -> bytes:
        raise OSError("connection refused")
    w = WikiSource(url="http://x", book="wiki", fetch=boom)
    assert w.search("anything") == []
    assert w.context("anything") is None


def test_context_renders_attributed_blocks() -> None:
    w = WikiSource(url="http://x", book="wiki",
                   fetch=_fetch([{"value": "Apple", "path": "A/Apple"}],
                                {"A/Apple": "<p>An apple is a fruit.</p>"}))
    ctx = w.context("apple")
    assert ctx and ctx.startswith("Apple:") and "fruit" in ctx


def test_no_book_or_empty_query_yields_nothing() -> None:
    assert WikiSource(url="http://x", book="", fetch=_fetch([], {})).search("apple") == []
    assert WikiSource(url="http://x", book="wiki", fetch=_fetch([], {})).search("  ") == []


class _StubWiki:
    def context(self, query: str) -> str:
        return "Stub: a reference result."


def test_brain_gates_smalltalk_from_wiki(brain: Mimir) -> None:
    brain._wiki = _StubWiki()
    assert brain._wiki_context("hi") is None                       # trivial → no lookup
    assert brain._wiki_context("thanks") is None
    assert brain._wiki_context("what is a quasar?") == "Stub: a reference result."
    assert brain._wiki_context("tell me about quasars please") == "Stub: a reference result."

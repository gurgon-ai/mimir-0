"""Discover Ollama nodes on the LAN — zero setup on the nodes (just ``ollama serve``).

The whole appeal of the distributed pool is that worker machines need *no* Mimir code: they just
run Ollama on :11434. Discovery here is purely client-side — probe the local host, any explicitly
declared nodes, and (when ``lan_backend`` is on) every host on a subnet — and keep the ones whose
``/api/tags`` answers. The probe is injectable so the scan is testable without touching a network.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from ..config import BackendConfig

log = logging.getLogger("mimir.discovery")

_DEFAULT_PORT = 11434
LOCAL_URL = f"http://127.0.0.1:{_DEFAULT_PORT}"


def normalize_url(node: str) -> str:
    """Normalize a host/url into ``http://host:port`` (default port 11434)."""
    text = node.strip()
    if not text:
        return ""
    if "://" not in text:
        text = "http://" + text
    parsed = urlparse(text)
    host = parsed.hostname
    if not host:
        return ""
    return f"{parsed.scheme or 'http'}://{host}:{parsed.port or _DEFAULT_PORT}"


def _http_probe(url: str, timeout: float) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=timeout) as resp:
            return bool(resp.status == 200)
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _local_subnet() -> str | None:
    """Best-effort detect the local /24 by finding this host's LAN IP."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))  # no packet sent for a UDP connect; just picks the route
        ip = sock.getsockname()[0]
    except OSError:
        return None
    finally:
        sock.close()
    return str(ipaddress.ip_network(f"{ip}/24", strict=False))


def discover_node_urls(
    backend: BackendConfig, *, probe: Callable[[str], bool] | None = None
) -> list[str]:
    """Return the Ollama node URLs to pool: localhost + declared nodes + (optional) scanned subnet.

    Localhost and explicitly-declared nodes are always included (the pool health-checks them);
    only subnet-scanned hosts are probed here, so a /24 doesn't create 254 dead endpoints.
    """
    do_probe = probe or (lambda u: _http_probe(u, backend.scan_timeout_s))

    urls: list[str] = [LOCAL_URL]
    for node in backend.nodes:
        url = normalize_url(node)
        if url:
            urls.append(url)

    if backend.lan_backend:
        subnet = backend.subnet or _local_subnet()
        if subnet:
            try:
                network = ipaddress.ip_network(subnet, strict=False)
            except ValueError as exc:
                log.warning("discovery: bad subnet %r: %s", subnet, exc)
                network = None
            hosts = (
                [f"http://{ip}:{_DEFAULT_PORT}" for ip in network.hosts()] if network else []
            )
            log.info("discovery: scanning %d host(s) on %s for Ollama", len(hosts), subnet)
            with ThreadPoolExecutor(max_workers=max(1, backend.scan_concurrency)) as pool:
                reachable = list(pool.map(do_probe, hosts))
            urls.extend(url for url, ok in zip(hosts, reachable, strict=True) if ok)
        else:
            log.warning("discovery: lan_backend on but no subnet given and auto-detect failed")

    # Dedupe, preserve order.
    seen: set[str] = set()
    deduped: list[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            deduped.append(url)
    log.info("discovery: %d candidate node(s): %s", len(deduped), deduped)
    return deduped

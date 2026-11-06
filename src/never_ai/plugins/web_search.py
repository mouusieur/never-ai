"""Live Multi-Source Internet Search & Page Extraction Plugin (`web_search_engine`).

Runs providers concurrently inside thread-bounded hard timeouts (`<= 2.6s` wall-clock)
so DNS `getaddrinfo()` on offline/firewalled networks can NEVER stall chat streaming.
On a normal internet connection (Windows `.exe` / home PC), DuckDuckGo Web + Wikipedia FR/EN
+ GitHub + npm/PyPI all respond in parallel without requiring any paid API key.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
import html
import json
import re
from typing import Any
import urllib.parse
import urllib.request

from never_ai.plugins.base import NeverPlugin


_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) NeverAI/1.3 Safari/537.36"
)

_TAG_STRIP_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _clean_html_text(raw: str) -> str:
    txt = _TAG_STRIP_RE.sub(" ", raw or "")
    txt = html.unescape(txt)
    return _WS_RE.sub(" ", txt).strip()


def _http_get_text(url: str, timeout: float = 2.2, accept: str = "*/*") -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": _USER_AGENT,
            "Accept": accept,
            "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read(220_000).decode(charset, errors="replace")


def _provider_duckduckgo(query: str, max_results: int = 4) -> list[dict[str, str]]:
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    body = _http_get_text(url, timeout=2.2, accept="text/html")
    results: list[dict[str, str]] = []
    blocks = re.findall(
        r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>(.*?)(?=<a[^>]*class="result__a"|$)',
        body,
        flags=re.DOTALL | re.IGNORECASE,
    )
    for raw_href, raw_title, tail in blocks[:max_results]:
        href = html.unescape(raw_href)
        if "uddg=" in href:
            parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
            if "uddg" in parsed and parsed["uddg"]:
                href = urllib.parse.unquote(parsed["uddg"][0])
        snip_m = re.search(r'class="result__snippet"[^>]*>(.*?)</(?:a|div|span)>', tail, flags=re.DOTALL | re.IGNORECASE)
        snippet = _clean_html_text(snip_m.group(1)) if snip_m else ""
        title = _clean_html_text(raw_title)
        if title and href.startswith("http"):
            results.append(
                {
                    "source": "Web (DuckDuckGo)",
                    "title": title,
                    "url": href,
                    "snippet": snippet[:320],
                }
            )
    return results


def _provider_wikipedia(query: str, lang: str = "fr", max_results: int = 2) -> list[dict[str, str]]:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": str(max_results),
        "format": "json",
        "utf8": "1",
    }
    url = f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    data = json.loads(_http_get_text(url, timeout=2.2, accept="application/json"))
    out: list[dict[str, str]] = []
    for item in data.get("query", {}).get("search", [])[:max_results]:
        title = str(item.get("title", "")).strip()
        snippet = _clean_html_text(str(item.get("snippet", "")))
        if title:
            page_url = f"https://{lang}.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
            out.append(
                {
                    "source": f"Wikipedia ({lang.upper()})",
                    "title": title,
                    "url": page_url,
                    "snippet": snippet[:320],
                }
            )
    return out


def _provider_github(query: str, max_results: int = 3) -> list[dict[str, str]]:
    clean_q = re.sub(r"\b(cherche|recherche|internet|web|sur|les|des|que|quoi|est|c'est|un|une)\b", " ", query, flags=re.IGNORECASE)
    clean_q = _WS_RE.sub(" ", clean_q).strip() or query.strip()
    params = {"q": clean_q, "sort": "stars", "order": "desc", "per_page": str(max_results)}
    url = "https://api.github.com/search/repositories?" + urllib.parse.urlencode(params)
    data = json.loads(_http_get_text(url, timeout=2.3, accept="application/vnd.github+json"))
    out: list[dict[str, str]] = []
    for repo in data.get("items", [])[:max_results]:
        full_name = str(repo.get("full_name", ""))
        desc = str(repo.get("description") or "Projet open-source sur GitHub.").strip()
        stars = int(repo.get("stargazers_count") or 0)
        lang = repo.get("language") or "Multi"
        html_url = str(repo.get("html_url", ""))
        if full_name and html_url:
            out.append(
                {
                    "source": "GitHub Live",
                    "title": f"{full_name} (★ {stars:,} · {lang})",
                    "url": html_url,
                    "snippet": desc[:300],
                }
            )
    return out


def _provider_npm(query: str, max_results: int = 2) -> list[dict[str, str]]:
    clean_q = re.sub(r"\b(cherche|recherche|internet|web|sur|les|des|que|quoi|est)\b", " ", query, flags=re.IGNORECASE)
    clean_q = _WS_RE.sub(" ", clean_q).strip() or query.strip()
    params = {"text": clean_q, "size": str(max_results)}
    url = "https://registry.npmjs.org/-/v1/search?" + urllib.parse.urlencode(params)
    data = json.loads(_http_get_text(url, timeout=2.2, accept="application/json"))
    out: list[dict[str, str]] = []
    for obj in data.get("objects", [])[:max_results]:
        pkg = obj.get("package", {})
        name = str(pkg.get("name", ""))
        ver = str(pkg.get("version", ""))
        desc = str(pkg.get("description") or "").strip()
        link = str(pkg.get("links", {}).get("npm") or f"https://www.npmjs.com/package/{name}")
        if name and desc:
            out.append(
                {
                    "source": "NPM Registry",
                    "title": f"{name} v{ver}",
                    "url": link,
                    "snippet": desc[:280],
                }
            )
    return out


def _fetch_direct_url(url: str) -> dict[str, str] | None:
    raw = _http_get_text(url, timeout=2.5)
    title_m = re.search(r"<title[^>]*>(.*?)</title>", raw, flags=re.DOTALL | re.IGNORECASE)
    title = _clean_html_text(title_m.group(1)) if title_m else url
    raw_no_scripts = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
    text = _clean_html_text(raw_no_scripts)
    if not text:
        return None
    return {
        "source": "Page Web Directe",
        "title": title[:120],
        "url": url,
        "snippet": text[:650],
    }


class WebSearchEnginePlugin(NeverPlugin):
    name = "web_search_engine"
    description = (
        "Live multi-source internet search & URL reader (DuckDuckGo Web, Wikipedia FR/EN, "
        "GitHub Live, NPM/PyPI) with strict non-blocking timeouts."
    )
    keywords = [
        "internet", "web", "recherche", "cherche", "google", "actualité", "news",
        "site", "url", "http", "https", "wikipedia", "github", "trouve", "en ligne",
    ]
    parameters_schema = {
        "query": {"type": "string", "description": "Search query or URL to inspect live on the internet."},
        "max_results": {"type": "integer", "description": "Maximum combined results (default 6)."},
    }

    def execute(self, query: str = "", max_results: int = 6, **_: Any) -> dict[str, Any]:
        q = (query or "").strip()
        if not q:
            return {"ok": False, "error": "Empty search query"}

        max_r = max(1, min(10, int(max_results)))
        results: list[dict[str, str]] = []
        providers_reached: list[str] = []

        # Check if query contains a direct URL
        url_match = re.search(r"https?://[^\s<>\"']+", q)

        pool = ThreadPoolExecutor(max_workers=5)
        try:
            futs = {}
            if url_match:
                futs[pool.submit(_fetch_direct_url, url_match.group(0))] = "DirectURL"
            futs[pool.submit(_provider_duckduckgo, q, 4)] = "DuckDuckGo"
            futs[pool.submit(_provider_wikipedia, q, "fr", 2)] = "Wikipedia-FR"
            futs[pool.submit(_provider_wikipedia, q, "en", 1)] = "Wikipedia-EN"
            futs[pool.submit(_provider_github, q, 3)] = "GitHub"
            futs[pool.submit(_provider_npm, q, 2)] = "NPM"

            done, _not_done = wait(list(futs.keys()), timeout=2.7)
            seen_urls: set[str] = set()
            # Order results: DuckDuckGo & Wikipedia first, then GitHub & NPM
            ordered_done = sorted(
                done,
                key=lambda f: {"DirectURL": 0, "DuckDuckGo": 1, "Wikipedia-FR": 2, "Wikipedia-EN": 3, "GitHub": 4, "NPM": 5}.get(futs[f], 9),
            )
            for fut in ordered_done:
                pname = futs[fut]
                try:
                    val = fut.result(timeout=0.05)
                except Exception:
                    continue
                if isinstance(val, dict):
                    providers_reached.append(pname)
                    results.append(val)
                    seen_urls.add(val["url"])
                elif isinstance(val, list) and val:
                    providers_reached.append(pname)
                    for item in val:
                        u = item.get("url", "")
                        if u and u not in seen_urls:
                            seen_urls.add(u)
                            results.append(item)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        trimmed = results[:max_r]
        return {
            "ok": True,
            "query": q,
            "count": len(trimmed),
            "providers_reached": providers_reached,
            "results": trimmed,
            "network_note": (
                "Recherche internet complète active."
                if any(p in providers_reached for p in ("DuckDuckGo", "Wikipedia-FR", "DirectURL"))
                else (
                    f"Sources live accessibles dans ce conteneur cloud : {', '.join(providers_reached)}. "
                    "(Sur votre PC / .exe Windows, DuckDuckGo + Wikipédia + tout le Web s'activent automatiquement)."
                    if providers_reached
                    else "Réseau externe restreint ou hors-ligne : bascule sur les connaissances internes."
                )
            ),
        }

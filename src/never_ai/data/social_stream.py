"""Raw, Zero-Directive Social Network Stream Harvester (X/Twitter, Microblogging & Reddit FR).

Feeds authentic, unedited human posts and reply chains directly into ChronoMorph-LM's
synaptic weights and Titans neural memory — with ZERO behavioral directives, ZERO labels,
and ZERO artificial modes, so Never-AI constructs its own emergent voice and temperament.
"""

from __future__ import annotations

import html
import json
import re
import urllib.parse
import urllib.request
from xml.etree import ElementTree
from typing import Any

from never_ai.core.special_tokens import BOS_TOKEN, EOS_TOKEN, NEVER_TOKEN, USER_TOKEN

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NeverAI-SovereignLearner/3.0"
_URL_CLEAN_RE = re.compile(r"https?://\S+")
_SPACE_RE = re.compile(r"[ \t]{2,}")


def _clean_social_text(raw: str) -> str:
    txt = html.unescape(raw or "")
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = _URL_CLEAN_RE.sub("", txt)
    txt = _SPACE_RE.sub(" ", txt).strip()
    return txt


class SocialStreamHarvester:
    """Harvests raw posts & reply chains from public social networks without any directive."""

    # Public Nitter / X RSS mirrors (best-effort when reachable from user's PC or Colab)
    X_MIRRORS: tuple[str, ...] = (
        "https://nitter.poast.org",
        "https://nitter.privacydev.net",
        "https://xcancel.com",
    )

    REDDIT_FR_SUBS: tuple[str, ...] = (
        "AskFrance",
        "PasDeQuestionIdiote",
        "discussionsbancales",
        "france",
        "TropPeurDeDemander",
    )

    BSKY_FR_QUERIES: tuple[str, ...] = (
        "lang:fr",
        "franchement lang:fr",
        "mdr lang:fr",
        "en vrai lang:fr",
    )

    def _http_get_json(self, url: str, timeout: float = 6.0) -> Any:
        req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))

    def _http_get_text(self, url: str, timeout: float = 6.0) -> str:
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")

    def fetch_x_public_search_or_accounts(
        self,
        query_or_handle: str = "lang:fr",
        max_items: int = 30,
    ) -> list[tuple[str, str | None]]:
        """Best-effort fetch of public X posts via syndication / Nitter RSS mirrors."""
        collected: list[tuple[str, str | None]] = []
        q = query_or_handle.strip()

        # 1) If user gave a @handle, try X syndication timeline + Nitter RSS
        if q.startswith("@") and len(q) > 2:
            handle = q.lstrip("@").strip()
            for mirror in self.X_MIRRORS:
                try:
                    rss_xml = self._http_get_text(f"{mirror}/{handle}/rss", timeout=5.0)
                    root = ElementTree.fromstring(rss_xml)
                    for item in root.findall(".//item"):
                        desc = _clean_social_text(item.findtext("description") or item.findtext("title") or "")
                        if len(desc) >= 12:
                            collected.append((desc, None))
                            if len(collected) >= max_items:
                                return collected
                except Exception:  # noqa: BLE001
                    continue

        # 2) Search RSS on public X mirrors
        enc = urllib.parse.quote(q)
        for mirror in self.X_MIRRORS:
            try:
                rss_xml = self._http_get_text(f"{mirror}/search/rss?f=tweets&q={enc}", timeout=5.0)
                root = ElementTree.fromstring(rss_xml)
                for item in root.findall(".//item"):
                    desc = _clean_social_text(item.findtext("description") or item.findtext("title") or "")
                    if len(desc) >= 12:
                        collected.append((desc, None))
                        if len(collected) >= max_items:
                            return collected
            except Exception:  # noqa: BLE001
                continue
        return collected

    def fetch_bluesky_fr_microblog(self, max_items: int = 60) -> list[tuple[str, str | None]]:
        """Fetch live real-time French microblogging posts & reply pairs from public ATProto/Bluesky."""
        out: list[tuple[str, str | None]] = []
        seen: set[str] = set()
        for q in self.BSKY_FR_QUERIES:
            if len(out) >= max_items:
                break
            try:
                enc = urllib.parse.quote(q)
                url = f"https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts?q={enc}&limit=40"
                data = self._http_get_json(url, timeout=6.0)
                for post in data.get("posts", []):
                    rec = post.get("record") or {}
                    txt = _clean_social_text(rec.get("text") or "")
                    uri = post.get("uri") or txt[:40]
                    if len(txt) < 10 or uri in seen:
                        continue
                    seen.add(uri)

                    # Try to fetch immediate thread parent/replies if post has replies!
                    reply_count = int(post.get("replyCount") or 0)
                    if reply_count > 0 and len(out) < max_items and post.get("uri"):
                        try:
                            t_uri = urllib.parse.quote(post["uri"])
                            tdata = self._http_get_json(
                                f"https://public.api.bsky.app/xrpc/app.bsky.feed.getPostThread?uri={t_uri}&depth=1",
                                timeout=4.0,
                            )
                            replies = (tdata.get("thread") or {}).get("replies") or []
                            added_pair = False
                            for rep_node in replies[:3]:
                                rtxt = _clean_social_text(((rep_node.get("post") or {}).get("record") or {}).get("text") or "")
                                if len(rtxt) >= 8:
                                    out.append((txt, rtxt))
                                    added_pair = True
                            if added_pair:
                                continue
                        except Exception:  # noqa: BLE001
                            pass

                    out.append((txt, None))
                    if len(out) >= max_items:
                        break
            except Exception:  # noqa: BLE001
                continue
        return out

    def fetch_reddit_fr_threads(self, max_pairs: int = 60) -> list[tuple[str, str | None]]:
        """Fetch live French Reddit threads + top spontaneous human replies."""
        pairs: list[tuple[str, str | None]] = []
        for sub in self.REDDIT_FR_SUBS:
            if len(pairs) >= max_pairs:
                break
            try:
                listing = self._http_get_json(f"https://www.reddit.com/r/{sub}/hot.json?limit=12", timeout=6.0)
                children = ((listing.get("data") or {}).get("children")) or []
                for child in children[:8]:
                    pdata = child.get("data") or {}
                    if pdata.get("stickied") or pdata.get("over_18"):
                        continue
                    title = _clean_social_text(pdata.get("title") or "")
                    selftext = _clean_social_text(pdata.get("selftext") or "")
                    permalink = pdata.get("permalink") or ""
                    prompt = f"{title}\n{selftext}".strip() if selftext else title
                    if len(prompt) < 12 or not permalink:
                        continue
                    try:
                        tree = self._http_get_json(f"https://www.reddit.com{permalink}.json?limit=6", timeout=5.0)
                        if isinstance(tree, list) and len(tree) >= 2:
                            comments = ((tree[1].get("data") or {}).get("children")) or []
                            for c in comments[:3]:
                                body = _clean_social_text((c.get("data") or {}).get("body") or "")
                                if len(body) >= 12 and "[deleted]" not in body and "[removed]" not in body:
                                    pairs.append((prompt[:900], body[:1200]))
                                    if len(pairs) >= max_pairs:
                                        break
                    except Exception:  # noqa: BLE001
                        pairs.append((prompt[:900], None))
                    if len(pairs) >= max_pairs:
                        break
            except Exception:  # noqa: BLE001
                continue
        return pairs

    @staticmethod
    def parse_raw_x_dump_or_text(raw_payload: str) -> list[tuple[str, str | None]]:
        """
        Parse raw pasted X/Twitter threads, JSONL, or X Archive `tweets.js` (`window.YTD.tweet.part0 = [...]`)
        into pure (post, optional_reply) items — zero directives, zero filtering of style.
        """
        text = (raw_payload or "").strip()
        if not text:
            return []

        # Case 1: Official X / Twitter data export `tweets.js` or JSON array
        json_candidate = re.sub(r"^window\.YTD\.tweets?\.part\d+\s*=\s*", "", text)
        if json_candidate.startswith("["):
            try:
                arr = json.loads(json_candidate)
                out: list[tuple[str, str | None]] = []
                for entry in arr:
                    tw = entry.get("tweet", entry) if isinstance(entry, dict) else {}
                    full_text = _clean_social_text(str(tw.get("full_text") or tw.get("text") or ""))
                    # Strip leading @mentions only if pure reply prefix, keep body intact
                    full_text = re.sub(r"^(?:@\w+\s+)+", "", full_text).strip()
                    if len(full_text) >= 6 and not full_text.startswith("RT @"):
                        out.append((full_text, None))
                if out:
                    return out
            except Exception:  # noqa: BLE001
                pass

        # Case 2: Raw pasted X thread / posts separated by blank lines or `---`
        blocks = [
            _clean_social_text(b)
            for b in re.split(r"\n\s*(?:---+|===+|\n)\s*", text)
            if len(_clean_social_text(b)) >= 6
        ]
        items: list[tuple[str, str | None]] = []
        # If multiple sequential posts in a thread/reply chain, pair consecutive ones AND keep standalone
        for idx, blk in enumerate(blocks):
            items.append((blk, None))
            if idx + 1 < len(blocks):
                items.append((blk, blocks[idx + 1]))
        return items

    @staticmethod
    def to_raw_training_sequences(items: list[tuple[str, str | None]]) -> list[str]:
        """Convert (post, optional_reply) into pure unconditioned ChronoMorph-LM sequences."""
        seqs: list[str] = []
        for post, reply in items:
            if reply:
                seqs.append(f"{BOS_TOKEN}{USER_TOKEN}{post}{NEVER_TOKEN}{reply}{EOS_TOKEN}")
            else:
                seqs.append(f"{BOS_TOKEN}{post}{EOS_TOKEN}")
        return seqs

    def harvest_live_bundle(self, x_query: str = "lang:fr", max_per_source: int = 40) -> dict[str, Any]:
        """Pull live social posts & replies across X mirrors, Bluesky FR, and Reddit FR."""
        x_items = self.fetch_x_public_search_or_accounts(x_query, max_items=max_per_source)
        bsky_items = self.fetch_bluesky_fr_microblog(max_items=max_per_source)
        reddit_items = self.fetch_reddit_fr_threads(max_pairs=max_per_source)
        all_items = x_items + bsky_items + reddit_items
        return {
            "counts": {
                "x_mirrors": len(x_items),
                "microblog_fr": len(bsky_items),
                "reddit_fr": len(reddit_items),
                "total": len(all_items),
            },
            "sequences": self.to_raw_training_sequences(all_items),
            "samples": [p if r is None else f"{p} → {r}" for p, r in all_items[:6]],
        }

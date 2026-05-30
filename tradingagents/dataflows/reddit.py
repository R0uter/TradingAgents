"""Reddit search fetcher using TikHub SDK (RedditApp).

Requires:
    TIKHUB_API_KEY in .env

Install:
    pip install tikhub
"""

from __future__ import annotations

import logging
import os
import time
from typing import Iterable

logger = logging.getLogger(__name__)

DEFAULT_SUBREDDITS = ("wallstreetbets", "stocks", "investing")

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    try:
        from tikhub import TikHub
    except ImportError:
        raise ImportError("tikhub not installed — run: pip install tikhub")
    api_key = os.environ.get("TIKHUB_API_KEY", "").strip()
    if not api_key:
        raise ValueError("TIKHUB_API_KEY not set in .env")
    _client = TikHub(api_key=api_key)
    return _client


def _extract_posts(data: dict) -> list[dict]:
    """Navigate TikHub's GraphQL-shaped response to a flat list of normalized post dicts."""
    if not isinstance(data, dict):
        return []
    try:
        edges = (
            data["data"]["search"]["dynamic"]["components"]["main"]["edges"]
        )
    except (KeyError, TypeError):
        return []

    posts = []
    for edge in edges:
        try:
            children = edge["node"]["children"]
        except (KeyError, TypeError):
            continue
        for child in children:
            raw = child.get("post")
            if not raw:
                continue
            # normalize to the field names the formatter expects
            created_utc = None
            created_at = raw.get("createdAt")
            if created_at:
                try:
                    import datetime
                    dt = datetime.datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    created_utc = dt.timestamp()
                except Exception:
                    pass
            selftext = ""
            content = raw.get("content") or {}
            selftext = content.get("markdown") or content.get("html") or ""

            posts.append({
                "title":        raw.get("postTitle", ""),
                "score":        raw.get("score", 0),
                "num_comments": raw.get("commentCount", 0),
                "created_utc":  created_utc,
                "selftext":     selftext,
                "permalink":    raw.get("permalink", ""),
            })
    return posts


def _fetch_subreddit(ticker: str, sub: str, limit: int, timeout: float) -> list[dict]:
    try:
        resp = _get_client().reddit_app.fetch_dynamic_search(
            query=f"{ticker} subreddit:{sub}",
            search_type="post",
            sort="NEW",
            time_range="week",
            safe_search="unset",
            allow_nsfw="0",
            need_format=False,
        )
        posts = _extract_posts(resp if isinstance(resp, dict) else {})
        return posts[:limit]
    except Exception as exc:
        logger.warning("Reddit fetch failed for r/%s · %s: %s", sub, ticker, exc)
        return []


def fetch_reddit_posts(
    ticker: str,
    subreddits: Iterable[str] = DEFAULT_SUBREDDITS,
    limit_per_sub: int = 5,
    timeout: float = 15.0,
    inter_request_delay: float = 0.4,
) -> str:
    """Fetch recent Reddit posts mentioning ``ticker`` via TikHub SDK."""
    try:
        _get_client()
    except (ImportError, ValueError) as exc:
        logger.warning("Reddit skipped: %s", exc)
        return f"<Reddit data unavailable: {exc}>"

    subreddits = list(subreddits)
    blocks = []
    total_posts = 0

    for i, sub in enumerate(subreddits):
        if i > 0:
            time.sleep(inter_request_delay)
        posts = _fetch_subreddit(ticker, sub, limit_per_sub, timeout)
        total_posts += len(posts)
        if not posts:
            blocks.append(f"r/{sub}: <no posts found mentioning {ticker.upper()} in the past 7 days>")
            continue

        lines = [f"r/{sub} — {len(posts)} recent posts mentioning {ticker.upper()}:"]
        for p in posts:
            title = (p.get("title") or "").replace("\n", " ").strip()
            score = p.get("score", 0)
            comments = p.get("num_comments", 0)
            created = p.get("created_utc")
            created_str = (
                time.strftime("%Y-%m-%d", time.gmtime(int(created))) if created else "?"
            )
            selftext = (p.get("selftext") or "").replace("\n", " ").strip()
            if len(selftext) > 240:
                selftext = selftext[:240] + "…"
            lines.append(
                f"  [{created_str} · {score:>4}↑ · {comments:>3}c] {title}"
                + (f"\n    body excerpt: {selftext}" if selftext else "")
            )
        blocks.append("\n".join(lines))

    if total_posts == 0:
        return (
            f"<no Reddit posts found mentioning {ticker.upper()} across "
            f"{', '.join(f'r/{s}' for s in subreddits)} in the past 7 days>"
        )
    return "\n\n".join(blocks)

"""
Search grounding — DuckDuckGo + Wikipedia. Used before generating the
"commentator" agents so their names/outlets come from real search results
rather than the model inventing something that merely sounds plausible.
Best-effort: every function catches broadly and returns an empty result
rather than raising — a failed search should degrade the commentator
persona to an honestly-labeled fictional one, not crash the simulation.
"""
from __future__ import annotations
import asyncio
import httpx
from ddgs import DDGS

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
REQUEST_TIMEOUT = 12.0


def _ddg_search_sync(query: str, max_results: int = 5) -> list[dict]:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [
            {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
            for r in results
        ]
    except Exception:
        return []


async def ddg_search(query: str, max_results: int = 5) -> list[dict]:
    return await asyncio.to_thread(_ddg_search_sync, query, max_results)


async def wikipedia_search(query: str, limit: int = 3) -> list[dict]:
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.get(
                WIKI_API,
                params={"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": limit},
                headers={"User-Agent": "ArkSimulator/1.0 (educational prototype)"},
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            return [
                {"title": r["title"], "snippet": r.get("snippet", "")}
                for r in data.get("query", {}).get("search", [])
            ]
    except Exception:
        return []


async def wikipedia_summary(title: str) -> dict | None:
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.get(
                WIKI_SUMMARY.format(title=title.replace(" ", "_")),
                headers={"User-Agent": "ArkSimulator/1.0 (educational prototype)"},
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            return {
                "title": data.get("title", title),
                "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                "summary": data.get("extract", ""),
            }
    except Exception:
        return None


async def research_topic(query: str, wiki_limit: int = 2, ddg_limit: int = 5) -> str:
    """Combined DDG + Wikipedia digest, formatted as plain text ready to
    drop into an LLM prompt. Returns "" (not an exception) if both sources
    come back empty."""
    ddg_task = ddg_search(query, ddg_limit)
    wiki_search_task = wikipedia_search(query, wiki_limit)
    ddg_results, wiki_hits = await asyncio.gather(ddg_task, wiki_search_task)

    wiki_summaries = await asyncio.gather(*(wikipedia_summary(h["title"]) for h in wiki_hits))
    wiki_summaries = [s for s in wiki_summaries if s]

    lines = []
    if wiki_summaries:
        lines.append("Wikipedia:")
        for s in wiki_summaries:
            lines.append(f"- {s['title']}: {s['summary'][:400]}")
    if ddg_results:
        lines.append("Web search:")
        for r in ddg_results:
            lines.append(f"- {r['title']}: {r['snippet'][:250]} ({r['url']})")
    return "\n".join(lines)

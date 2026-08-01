"""
LLM router — tries providers in order until one succeeds, falls back to a
deterministic offline generator if none are configured or all fail.
Config-driven fallback chain (see config.py) rather than a single hardcoded
provider/model, since free-tier catalogs churn without notice.

Every provider failure is logged (not silently swallowed) — this is the
single biggest diagnostic tool for "why did my simulation only produce a
bland one-event fallback": check the terminal for these warnings.
"""
from __future__ import annotations
import json
import logging
import random
import httpx
from . import config

log = logging.getLogger("ark.llm_router")

REQUEST_TIMEOUT = config.REQUEST_TIMEOUT_SECONDS

# Tools offered to agents during post generation. Whether an agent calls
# these is a real per-agent LLM decision (see graph.py), not a keyword
# heuristic — a press wire deciding to post a photo, or genuinely replying
# to a specific other post, looks different from doing it on autopilot.

ATTACH_MEDIA_TOOL = {
    "name": "attach_media",
    "description": (
        "Attach a photo or video to this specific post. Only call this if it is "
        "natural for YOU, given your specific role, to be the one posting visual "
        "media about this event right now (e.g. a press wire releasing a photograph, "
        "a broadcaster releasing footage, a photographer on the scene). Most posts "
        "should NOT call this — real social posts are mostly text-only. Do not call "
        "this on every post."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["photo", "video"]},
            "caption": {
                "type": "string",
                "description": "A short, vivid, concrete description of exactly what the image/video depicts, written for an image generator to render.",
            },
        },
        "required": ["kind", "caption"],
    },
}

REPLY_TO_TOOL = {
    "name": "reply_to",
    "description": (
        "Mark this post as a direct reply to one specific earlier post — only call "
        "this if your post is genuinely responding to what a specific other agent "
        "just said, the way a real reply does. Most posts are NOT replies; they're "
        "independent reactions to the event itself. Don't call this just because "
        "someone else already posted about the same event."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "post_id": {
                "type": "string",
                "description": "The exact post id shown in brackets next to the post you're replying to, copied verbatim from the 'Recent posts about this' list.",
            },
        },
        "required": ["post_id"],
    },
}


def _to_gemini_schema(schema):
    """Gemini's function-calling REST API expects Schema.type as an
    upper-case enum (STRING/OBJECT/...), unlike the lowercase JSON Schema
    convention every other provider here uses. Deep-copy + uppercase rather
    than maintaining a second schema definition."""
    if isinstance(schema, dict):
        out = {}
        for k, v in schema.items():
            out[k] = v.upper() if (k == "type" and isinstance(v, str)) else _to_gemini_schema(v)
        return out
    if isinstance(schema, list):
        return [_to_gemini_schema(v) for v in schema]
    return schema


class ProviderError(Exception):
    pass


async def _call_openai_compatible(base_url: str, api_key: str, model: str, system: str, user: str) -> str:
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.9,
                "max_tokens": 700,
            },
        )
        if resp.status_code != 200:
            raise ProviderError(f"{base_url} -> {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        return data["choices"][0]["message"]["content"]


async def _call_groq(system: str, user: str) -> str:
    if not config.GROQ_API_KEY:
        raise ProviderError("no GROQ_API_KEY")
    return await _call_openai_compatible(
        "https://api.groq.com/openai/v1", config.GROQ_API_KEY, config.GROQ_MODEL, system, user
    )


async def _call_cerebras(system: str, user: str) -> str:
    if not config.CEREBRAS_API_KEY:
        raise ProviderError("no CEREBRAS_API_KEY")
    return await _call_openai_compatible(
        "https://api.cerebras.ai/v1", config.CEREBRAS_API_KEY, config.CEREBRAS_MODEL, system, user
    )


async def _call_openrouter(system: str, user: str) -> str:
    if not config.OPENROUTER_API_KEY:
        raise ProviderError("no OPENROUTER_API_KEY")
    return await _call_openai_compatible(
        "https://openrouter.ai/api/v1", config.OPENROUTER_API_KEY, config.OPENROUTER_MODEL, system, user
    )


async def _call_gemini(system: str, user: str) -> str:
    if not config.GEMINI_API_KEY:
        raise ProviderError("no GEMINI_API_KEY")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{config.GEMINI_MODEL}:generateContent?key={config.GEMINI_API_KEY}"
    )
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.post(
            url,
            json={
                "system_instruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": 0.9, "maxOutputTokens": 700},
            },
        )
        if resp.status_code != 200:
            raise ProviderError(f"gemini -> {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]


async def _call_anthropic(system: str, user: str) -> str:
    if not config.ANTHROPIC_API_KEY:
        raise ProviderError("no ANTHROPIC_API_KEY")
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": config.ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": config.ANTHROPIC_MODEL,
                "max_tokens": 700,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
        )
        if resp.status_code != 200:
            raise ProviderError(f"anthropic -> {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        return data["content"][0]["text"]


async def _call_pollinations(system: str, user: str) -> str:
    if not config.POLLINATIONS_API_KEY:
        raise ProviderError("no POLLINATIONS_API_KEY")
    # Pollinations' unified API (gen.pollinations.ai) is OpenAI-compatible,
    # so this reuses the same helper as Groq/Cerebras/OpenRouter rather
    # than needing bespoke request/response handling.
    return await _call_openai_compatible(
        "https://gen.pollinations.ai/v1", config.POLLINATIONS_API_KEY, config.POLLINATIONS_MODEL, system, user
    )


_DEMO_OPENERS = [
    "Watching this unfold and I have thoughts.",
    "Can confirm what people are saying.",
    "This changes things.",
    "Statement incoming:",
    "Everyone needs to see this.",
    "I was there. Here's what actually happened.",
]


def _demo_fallback(system: str, user: str) -> str:
    """Offline, deterministic-ish generator so the app runs with zero API keys.
    Not an LLM — just enough signal to exercise the full pipeline and UI."""
    seed = abs(hash(user)) % len(_DEMO_OPENERS)
    opener = _DEMO_OPENERS[seed]
    tail = user.strip().splitlines()[-1][:140] if user.strip() else ""
    return f"{opener} {tail}".strip()


def _demo_fallback_tool_calls(user: str) -> list[dict]:
    """Deterministic-ish: roughly 1 in 4 offline generations 'attaches media',
    so the demo path still exercises the image pipeline end to end."""
    seed = abs(hash(user + "::media")) % 4
    if seed == 0:
        return [{"name": "attach_media", "arguments": {
            "kind": "photo", "caption": "a period-appropriate press photograph of the moment"
        }}]
    return []


# ---------------------------------------------------------------------------
# Tool-calling variants — used for post generation, where an agent may
# optionally call attach_media and/or reply_to. Each returns
# (content, tool_calls) where tool_calls is a normalized list of
# {"name": ..., "arguments": {...}}.
# ---------------------------------------------------------------------------

async def _call_openai_compatible_tools(base_url: str, api_key: str, model: str, system: str, user: str, tools: list[dict]) -> tuple[str, list[dict]]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.9,
        "max_tokens": 700,
    }
    if tools:
        payload["tools"] = [{"type": "function", "function": t} for t in tools]
        payload["tool_choice"] = "auto"
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
        )
        if resp.status_code != 200:
            raise ProviderError(f"{base_url} -> {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        message = data["choices"][0]["message"]
        content = message.get("content") or ""
        tool_calls = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {}
            tool_calls.append({"name": fn.get("name"), "arguments": args})
        return content, tool_calls


async def _call_groq_tools(system: str, user: str, tools: list[dict]) -> tuple[str, list[dict]]:
    if not config.GROQ_API_KEY:
        raise ProviderError("no GROQ_API_KEY")
    return await _call_openai_compatible_tools(
        "https://api.groq.com/openai/v1", config.GROQ_API_KEY, config.GROQ_MODEL, system, user, tools
    )


async def _call_cerebras_tools(system: str, user: str, tools: list[dict]) -> tuple[str, list[dict]]:
    if not config.CEREBRAS_API_KEY:
        raise ProviderError("no CEREBRAS_API_KEY")
    return await _call_openai_compatible_tools(
        "https://api.cerebras.ai/v1", config.CEREBRAS_API_KEY, config.CEREBRAS_MODEL, system, user, tools
    )


async def _call_openrouter_tools(system: str, user: str, tools: list[dict]) -> tuple[str, list[dict]]:
    if not config.OPENROUTER_API_KEY:
        raise ProviderError("no OPENROUTER_API_KEY")
    return await _call_openai_compatible_tools(
        "https://openrouter.ai/api/v1", config.OPENROUTER_API_KEY, config.OPENROUTER_MODEL, system, user, tools
    )


async def _call_gemini_tools(system: str, user: str, tools: list[dict]) -> tuple[str, list[dict]]:
    if not config.GEMINI_API_KEY:
        raise ProviderError("no GEMINI_API_KEY")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{config.GEMINI_MODEL}:generateContent?key={config.GEMINI_API_KEY}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"temperature": 0.9, "maxOutputTokens": 700},
    }
    if tools:
        payload["tools"] = [{"function_declarations": [
            {"name": t["name"], "description": t["description"], "parameters": _to_gemini_schema(t["parameters"])}
            for t in tools
        ]}]
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.post(url, json=payload)
        if resp.status_code != 200:
            raise ProviderError(f"gemini -> {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        parts = data["candidates"][0]["content"]["parts"]
        content = ""
        tool_calls = []
        for part in parts:
            if "text" in part:
                content += part["text"]
            fc = part.get("functionCall")
            if fc:
                tool_calls.append({"name": fc.get("name"), "arguments": fc.get("args", {})})
        return content, tool_calls


async def _call_anthropic_tools(system: str, user: str, tools: list[dict]) -> tuple[str, list[dict]]:
    if not config.ANTHROPIC_API_KEY:
        raise ProviderError("no ANTHROPIC_API_KEY")
    payload = {
        "model": config.ANTHROPIC_MODEL,
        "max_tokens": 700,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if tools:
        payload["tools"] = [
            {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools
        ]
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": config.ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
        )
        if resp.status_code != 200:
            raise ProviderError(f"anthropic -> {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        content = ""
        tool_calls = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls.append({"name": block.get("name"), "arguments": block.get("input", {})})
        return content, tool_calls


async def _call_pollinations_tools(system: str, user: str, tools: list[dict]) -> tuple[str, list[dict]]:
    if not config.POLLINATIONS_API_KEY:
        raise ProviderError("no POLLINATIONS_API_KEY")
    return await _call_openai_compatible_tools(
        "https://gen.pollinations.ai/v1", config.POLLINATIONS_API_KEY, config.POLLINATIONS_MODEL, system, user, tools
    )


PROVIDERS = {
    "groq": _call_groq,
    "cerebras": _call_cerebras,
    "openrouter": _call_openrouter,
    "gemini": _call_gemini,
    "anthropic": _call_anthropic,
    "pollinations": _call_pollinations,
}

TOOL_PROVIDERS = {
    "groq": _call_groq_tools,
    "cerebras": _call_cerebras_tools,
    "openrouter": _call_openrouter_tools,
    "gemini": _call_gemini_tools,
    "anthropic": _call_anthropic_tools,
    "pollinations": _call_pollinations_tools,
}


async def complete(system: str, user: str, chain: list[str]) -> tuple[str, str]:
    """Try providers in `chain` order. Returns (text, provider_used).
    Falls back to the offline demo generator if every real provider fails
    or none are configured — the app should never hard-fail on a 429."""
    last_err = None
    for name in chain:
        if name == "demo":
            continue
        fn = PROVIDERS.get(name)
        if fn is None:
            continue
        try:
            text = await fn(system, user)
            if text and text.strip():
                return text.strip(), name
        except Exception as e:
            log.warning("provider %r failed, trying next in chain: %s", name, e)
            last_err = e
            continue
    log.warning("all providers in chain %r failed — falling back to offline demo mode (last error: %s)", chain, last_err)
    return _demo_fallback(system, user), "demo"


async def complete_with_tools(system: str, user: str, chain: list[str], tools: list[dict]) -> tuple[str, list[dict], str]:
    """Same fallback-chain philosophy as complete(), but for calls where the
    model may optionally invoke a tool. Returns (content, tool_calls, provider).
    Falls back to a deterministic offline generator (which occasionally
    "attaches media" too) if every real provider fails."""
    for name in chain:
        if name == "demo":
            continue
        fn = TOOL_PROVIDERS.get(name)
        if fn is None:
            continue
        try:
            content, tool_calls = await fn(system, user, tools)
            if (content and content.strip()) or tool_calls:
                return (content or "").strip(), tool_calls, name
        except Exception as e:
            log.warning("provider %r failed (tool call), trying next in chain: %s", name, e)
            continue
    log.warning("all providers in chain %r failed (tool call) — falling back to offline demo mode", chain)
    return _demo_fallback(system, user), _demo_fallback_tool_calls(user), "demo"


def extract_json(text: str) -> dict | list:
    """Strip markdown code fences etc, then parse JSON. Providers are asked
    for raw JSON but models love wrapping it in ```json fences anyway."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()
    start = min([i for i in [cleaned.find("{"), cleaned.find("[")] if i != -1], default=-1)
    end = max(cleaned.rfind("}"), cleaned.rfind("]"))
    if start != -1 and end != -1:
        cleaned = cleaned[start:end + 1]
    return json.loads(cleaned)
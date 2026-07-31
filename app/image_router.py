"""
Image generation router — Gemini first, Hugging Face (FLUX.1-schnell, free
serverless inference) as fallback. Catches broadly and returns None on
total failure rather than raising — a missing image should degrade to a
text-only post, never crash the simulation.
"""
from __future__ import annotations
import base64
import httpx
from . import config

REQUEST_TIMEOUT = 40.0


class ImageProviderError(Exception):
    pass


async def _gemini_image(prompt: str) -> bytes:
    if not config.GEMINI_API_KEY:
        raise ImageProviderError("no GEMINI_API_KEY")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{config.GEMINI_IMAGE_MODEL}:generateContent?key={config.GEMINI_API_KEY}"
    )
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.post(
            url,
            json={
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"responseModalities": ["IMAGE"]},
            },
        )
        if resp.status_code != 200:
            raise ImageProviderError(f"gemini image -> {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        parts = data["candidates"][0]["content"]["parts"]
        for part in parts:
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                return base64.b64decode(inline["data"])
        raise ImageProviderError("gemini image -> no inline image data in response")


async def _huggingface_image(prompt: str) -> bytes:
    if not config.HF_API_KEY:
        raise ImageProviderError("no HF_API_KEY")
    url = f"https://api-inference.huggingface.co/models/{config.HF_IMAGE_MODEL}"
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.post(
            url,
            headers={"Authorization": f"Bearer {config.HF_API_KEY}"},
            json={"inputs": prompt},
        )
        if resp.status_code != 200:
            raise ImageProviderError(f"huggingface image -> {resp.status_code}: {resp.text[:200]}")
        content_type = resp.headers.get("content-type", "")
        if "image" not in content_type:
            raise ImageProviderError(f"huggingface image -> unexpected content-type {content_type}")
        return resp.content


IMAGE_PROVIDERS = {
    "gemini": _gemini_image,
    "huggingface": _huggingface_image,
}


async def generate_image(prompt: str, chain: list[str]) -> tuple[bytes, str] | None:
    """Try providers in order. Returns (image_bytes, provider_name) or None
    if every provider failed / wasn't configured — never raises."""
    for name in chain:
        fn = IMAGE_PROVIDERS.get(name)
        if fn is None:
            continue
        try:
            image_bytes = await fn(prompt)
            if image_bytes:
                return image_bytes, name
        except Exception:
            continue
    return None

"""
Auth — deliberately minimal: username/password (stdlib PBKDF2 hashing, no
bcrypt/passlib dependency) plus Google OAuth, both landing on the same thing
— a signed session cookie holding a user_id. No email verification, no
password reset flow, no account linking between the two paths. This is
enough to stop strangers from seeing each other's simulations, which was
the actual problem; it is not a production-grade identity system.
"""
from __future__ import annotations
import hashlib
import hmac
import os
import secrets
from urllib.parse import urlencode
import httpx
from fastapi import Request, HTTPException
from starlette.websockets import WebSocket

from . import config, storage

PBKDF2_ITERATIONS = 260_000


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), PBKDF2_ITERATIONS)
    return f"{salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt, digest_hex = stored.split("$", 1)
    except ValueError:
        return False
    check = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), PBKDF2_ITERATIONS)
    return hmac.compare_digest(check.hex(), digest_hex)


def current_user_id(request_or_ws) -> str | None:
    """Works for both a Request and a WebSocket — SessionMiddleware
    populates scope['session'] before routing regardless of connection
    type, so both expose the same .session attribute."""
    try:
        return request_or_ws.session.get("user_id")
    except Exception:
        return None


async def require_user(request: Request) -> dict:
    """FastAPI dependency for HTTP routes — 401s cleanly for API calls
    instead of trying to redirect, since these are called from fetch()."""
    user_id = current_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="not authenticated")
    user = await storage.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="not authenticated")
    return user


async def require_user_ws(websocket: WebSocket) -> dict | None:
    """Same idea for a WebSocket — caller closes the socket if this
    returns None, since you can't raise HTTPException on a WS connection."""
    user_id = current_user_id(websocket)
    if not user_id:
        return None
    return await storage.get_user_by_id(user_id)


# ---------------------------------------------------------------------------
# Google OAuth — plain Authorization Code flow over httpx, no SDK. Fixed,
# stable Google endpoints rather than doing OpenID discovery, to keep this
# genuinely simple.
# ---------------------------------------------------------------------------

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"


def build_redirect_uri(request: Request) -> str:
    if config.GOOGLE_REDIRECT_URI:
        return config.GOOGLE_REDIRECT_URI
    return str(request.url_for("google_callback"))


def google_authorize_url(request: Request, state: str) -> str:
    redirect_uri = build_redirect_uri(request)
    params = {
        "client_id": config.GOOGLE_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "prompt": "select_account",
    }
    return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"


async def google_exchange_code(request: Request, code: str) -> dict:
    redirect_uri = build_redirect_uri(request)
    async with httpx.AsyncClient(timeout=15) as client:
        token_resp = await client.post(GOOGLE_TOKEN_URL, data={
            "client_id": config.GOOGLE_CLIENT_ID,
            "client_secret": config.GOOGLE_CLIENT_SECRET,
            "code": code,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        })
        token_resp.raise_for_status()
        access_token = token_resp.json()["access_token"]

        userinfo_resp = await client.get(GOOGLE_USERINFO_URL, headers={"Authorization": f"Bearer {access_token}"})
        userinfo_resp.raise_for_status()
        return userinfo_resp.json()

"""
Ark — monolithic FastAPI app.

One process serves: the HTML/CSS/JS frontend, the REST API to create/list
simulations, a WebSocket that streams posts live, and now auth (session
cookie, username/password or Google) so simulations are private per user.
"""
from __future__ import annotations
import asyncio
import logging
import os
import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from starlette.middleware.sessions import SessionMiddleware

from . import storage, config, auth
from .graph import ARK_GRAPH, ArkState
from .models import SimulationCreateRequest, SignupRequest, LoginRequest, new_id

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ark")

if config.SESSION_SECRET_IS_EPHEMERAL:
    log.warning(
        "ARK_SESSION_SECRET is not set — using a random key generated at process "
        "start. Every restart invalidates all existing sessions. Set ARK_SESSION_SECRET "
        "explicitly in production so logins survive a redeploy."
    )
if not config.GOOGLE_AUTH_ENABLED:
    log.info("Google OAuth not configured (GOOGLE_CLIENT_ID/GOOGLE_CLIENT_SECRET unset) — only username/password login is available.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await storage.init()
    yield


# Created before mount time (not in the lifespan hook) because StaticFiles
# requires its directory to exist at import time. ARK_MEDIA_DIR can point
# anywhere writable — including a Railway Volume — since /media is its own
# mount, decoupled from /static.
os.makedirs(config.MEDIA_DIR, exist_ok=True)

app = FastAPI(title="Ark", lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=config.SESSION_SECRET, same_site="lax")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/media", StaticFiles(directory=config.MEDIA_DIR), name="media")
templates = Jinja2Templates(directory="app/templates")


# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    def __init__(self):
        self._conns: dict[str, list[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, sim_id: str, ws: WebSocket):
        await ws.accept()
        await self.register(sim_id, ws)

    async def register(self, sim_id: str, ws: WebSocket):
        """Like connect(), but for a socket that's already been accept()ed —
        used here because auth has to happen post-accept (see ws_endpoint)."""
        async with self._lock:
            self._conns.setdefault(sim_id, []).append(ws)

    async def disconnect(self, sim_id: str, ws: WebSocket):
        async with self._lock:
            if sim_id in self._conns and ws in self._conns[sim_id]:
                self._conns[sim_id].remove(ws)

    async def broadcast(self, sim_id: str, message: dict):
        dead = []
        for ws in self._conns.get(sim_id, []):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect(sim_id, ws)


manager = ConnectionManager()


# ---------------------------------------------------------------------------
# Background simulation runner
# ---------------------------------------------------------------------------

async def run_simulation(sim_id: str, prompt: str, title: str):
    initial_state: ArkState = {
        "sim_id": sim_id,
        "prompt": prompt,
        "title": title or "",
        "era_summary": "",
        "entities": [],
        "research_digest": "",
        "commentator_brief": "",
        "roster": [],
        "timeline": [],
        "cursor": 0,
        "posts": [],
    }
    run_config = {"configurable": {"thread_id": sim_id}}
    sent_roster = False
    sent_timeline = False
    last_post_count = 0
    try:
        await storage.set_status(sim_id, "planning")
        async for state in ARK_GRAPH.astream(initial_state, config=run_config, stream_mode="values"):
            if not sent_roster and state.get("roster"):
                sent_roster = True
                await manager.broadcast(sim_id, {
                    "type": "roster",
                    "title": state.get("title"),
                    "era_summary": state.get("era_summary"),
                    "roster": state["roster"],
                })
            if not sent_timeline and state.get("timeline"):
                sent_timeline = True
                await storage.set_status(sim_id, "streaming")
                await manager.broadcast(sim_id, {"type": "timeline", "timeline": state["timeline"]})
            posts = state.get("posts", [])
            if len(posts) > last_post_count:
                for post in posts[last_post_count:]:
                    await manager.broadcast(sim_id, {"type": "post", "post": post})
                last_post_count = len(posts)
        await storage.set_status(sim_id, "done")
        await manager.broadcast(sim_id, {"type": "done"})
    except Exception as e:  # noqa: BLE001
        log.exception("simulation %s failed", sim_id)
        await storage.set_status(sim_id, "error", error=str(e))
        await manager.broadcast(sim_id, {"type": "error", "message": str(e)})


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if auth.current_user_id(request):
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse(request, "login.html", {"google_enabled": config.GOOGLE_AUTH_ENABLED})


@app.post("/auth/signup")
async def signup(payload: SignupRequest, request: Request):
    existing = await storage.get_user_by_username(payload.username)
    if existing:
        raise HTTPException(status_code=400, detail="that username is already taken")
    password_hash = auth.hash_password(payload.password)
    user = await storage.create_user_local(payload.username, password_hash)
    if not user:
        raise HTTPException(status_code=400, detail="that username is already taken")
    request.session["user_id"] = user["id"]
    return {"ok": True}


@app.post("/auth/login")
async def login(payload: LoginRequest, request: Request):
    user = await storage.get_user_by_username(payload.username)
    if not user or not user.get("password_hash") or not auth.verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="incorrect username or password")
    request.session["user_id"] = user["id"]
    return {"ok": True}


@app.get("/auth/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=302)


@app.get("/auth/google/login")
async def google_login(request: Request):
    if not config.GOOGLE_AUTH_ENABLED:
        raise HTTPException(status_code=404, detail="Google sign-in is not configured")
    state = secrets.token_urlsafe(24)
    request.session["oauth_state"] = state
    return RedirectResponse(auth.google_authorize_url(request, state), status_code=302)


@app.get("/auth/google/callback")
async def google_callback(request: Request, code: str = "", state: str = "", error: str = ""):
    if error:
        return RedirectResponse("/login?error=google_denied", status_code=302)
    expected_state = request.session.pop("oauth_state", None)
    if not expected_state or state != expected_state:
        return RedirectResponse("/login?error=state_mismatch", status_code=302)
    try:
        userinfo = await auth.google_exchange_code(request, code)
    except Exception:
        log.exception("google oauth exchange failed")
        return RedirectResponse("/login?error=google_failed", status_code=302)
    google_id = userinfo.get("sub")
    email = userinfo.get("email", "")
    name = userinfo.get("name", email or "Google user")
    if not google_id:
        return RedirectResponse("/login?error=google_failed", status_code=302)
    user = await storage.get_or_create_google_user(google_id, email, name)
    request.session["user_id"] = user["id"]
    return RedirectResponse("/", status_code=302)


# ---------------------------------------------------------------------------
# App routes (all require login)
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    user_id = auth.current_user_id(request)
    if not user_id:
        return RedirectResponse("/login", status_code=302)
    user = await storage.get_user_by_id(user_id)
    if not user:
        request.session.clear()
        return RedirectResponse("/login", status_code=302)
    sims = await storage.list_simulations(user_id)
    display_name = user.get("display_name") or user.get("username") or user.get("email") or "there"
    return templates.TemplateResponse(request, "index.html", {"simulations": sims, "display_name": display_name})


@app.post("/api/simulations")
async def create_simulation(payload: SimulationCreateRequest, user: dict = Depends(auth.require_user)):
    sim_id = new_id("sim")
    title = payload.title or payload.prompt[:60]
    await storage.create_simulation(sim_id, user["id"], title, payload.prompt)
    asyncio.create_task(run_simulation(sim_id, payload.prompt, title))
    return {"id": sim_id, "title": title, "status": "planning"}


@app.get("/api/simulations")
async def list_simulations(user: dict = Depends(auth.require_user)):
    sims = await storage.list_simulations(user["id"])
    return [
        {
            "id": s["id"],
            "title": s["title"],
            "status": s["status"],
            "agent_count": s["agent_count"],
            "event_count": s["event_count"],
        }
        for s in sims
    ]


@app.get("/api/simulations/{sim_id}")
async def get_simulation(sim_id: str, user: dict = Depends(auth.require_user)):
    sim = await storage.get_simulation(sim_id)
    if not sim:
        raise HTTPException(status_code=404, detail="simulation not found")
    if sim.get("owner_id") != user["id"]:
        raise HTTPException(status_code=404, detail="simulation not found")
    return sim


@app.get("/api/simulations/{sim_id}/agents/{agent_id}")
async def get_agent_profile(sim_id: str, agent_id: str, user: dict = Depends(auth.require_user)):
    sim = await storage.get_simulation(sim_id)
    if not sim or sim.get("owner_id") != user["id"]:
        raise HTTPException(status_code=404, detail="simulation not found")
    agent = next((a for a in sim["agents"] if a["id"] == agent_id), None)
    if not agent:
        raise HTTPException(status_code=404, detail="agent not found")
    posts = [p for p in sim["posts"] if p["agent_id"] == agent_id]
    return {"agent": agent, "posts": posts}


@app.websocket("/ws/{sim_id}")
async def ws_endpoint(websocket: WebSocket, sim_id: str):
    # Accept first, then reject via an application-level message + close —
    # rejecting BEFORE accept() surfaces to browsers as a bare HTTP 403 at
    # the handshake level, which loses any custom close code/reason we'd
    # try to set, so the client can't distinguish "not logged in" from
    # "network error." This way the client's normal message handling
    # (already built for {"type": "error", ...}) covers this case too.
    await websocket.accept()
    user = await auth.require_user_ws(websocket)
    if not user:
        await websocket.send_json({"type": "error", "message": "not authenticated", "auth_error": True})
        await websocket.close()
        return
    sim = await storage.get_simulation(sim_id)
    if not sim or sim.get("owner_id") != user["id"]:
        await websocket.send_json({"type": "error", "message": "simulation not found"})
        await websocket.close()
        return

    await manager.register(sim_id, websocket)
    try:
        # Replay backlog so a client connecting mid-run (or reconnecting)
        # still sees everything generated so far, before subscribing live.
        if sim["agents"]:
            await websocket.send_json({
                "type": "roster", "title": sim["title"], "era_summary": "", "roster": sim["agents"],
            })
        if sim["events"]:
            await websocket.send_json({"type": "timeline", "timeline": sim["events"]})
        for post in sim["posts"]:
            await websocket.send_json({"type": "post", "post": post})
        if sim["status"] == "done":
            await websocket.send_json({"type": "done"})
        elif sim["status"] == "error":
            await websocket.send_json({"type": "error", "message": sim.get("error", "unknown error")})
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(sim_id, websocket)

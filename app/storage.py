"""
Storage layer — SQLite as the authoritative World Event Log, plus a users
table for auth. Every write goes through asyncio.to_thread so sqlite3's
blocking calls never stall the event loop.
"""
from __future__ import annotations
import asyncio
import json
import sqlite3
import time
from pathlib import Path
from . import config

DB_PATH = Path(config.DB_PATH).resolve()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE,
    password_hash TEXT,
    google_id TEXT UNIQUE,
    email TEXT,
    display_name TEXT,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS simulations (
    id TEXT PRIMARY KEY,
    owner_id TEXT,
    title TEXT NOT NULL,
    prompt TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'planning',
    error TEXT,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY,
    sim_id TEXT NOT NULL,
    data TEXT NOT NULL,
    FOREIGN KEY(sim_id) REFERENCES simulations(id)
);

CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    sim_id TEXT NOT NULL,
    order_index INTEGER NOT NULL,
    data TEXT NOT NULL,
    FOREIGN KEY(sim_id) REFERENCES simulations(id)
);

CREATE TABLE IF NOT EXISTS posts (
    id TEXT PRIMARY KEY,
    sim_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    created_order INTEGER NOT NULL,
    data TEXT NOT NULL,
    FOREIGN KEY(sim_id) REFERENCES simulations(id)
);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_sync():
    conn = _connect()
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()


async def init():
    await asyncio.to_thread(_init_sync)


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def _new_user_id() -> str:
    import uuid
    return f"user_{uuid.uuid4().hex[:10]}"


def _create_user_local_sync(username: str, password_hash: str) -> dict | None:
    conn = _connect()
    try:
        existing = conn.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
        if existing:
            return None
        user_id = _new_user_id()
        conn.execute(
            "INSERT INTO users (id, username, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (user_id, username, password_hash, time.time()),
        )
        conn.commit()
        return {"id": user_id, "username": username}
    finally:
        conn.close()


async def create_user_local(username: str, password_hash: str) -> dict | None:
    return await asyncio.to_thread(_create_user_local_sync, username, password_hash)


def _get_user_by_username_sync(username: str) -> dict | None:
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


async def get_user_by_username(username: str) -> dict | None:
    return await asyncio.to_thread(_get_user_by_username_sync, username)


def _get_user_by_id_sync(user_id: str) -> dict | None:
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


async def get_user_by_id(user_id: str) -> dict | None:
    return await asyncio.to_thread(_get_user_by_id_sync, user_id)


def _get_or_create_google_user_sync(google_id: str, email: str, display_name: str) -> dict:
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM users WHERE google_id=?", (google_id,)).fetchone()
        if row:
            return dict(row)
        user_id = _new_user_id()
        conn.execute(
            "INSERT INTO users (id, google_id, email, display_name, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, google_id, email, display_name, time.time()),
        )
        conn.commit()
        return {"id": user_id, "google_id": google_id, "email": email, "display_name": display_name}
    finally:
        conn.close()


async def get_or_create_google_user(google_id: str, email: str, display_name: str) -> dict:
    return await asyncio.to_thread(_get_or_create_google_user_sync, google_id, email, display_name)


# ---------------------------------------------------------------------------
# Simulations (owner-scoped) / agents / events / posts
# ---------------------------------------------------------------------------

def _create_simulation_sync(sim_id: str, owner_id: str, title: str, prompt: str):
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO simulations (id, owner_id, title, prompt, status, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (sim_id, owner_id, title, prompt, "planning", time.time()),
        )
        conn.commit()
    finally:
        conn.close()


async def create_simulation(sim_id: str, owner_id: str, title: str, prompt: str):
    await asyncio.to_thread(_create_simulation_sync, sim_id, owner_id, title, prompt)


def _set_status_sync(sim_id: str, status: str, error: str | None = None):
    conn = _connect()
    try:
        conn.execute("UPDATE simulations SET status=?, error=? WHERE id=?", (status, error, sim_id))
        conn.commit()
    finally:
        conn.close()


async def set_status(sim_id: str, status: str, error: str | None = None):
    await asyncio.to_thread(_set_status_sync, sim_id, status, error)


def _save_agents_sync(sim_id: str, agents: list[dict]):
    conn = _connect()
    try:
        conn.executemany(
            "INSERT OR REPLACE INTO agents (id, sim_id, data) VALUES (?, ?, ?)",
            [(a["id"], sim_id, json.dumps(a)) for a in agents],
        )
        conn.commit()
    finally:
        conn.close()


async def save_agents(sim_id: str, agents: list[dict]):
    if agents:
        await asyncio.to_thread(_save_agents_sync, sim_id, agents)


def _save_events_sync(sim_id: str, events: list[dict]):
    conn = _connect()
    try:
        conn.executemany(
            "INSERT OR REPLACE INTO events (id, sim_id, order_index, data) VALUES (?, ?, ?, ?)",
            [(e["id"], sim_id, e["order"], json.dumps(e)) for e in events],
        )
        conn.commit()
    finally:
        conn.close()


async def save_events(sim_id: str, events: list[dict]):
    if events:
        await asyncio.to_thread(_save_events_sync, sim_id, events)


def _append_post_sync(sim_id: str, post: dict):
    conn = _connect()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO posts (id, sim_id, event_id, created_order, data) VALUES (?, ?, ?, ?, ?)",
            (post["id"], sim_id, post["event_id"], post["created_order"], json.dumps(post)),
        )
        conn.commit()
    finally:
        conn.close()


async def append_post(sim_id: str, post: dict):
    await asyncio.to_thread(_append_post_sync, sim_id, post)


def _get_simulation_sync(sim_id: str) -> dict | None:
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM simulations WHERE id=?", (sim_id,)).fetchone()
        if not row:
            return None
        sim = dict(row)
        agents = [json.loads(r["data"]) for r in conn.execute("SELECT data FROM agents WHERE sim_id=?", (sim_id,))]
        events = [
            json.loads(r["data"])
            for r in conn.execute("SELECT data FROM events WHERE sim_id=? ORDER BY order_index", (sim_id,))
        ]
        posts = [
            json.loads(r["data"])
            for r in conn.execute("SELECT data FROM posts WHERE sim_id=? ORDER BY created_order", (sim_id,))
        ]
        sim["agents"] = agents
        sim["events"] = events
        sim["posts"] = posts
        return sim
    finally:
        conn.close()


async def get_simulation(sim_id: str) -> dict | None:
    return await asyncio.to_thread(_get_simulation_sync, sim_id)


def _list_simulations_sync(owner_id: str) -> list[dict]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM simulations WHERE owner_id=? ORDER BY created_at DESC", (owner_id,)
        ).fetchall()
        out = []
        for row in rows:
            sim = dict(row)
            sim["agent_count"] = conn.execute(
                "SELECT COUNT(*) c FROM agents WHERE sim_id=?", (sim["id"],)
            ).fetchone()["c"]
            sim["event_count"] = conn.execute(
                "SELECT COUNT(*) c FROM events WHERE sim_id=?", (sim["id"],)
            ).fetchone()["c"]
            out.append(sim)
        return out
    finally:
        conn.close()


async def list_simulations(owner_id: str) -> list[dict]:
    return await asyncio.to_thread(_list_simulations_sync, owner_id)

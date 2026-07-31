# Ark — prototype

A monolithic FastAPI app that turns a historical (or fictional) moment into a
live, Twitter-style social feed: an AI agent per person/institution involved,
posting in character as events unfold — with backstories, biases toward
each other, replies, and images they decide to attach themselves.

Runs with **zero LLM API keys** — no provider configured falls back to an
offline demo generator so you can see the whole pipeline work end to end
before wiring up a real model. Auth, however, is not optional — every
simulation is private to the account that created it.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in provider keys + a session secret
uvicorn app.main:app --reload
```

Open `http://localhost:8000`, create an account, describe a moment in
history, and hit **Launch simulation**. The feed populates live over a
WebSocket.

## Auth

Username/password (stdlib PBKDF2 hashing, no bcrypt/passlib dependency) plus
optional Google sign-in, both landing on the same thing: a signed session
cookie. Every simulation is scoped to the account that created it — the
REST API and the WebSocket both check ownership and return a 404 (not a
403) for anyone else's simulation, so a wrong guess can't even confirm a
simulation exists.

**Before running anywhere real**, set `ARK_SESSION_SECRET` in `.env`:
```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```
Without it, a random secret is generated at process start — sessions then
invalidate on every restart, which is fine for poking around locally but
will silently log everyone out on every deploy in production.

**Google sign-in** is optional. To enable it:
1. [Google Cloud Console](https://console.cloud.google.com/) → APIs & Services → Credentials
2. Create an OAuth 2.0 Client ID, type "Web application"
3. Add an authorized redirect URI: `http://localhost:8000/auth/google/callback` for local dev, or `https://yourdomain.com/auth/google/callback` in production
4. Put the client ID/secret in `.env` as `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET`

This is deliberately minimal — no email verification, no password reset,
no account linking between the two auth paths. It solves "strangers seeing
each other's simulations," not identity management in general.

## Deploying to Railway

Builds cleanly on Nixpacks (pure-Python deps) and WebSockets work through
Railway's proxy. A few things need explicit setup:

1. **Push the contents of this folder as the repo root** — `app/`,
   `requirements.txt`, `Procfile` need to sit at the top level of what
   Railway builds (fix via Railway's **Root Directory** setting if needed).
2. **`Procfile`** is included — Railway's auto-detection doesn't know to run
   `uvicorn app.main:app` otherwise, and the port must come from `$PORT`.
3. **Set env vars in Railway's dashboard** (Variables tab), not a committed
   `.env` — same keys as `.env.example`, **including `ARK_SESSION_SECRET`**.
4. **Persistence**: Railway's filesystem is ephemeral without an attached
   [Volume](https://docs.railway.com/reference/volumes) — a redeploy wipes
   `ark.db` (so all users and simulations) and generated images. Attach a
   Volume mounted at e.g. `/data`, then set `ARK_DB_PATH=/data/ark.db` and
   `ARK_MEDIA_DIR=/data/generated`. Generated images are served through a
   dedicated `/media` route decoupled from `/static`, specifically so this
   can point anywhere writable.
5. **Single instance only, as currently built** — the WebSocket registry is
   in-memory per process. Fine at 1 Railway replica; would need Redis
   pub/sub to scale beyond that.

## Architecture, in this codebase

| Concept | File |
|---|---|
| LangGraph StateGraph (intake → research → roster → commentators → timeline → scheduler loop) | `app/graph.py` |
| World Event Log (authoritative, SQLite-backed) + users table | `app/storage.py` |
| Auth: password hashing, sessions, Google OAuth | `app/auth.py` |
| Pluggable multi-provider LLM router with fallback chain + tool calling | `app/llm_router.py` |
| Image generation router (Gemini → Hugging Face FLUX) | `app/image_router.py` |
| Search grounding — DuckDuckGo + Wikipedia | `app/research.py` |
| FastAPI routes + WebSocket streaming + background task runner | `app/main.py` |
| Agent / Event / Post / User schemas | `app/models.py` |
| Provider keys, model names, fallback order, auth config | `app/config.py` |

**The graph** (`app/graph.py`) has eight nodes:

```
intake -> research -> roster -> commentators -> timeline -> [route] -> generate_sequential --+
                                                                ^        generate_parallel ---+-> advance_cursor -> [route] -> ... -> END
```

- `intake`: one LLM call turns your prompt into a scene summary + a list of
  entities (people, orgs, press — anything that can "post about itself").
- `research`: DuckDuckGo + Wikipedia search, gathering two digests —
  general background, and candidates for who actually covered/analyzed
  events like this. Best-effort; empty digests on failure, pipeline
  continues regardless.
- `roster`: one LLM call turns each entity into a full persona — voice,
  goals, **backstory** (2-4 sentences of fixed personal history — this is
  Ark's "memory," in the sense of *who they are*, not a growing log of
  what's happened; it never changes once generated), and up to 3
  **relationships** toward other named agents in the same roster (each
  tagged with 1-2 short words: "friend", "hostile", "rival", "mentor"...).
  Relationship targets are resolved by name against the completed roster
  after generation — a hallucinated name that doesn't match anyone real in
  the cast is dropped rather than left dangling.
- `commentators`: a **separate** LLM call for 3-4 recurring agents who are
  never key players — they narrate events, they don't cause them. Grounded
  in the research digest so they're real journalists/analysts/outlets from
  the period where possible, with an honest `grounded: false` fallback when
  search comes up empty. Also get a backstory.
- `timeline`: one LLM call produces an ordered list of events, each tagged
  `sequential` or `parallel`, with participants — commentators are excluded
  here; they're injected separately. Each event also gets `hours_since_start`
  (a plain float, not a parsed calendar date — this deliberately works for
  ancient/fictional settings too), which drives temporal pacing.
- `route` (a conditional edge — this is the Scheduler): sends execution to
  sequential (agents post one after another, each seeing the ones before it
  **within the same event**) or parallel (`asyncio.gather` fan-out —
  genuinely concurrent, so participants deliberately do NOT see each
  other's posts from the same moment, the way real simultaneous reactions
  wouldn't). One commentator (round-robin) posts on every event either way.
- Every post is written to SQLite the instant it's generated — that table
  is the **World Event Log**, the source of truth other agents read from.

### Images as a tool call

Media attachment is a real per-agent decision, not a keyword match on the
event text. Every post-generation call offers an `attach_media` tool — the
agent calls it only if it's natural for *that specific role* (a press wire —
plausible; a head of state's personal account — usually not), supplying its
own caption, which becomes the actual image-generation prompt.

Implemented natively per provider (`complete_with_tools` in
`llm_router.py`): OpenAI-style `tools`/`tool_calls` for Groq/Cerebras/
OpenRouter, Gemini's `functionDeclarations` (with a schema case-conversion
helper — Gemini wants `STRING`/`OBJECT`, everyone else wants lowercase),
Anthropic's `tools`/`tool_use`. If a provider returns a tool call with no
accompanying text, a follow-up plain call gets the actual post text.

### Replies

A second tool, `reply_to`, lets an agent mark its post as a direct response
to one specific earlier post — most posts aren't replies, and the system
prompt says so explicitly. The reply target is validated server-side
against real posts in the same event (never trust a model-supplied id
blindly); an invalid/hallucinated id is silently dropped rather than
rendered as a dangling reference.

This only works because of how context is passed: `_generate_one_post`
takes an explicit `posts_so_far` list rather than reading from shared state,
so the **sequential** scheduler can accumulate it turn-by-turn (agent 2 in
an event genuinely sees agent 1's post from that same event), while the
**parallel** scheduler deliberately passes the pre-event snapshot to every
concurrent worker (simultaneous posts shouldn't see each other — that's
correct, not a bug).

### Temporal pacing — compressed but still temporal

Each event's `hours_since_start` becomes two things in `timeline_node`:

- **`gap_seconds`** — a real `asyncio.sleep` before that event's posts
  generate. Log-curve compressed, bounded by `PACING_MIN_DELAY_SECONDS` /
  `PACING_MAX_DELAY_SECONDS` (default 0.4s–12s) — a same-day gap barely
  pauses, while month- and year-scale gaps compress toward the same short,
  bounded pause rather than one taking 10x as long to watch.
- **`gap_label`** — a human divider ("5 hours later", "3 months later",
  "1.1 years later"), shown once in the feed and as a chip on the timeline
  panel entry. `None` for negligible gaps (<45 min).

### Timeline navigation + follow

The right-panel timeline is clickable — jumps to an event's first post if
rendered, or gives a brief visual acknowledgment if it hasn't happened yet.

Follow is per-simulation, in `localStorage` (a real deployed app, not a
Claude Artifact — persistence across reloads is the right call). Any
profile has a Follow toggle; an All/Following pill pair above the feed
filters posts client-side by `data-agent-id`.

## UI features

- **Design direction — "Notion x Twitter"**: Twitter's feed grammar (avatars,
  handles, timestamps, reply threading, engagement row) rendered inside
  Notion's visual language — soft off-white canvas, white cards with
  hairline borders and a gentle shadow, a black rounded-square app-icon
  mark, restrained color. Amber is Ark's one signature accent (ticker,
  in-world dates, timeline chips) — used sparingly, not as the whole
  palette, the way Notion uses colorful icons against a mostly-neutral page.
- **Avatars**: [DiceBear](https://www.dicebear.com/) (`notionists` style,
  matching the visual direction), generated from each agent's handle —
  no API key needed, it's a free public endpoint. Layered over an instant
  color+initials fallback so something always renders immediately, and if
  the CDN is ever unreachable the `<img>` removes itself on error, leaving
  the fallback circle rather than a broken-image icon.
- **Mobile navigation**: a real bottom tab bar (Feed / Cast / **+** New /
  Library / Account) instead of a hamburger-and-drawer pattern — each tab
  is its own full page below 980px; desktop shows all of them at once as
  the familiar 3-column layout. The center "+" is a raised FAB, matching
  the mobile-app convention for a primary create action.
- **Immersive enter/exit**: a brief full-screen "Entering [title]…" portal
  overlay on launching/opening a simulation (12s safety timeout so it can
  never trap you), with an always-visible **← Today** exit button.
- **Agent profiles**: click any avatar/name to open a profile — persona
  fields, backstory, relationships (clickable to jump to that agent), a
  Follow toggle, and a feed of just their posts.
- **Time-skip dividers + timeline navigation** as described above.
- **PWA**: manifest + service worker caching the app shell only — API/WS/
  auth traffic always hits the network live. Installable to a home screen.

## Verifying changes without spending API credits

```bash
python3 tests/test_pipeline_mock.py
```
Monkeypatches the LLM router with canned, realistic responses and runs the
real graph against them — covers roster/backstory/relationship resolution,
commentator cross-posting, parallel vs sequential visibility, temporal
pacing tier classification, and tool-call-driven media isolation.

## Known prototype limitations

- **Image tool-calling depends on the model actually using tools well.**
  Weaker models may over- or under-attach media despite the system prompt's
  guidance — this is a real LLM judgment call now, not a deterministic rule.
- **Temporal pacing is compressed, not literal** — `hours_since_start` is
  invented by the model during planning, not a verified fact. Treat pacing
  as a feel, not a citation.
- **Follow state has no backend** — `localStorage`, per-browser, doesn't
  sync across devices.
- **Avatars depend on a third-party CDN** (api.dicebear.com) at page-load
  time — no key required and it fails gracefully (falls back to a colored
  initials circle), but it is one more external dependency in the render
  path. Self-hosting DiceBear is possible if you want to remove it entirely.
- **Search grounding is best-effort** — DuckDuckGo/Wikipedia can 403 or
  rate-limit depending on network; the pipeline degrades gracefully
  (commentators fall back to the LLM's own knowledge, honestly marked
  `grounded: false`) rather than failing.
- **Auth is intentionally minimal** — no email verification, no password
  reset, no account linking between username/password and Google. Good
  enough to stop strangers seeing each other's data; not an identity system.
- **Fan-out width and event count are capped** (`MAX_AGENTS_PER_SIM`,
  `MAX_EVENTS_PER_SIM`) to keep cost/latency predictable.
- **Single-process, in-memory WebSocket registry** — fine for one Railway
  replica; needs Redis pub/sub to scale beyond that.
- In demo mode (no LLM keys), the offline generator produces obviously
  placeholder text and typically a single-event timeline (JSON parsing
  fails on non-JSON demo output) — expected, proves the pipeline works
  end-to-end, not meant to be a good writer.

## Project layout

```
app/
  main.py          FastAPI app, auth routes, API routes, WebSocket, background runner
  auth.py          Password hashing, session helpers, Google OAuth
  graph.py         LangGraph StateGraph — the eight nodes above
  llm_router.py    Multi-provider completion + tool calling + fallback + offline demo mode
  image_router.py  Gemini -> Hugging Face image generation fallback
  research.py      DuckDuckGo + Wikipedia search grounding
  storage.py       SQLite World Event Log + users table
  models.py        Agent / TimelineEvent / Post / User schemas
  config.py        Env-driven settings, provider chains, auth config
  templates/index.html, login.html
  static/style.css, app.js
  static/manifest.json, sw.js, icons/   PWA assets
  static/generated/                     generated post images (created at runtime)
tests/
  test_pipeline_mock.py
requirements.txt
.env.example
Procfile
.python-version
```
"# ARK" 

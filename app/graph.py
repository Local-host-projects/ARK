"""
Ark's core orchestration graph, built with LangGraph.

Graph shape:

    intake -> research -> roster -> commentators -> timeline -> [route] -> generate_sequential --+
                                                                    ^        generate_parallel ---+-> advance_cursor -> [route] -> ... -> END

`route` is a conditional edge: it inspects state["cursor"] against the
timeline and sends execution to the sequential generator, the parallel
(fan-out) generator, or END — this is the "Scheduler" from the architecture
doc, a supervisor deciding what happens next, not agents calling each other.
"""
from __future__ import annotations
import asyncio
import json
import logging
import math
import os
import random
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from . import llm_router, storage, config, research, image_router
from .models import Agent, TimelineEvent, Post, Relationship, new_id

log = logging.getLogger("ark.graph")


class ArkState(TypedDict):
    sim_id: str
    prompt: str
    title: str
    era_summary: str
    entities: list[dict]
    research_digest: str
    commentator_brief: str
    roster: list[dict]
    timeline: list[dict]
    cursor: int
    posts: list[dict]


# ---------------------------------------------------------------------------
# Node 1: intake — parse the source into a scene-setting summary + entity list
# ---------------------------------------------------------------------------

INTAKE_SYSTEM = """You are the intake analyst for Ark, a history-through-social-media simulator.
Given a user's topic/prompt/source text, identify the setting and the entities who would post
about it — people, organizations, and press/media outlets, anything that "disseminates
information about itself." Respond with ONLY raw JSON, no markdown fences, no prose. Schema:
{
  "title": "short punchy title for this simulation",
  "era_summary": "2-3 sentences setting the scene: time, place, stakes",
  "entities": [
    {"name": "...", "role": "...", "kind": "person|org|press"}
  ]
}
Include 6 to 12 entities. Prefer named, verifiable figures/institutions."""


async def intake_node(state: ArkState) -> dict:
    text, provider = await llm_router.complete(
        INTAKE_SYSTEM, state["prompt"], config.PLANNING_PROVIDER_CHAIN
    )
    try:
        data = llm_router.extract_json(text)
    except Exception as e:
        log.warning("intake_node: failed to parse JSON from provider %r (%s) — using minimal fallback entities", provider, e)
        data = {
            "title": state["prompt"][:60],
            "era_summary": state["prompt"][:300],
            "entities": [
                {"name": "Wire Report", "role": "News Wire", "kind": "press"},
                {"name": "Eyewitness", "role": "Public Commentator", "kind": "person"},
            ],
        }
    entities = data.get("entities", [])[: config.MAX_AGENTS_PER_SIM]
    return {
        "title": data.get("title") or state["prompt"][:60],
        "era_summary": data.get("era_summary", ""),
        "entities": entities,
    }


# ---------------------------------------------------------------------------
# Node 1.5: research — ground the scene + candidate commentators in real
# search results (DuckDuckGo + Wikipedia) before anyone's persona is written
# ---------------------------------------------------------------------------

async def research_node(state: ArkState) -> dict:
    if not config.ENABLE_RESEARCH:
        return {"research_digest": "", "commentator_brief": ""}
    topic = state.get("title") or state["prompt"][:80]
    general_task = research.research_topic(topic)
    commentator_task = research.research_topic(f"journalists correspondents analysts who covered {topic}")
    general_digest, commentator_brief = await asyncio.gather(general_task, commentator_task)
    return {"research_digest": general_digest, "commentator_brief": commentator_brief}


# ---------------------------------------------------------------------------
# Node 2: roster — build a full persona per entity, including a static
# backstory ("memory" that never changes — who they are, not a log) and up
# to 3 explicit relationships toward other named agents in the same roster.
# ---------------------------------------------------------------------------

ROSTER_SYSTEM = """You build character personas for Ark, a history-through-social-media simulator.
Given a scene summary, a list of entities, and (if present) background research from web/Wikipedia
search, write a posting persona for each entity. Use the background research to keep facts accurate
where it's relevant. Respond with ONLY raw JSON, no markdown fences: a list of objects matching:
{
  "name": "...",
  "handle": "@lowercase_no_spaces",
  "role": "... (their title/function)",
  "kind": "person|org|press",
  "personality": "one vivid sentence on voice/tone/temperament",
  "goals": "one sentence: what they want, what they're protecting or pursuing",
  "era_context": "one sentence: what they know/believe at this point in the story",
  "backstory": "2-4 sentences of personal history/background — where they came from, formative
      experiences, why they think the way they do. This is fixed at creation and never changes;
      it's who they ARE, not a log of what happens during the simulation.",
  "relationships": [
      {"name": "exact name of another entity in this same list", "tags": ["friend"]}
  ] — up to 3 entries max, only toward OTHER entities that also appear in this entity list. Each
      tags list is 1-2 short words like "friend", "rival", "hostile", "partner", "mentor",
      "distrusts", "respects". Omit entirely (empty list) for entities with no strong personal
      opinion of anyone else — not everyone needs 3.
}
Keep handles short and distinct. Return exactly one object per entity, same order."""


async def roster_node(state: ArkState) -> dict:
    user = json.dumps({
        "era_summary": state["era_summary"],
        "entities": state["entities"],
        "background_research": state.get("research_digest", "")[:2000],
    })
    text, provider = await llm_router.complete(ROSTER_SYSTEM, user, config.PLANNING_PROVIDER_CHAIN)
    try:
        raw_roster = llm_router.extract_json(text)
        if isinstance(raw_roster, dict):
            raw_roster = raw_roster.get("roster") or raw_roster.get("agents") or []
    except Exception as e:
        log.warning("roster_node: failed to parse JSON from provider %r (%s) — using bland fallback personas", provider, e)
        raw_roster = [
            {
                "name": ent.get("name", "Unknown"),
                "handle": "@" + ent.get("name", "unknown").lower().replace(" ", "_")[:15],
                "role": ent.get("role", ""),
                "kind": ent.get("kind", "person"),
                "personality": "Speaks plainly about events as they unfold.",
                "goals": "Wants the truth of the moment recorded.",
                "era_context": state["era_summary"][:200],
                "backstory": "",
                "relationships": [],
            }
            for ent in state["entities"]
        ]

    # Pass 1: build every agent, keeping relationship target names
    # unresolved for now — a target might appear later in the list, so
    # name->id resolution has to happen after everyone exists.
    agents = []
    pending_relationships = []
    for item in raw_roster:
        try:
            agent = Agent(
                name=item.get("name", "Unknown"),
                handle=item.get("handle") or ("@" + item.get("name", "unknown").lower().replace(" ", "_")[:15]),
                role=item.get("role", ""),
                kind=item.get("kind", "person") if item.get("kind") in ("person", "org", "press") else "person",
                personality=item.get("personality", ""),
                goals=item.get("goals", ""),
                era_context=item.get("era_context", ""),
                backstory=item.get("backstory", ""),
            )
            agents.append(agent)
            raw_rels = item.get("relationships", [])
            if isinstance(raw_rels, list) and raw_rels:
                pending_relationships.append((agent.id, raw_rels[:3]))
        except Exception:
            continue

    # Pass 2: resolve relationship target names against the now-complete
    # roster. Unresolved targets (typos, hallucinated names) are dropped.
    name_to_agent = {a.name.lower(): a for a in agents}
    by_id = {a.id: a for a in agents}
    for agent_id, raw_rels in pending_relationships:
        resolved = []
        for rel in raw_rels:
            target_name = str(rel.get("name", "")).strip()
            target = name_to_agent.get(target_name.lower())
            if not target or target.id == agent_id:
                continue
            tags = [str(t)[:20] for t in rel.get("tags", [])][:2]
            if not tags:
                continue
            resolved.append(Relationship(target_name=target.name, target_id=target.id, tags=tags))
        by_id[agent_id].relationships = resolved[:3]

    agents_dumped = [a.model_dump() for a in agents]
    await storage.save_agents(state["sim_id"], agents_dumped)
    return {"roster": agents_dumped}


# ---------------------------------------------------------------------------
# Node 2.5: commentators — 3-4 recurring agents who narrate across the whole
# timeline without being key players. Grounded in real search results where
# possible (real journalists/analysts from the period), not generic
# placeholder labels.
# ---------------------------------------------------------------------------

COMMENTATOR_SYSTEM = """You identify recurring commentary voices for Ark, a history-through-social-media
simulator. These agents are NOT key players and never cause events — they are observers whose job is
to stay informed and narrate what's happening: think a specific named war correspondent, a specific
named financial analyst, a specific wire-service desk. They post across most events in the timeline,
reacting and interpreting, never driving the plot.

You are given real search results (web + Wikipedia) about who actually covered or analyzed this kind
of moment. Ground each persona in a REAL, SPECIFIC person or outlet named in those results where you
can — use their real name, real outlet, and real title from the sources. Do not invent a fake named
individual and present them as real. If the sources don't give you a confident specific real name for
a slot, use a real outlet/institution from the era as the poster instead of a person, or a
generic-but-honest role tied to a real institution — never a bare template label like "The Journalist"
with no specific grounding.

Respond with ONLY raw JSON, no markdown fences: a list of {count} objects matching:
{{
  "name": "real person or real outlet name",
  "handle": "@lowercase_no_spaces",
  "role": "their specific real title/function — not a generic label",
  "kind": "person|org|press",
  "personality": "one vivid sentence on voice/tone/temperament",
  "goals": "one sentence — what they're trying to inform people of, not to achieve in the events",
  "era_context": "one sentence: their real vantage point on this story",
  "backstory": "2-4 sentences of real (or, if ungrounded, plausible) personal/professional
      history — how they came to cover stories like this, what shaped their voice. Fixed at
      creation, never changes.",
  "grounded": true or false — true only if this maps to a specific real name/outlet in the sources
}}"""


async def commentator_node(state: ArkState) -> dict:
    if config.NUM_COMMENTATORS <= 0:
        return {}
    brief = state.get("commentator_brief", "")
    user = json.dumps({
        "era_summary": state["era_summary"],
        "search_results": brief[:3000] if brief else "(no search results available — use your own knowledge, but stay conservative about naming a specific real individual you're not confident about)",
    })
    system = COMMENTATOR_SYSTEM.format(count=config.NUM_COMMENTATORS)
    text, provider = await llm_router.complete(system, user, config.PLANNING_PROVIDER_CHAIN)
    try:
        raw = llm_router.extract_json(text)
        if isinstance(raw, dict):
            raw = raw.get("commentators") or raw.get("agents") or []
    except Exception as e:
        log.warning("commentator_node: failed to parse JSON from provider %r (%s) — using generic fallback commentators", provider, e)
        raw = [
            {"name": "Wire Desk", "handle": "@wire_desk", "role": "News Wire", "kind": "press",
             "personality": "Terse, fact-first.", "goals": "Get confirmed details out fast.",
             "era_context": state["era_summary"][:150],
             "backstory": "A wire service desk with no single named voice — reports what's confirmed, fast, without embellishment.",
             "grounded": False},
            {"name": "Field Correspondent", "handle": "@field_corr", "role": "Correspondent", "kind": "person",
             "personality": "On the ground, first-person observations.", "goals": "Document what's actually happening.",
             "era_context": state["era_summary"][:150],
             "backstory": "A journalist who built a career getting close to the story, trusting what they can see over what they're told.",
             "grounded": False},
        ][: config.NUM_COMMENTATORS]

    existing_handles = {a["handle"] for a in state["roster"]}
    commentators = []
    for item in raw[: config.NUM_COMMENTATORS]:
        handle = item.get("handle") or ("@" + item.get("name", "correspondent").lower().replace(" ", "_")[:15])
        if handle in existing_handles:
            handle = handle + "_desk"
        try:
            agent = Agent(
                name=item.get("name", "Wire Desk"),
                handle=handle,
                role=item.get("role", "Correspondent"),
                kind=item.get("kind", "press") if item.get("kind") in ("person", "org", "press") else "press",
                narrative_role="commentator",
                personality=item.get("personality", "Observant, informs rather than participates."),
                goals=item.get("goals", "Keep the public informed of what's actually happening."),
                era_context=item.get("era_context", ""),
                backstory=item.get("backstory", ""),
                grounded=bool(item.get("grounded", False)),
            )
            commentators.append(agent.model_dump())
            existing_handles.add(handle)
        except Exception:
            continue

    full_roster = state["roster"] + commentators
    await storage.save_agents(state["sim_id"], commentators)
    return {"roster": full_roster}


# ---------------------------------------------------------------------------
# Node 3: timeline — coarse event skeleton, sequential vs parallel, plus a
# numeric hours_since_start per event that drives temporal pacing.
# ---------------------------------------------------------------------------

TIMELINE_SYSTEM = """You build the event skeleton for Ark, a history-through-social-media simulator.
Given a scene summary and a roster of agents, produce an ordered list of events that will each
become a burst of social-media posts. Some events are "sequential" (one thing causes the next,
one or two agents post about it), others are "parallel" (many agents react to the same moment
at once — a genuine news event everyone comments on). Respond with ONLY raw JSON, no markdown
fences: a list of objects matching:
{
  "title": "short event title",
  "description": "1-2 sentences: what happens in this event, for agents to react to",
  "mode": "sequential|parallel",
  "sim_date": "an in-world date/time label, e.g. '7 Dec 1941, 08:10'",
  "hours_since_start": number — hours elapsed since the FIRST event (the first event is 0).
      Must never decrease across the list. Be HONEST about real pacing: minutes or hours apart
      for a fast-moving sequence, but weeks/months/years apart if the true story spans that long.
      This number drives real playback timing, so scale it to the ACTUAL time elapsed, not just
      narrative order — a story spanning years should have large jumps in this number.
  "participants": ["agent name", "agent name"]
}
Order matters — earlier events should be causally/temporally prior. Use participant names
exactly as given in the roster. Produce between 6 and 14 events. Mix sequential and parallel;
at least 2 parallel events where multiple agents react to the same trigger."""


# ---------------------------------------------------------------------------
# Temporal pacing — "compressed but still temporal": a year-scale gap and a
# ten-year-scale gap both land near the same (bounded) real-time pause
# rather than one taking literally 10x as long, since this is a live UI.
# The log curve still preserves *relative* feel — same-day events barely
# pause, month-scale gaps pause noticeably more, year-scale gaps hit the
# ceiling — which is what "years compress to weeks, months compress to
# within a week" means in practice: every tier above ~a month compresses
# into the same short, bounded pause, distinguished mainly by the divider
# label shown in the feed rather than by literally waiting longer.
# ---------------------------------------------------------------------------

def _delay_seconds(gap_hours: float) -> float:
    if gap_hours <= 0:
        return 0.0
    delay = 0.6 + math.log1p(gap_hours) * 1.15
    return min(max(delay, config.PACING_MIN_DELAY_SECONDS), config.PACING_MAX_DELAY_SECONDS)


def _humanize_gap(gap_hours: float) -> str | None:
    if gap_hours < 0.75:
        return None  # near-simultaneous — no divider needed
    if gap_hours < 24:
        h = max(1, round(gap_hours))
        return f"{h} hour{'s' if h != 1 else ''} later"
    days = gap_hours / 24
    if days < 30:
        d = max(1, round(days))
        return f"{d} day{'s' if d != 1 else ''} later"
    months = days / 30.44
    if months < 12:
        m = max(1, round(months))
        return f"{m} month{'s' if m != 1 else ''} later"
    years = days / 365.25
    y = round(years, 1) if years < 10 else round(years)
    return f"{y} year{'s' if y != 1 else ''} later"


def _pacing_for_gap(gap_hours: float) -> tuple[float, str | None]:
    return _delay_seconds(gap_hours), _humanize_gap(gap_hours)


async def timeline_node(state: ArkState) -> dict:
    key_players = [a for a in state["roster"] if a.get("narrative_role") != "commentator"]
    roster_brief = [{"name": a["name"], "role": a["role"]} for a in key_players]
    user = json.dumps({"era_summary": state["era_summary"], "roster": roster_brief})
    text, provider = await llm_router.complete(TIMELINE_SYSTEM, user, config.PLANNING_PROVIDER_CHAIN)
    name_to_id = {a["name"].lower(): a["id"] for a in state["roster"]}

    def resolve(names: list[str]) -> list[str]:
        ids = []
        for n in names:
            aid = name_to_id.get(n.lower())
            if aid:
                ids.append(aid)
        return ids or ([state["roster"][0]["id"]] if state["roster"] else [])

    try:
        raw_events = llm_router.extract_json(text)
        if isinstance(raw_events, dict):
            raw_events = raw_events.get("events") or raw_events.get("timeline") or []
    except Exception as e:
        log.warning(
            "timeline_node: failed to parse JSON from provider %r (%s) — "
            "falling back to a SINGLE emergency event. This is the #1 cause of "
            "'the timeline only has one event' — check the warning above this "
            "one for which provider actually failed and why.", provider, e
        )
        raw_events = [
            {
                "title": "Opening moment",
                "description": state["era_summary"][:200],
                "mode": "parallel",
                "sim_date": "",
                "hours_since_start": 0,
                "participants": [a["name"] for a in state["roster"][:4]],
            }
        ]
    events = []
    prev_hours = None
    for i, item in enumerate(raw_events[: config.MAX_EVENTS_PER_SIM]):
        participants = resolve(item.get("participants", []))
        try:
            hours = float(item.get("hours_since_start", i))
        except (TypeError, ValueError):
            hours = float(i)
        if prev_hours is not None and hours < prev_hours:
            hours = prev_hours  # clamp: model must not report time moving backward
        gap_hours = 0.0 if prev_hours is None else max(0.0, hours - prev_hours)
        gap_seconds, gap_label = _pacing_for_gap(gap_hours)
        prev_hours = hours
        event = TimelineEvent(
            title=item.get("title", f"Event {i+1}"),
            description=item.get("description", ""),
            order=i,
            mode="parallel" if item.get("mode") == "parallel" else "sequential",
            participant_ids=participants,
            sim_date=item.get("sim_date", ""),
            hours_since_start=hours,
            gap_seconds=gap_seconds,
            gap_label=gap_label,
        )
        events.append(event.model_dump())
    await storage.save_events(state["sim_id"], events)
    return {"timeline": events, "cursor": 0, "posts": []}


# ---------------------------------------------------------------------------
# Scheduler / router
# ---------------------------------------------------------------------------

def route_next(state: ArkState) -> str:
    if state["cursor"] >= len(state["timeline"]):
        return "end"
    event = state["timeline"][state["cursor"]]
    return "parallel" if event["mode"] == "parallel" else "sequential"


POST_SYSTEM_TEMPLATE = """You are {name} ({handle}), {role}, posting on Ark — a live social feed.
Voice/personality: {personality}
What you want: {goals}
What you currently know/believe: {era_context}
Backstory (who you are — this never changes): {backstory}
Your relationships: {relationships_text}

Write exactly ONE short in-character post reacting to the current moment. Under 260 characters.
No hashtags, no "As an AI", no meta-commentary, no quotation marks around the whole post.
Sound like a real person or institution posting in the moment, not a historian summarizing it.
Let your backstory and relationships genuinely color your tone — warmer or more deferential toward
someone tagged as a friend/partner/mentor, sharper or more guarded toward someone tagged hostile —
without narrating the relationship itself ("as my friend..." reads as fake; just let it show).

You may optionally call the attach_media tool — but only if it is genuinely natural for someone
in YOUR specific role to be posting a photo or video right now. Most posts should stay text-only.

You may optionally call the reply_to tool if your post is a direct, genuine response to one specific
post shown below — most posts are not replies, they're independent reactions to the event itself."""


def _format_relationships(agent: dict) -> str:
    rels = agent.get("relationships") or []
    if not rels:
        return "(no strong personal relationships defined for this event)"
    parts = []
    for r in rels:
        tags = ", ".join(r.get("tags", []))
        parts.append(f"{r.get('target_name', '?')} ({tags})")
    return "; ".join(parts)


def _recent_context(posts: list[dict], event_id: str, limit: int = 4) -> str:
    same_event = [p for p in posts if p["event_id"] == event_id]
    others = same_event[-limit:]
    if not others:
        return "(no one has posted about this yet — you're first)"
    return "\n".join(f"[{p['id']}] {p['agent_name']} ({p['agent_handle']}): {p['content']}" for p in others)


def _commentators(state: ArkState) -> list[dict]:
    return [a for a in state["roster"] if a.get("narrative_role") == "commentator"]


GENERATED_MEDIA_DIR = config.MEDIA_DIR


async def _maybe_generate_media(post_id: str, caption: str, media_hint: str) -> str | None:
    if not config.GENERATE_MEDIA:
        return None
    style = "a period-accurate press photograph" if media_hint == "Photo" else "a still frame from period newsreel footage"
    prompt = f"{style}: {caption}"
    result = await image_router.generate_image(prompt, config.IMAGE_PROVIDER_CHAIN)
    if not result:
        return None
    image_bytes, provider = result
    os.makedirs(GENERATED_MEDIA_DIR, exist_ok=True)
    filename = f"{post_id}.png"
    path = os.path.join(GENERATED_MEDIA_DIR, filename)
    try:
        with open(path, "wb") as f:
            f.write(image_bytes)
    except Exception:
        return None
    return f"/media/{filename}"


async def _generate_one_post(roster: list[dict], posts_so_far: list[dict], event: dict, agent_id: str, order_counter: list[int]) -> dict:
    """Takes roster/posts_so_far explicitly (rather than pulling from a full
    ArkState) so callers control exactly what context an agent can see —
    critical for replies to work: a sequential caller passes an
    accumulating list so agent 2 can see agent 1's post from THIS event;
    a parallel caller intentionally passes the pre-event snapshot, since
    genuinely simultaneous posts shouldn't see each other."""
    agent = next((a for a in roster if a["id"] == agent_id), None)
    if agent is None:
        return None
    system = POST_SYSTEM_TEMPLATE.format(relationships_text=_format_relationships(agent), **agent)
    context = _recent_context(posts_so_far, event["id"])
    user = (
        f"Event: {event['title']}\n"
        f"What's happening: {event['description']}\n"
        f"In-world date: {event.get('sim_date', '')}\n"
        f"Recent posts about this (format: [post_id] name: content):\n{context}\n\n"
        f"Write your post now."
    )

    tools = [llm_router.REPLY_TO_TOOL]
    if config.GENERATE_MEDIA:
        tools.append(llm_router.ATTACH_MEDIA_TOOL)

    text, tool_calls, provider = await llm_router.complete_with_tools(system, user, config.POST_PROVIDER_CHAIN, tools)
    if not text or not text.strip():
        # Some providers return only a tool call with no text content in
        # the same turn — get the actual post text with a plain follow-up
        # call rather than leaving the post empty.
        text, _provider2 = await llm_router.complete(system, user, config.POST_PROVIDER_CHAIN)

    order_counter[0] += 1
    media_hint = None
    media_caption = None
    attach_call = next((tc for tc in tool_calls if tc.get("name") == "attach_media"), None)
    if attach_call:
        args = attach_call.get("arguments", {})
        kind = args.get("kind")
        caption = args.get("caption")
        if kind in ("photo", "video") and caption:
            media_hint = "Photo" if kind == "photo" else "Video"
            media_caption = str(caption)[:300]

    # Validate the reply target against the SAME posts_so_far the model was
    # actually shown — never trust a model-supplied id blindly.
    reply_to_post_id = None
    reply_call = next((tc for tc in tool_calls if tc.get("name") == "reply_to"), None)
    if reply_call:
        candidate_id = reply_call.get("arguments", {}).get("post_id")
        same_event_ids = {p["id"] for p in posts_so_far if p["event_id"] == event["id"]}
        if candidate_id in same_event_ids:
            reply_to_post_id = candidate_id

    post_obj = Post(
        event_id=event["id"],
        agent_id=agent["id"],
        agent_name=agent["name"],
        agent_handle=agent["handle"],
        agent_role=agent["role"],
        content=text[:280],
        media_hint=media_hint,
        media_caption=media_caption,
        reply_to_post_id=reply_to_post_id,
        sim_date=event.get("sim_date", ""),
        created_order=order_counter[0],
    )
    if media_hint and media_caption:
        post_obj.media_url = await _maybe_generate_media(post_obj.id, media_caption, media_hint)
    post = post_obj.model_dump()
    # Note: writing to the World Event Log (storage.append_post) is the
    # CALLER's responsibility, not this function's — this function
    # deliberately doesn't receive sim_id, only roster/posts_so_far, so
    # that both the sequential and parallel schedulers stay in full control
    # of exactly when/whether a post is persisted.
    return post


async def generate_sequential_node(state: ArkState) -> dict:
    event = state["timeline"][state["cursor"]]
    gap = event.get("gap_seconds", 0.0)
    if gap:
        await asyncio.sleep(gap)
    order_counter = [len(state["posts"])]
    # Accumulates as we go — agent 2 in this event needs to see agent 1's
    # post from THIS SAME event, which state["posts"] alone can't provide
    # (it's only updated once the whole node returns).
    posts_so_far = list(state["posts"])
    for agent_id in event["participant_ids"][:3]:  # cap chatter per event
        post = await _generate_one_post(state["roster"], posts_so_far, event, agent_id, order_counter)
        if post:
            await storage.append_post(state["sim_id"], post)
            posts_so_far.append(post)
    commentators = _commentators(state)
    if commentators:
        pick = commentators[state["cursor"] % len(commentators)]
        if pick["id"] not in event["participant_ids"]:
            post = await _generate_one_post(state["roster"], posts_so_far, event, pick["id"], order_counter)
            if post:
                await storage.append_post(state["sim_id"], post)
                posts_so_far.append(post)
    return {"posts": posts_so_far}


async def generate_parallel_node(state: ArkState) -> dict:
    event = state["timeline"][state["cursor"]]
    gap = event.get("gap_seconds", 0.0)
    if gap:
        await asyncio.sleep(gap)
    order_counter = [len(state["posts"])]
    participant_ids = list(event["participant_ids"][:8])  # cap fan-out width
    commentators = _commentators(state)
    if commentators:
        pick = commentators[state["cursor"] % len(commentators)]
        if pick["id"] not in participant_ids:
            participant_ids.append(pick["id"])

    # Genuinely concurrent — each worker sees the pre-event snapshot only,
    # not each other's posts, since real simultaneous reactions don't see
    # each other yet either. Slight jitter keeps the feed from looking
    # mechanically simultaneous.
    base_posts = state["posts"]

    async def worker(aid):
        await asyncio.sleep(random.uniform(0, 0.4))
        return await _generate_one_post(state["roster"], base_posts, event, aid, order_counter)

    results = await asyncio.gather(*(worker(aid) for aid in participant_ids))
    fresh = [p for p in results if p]
    for p in fresh:
        await storage.append_post(state["sim_id"], p)
    new_posts = list(state["posts"]) + fresh
    new_posts.sort(key=lambda p: p["created_order"])
    return {"posts": new_posts}


async def advance_cursor_node(state: ArkState) -> dict:
    return {"cursor": state["cursor"] + 1}


def build_graph():
    graph = StateGraph(ArkState)
    graph.add_node("intake", intake_node)
    graph.add_node("research", research_node)
    graph.add_node("roster", roster_node)
    graph.add_node("commentators", commentator_node)
    graph.add_node("timeline", timeline_node)
    graph.add_node("generate_sequential", generate_sequential_node)
    graph.add_node("generate_parallel", generate_parallel_node)
    graph.add_node("advance_cursor", advance_cursor_node)

    graph.set_entry_point("intake")
    graph.add_edge("intake", "research")
    graph.add_edge("research", "roster")
    graph.add_edge("roster", "commentators")
    graph.add_edge("commentators", "timeline")

    routes = {"sequential": "generate_sequential", "parallel": "generate_parallel", "end": END}
    graph.add_conditional_edges("timeline", route_next, routes)
    graph.add_conditional_edges("advance_cursor", route_next, routes)

    graph.add_edge("generate_sequential", "advance_cursor")
    graph.add_edge("generate_parallel", "advance_cursor")

    return graph.compile(checkpointer=MemorySaver())


ARK_GRAPH = build_graph()

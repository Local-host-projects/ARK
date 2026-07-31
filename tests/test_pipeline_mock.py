"""
Standalone test: monkeypatch llm_router with canned, realistic responses so
we can verify graph.py's parsing, scheduling, pacing, and tool-call logic
end-to-end WITHOUT needing a live provider key.

Run: python3 tests/test_pipeline_mock.py
"""
import asyncio
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ARK_DB_PATH", str(ROOT / "ark_test.db"))

from app import llm_router, graph, storage, config

INTAKE_JSON = json.dumps({
    "title": "Pearl Harbor: The Morning Of",
    "era_summary": "Dec 7 1941. Japanese carrier aircraft strike the US Pacific Fleet at Pearl Harbor without warning.",
    "entities": [
        {"name": "Franklin D. Roosevelt", "role": "US President", "kind": "person"},
        {"name": "Isoroku Yamamoto", "role": "Japanese Admiral", "kind": "person"},
        {"name": "Associated Press", "role": "News Wire", "kind": "press"},
        {"name": "Honolulu Star-Bulletin", "role": "Local Newspaper", "kind": "press"},
    ],
})

ROSTER_JSON = json.dumps([
    {"name": "Franklin D. Roosevelt", "handle": "@fdr", "role": "US President", "kind": "person",
     "personality": "Measured, resolute, speaks to reassure a nation.",
     "goals": "Rally the country and get a declaration of war through Congress.",
     "era_context": "Just informed of the attack; preparing an address.",
     "backstory": "Rose to the presidency through the Depression; no stranger to steering a nation through crisis.",
     "relationships": [{"name": "Isoroku Yamamoto", "tags": ["hostile"]}]},
    {"name": "Isoroku Yamamoto", "handle": "@yamamoto_iso", "role": "Japanese Admiral", "kind": "person",
     "personality": "Calculating, privately wary of a long war with the US.",
     "goals": "Cripple the Pacific Fleet before it can respond.",
     "era_context": "Planned the strike; aware of its strategic gamble.",
     "backstory": "Studied in the United States as a young officer; understands American industrial capacity better than most of his peers.",
     "relationships": [{"name": "Franklin D. Roosevelt", "tags": ["rival"]}]},
    {"name": "Associated Press", "handle": "@AP", "role": "News Wire", "kind": "press",
     "personality": "Terse, fact-first wire service voice.",
     "goals": "Get confirmed details out fast.",
     "era_context": "Receiving fragmented first reports from Hawaii.",
     "backstory": "", "relationships": []},
    {"name": "Honolulu Star-Bulletin", "handle": "@honolulu_sb", "role": "Local Newspaper", "kind": "press",
     "personality": "Local, urgent, eyewitness-driven.",
     "goals": "Cover the attack on their own doorstep.",
     "era_context": "Reporters watching smoke rise from the harbor.",
     "backstory": "", "relationships": []},
])

COMMENTATOR_JSON = json.dumps([
    {"name": "Edward R. Murrow", "handle": "@ed_murrow", "role": "CBS Radio Correspondent", "kind": "person",
     "personality": "Measured, gravitas-heavy broadcast voice.",
     "goals": "Explain the scale of what just happened to a listening public.",
     "era_context": "Reporting from CBS on the unfolding attack and its aftermath.",
     "backstory": "Built his reputation broadcasting from London during the Blitz.", "grounded": True},
    {"name": "United Press", "handle": "@united_press", "role": "Wire Service", "kind": "press",
     "personality": "Fast, competitive with AP, fact-first.",
     "goals": "Beat the competition on confirmed details.",
     "era_context": "Running fragmentary cables from Hawaii correspondents.",
     "backstory": "", "grounded": True},
])

TIMELINE_JSON = json.dumps([
    {"title": "First bombs fall", "description": "Japanese aircraft begin bombing Pearl Harbor without warning.",
     "mode": "parallel", "sim_date": "7 Dec 1941, 07:55", "hours_since_start": 0,
     "participants": ["Associated Press", "Honolulu Star-Bulletin"]},
    {"title": "Yamamoto reflects privately", "description": "Yamamoto privately notes the strategic risk of the strike.",
     "mode": "sequential", "sim_date": "7 Dec 1941, 08:00", "hours_since_start": 5,
     "participants": ["Isoroku Yamamoto"]},
    {"title": "Roosevelt briefed", "description": "Roosevelt receives word of the scale of the attack.",
     "mode": "sequential", "sim_date": "7 Dec 1941, 14:30", "hours_since_start": 30,
     "participants": ["Franklin D. Roosevelt"]},
    {"title": "Infamy speech", "description": "Roosevelt addresses Congress calling Dec 7 a date which will live in infamy.",
     "mode": "parallel", "sim_date": "8 Dec 1941, 12:30", "hours_since_start": 9600,  # ~400 days -> year-scale tier
     "participants": ["Franklin D. Roosevelt", "Associated Press", "Honolulu Star-Bulletin"]},
])

POST_TEXT = "In-character reaction to the moment, under 260 characters, matching the persona."


async def fake_complete(system, user, chain):
    if "intake analyst" in system:
        return INTAKE_JSON, "mock"
    if "character personas" in system:
        return ROSTER_JSON, "mock"
    if "recurring commentary voices" in system:
        return COMMENTATOR_JSON, "mock"
    if "event skeleton" in system:
        return TIMELINE_JSON, "mock"
    return f"{POST_TEXT} — from {system.split()[2]}", "mock"


async def fake_complete_with_tools(system, user, chain, tools=None):
    text = f"{POST_TEXT} — from {system.split()[2]}"
    tool_calls = []
    if "Honolulu Star-Bulletin" in system:
        tool_calls.append({"name": "attach_media", "arguments": {
            "kind": "photo", "caption": "Smoke rising over Pearl Harbor, taken from Honolulu"
        }})
        # Also test reply_to: find an Associated Press post id in the
        # context this call was actually given, and reply to it if visible.
        # This exercises the SAME-EVENT visibility fix — Honolulu is the
        # second participant in event 0 (sequential-style within a
        # parallel event? no — event 0 is parallel, so AP and Honolulu run
        # concurrently and should NOT see each other; this assertion is
        # instead checked against a sequential case below).
        match = re.search(r"\[(post_[a-f0-9]+)\] Associated Press", user)
        if match:
            tool_calls.append({"name": "reply_to", "arguments": {"post_id": match.group(1)}})
    return text, tool_calls, "mock"


async def fake_research_topic(query, wiki_limit=2, ddg_limit=5):
    return "Wikipedia:\n- Pearl Harbor attack: Historical background on war correspondents.\n"


async def main():
    llm_router.complete = fake_complete
    llm_router.complete_with_tools = fake_complete_with_tools
    graph.research.research_topic = fake_research_topic
    graph.image_router.generate_image = lambda *a, **k: asyncio.sleep(0, result=None)

    # Truncate the real pacing sleeps so the test doesn't actually wait up
    # to 12s per event — still exercises the code path (gap_seconds get
    # computed and slept on), just fast.
    real_sleep = asyncio.sleep
    async def fast_sleep(seconds, result=None):
        return await real_sleep(min(seconds, 0.01), result=result)
    asyncio.sleep = fast_sleep

    await storage.init()
    sim_id = "sim_test_mock"
    initial_state = {
        "sim_id": sim_id, "prompt": "Pearl Harbor", "title": "", "era_summary": "",
        "entities": [], "research_digest": "", "commentator_brief": "",
        "roster": [], "timeline": [], "cursor": 0, "posts": [],
    }
    final_state = None
    events_seen = []
    async for state in graph.ARK_GRAPH.astream(initial_state, config={"configurable": {"thread_id": sim_id}}, stream_mode="values"):
        final_state = state
        if state.get("timeline") and not events_seen:
            events_seen = state["timeline"]
            print(f"TIMELINE ({len(events_seen)} events):")
            for e in events_seen:
                print(f"  [{e['mode']:10s}] {e['sim_date']:20s} {e['title']} -> participants={len(e['participant_ids'])}")

    print(f"\nROSTER ({len(final_state['roster'])} agents):")
    for a in final_state["roster"]:
        rel_str = "; ".join(f"{r['target_name']}({','.join(r['tags'])})" for r in a.get("relationships", []))
        print(f"  {a['name']:25s} {a['handle']:15s} {a['role']:20s} backstory={'yes' if a['backstory'] else 'no':3s} rel=[{rel_str}]")

    print(f"\nPOSTS ({len(final_state['posts'])} total), in generation order:")
    for p in final_state["posts"]:
        media_note = f" [media: {p['media_hint']} — {p['media_caption'][:40]}]" if p.get("media_hint") else ""
        reply_note = f" [reply_to: {p['reply_to_post_id']}]" if p.get("reply_to_post_id") else ""
        print(f"  [{p['created_order']:2d}] {p['agent_name']:25s} ({p['sim_date']:20s}): {p['content'][:50]}{media_note}{reply_note}")

    print("\nTIMELINE PACING:")
    timeline_by_order = sorted(final_state["timeline"], key=lambda e: e["order"])
    for e in timeline_by_order:
        print(f"  order={e['order']} hours_since_start={e['hours_since_start']:.1f} gap_seconds={e['gap_seconds']:.2f} gap_label={e['gap_label']!r}")

    # --- roster shape ---
    key_players = [a for a in final_state["roster"] if a["narrative_role"] == "participant"]
    commentators = [a for a in final_state["roster"] if a["narrative_role"] == "commentator"]
    assert len(key_players) == 4, "expected 4 key-player agents"
    assert len(commentators) == 2, "expected 2 commentator agents"
    assert len(final_state["timeline"]) == 4, "expected 4 events"
    assert len(final_state["posts"]) == 11, f"expected 11 posts total, got {len(final_state['posts'])}"

    # --- backstory + relationships resolution ---
    fdr = next(a for a in final_state["roster"] if a["name"] == "Franklin D. Roosevelt")
    yamamoto = next(a for a in final_state["roster"] if a["name"] == "Isoroku Yamamoto")
    assert fdr["backstory"], "FDR should have a non-empty backstory"
    assert len(fdr["relationships"]) == 1, "FDR should have exactly 1 resolved relationship"
    assert fdr["relationships"][0]["target_id"] == yamamoto["id"], "FDR's relationship should resolve to Yamamoto's real agent id, not just a name"
    assert fdr["relationships"][0]["tags"] == ["hostile"]

    # --- parallel event posts ---
    parallel_event_id = events_seen[0]["id"]
    parallel_posts = [p for p in final_state["posts"] if p["event_id"] == parallel_event_id]
    assert len(parallel_posts) == 3, f"first parallel event should have 3 posts, got {len(parallel_posts)}"

    # --- commentators post across ALL events (connective tissue) ---
    commentator_ids = {a["id"] for a in commentators}
    commentator_post_events = {p["event_id"] for p in final_state["posts"] if p["agent_id"] in commentator_ids}
    assert len(commentator_post_events) == 4, "commentators should post across all 4 events"

    # --- temporal pacing tiers ---
    assert timeline_by_order[0]["gap_label"] is None, "first event should have no gap divider"
    assert "hour" in timeline_by_order[1]["gap_label"]
    assert "day" in timeline_by_order[2]["gap_label"]
    assert "year" in timeline_by_order[3]["gap_label"]
    for e in timeline_by_order:
        assert 0 <= e["gap_seconds"] <= config.PACING_MAX_DELAY_SECONDS, "gap_seconds must stay bounded"

    # --- tool-call-driven media: only Honolulu Star-Bulletin's post(s) ---
    media_posts = [p for p in final_state["posts"] if p.get("media_hint")]
    assert len(media_posts) >= 1, "expected at least one tool-call-triggered media post"
    assert all(p["agent_name"] == "Honolulu Star-Bulletin" for p in media_posts), \
        "only the press agent's mock triggers attach_media — media leaked to another agent"
    assert all(p.get("media_caption") for p in media_posts)

    # --- same-event visibility fix: event 3 (the second parallel event)
    # has Honolulu as a participant AND an Associated Press post generated
    # in the SAME event. Since event 3 is parallel, concurrent participants
    # deliberately do NOT see each other (that's correct — simultaneous
    # posts shouldn't see each other), so Honolulu's reply_to in event 3
    # should NOT resolve (AP's post from event 3 doesn't exist yet when
    # Honolulu's context is built). Confirm no invalid/dangling reply
    # slipped through validation regardless. ---
    honolulu_posts = [p for p in final_state["posts"] if p["agent_name"] == "Honolulu Star-Bulletin"]
    for p in honolulu_posts:
        if p.get("reply_to_post_id"):
            same_event_ids = {q["id"] for q in final_state["posts"] if q["event_id"] == p["event_id"]}
            assert p["reply_to_post_id"] in same_event_ids, "reply_to_post_id must always resolve to a real post in the same event"

    print("\nALL ASSERTIONS PASSED")


asyncio.run(main())

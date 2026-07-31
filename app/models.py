from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field
import uuid


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class Relationship(BaseModel):
    target_name: str          # the other agent's name, as written by the model
    target_id: str = ""       # resolved to a real roster agent id after generation; blank if unresolved
    tags: list[str] = Field(default_factory=list)  # 1-2 short words: "friend", "hostile", "rival", "partner", "mentor"


class Agent(BaseModel):
    id: str = Field(default_factory=lambda: new_id("agent"))
    name: str
    handle: str
    role: str                      # e.g. "Head of State", "War Correspondent", "News Wire"
    kind: Literal["person", "org", "press"] = "person"
    narrative_role: Literal["participant", "commentator"] = "participant"
    personality: str                # short voice/personality description
    goals: str                      # what this agent wants / is trying to achieve
    era_context: str = ""           # background facts this agent knows at sim start
    backstory: str = ""             # a real paragraph of personal history — fixed at creation, never
                                     # updated; this is Ark's "memory": who they are, not a growing log
    relationships: list[Relationship] = Field(default_factory=list)  # max 3, resolved against the roster
    avatar_seed: str = ""           # used client-side to generate a deterministic avatar
    grounded: bool = False          # True if a commentator was matched to a real search result

    def __init__(self, **data):
        super().__init__(**data)
        if not self.avatar_seed:
            self.avatar_seed = self.handle


class TimelineEvent(BaseModel):
    id: str = Field(default_factory=lambda: new_id("evt"))
    title: str
    description: str
    order: int                      # position in the timeline
    mode: Literal["sequential", "parallel"] = "sequential"
    participant_ids: list[str] = Field(default_factory=list)
    sim_date: str = ""               # in-world date label, e.g. "7 Dec 1941"
    hours_since_start: float = 0.0   # numeric clock, hours since event 0 — drives pacing
    gap_seconds: float = 0.0         # real playback delay before this event, compressed from hours_since_start
    gap_label: Optional[str] = None  # human label for the gap ("3 months later"); None if negligible


class Post(BaseModel):
    id: str = Field(default_factory=lambda: new_id("post"))
    event_id: str
    agent_id: str
    agent_name: str
    agent_handle: str
    agent_role: str
    content: str
    media_hint: Optional[str] = None   # short description of an image/video the post "attaches"
    media_url: Optional[str] = None    # populated if image generation succeeded
    media_caption: Optional[str] = None  # the agent's own tool-call caption used to generate media
    reply_to_post_id: Optional[str] = None  # set if the agent's own reply_to tool call targeted a real prior post
    sim_date: str = ""
    created_order: int = 0


class SimulationCreateRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=8000)
    title: Optional[str] = None


class SimulationSummary(BaseModel):
    id: str
    title: str
    status: Literal["planning", "ready", "streaming", "done", "error"]
    agent_count: int = 0
    event_count: int = 0
    error: Optional[str] = None


class SignupRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    password: str = Field(..., min_length=8, max_length=200)


class LoginRequest(BaseModel):
    username: str
    password: str

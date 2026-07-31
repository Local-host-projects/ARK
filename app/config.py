import os
import secrets
from dotenv import load_dotenv

load_dotenv()

# --- LLM provider keys (all optional; router falls back in order) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
HF_API_KEY = os.getenv("HF_API_KEY", "")

# --- Model choices per provider (override via env if a model gets retired) ---
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "llama-3.3-70b")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-r1:free")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# --- Provider fallback order ---
POST_PROVIDER_CHAIN = ["groq", "cerebras", "openrouter", "gemini", "anthropic", "demo"]
PLANNING_PROVIDER_CHAIN = ["gemini", "anthropic", "openrouter", "groq", "demo"]

# --- Image generation: Gemini first (as requested), Hugging Face free
# serverless FLUX.1-schnell as fallback. ---
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
HF_IMAGE_MODEL = os.getenv("HF_IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell")
IMAGE_PROVIDER_CHAIN = ["gemini", "huggingface"]
GENERATE_MEDIA = os.getenv("ARK_GENERATE_MEDIA", "true").lower() != "false"

# --- Search grounding (DuckDuckGo + Wikipedia) for commentator agents ---
ENABLE_RESEARCH = os.getenv("ARK_ENABLE_RESEARCH", "true").lower() != "false"
NUM_COMMENTATORS = int(os.getenv("ARK_NUM_COMMENTATORS", "3"))

# --- Temporal pacing ---
PACING_MIN_DELAY_SECONDS = float(os.getenv("ARK_PACING_MIN_DELAY", "0.4"))
PACING_MAX_DELAY_SECONDS = float(os.getenv("ARK_PACING_MAX_DELAY", "12.0"))

# --- App behavior ---
DB_PATH = os.getenv("ARK_DB_PATH", os.path.join(os.path.dirname(__file__), "..", "ark.db"))
MEDIA_DIR = os.getenv("ARK_MEDIA_DIR", os.path.join(os.path.dirname(__file__), "static", "generated"))
MAX_AGENTS_PER_SIM = int(os.getenv("ARK_MAX_AGENTS", "14"))
MAX_EVENTS_PER_SIM = int(os.getenv("ARK_MAX_EVENTS", "24"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("ARK_LLM_TIMEOUT", "30"))

# --- Auth ---
# Session cookies are signed with this key. If unset, a random key is
# generated at process start — sessions then invalidate on every restart,
# which is fine for local dev but you should set ARK_SESSION_SECRET
# explicitly in production (e.g. Railway) so logins survive a redeploy.
SESSION_SECRET = os.getenv("ARK_SESSION_SECRET") or secrets.token_hex(32)
SESSION_SECRET_IS_EPHEMERAL = not os.getenv("ARK_SESSION_SECRET")

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
# If unset, built from the incoming request's own base URL at call time
# (see main.py) — only set this explicitly if you need to force a specific
# host (e.g. behind a proxy that mangles the scheme/host).
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "")
GOOGLE_AUTH_ENABLED = bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)

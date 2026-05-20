# Save as: director_engine/config.py
import os
from enum import Enum, auto
from pathlib import Path


def _load_sibling_secrets() -> None:
    secrets_dir = Path(__file__).resolve().parent.parent / "director_ui" / "secrets"
    if not secrets_dir.is_dir():
        return
    for path in sorted(secrets_dir.glob("*.env")):
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


if "GEMINI_API_KEY_NAMI" not in os.environ:
    _load_sibling_secrets()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY_NAMI")
if not GEMINI_API_KEY:
    print("⚠️ Warning: GEMINI_API_KEY_NAMI not set. Visual summarization will be disabled.")

class InputSource(Enum):
    DIRECT_MICROPHONE = auto()
    TWITCH_MENTION = auto()
    AMBIENT_AUDIO = auto()
    VISUAL_CHANGE = auto()
    TWITCH_CHAT = auto()
    MICROPHONE = auto()
    BOT_TWITCH_REPLY = auto()
    SYSTEM_PATTERN = auto()
    INTERNAL_THOUGHT = auto()

SOURCE_WEIGHTS = {
    InputSource.DIRECT_MICROPHONE: 0.8,
    InputSource.TWITCH_MENTION: 0.8,
    InputSource.AMBIENT_AUDIO: 0.7,
    InputSource.VISUAL_CHANGE: 0.95,
    InputSource.TWITCH_CHAT: 0.3,
    InputSource.MICROPHONE: 0.5,
    InputSource.BOT_TWITCH_REPLY: 0.0,
    InputSource.SYSTEM_PATTERN: 0.95,
    InputSource.INTERNAL_THOUGHT: 0.95
}

class ConversationState(Enum):
    IDLE = "quiet, waiting for something to happen"
    ENGAGED = "active conversation happening"
    STORYTELLING = "user sharing experience"
    TEACHING = "explaining something to chat"
    FRUSTRATED = "user struggling"
    CELEBRATORY = "user succeeded at something"

class FlowState(Enum):
    NATURAL = "natural flow"
    DRIFTING = "topic drifting"
    STACCATO = "rapid fire / chaotic"
    DOMINATED = "user dominating"
    DEAD_AIR = "awkward silence"

class UserIntent(Enum):
    CASUAL = "just hanging out"
    HELP_SEEKING = "needs assistance/backseating"
    VALIDATION = "wants praise/reaction"
    ENTERTAINMENT = "performing for stream"
    INFO_SEEKING = "fishing for specific info"
    PROVOKING = "trying to trigger bot"

class BotGoal(Enum):
    OBSERVE = "passively watch and learn"
    ENTERTAIN = "make jokes and keep energy up"
    SUPPORT = "help the user succeed"
    INVESTIGATE = "learn more about the user"
    TROLL = "create chaos and banter"

class HostState(Enum):
    ACTIVE = "talking regularly"
    FADING = "slowing down or sparse"
    QUIET = "silent"
    UNKNOWN = "no signal yet"

# Sliding-window mic-rate thresholds for HostState
HOST_STATE_WINDOW_SECONDS = 60.0       # full window we keep timestamps for
HOST_STATE_ACTIVE_WINDOW = 30.0        # short window for "talking right now"
HOST_STATE_ACTIVE_THRESHOLD = 3        # >= N transcripts in short window => ACTIVE
HOST_STATE_FRESH_STARTUP_SECONDS = 30.0  # before this, no-events => UNKNOWN (not QUIET)

class SceneType(Enum):
    CHILL_CHATTING = "Just Chatting / Low Intensity"
    EXPLORATION = "Gameplay Exploration / Wandering"
    COMBAT_HIGH = "Intense Combat / Boss Fight"
    MENUING = "In Menus / Inventory Management"
    HORROR_TENSION = "Spooky / High Tension"
    COMEDY_MOMENT = "Funny / Meme / Laughter"
    TECHNICAL_DOWNTIME = "Loading / Tech Issues"
    HOST_FOCUSED_QUIET = "Host locked in — game audio busy, mic quiet"
    HOST_LOW_ENERGY = "Host fading — mic and game audio both quiet"

# --- Game (desktop) audio activity thresholds ---
# Counts AMBIENT_AUDIO events in the recent window (WINDOW_RECENT seconds).
GAME_AUDIO_BUSY_THRESHOLD = 3

# --- Identity ---
# The Twitch username of the bot's owner.
# Used to determine whether Nami is watching her owner stream,
# or watching a third-party stream alongside her owner.
OWNER_STREAMER_ID = "peepingotter"

# --- Energy System Config ---
ENERGY_MAX = 100.0
ENERGY_REGEN_PER_SEC = 5.0
ENERGY_COST_INTERJECTION = 5.0
ENERGY_COST_REPLY = 5.0
ENERGY_COST_PATTERN = 5.0

# --- Memory & Proactivity Config ---
CURIOSITY_INTERVAL = 10.0
CALLBACK_INTERVAL = 45.0
COMPRESSION_INTERVAL = 30.0
POST_SPEECH_COOLDOWN = 5.0

# --- Director Config ---
DIRECTOR_PORT = 8006
HUB_URL = "http://localhost:8002"
DIRECTOR_HOST = "0.0.0.0"
WINDOW_IMMEDIATE = 10.0
WINDOW_RECENT = 30.0
WINDOW_BACKGROUND = 300.0
SUMMARY_INTERVAL_SECONDS = 5.0
INTERJECTION_THRESHOLD = 0.75
VALID_MOODS = ["Neutral", "Happy", "Annoyed", "Scared", "Horny", "Tired"]
DEFAULT_MOOD = "Neutral"
MOOD_WINDOW_SIZE = 5
OLLAMA_TRIGGER_THRESHOLD = 0.5
OLLAMA_MODEL = 'llama3.2:latest'
OLLAMA_HOST = 'http://localhost:11434'

# --- Ollama throughput tuning ---
# A 32B-class model on the M4 takes ~10-14s per call at modest prompt sizes
# and exceeds 15s once prompts cross ~2k tokens. Two concurrent calls compound
# and most start failing the timeout. These knobs gate that.
OLLAMA_MAX_CONCURRENT = 2          # global cap on in-flight Ollama calls
OLLAMA_TIMEOUT_THOUGHT = 45.0      # generate_thought
OLLAMA_TIMEOUT_ANALYZE = 45.0      # analyze_and_update_event
OLLAMA_TIMEOUT_SUMMARY = 45.0      # generate_summary
OLLAMA_TIMEOUT_CTX_INFER = 30.0    # _do_context_inference
OLLAMA_TIMEOUT_COMPRESS = 60.0     # _compress_recent / _compress_ancient

PROMPT_SERVICE_URL = "http://localhost:8001"
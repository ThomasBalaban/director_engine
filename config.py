# Save as: director_engine/config.py
from enum import Enum, auto

# --- NEW: Import API Key ---
try:
    from private_key import GEMINI_API_KEY
except ImportError:
    GEMINI_API_KEY = None
    print("⚠️ Warning: private_key.py not found. Visual summarization will be disabled.")

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
    InputSource.AMBIENT_AUDIO: 0.7,      # INCREASED: Listen to game audio
    InputSource.VISUAL_CHANGE: 0.95,     # MAXED: Vision is primary
    InputSource.TWITCH_CHAT: 0.3,
    InputSource.MICROPHONE: 0.5,
    InputSource.BOT_TWITCH_REPLY: 0.0,
    InputSource.SYSTEM_PATTERN: 0.95,
    InputSource.INTERNAL_THOUGHT: 0.95   # MAXED: Thoughts must be voiced
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

class SceneType(Enum):
    CHILL_CHATTING = "Just Chatting / Low Intensity"
    EXPLORATION = "Gameplay Exploration / Wandering"
    COMBAT_HIGH = "Intense Combat / Boss Fight"
    MENUING = "In Menus / Inventory Management"
    HORROR_TENSION = "Spooky / High Tension"
    COMEDY_MOMENT = "Funny / Meme / Laughter"
    TECHNICAL_DOWNTIME = "Loading / Tech Issues"

# --- Energy System Config ---
ENERGY_MAX = 100.0
ENERGY_REGEN_PER_SEC = 5.0       # INCREASED: Unlimited yapping
ENERGY_COST_INTERJECTION = 5.0   # LOWERED: Very cheap to speak
ENERGY_COST_REPLY = 5.0
ENERGY_COST_PATTERN = 5.0

# --- Memory & Proactivity Config ---
MEMORY_DECAY_RATE = 0.05 
MEMORY_RETRIEVAL_LIMIT = 5
CURIOSITY_INTERVAL = 10.0        # Fill silence after ~10s of nothing
CALLBACK_INTERVAL = 45.0 
COMPRESSION_INTERVAL = 30.0 
POST_SPEECH_COOLDOWN = 5.0       # Brief breather after speaking (anti machine-gun)

# --- Director Config ---
DIRECTOR_PORT = 8002
DIRECTOR_HOST = "0.0.0.0"
WINDOW_IMMEDIATE = 10.0
WINDOW_RECENT = 30.0
WINDOW_BACKGROUND = 300.0 
SUMMARY_INTERVAL_SECONDS = 5.0
INTERJECTION_THRESHOLD = 0.75
NAMI_INTERJECT_URL = "http://localhost:8000/funnel/interject"
MEMORY_THRESHOLD = 0.85 
PRIMARY_MEMORY_COUNT = 5
PROFILES_DIR = "profiles"
DEFAULT_RELATIONSHIP_TIER = "Stranger"
DEFAULT_AFFINITY = 0
VALID_MOODS = ["Neutral", "Happy", "Annoyed", "Scared", "Horny", "Tired"]
DEFAULT_MOOD = "Neutral"
MOOD_WINDOW_SIZE = 5 
OLLAMA_TRIGGER_THRESHOLD = 0.5
OLLAMA_MODEL = 'llama3.2:latest'
OLLAMA_HOST = 'http://localhost:11434'

PROMPT_SERVICE_URL = "http://localhost:8001"
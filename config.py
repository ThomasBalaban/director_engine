# Save as: director_engine/config.py
from enum import Enum, auto

class InputSource(Enum):
    DIRECT_MICROPHONE = auto()
    TWITCH_MENTION = auto()
    AMBIENT_AUDIO = auto()
    VISUAL_CHANGE = auto()
    TWITCH_CHAT = auto()
    MICROPHONE = auto()
    BOT_TWITCH_REPLY = auto()
    SYSTEM_PATTERN = auto()

SOURCE_WEIGHTS = {
    InputSource.DIRECT_MICROPHONE: 0.8,
    InputSource.TWITCH_MENTION: 0.7,
    InputSource.AMBIENT_AUDIO: 0.3,
    InputSource.VISUAL_CHANGE: 0.4,
    InputSource.TWITCH_CHAT: 0.2,
    InputSource.MICROPHONE: 0.5,
    InputSource.BOT_TWITCH_REPLY: 0.0,
    InputSource.SYSTEM_PATTERN: 0.95
}

# --- [NEW] Conversation State Definition ---
class ConversationState(Enum):
    IDLE = "quiet, waiting for something to happen"
    ENGAGED = "active conversation happening"
    STORYTELLING = "user sharing experience"
    TEACHING = "explaining something to chat"
    FRUSTRATED = "user struggling"
    CELEBRATORY = "user succeeded at something"

# --- Director Config ---
DIRECTOR_PORT = 8002
DIRECTOR_HOST = "0.0.0.0"

# --- Context Hierarchy Config ---
WINDOW_IMMEDIATE = 10.0
WINDOW_RECENT = 30.0
WINDOW_BACKGROUND = 300.0 

# --- Summary Generation ---
SUMMARY_INTERVAL_SECONDS = 8.0

# --- Tier 2 "Interjection" Config ---
INTERJECTION_THRESHOLD = 0.9
NAMI_INTERJECT_URL = "http://localhost:8000/funnel/interject"

# --- MEMORY CONFIG ---
MEMORY_THRESHOLD = 0.85 
PRIMARY_MEMORY_COUNT = 5

# --- USER PROFILES ---
PROFILES_DIR = "profiles"
DEFAULT_RELATIONSHIP_TIER = "Stranger"
DEFAULT_AFFINITY = 0

# --- MOOD ENGINE ---
VALID_MOODS = ["Neutral", "Happy", "Annoyed", "Scared", "Horny", "Tired"]
DEFAULT_MOOD = "Neutral"
MOOD_WINDOW_SIZE = 5 

# --- Analyst (Ollama) Config ---
OLLAMA_TRIGGER_THRESHOLD = 0.5
OLLAMA_MODEL = 'llama3.2:latest'
OLLAMA_HOST = 'http://localhost:11434'
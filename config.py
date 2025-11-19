# Save as: director_engine/config.py
from enum import Enum, auto

# --- Copied from nami/input_systems/priority_core.py ---
class InputSource(Enum):
    DIRECT_MICROPHONE = auto()
    TWITCH_MENTION = auto()
    AMBIENT_AUDIO = auto()
    VISUAL_CHANGE = auto()
    TWITCH_CHAT = auto()
    MICROPHONE = auto()
    BOT_TWITCH_REPLY = auto()

# --- Copied from nami/input_systems/priority_core.py ---
SOURCE_WEIGHTS = {
    InputSource.DIRECT_MICROPHONE: 0.8,
    InputSource.TWITCH_MENTION: 0.7,
    InputSource.AMBIENT_AUDIO: 0.3,
    InputSource.VISUAL_CHANGE: 0.4,
    InputSource.TWITCH_CHAT: 0.2,
    InputSource.MICROPHONE: 0.5,
    InputSource.BOT_TWITCH_REPLY: 0.0
}

# --- New Configuration for the Director ---
DIRECTOR_PORT = 8002
DIRECTOR_HOST = "0.0.0.0"
CONTEXT_TIME_WINDOW_SECONDS = 30.0

# --- Summary Generation ---
SUMMARY_INTERVAL_SECONDS = 8.0

# --- Tier 2 "Interjection" Config ---
INTERJECTION_THRESHOLD = 0.9
NAMI_INTERJECT_URL = "http://localhost:8000/funnel/interject"

# --- MEMORY & PREDICTION CONFIG ---
# Events with a score above this are saved to "Long Term Memory"
# *** CHANGED TO 0.1 FOR TESTING (Normally 0.85) ***
MEMORY_THRESHOLD = 0.1 
# How many top memories to send to Brain 2 (Active/Primary memories)
PRIMARY_MEMORY_COUNT = 5

# --- Analyst (Ollama) Config ---
OLLAMA_TRIGGER_THRESHOLD = 0.5
OLLAMA_MODEL = 'llama3.2:latest'
OLLAMA_HOST = 'http://localhost:11434'
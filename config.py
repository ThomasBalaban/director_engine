# director_engine/config.py
from enum import Enum, auto

# --- Copied from nami/input_systems/priority_core.py ---
class InputSource(Enum):
    DIRECT_MICROPHONE = auto()
    TWITCH_MENTION = auto()
    AMBIENT_AUDIO = auto()
    VISUAL_CHANGE = auto()
    TWITCH_CHAT = auto()
    MICROPHONE = auto()

# --- Copied from nami/input_systems/priority_core.py ---
SOURCE_WEIGHTS = {
    InputSource.DIRECT_MICROPHONE: 0.8,
    InputSource.TWITCH_MENTION: 0.7,
    InputSource.AMBIENT_AUDIO: 0.3,
    InputSource.VISUAL_CHANGE: 0.4,
    InputSource.TWITCH_CHAT: 0.2,
    InputSource.MICROPHONE: 0.5
}

# --- New Configuration for the Director ---

# Port for this "Director" server
# CHANGED TO 8002 to match the hardcoded port in your Nami UI & TTS files
DIRECTOR_PORT = 8002
DIRECTOR_HOST = "0.0.0.0"

# How long to keep events in memory (in seconds)
CONTEXT_TIME_WINDOW_SECONDS = 30.0

# --- Tier 2 "Interjection" Config ---
# The "dumb" score an event must have to be considered for an interjection
INTERJECTION_THRESHOLD = 0.9
# The endpoint on Nami's (Brain 2) app that we call to *trigger* an interjection.
NAMI_INTERJECT_URL = "http://localhost:8000/funnel/interject" # Placeholder URL

# --- New "Analyst" (Ollama) Config ---
# The "dumb" score an event must have to trigger a "second look" from the LLM.
OLLAMA_TRIGGER_THRESHOLD = 0.5
# The Ollama model to use for analysis (must be small and fast)
OLLAMA_MODEL = 'llama3.2:latest'
# The host for your local Ollama server
OLLAMA_HOST = 'http://localhost:11434'
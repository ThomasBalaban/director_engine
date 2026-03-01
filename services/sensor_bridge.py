# director_engine/services/sensor_bridge.py
"""
Sensor Bridge — Receives data from the three microservice sockets and
routes each to the correct InputSource.

Socket → InputSource mapping (one-to-one, no ambiguity):
  vision_context       → VISUAL_CHANGE     (screen capture / vision service)
  audio_context        → AMBIENT_AUDIO     (stream audio service)
  spoken_word_context  → DIRECT_MICROPHONE (if "nami" heard) or MICROPHONE (mic service)

The old source-routing logic via metadata has been removed.
Each socket IS the source — no inference needed.
"""

import asyncio
import re
from pydantic import BaseModel
from config import InputSource
import shared
import core_logic
import services.prompt_client as prompt_client


class EventPayload(BaseModel):
    source_str: str
    text: str
    metadata: dict = {}
    username: str = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text(payload: dict, *keys) -> str:
    """Pull the first non-empty string from a list of candidate keys."""
    for key in keys:
        val = payload.get(key, "")
        if isinstance(val, str):
            val = val.strip()
        if val:
            return val
    return ""


def _is_silence(text: str) -> bool:
    return text.lower() in {"[silence]", "none", "n/a", "silence", ""}


# ---------------------------------------------------------------------------
# SensorBridge
# ---------------------------------------------------------------------------

class SensorBridge:
    """
    Registers Socket.IO handlers on the shared Hub connection.
    Each handler maps a single socket event to a single InputSource.
    """

    def __init__(self, event_callback=None):
        self.callback = event_callback or self._default_callback
        self.running = False
        self._register_handlers()

    async def _default_callback(self, source, text, metadata, username=None):
        await core_logic.process_engine_event(source, text, metadata, username)

    async def run(self):
        """Kept for backwards compatibility."""
        self.running = True
        while self.running:
            await asyncio.sleep(3600)

    # -----------------------------------------------------------------------
    # Handler registration
    # -----------------------------------------------------------------------

    def _register_handlers(self):

        # --- Hub lifecycle ---

        @shared.sio.on("connect")
        async def on_hub_connect():
            print("✅ [SensorBridge] Connected to Central Hub")

        # --- Vision microservice ---
        # Always VISUAL_CHANGE. The vision service owns screen analysis.
        # Supports plain text, XML-tagged blocks, and legacy label prefixes.

        @shared.sio.on("vision_context")
        async def handle_vision(payload: dict):
            text = _extract_text(payload, "context", "content", "text")
            if not text or _is_silence(text):
                return

            events = _parse_vision_payload(text)
            for content, metadata in events:
                if content:
                    await self.callback(
                        source=InputSource.VISUAL_CHANGE,
                        text=content,
                        metadata=metadata,
                    )

        # --- Stream audio microservice ---
        # Always AMBIENT_AUDIO. No source detection needed.

        @shared.sio.on("audio_context")
        async def handle_stream_audio(payload: dict):
            text = _extract_text(payload, "context", "text", "content")
            if not text or _is_silence(text):
                return

            metadata = {
                "confidence": payload.get("confidence", 1.0),
                "type": "stream_audio",
                "is_partial": payload.get("is_partial", False),
            }
            await self.callback(
                source=InputSource.AMBIENT_AUDIO,
                text=text,
                metadata=metadata,
            )

        # --- Microphone service ---
        # Routes to DIRECT_MICROPHONE if "nami" is heard, otherwise MICROPHONE.

        @shared.sio.on("spoken_word_context")
        async def handle_microphone(payload: dict):
            text = _extract_text(payload, "context", "text", "content")
            if not text or _is_silence(text):
                return

            source = (
                InputSource.DIRECT_MICROPHONE
                if "nami" in text.lower()
                else InputSource.MICROPHONE
            )

            metadata = {
                "confidence": payload.get("confidence", payload.get("metadata", {}).get("confidence", 1.0)),
                "type": "microphone_transcription",
                "is_partial": payload.get("is_partial", False),
            }
            await self.callback(
                source=source,
                text=text,
                metadata=metadata,
            )

        # --- Bot reply (from Nami back to director) ---

        @shared.sio.on("bot_reply")
        async def receive_bot_reply(payload: dict):
            try:
                reply_text = payload.get("reply", "")
                active_thread = shared.store.thread_manager.get_active_thread()
                if active_thread:
                    resolves = "?" not in reply_text
                    shared.store.thread_manager.track_nami_response(
                        text=reply_text, resolves_thread=resolves
                    )
                asyncio.create_task(prompt_client.notify_bot_response())
            except Exception as e:
                print(f"⚠️ [SensorBridge] Error handling bot_reply: {e}")

        # --- Generic event injection (manual / test harness) ---

        @shared.sio.on("event")
        async def ingest_event(payload: dict):
            try:
                model = EventPayload(**payload)
                await self.callback(
                    InputSource[model.source_str],
                    model.text,
                    model.metadata,
                    model.username,
                )
            except Exception as e:
                print(f"⚠️ [SensorBridge] Invalid event payload: {e}")


# ---------------------------------------------------------------------------
# Vision payload parser
# ---------------------------------------------------------------------------
# Kept as a module-level function so it can be unit-tested independently.

# Regex strips filler phrases that vision models commonly prepend
_AI_FILLER_RE = re.compile(
    r"^(Okay,?\s*(let'?s?\s*)?(describe|analyze|break\s*down)\s*(what'?s?\s*going\s*on\s*in\s*this\s*image|this\s*image[:\s]*)|"
    r"Here'?s?\s*the\s*screen\s*content\s*analysis[:\s]*|"
    r"Alright,?\s*here'?s?\s*the\s*rundown\s*of\s*what\s*I'?m\s*seeing[:\s]*|"
    r"Alright,?\s*let'?s?\s*(break\s*it\s*down|analyze\s*this\s*image)[:\s]*|"
    r"This\s*is\s*a\s*cartoon\s*(still|frame)[,\s]*)",
    flags=re.IGNORECASE,
)

# XML-tagged blocks: <visual_context>...</visual_context> etc.
_XML_RE = re.compile(r"<(\w+)>([\s\S]*?)<\/\1>")

# Legacy label prefixes: [VISUAL]: ... or [AUDIO]: ...
_LEGACY_RE = re.compile(
    r"\[(AUDIO|VISUAL|ACTION|CHARACTERS|DIALOGUE)\]:?\s*(.*?)(?=(?:\n|\[|$))",
    re.DOTALL | re.IGNORECASE,
)


def _parse_vision_payload(raw: str) -> list[tuple[str, dict]]:
    """
    Parse a vision service payload into a list of (text, metadata) tuples.

    Priority:
      1. XML-tagged blocks  → each tag becomes one VISUAL_CHANGE event
      2. Legacy label lines → each label becomes one VISUAL_CHANGE event
      3. Plain text         → single VISUAL_CHANGE event

    Note: audio tags inside vision payloads are IGNORED. The stream audio
    microservice is the authoritative source for AMBIENT_AUDIO events.
    """
    results: list[tuple[str, dict]] = []

    # 1. XML blocks
    xml_matches = list(_XML_RE.finditer(raw))
    if xml_matches:
        for m in xml_matches:
            tag = m.group(1).lower()
            content = m.group(2).strip()
            if not content or _is_silence(content):
                continue
            # Skip audio tags — that's the audio service's job
            if tag in {"audio_context", "audio"}:
                continue
            content = _AI_FILLER_RE.sub("", content).strip()
            results.append((content, {"type": "vision_xml", "xml_tag": tag}))
        if results:
            return results

    # 2. Legacy label prefixes
    legacy_matches = list(_LEGACY_RE.finditer(raw))
    if legacy_matches:
        for m in legacy_matches:
            tag = m.group(1).upper()
            content = m.group(2).strip()
            if not content or _is_silence(content):
                continue
            if tag == "AUDIO":
                continue  # Same as above — audio service owns this
            content = _AI_FILLER_RE.sub("", content).strip()
            results.append((content, {"type": "vision_legacy", "tag": tag}))
        if results:
            return results

    # 3. Plain text fallback
    cleaned = _AI_FILLER_RE.sub("", raw).strip()
    if cleaned and not _is_silence(cleaned):
        results.append((cleaned, {"type": "vision_plain"}))

    return results
# Save as: director_engine/context/context_compression.py
import asyncio
import json
import time
from typing import Any, Optional

import google.generativeai as genai  # type: ignore
from google.generativeai.types import HarmCategory, HarmBlockThreshold  # type: ignore

from context.context_store import ContextStore, EventItem
from config import OLLAMA_TIMEOUT_COMPRESS, COMPRESSION_INTERVAL, InputSource, GEMINI_API_KEY
from diagnostics import log_error

# Switched from Ollama to Gemini 2.0 Flash for narrative/ancient compression.
# Rationale:
#  - Same hallucination class as the summary path: loose prompt + sparse data
#    causes the model to invent quotes and details (notes.md 2026-05-21).
#    Concrete observed failure: user asked "what is the square root of pi",
#    Nami answered with the airhorn pie joke, narrative_log captured an
#    invented "silence and slow Uh" response that never happened.
#  - JSON-structured output lets us return a strict is_memorable boolean,
#    so the model can explicitly decline instead of always inventing
#    something to satisfy the "Memorable moment:" completion.
#  - BOT_TWITCH_REPLY now has its own source label so the LLM can tell
#    question from answer instead of guessing.

_gemini_compress_model: Optional[Any] = None


def _get_gemini_model() -> Optional[Any]:
    global _gemini_compress_model
    if _gemini_compress_model is None and GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            _gemini_compress_model = genai.GenerativeModel(
                'gemini-2.0-flash',
                safety_settings=safety_settings,
            )
            print("[Compressor] ✅ Gemini compression model initialized (gemini-2.0-flash)")
        except Exception as e:
            print(f"[Compressor] ❌ Failed to init Gemini compression model: {e}")
    return _gemini_compress_model


_SOURCE_LABELS = {
    InputSource.MICROPHONE: "Streamer said",
    InputSource.DIRECT_MICROPHONE: "Streamer said",
    InputSource.TWITCH_CHAT: "Chat",
    InputSource.TWITCH_MENTION: "Chat @Nami",
    InputSource.BOT_TWITCH_REPLY: "Nami replied",
    InputSource.VISUAL_CHANGE: "On screen",
    InputSource.AMBIENT_AUDIO: "Audio",
    InputSource.SYSTEM_PATTERN: "Event",
}


_RECENT_SCHEMA = {
    "type": "object",
    "properties": {
        "is_memorable": {"type": "boolean"},
        "narrative": {"type": "string"},
    },
    "required": ["is_memorable", "narrative"],
}

_ANCIENT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
    },
    "required": ["summary"],
}

class ContextCompressor:
    def __init__(self):
        self.last_compression_time = 0 
        self.last_ancient_compression_time = 0
        
    async def run_compression_cycle(self, store: ContextStore):
        now = time.time()
        
        # 1. Standard Narrative Compression (Recent Events -> Narrative Log)
        if now - self.last_compression_time >= COMPRESSION_INTERVAL:
            self.last_compression_time = now
            await self._compress_recent(store)

        # 2. Ancient History Compression (Narrative Log -> Ancient History Log)
        if now - self.last_ancient_compression_time >= 300: 
            self.last_ancient_compression_time = now
            await self._compress_ancient(store)
            
    async def _compress_recent(self, store: ContextStore):
        layers = store.get_all_events_for_summary()
        background_events = layers['background']
        if not background_events and len(layers['recent']) > 5:
            background_events = layers['recent']

        if not background_events:
            return

        model = _get_gemini_model()
        if model is None:
            print("[Compressor] ⚠️  Gemini unavailable — skipping narrative compression")
            return

        # Format events with explicit source labels. BOT_TWITCH_REPLY is now
        # labeled "Nami replied" so the LLM can match questions to actual
        # responses instead of inventing them.
        context_lines = []
        for e in background_events:
            label = _SOURCE_LABELS.get(e.source, "Event")
            context_lines.append(f"- [{label}] {e.text}")
        context_text = "\n".join(context_lines)

        prompt = (
            "You extract one memorable moment from a livestream for later callback.\n\n"
            f"Events (verbatim, in order):\n{context_text}\n\n"
            "RULES:\n"
            "1. The narrative MUST be derived from the events above. Do not invent\n"
            "   actions, quotes, reactions, characters, or details that are not\n"
            "   explicitly present in an event line.\n"
            "2. When referencing what someone said, use only what appears after\n"
            "   the source label. Never paraphrase a 'Nami replied' line into\n"
            "   something Nami didn't actually say. Never invent a Nami response\n"
            "   if no 'Nami replied' line is present.\n"
            "3. Do not infer emotions, tones, or delivery (\"flat tone\", \"awkward\n"
            "   pause\", \"sarcastically\") unless an event line literally states it.\n"
            "4. If the events show nothing genuinely memorable — routine chat,\n"
            "   ambient noise, idle game state, or just a single isolated line —\n"
            "   return is_memorable=false. Routine is acceptable; inventing is not.\n"
            "5. If is_memorable=true, the narrative is ONE sentence describing\n"
            "   what specifically happened, grounded in the event lines.\n"
            "6. If is_memorable=false, narrative MUST be the empty string.\n\n"
            "Return a JSON object with: is_memorable (boolean), narrative (string)."
        )

        try:
            response = await asyncio.wait_for(
                model.generate_content_async(
                    prompt,
                    generation_config={
                        "temperature": 0.1,
                        "response_mime_type": "application/json",
                        "response_schema": _RECENT_SCHEMA,
                    },
                ),
                timeout=OLLAMA_TIMEOUT_COMPRESS,
            )
        except asyncio.TimeoutError:
            print(f"⏱️  [Compressor] Recent-compress timed out after {OLLAMA_TIMEOUT_COMPRESS}s")
            log_error("compressor.recent", "gemini timeout", timeout_s=OLLAMA_TIMEOUT_COMPRESS)
            return
        except Exception as e:
            kind = type(e).__name__
            detail = str(e) or repr(e)
            print(f"❌ [Compressor] Error ({kind}): {detail}")
            log_error("compressor.recent", "gemini error", exc=e)
            return

        try:
            data = json.loads(response.text)
        except (json.JSONDecodeError, AttributeError, ValueError) as e:
            print(f"[Compressor] Recent JSON parse failed: {e}")
            return

        if not data.get("is_memorable"):
            print("📖 [Compressor] No memorable moment found, skipping.")
            return

        narrative = (data.get("narrative") or "").strip()
        if narrative.startswith('"') and narrative.endswith('"'):
            narrative = narrative[1:-1]
        if narrative.startswith("'") and narrative.endswith("'"):
            narrative = narrative[1:-1]

        if narrative and len(narrative) > 10:
            store.add_narrative_segment(narrative)
            print(f"📖 [Compressor] Added: {narrative[:60]}...")
        else:
            print(f"📖 [Compressor] is_memorable=true but narrative too short, skipping: {narrative!r}")

    async def _compress_ancient(self, store: ContextStore):
        """
        Takes the oldest chunk of the narrative log and compresses it into a single 'Ancient' summary.
        """
        with store.lock:
            if len(store.narrative_log) < 10:
                return
            chunk_to_compress = store.narrative_log[:5]

        print(f"📚 [Compressor] Compressing {len(chunk_to_compress)} narrative segments into Ancient History...")

        model = _get_gemini_model()
        if model is None:
            print("[Compressor] ⚠️  Gemini unavailable — skipping ancient compression")
            return

        context_text = "\n".join(f"- {seg}" for seg in chunk_to_compress)

        prompt = (
            "Combine these earlier memorable moments into one sentence for long-term recall.\n\n"
            f"Moments:\n{context_text}\n\n"
            "RULES:\n"
            "1. Use ONLY information present in the moments above. Do not add new\n"
            "   details, characters, jokes, or events.\n"
            "2. Preserve names and specifics that appear verbatim.\n"
            "3. One sentence. No quotes around it. No preamble like 'Earlier,'.\n\n"
            "Return JSON with: summary (string)."
        )

        try:
            response = await asyncio.wait_for(
                model.generate_content_async(
                    prompt,
                    generation_config={
                        "temperature": 0.1,
                        "response_mime_type": "application/json",
                        "response_schema": _ANCIENT_SCHEMA,
                    },
                ),
                timeout=OLLAMA_TIMEOUT_COMPRESS,
            )
        except asyncio.TimeoutError:
            print(f"⏱️  [Compressor] Ancient-compress timed out after {OLLAMA_TIMEOUT_COMPRESS}s")
            log_error("compressor.ancient", "gemini timeout", timeout_s=OLLAMA_TIMEOUT_COMPRESS)
            return
        except Exception as e:
            kind = type(e).__name__
            detail = str(e) or repr(e)
            print(f"❌ [Compressor] Ancient compression error ({kind}): {detail}")
            log_error("compressor.ancient", "gemini error", exc=e)
            return

        try:
            data = json.loads(response.text)
        except (json.JSONDecodeError, AttributeError, ValueError) as e:
            print(f"[Compressor] Ancient JSON parse failed: {e}")
            return

        ancient_summary = (data.get("summary") or "").strip()
        if ancient_summary.startswith('"') and ancient_summary.endswith('"'):
            ancient_summary = ancient_summary[1:-1]

        if ancient_summary and len(ancient_summary) > 15:
            with store.lock:
                store.narrative_log = store.narrative_log[5:]
                store.archive_ancient_history(ancient_summary)
            print(f"📜 [History] Archived: {ancient_summary[:60]}...")
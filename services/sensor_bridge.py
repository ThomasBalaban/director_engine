# director_engine/services/sensor_bridge.py
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

class SensorBridge:
    """
    In the Star Topology, SensorBridge is no longer a direct WebSocket client.
    Instead, it registers event handlers on the shared Socket.IO client to process
    incoming data broadcasted from the Central Hub.
    """
    def __init__(self, event_callback=None):
        self.callback = event_callback or self._default_callback
        self.running = False
        self._register_handlers()

    async def _default_callback(self, source, text, metadata, username=None):
        await core_logic.process_engine_event(source, text, metadata, username)

    async def run(self):
        """Kept for backwards compatibility if process_manager calls bridge.run()"""
        self.running = True
        while self.running:
            await asyncio.sleep(3600)
            
    def _register_handlers(self):
        @shared.sio.on("connect")
        async def on_hub_connect():
            print("✅ [SensorBridge] Connected to Central Hub")

        @shared.sio.on("vision_context")
        async def handle_vision_context(payload: dict):
            # Desktop Monitor sends text in 'context' or 'content' key
            content = payload.get("context", payload.get("content", "")).strip()
            if not content:
                return
            print(f"👁️ [SensorBridge] vision_context received: {len(content)} chars")
            await self._parse_gemini_content(content)

        @shared.sio.on("audio_context")
        async def handle_audio_context(payload: dict):
            await self._parse_whisper_content(payload)

        @shared.sio.on("event")
        async def ingest_event(payload: dict):
            try:
                model = EventPayload(**payload)
                await self.callback(
                    InputSource[model.source_str],
                    model.text,
                    model.metadata,
                    model.username
                )
            except Exception as e: 
                print(f"⚠️ [SensorBridge] Invalid event payload: {e}")

        @shared.sio.on("bot_reply")
        async def receive_bot_reply(payload: dict):
            try:
                reply_text = payload.get('reply', '')
                active_thread = shared.store.thread_manager.get_active_thread()
                if active_thread:
                    resolves = "?" not in reply_text
                    shared.store.thread_manager.track_nami_response(text=reply_text, resolves_thread=resolves)
                shared.emit_bot_reply(reply_text, payload.get('prompt', ''), payload.get('is_censored', False))
                asyncio.create_task(prompt_client.notify_bot_response())
            except Exception as e: 
                print(f"⚠️ [SensorBridge] Error handling bot_reply: {e}")

    # --- PARSERS ---

    async def _parse_whisper_content(self, data):
        text = data.get("context", data.get("text", "")).strip()
        if not text:
            return

        metadata = data.get("metadata", {})
        source_str = metadata.get("source", data.get("source", "desktop"))
        
        # Only interrupt current thought if "nami" is specifically mentioned
        if source_str == "microphone":
            if "nami" in text.lower():
                source = InputSource.DIRECT_MICROPHONE
            else:
                source = InputSource.MICROPHONE
        else:
            source = InputSource.AMBIENT_AUDIO

        meta = {
            "confidence": metadata.get("confidence", 1.0), 
            "type": "fast_transcription",
            "is_partial": data.get("is_partial", False) 
        }
        
        await self.callback(source=source, text=text, metadata=meta)

    async def _parse_gemini_content(self, text):
        clean_text = text.strip()
        if not clean_text or clean_text.lower() in ["[silence]", "none", "n/a", "silence"]: 
            return

        # 1. XML Pattern parsing
        xml_pattern = r"<(\w+)>([\s\S]*?)<\/\1>"
        matches = list(re.finditer(xml_pattern, clean_text))
        
        if matches:
            for match in matches:
                tag = match.group(1).lower()
                content = match.group(2).strip()
                if not content or content.lower() in ["[silence]", "none"]: 
                    continue

                source = InputSource.VISUAL_CHANGE
                if tag == "audio_context":
                    source = InputSource.AMBIENT_AUDIO
                
                await self.callback(
                    source=source,
                    text=content,
                    metadata={"raw_tag": tag.upper(), "confidence": 1.0, "type": "hub_vision_xml", "xml_tag": tag}
                )
            return

        # 2. Legacy Pattern parsing
        legacy_pattern = r"\[(AUDIO|VISUAL|ACTION|CHARACTERS|DIALOGUE|AUDIO/DIALOGUE)\]:?\s*(.*?)(?=(?:\n\d+\.|\[|$))"
        legacy_matches = list(re.finditer(legacy_pattern, clean_text, re.DOTALL | re.IGNORECASE))
        
        if legacy_matches:
            for match in legacy_matches:
                tag = match.group(1).upper()
                content = match.group(2).strip()
                if not content or content == "[SILENCE]": continue

                source = InputSource.VISUAL_CHANGE
                if "AUDIO" in tag:
                    source = InputSource.AMBIENT_AUDIO
                
                await self.callback(
                    source=source,
                    text=content,
                    metadata={"raw_tag": tag, "confidence": 1.0, "type": "hub_vision_legacy"}
                )
            return

        # 3. Fallback: Plain Text Narrative
        await self.callback(
            source=InputSource.VISUAL_CHANGE,
            text=clean_text,
            metadata={"type": "hub_vision_plain"}
        )
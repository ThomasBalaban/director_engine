import asyncio
import websockets
import json
import re
from config import InputSource

class SensorBridge:
    def __init__(self, vision_uri="ws://localhost:8003", hearing_uri="ws://localhost:8003", event_callback=None):
        self.vision_uri = vision_uri
        self.hearing_uri = hearing_uri
        self.callback = event_callback
        self.running = False

    async def run(self):
        self.running = True
        print(f"🔌 [Bridge] Initializing Dual-Sensors...")
        
        # Create two parallel tasks for the two senses
        vision_task = asyncio.create_task(self._vision_loop())
        hearing_task = asyncio.create_task(self._hearing_loop())
        
        await asyncio.gather(vision_task, hearing_task)

    # --- LOOP 1: VISION (Gemini) ---
    async def _vision_loop(self):
        while self.running:
            try:
                print(f"👁️ [Bridge] Connecting to Vision (Port 8003)...")
                async with websockets.connect(self.vision_uri) as ws:
                    print("👁️ [Bridge] Vision Connected!")
                    async for message in ws:
                        data = json.loads(message)
                        if data.get("type") == "text_update":
                            await self._parse_gemini_content(data.get("content", ""))
                            
            except (ConnectionRefusedError, OSError):
                print("⚠️ [Bridge] Vision lost. Retrying in 5s...")
                await asyncio.sleep(5)
            except Exception as e:
                print(f"❌ [Bridge] Vision Error: {e}")
                await asyncio.sleep(5)

    # --- LOOP 2: HEARING (Whisper) ---
    async def _hearing_loop(self):
        while self.running:
            try:
                print(f"👂 [Bridge] Connecting to Hearing (Port 8003)...")
                async with websockets.connect(self.hearing_uri) as ws:
                    print("👂 [Bridge] Hearing Connected!")
                    async for message in ws:
                        # audio_mon sends simple JSON: { "source": "desktop", "text": "..." }
                        data = json.loads(message)
                        await self._parse_whisper_content(data)
                        
            except (ConnectionRefusedError, OSError):
                # Whisper might not be running yet, quiet retry
                await asyncio.sleep(5)
            except Exception as e:
                print(f"❌ [Bridge] Hearing Error: {e}")
                await asyncio.sleep(5)

    # --- PARSERS ---

    async def _parse_whisper_content(self, data):
        """Handle raw subtitles from the fast local transcriber."""
        if not self.callback: return
        
        text = data.get("text", "").strip()
        source_str = data.get("source", "desktop")
        
        if not text: return

        # Map to Director InputSource
        if source_str == "microphone":
            # This is YOU (The User)
            source = InputSource.DIRECT_MICROPHONE
        else:
            # This is GAME/SYSTEM Audio
            source = InputSource.AMBIENT_AUDIO

        await self.callback(
            source=source,
            text=text,
            metadata={"confidence": data.get("confidence", 1.0), "type": "fast_transcription"}
        )

    async def _parse_gemini_content(self, text):
        """
        Handle deep context from the slow vision AI.
        Parses XML tags from the updated system prompt:
        <summary>, <scene_and_entities>, <characters_and_appeal>, 
        <text_and_ui>, <audio_context>, <actionable_events>
        """
        if not self.callback: return

        # Regex to match XML-style tags <tag>content</tag>
        # [\s\S]*? ensures we match across newlines non-greedily
        xml_pattern = r"<(\w+)>([\s\S]*?)<\/\1>"
        matches = list(re.finditer(xml_pattern, text))
        
        found_xml = False
        
        for match in matches:
            found_xml = True
            tag = match.group(1).lower()
            content = match.group(2).strip()
            
            # Skip empty or silence markers
            if not content or content.lower() in ["[silence]", "none", "n/a", "silence"]: 
                continue

            # Map tags to Director InputSources
            source = InputSource.VISUAL_CHANGE
            
            if tag == "audio_context":
                source = InputSource.AMBIENT_AUDIO
            elif tag in ["summary", "scene_and_entities", "characters_and_appeal", "text_and_ui", "actionable_events"]:
                source = InputSource.VISUAL_CHANGE
            
            # Create readable tags for metadata/logs
            raw_tag_map = {
                "summary": "SUMMARY",
                "scene_and_entities": "SCENE",
                "characters_and_appeal": "CHARACTERS",
                "text_and_ui": "UI_TEXT",
                "audio_context": "AUDIO_AI",
                "actionable_events": "ACTION"
            }
            display_tag = raw_tag_map.get(tag, tag.upper())

            await self.callback(
                source=source,
                text=content,
                metadata={
                    "raw_tag": display_tag, 
                    "confidence": 1.0, 
                    "type": "gemini_analysis", 
                    "xml_tag": tag
                }
            )

        # Fallback: If no XML tags are found, try the legacy bracket format or raw text
        if not found_xml and text.strip():
            legacy_pattern = r"\[(AUDIO|VISUAL|ACTION|CHARACTERS|DIALOGUE|AUDIO/DIALOGUE)\]:?\s*(.*?)(?=(?:\n\d+\.|\[|$))"
            legacy_matches = list(re.finditer(legacy_pattern, text, re.DOTALL | re.IGNORECASE))
            
            if legacy_matches:
                for match in legacy_matches:
                    tag = match.group(1).upper()
                    content = match.group(2).strip()
                    
                    if not content or content == "[SILENCE]": continue

                    if "AUDIO" in tag or "DIALOGUE" in tag:
                        source = InputSource.AMBIENT_AUDIO
                    else:
                        source = InputSource.VISUAL_CHANGE
                    
                    await self.callback(
                        source=source,
                        text=content,
                        metadata={"raw_tag": tag, "confidence": 1.0, "type": "gemini_analysis_legacy"}
                    )
            # Extremely raw fallback (if it's just plain text)
            elif len(text) > 5:
                 await self.callback(
                    source=InputSource.VISUAL_CHANGE,
                    text=text.strip(),
                    metadata={"raw_tag": "RAW", "confidence": 0.5, "type": "gemini_raw"}
                )
# Save as: director_engine/services/sensor_bridge.py
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
        self.message_count = 0  # Debug counter

    async def run(self):
        self.running = True
        print(f"🔌 [Bridge] Initializing Dual-Sensors...")
        
        # Wait for the vision app to start up
        print(f"🔌 [Bridge] Waiting 5s for Vision subsystem to initialize...")
        await asyncio.sleep(5)
        
        await self._unified_loop()

    async def _unified_loop(self):
        """Single connection that handles both vision and hearing."""
        retry_count = 0
        while self.running:
            try:
                print(f"🔌 [Bridge] Connecting to Sensor Server (Port 8003)...")
                async with websockets.connect(self.vision_uri, ping_interval=20, ping_timeout=10) as ws:
                    print("🔌 [Bridge] Connected!")
                    retry_count = 0
                    
                    async for message in ws:
                        if not self.running:
                            break
                        
                        self.message_count += 1
                        
                        try:
                            data = json.loads(message)
                            msg_type = data.get("type", "unknown")
                            
                            # DEBUG: Print ALL messages (remove heartbeat filter temporarily)
                            print(f"📨 [Bridge #{self.message_count}] type={msg_type}, keys={list(data.keys())}")
                            
                            # Handle vision updates
                            if msg_type == "text_update":
                                content = data.get("content", "")
                                print(f"👁️ [Bridge] Vision text_update received: {len(content)} chars")
                                if content:
                                    print(f"👁️ [Bridge] Content preview: {content[:100]}...")
                                    await self._parse_gemini_content(content)
                            
                            # Handle transcripts
                            elif msg_type in ["transcript", "partial_transcript"]:
                                text = data.get('text', '')
                                source = data.get('source', 'unknown')
                                print(f"👂 [Bridge] Transcript received: source={source}, text={text[:60] if text else '(empty)'}...")
                                await self._parse_whisper_content(data)
                            
                            # DEBUG: Log unhandled message types
                            elif msg_type != "heartbeat":
                                print(f"❓ [Bridge] Unhandled message type: {msg_type}")
                                print(f"   Data: {str(data)[:200]}")
                                
                        except json.JSONDecodeError as e:
                            print(f"⚠️ [Bridge] JSON decode error: {e}")
                            print(f"   Raw message: {message[:200]}")
                        except Exception as e:
                            print(f"⚠️ [Bridge] Message handling error: {e}")
                            import traceback
                            traceback.print_exc()
                            
            except asyncio.CancelledError:
                print("🔌 [Bridge] Loop cancelled")
                break
            except (ConnectionRefusedError, OSError) as e:
                retry_count += 1
                wait_time = min(5 * retry_count, 30)
                print(f"⚠️ [Bridge] Connection failed (attempt {retry_count}): {e}")
                print(f"⚠️ [Bridge] Retrying in {wait_time}s...")
                try:
                    await asyncio.sleep(wait_time)
                except asyncio.CancelledError:
                    break
            except Exception as e:
                print(f"❌ [Bridge] Error: {e}")
                import traceback
                traceback.print_exc()
                try:
                    await asyncio.sleep(5)
                except asyncio.CancelledError:
                    break

    # --- PARSERS ---

    async def _parse_whisper_content(self, data):
        """Handle raw subtitles from the fast local transcriber."""
        if not self.callback:
            print("⚠️ [Bridge] No callback set!")
            return
        
        text = data.get("text", "").strip()
        source_str = data.get("source", "desktop")
        
        if not text:
            print("⚠️ [Bridge] Empty transcript text, skipping")
            return

        # Map to Director InputSource
        if source_str == "microphone":
            source = InputSource.DIRECT_MICROPHONE
        else:
            source = InputSource.AMBIENT_AUDIO

        meta = {
            "confidence": data.get("confidence", 1.0), 
            "type": "fast_transcription",
            "is_partial": data.get("is_partial", False) 
        }

        print(f"✅ [Bridge] Calling callback with: source={source.name}, text={text[:40]}...")
        
        try:
            await self.callback(
                source=source,
                text=text,
                metadata=meta
            )
            print(f"✅ [Bridge] Callback completed successfully")
        except Exception as e:
            print(f"❌ [Bridge] Callback error: {e}")
            import traceback
            traceback.print_exc()

    async def _parse_gemini_content(self, text):
        """Handle deep context from the slow vision AI."""
        if not self.callback:
            print("⚠️ [Bridge] No callback set for vision!")
            return
        
        clean_text = text.strip()
        if not clean_text or clean_text.lower() in ["[silence]", "none", "n/a", "silence"]: 
            print(f"⚠️ [Bridge] Skipping empty/silence vision text")
            return

        # 1. Try XML Parsing
        xml_pattern = r"<(\w+)>([\s\S]*?)<\/\1>"
        matches = list(re.finditer(xml_pattern, clean_text))
        
        if matches:
            print(f"📋 [Bridge] Found {len(matches)} XML tags in vision content")
            for match in matches:
                tag = match.group(1).lower()
                content = match.group(2).strip()
                
                if not content or content.lower() in ["[silence]", "none"]: 
                    continue

                source = InputSource.VISUAL_CHANGE
                
                if tag == "audio_context":
                    source = InputSource.AMBIENT_AUDIO
                
                print(f"✅ [Bridge] Sending XML vision to callback: tag={tag}, source={source.name}")
                
                try:
                    await self.callback(
                        source=source,
                        text=content,
                        metadata={
                            "raw_tag": tag.upper(), 
                            "confidence": 1.0, 
                            "type": "gemini_analysis", 
                            "xml_tag": tag
                        }
                    )
                    print(f"✅ [Bridge] Vision callback completed for tag={tag}")
                except Exception as e:
                    print(f"❌ [Bridge] Vision callback error: {e}")
                    import traceback
                    traceback.print_exc()
            return

        # 2. Try Legacy Parsing
        legacy_pattern = r"\[(AUDIO|VISUAL|ACTION|CHARACTERS|DIALOGUE|AUDIO/DIALOGUE)\]:?\s*(.*?)(?=(?:\n\d+\.|\[|$))"
        legacy_matches = list(re.finditer(legacy_pattern, clean_text, re.DOTALL | re.IGNORECASE))
        
        if legacy_matches:
            print(f"📋 [Bridge] Found {len(legacy_matches)} legacy tags in vision content")
            for match in legacy_matches:
                tag = match.group(1).upper()
                content = match.group(2).strip()
                if not content or content == "[SILENCE]": continue

                source = InputSource.VISUAL_CHANGE
                if "AUDIO" in tag:
                    source = InputSource.AMBIENT_AUDIO
                
                print(f"✅ [Bridge] Sending legacy vision to callback: tag={tag}")
                
                try:
                    await self.callback(
                        source=source,
                        text=content,
                        metadata={"raw_tag": tag, "confidence": 1.0, "type": "gemini_analysis_legacy"}
                    )
                except Exception as e:
                    print(f"❌ [Bridge] Legacy callback error: {e}")
            return

        # 3. Fallback: Plain Text
        print(f"✅ [Bridge] Sending plain vision to callback: {len(clean_text)} chars")
        
        try:
            await self.callback(
                source=InputSource.VISUAL_CHANGE,
                text=clean_text,
                metadata={
                    "raw_tag": "VISUAL", 
                    "confidence": 1.0, 
                    "type": "gemini_narrative"
                }
            )
            print(f"✅ [Bridge] Plain vision callback completed")
        except Exception as e:
            print(f"❌ [Bridge] Plain vision callback error: {e}")
            import traceback
            traceback.print_exc()
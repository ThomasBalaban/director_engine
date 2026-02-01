# Save as: director_engine/context/context_compression.py
import time
import ollama
from context.context_store import ContextStore, EventItem
from config import OLLAMA_MODEL, COMPRESSION_INTERVAL, InputSource

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
        
        # Build richer context with source info
        context_lines = []
        for e in background_events:
            source_label = {
                InputSource.MICROPHONE: "Streamer said",
                InputSource.DIRECT_MICROPHONE: "Streamer said",
                InputSource.TWITCH_CHAT: "Chat",
                InputSource.TWITCH_MENTION: "Chat @Nami",
                InputSource.VISUAL_CHANGE: "On screen",
                InputSource.AMBIENT_AUDIO: "Audio",
                InputSource.SYSTEM_PATTERN: "Event",
            }.get(e.source, "Event")
            
            context_lines.append(f"- [{source_label}] {e.text}")
        
        context_text = "\n".join(context_lines)
        
        # --- IMPROVED PROMPT: Extract specific memorable moments ---
        prompt = (
            f"You are extracting memorable moments from a livestream for later callback.\n\n"
            f"Stream events:\n{context_text}\n\n"
            f"Extract ONE specific memorable moment that Nami (the AI) could reference later.\n"
            f"Format: '[What happened] - [The funny/notable detail]'\n\n"
            f"Good examples:\n"
            f"- 'Otter rage-quit after spinning a 1 three times in a row'\n"
            f"- 'Chat convinced Otter to pick the Space Janitor job'\n"
            f"- 'Otter celebrated winning then immediately got hit with 50k taxes'\n"
            f"- 'Someone in chat called Otter a skill issue and he malded'\n\n"
            f"Bad examples (too vague/meta):\n"
            f"- 'The streamer played a game and had reactions'\n"
            f"- 'Various events occurred during gameplay'\n"
            f"- 'The user engaged with chat while gaming'\n\n"
            f"Be specific. Use names/usernames when available. Include the emotion or punchline.\n"
            f"If nothing memorable happened, write: [SKIP]\n\n"
            f"Memorable moment:"
        )

        try:
            client = ollama.AsyncClient()
            response = await client.generate(model=OLLAMA_MODEL, prompt=prompt)
            narrative = response['response'].strip()
            
            # Skip if nothing memorable
            if "[SKIP]" in narrative.upper() or not narrative:
                print(f"📖 [Compressor] No memorable moment found, skipping.")
                return
            
            # Post-processing cleanup
            # Remove common AI preambles
            cleanup_phrases = [
                "here is", "here's", "the memorable moment is", 
                "memorable moment:", "one memorable moment"
            ]
            narrative_lower = narrative.lower()
            for phrase in cleanup_phrases:
                if narrative_lower.startswith(phrase):
                    narrative = narrative[len(phrase):].strip()
                    if narrative.startswith(":"):
                        narrative = narrative[1:].strip()
                    break
            
            # Remove quotes if wrapped
            if narrative.startswith('"') and narrative.endswith('"'):
                narrative = narrative[1:-1]
            if narrative.startswith("'") and narrative.endswith("'"):
                narrative = narrative[1:-1]
                
            if narrative and len(narrative) > 10:
                store.add_narrative_segment(narrative)
                print(f"📖 [Compressor] Added: {narrative[:60]}...")
            else:
                print(f"📖 [Compressor] Result too short, skipping: {narrative}")
                
        except Exception as e:
            print(f"❌ [Compressor] Error: {e}")

    async def _compress_ancient(self, store: ContextStore):
        """
        Takes the oldest chunk of the narrative log and compresses it into a single 'Ancient' summary.
        """
        with store.lock:
            if len(store.narrative_log) < 10:
                return

            chunk_to_compress = store.narrative_log[:5]
            
        print(f"📚 [Compressor] Compressing {len(chunk_to_compress)} narrative segments into Ancient History...")
        
        context_text = "\n".join([f"- {seg}" for seg in chunk_to_compress])
        
        # --- IMPROVED PROMPT for ancient history ---
        prompt = (
            f"These are memorable moments from earlier in a livestream:\n"
            f"{context_text}\n\n"
            f"Combine these into ONE sentence that captures the key callbacks.\n"
            f"Focus on: names, specific events, running jokes, or notable fails/wins.\n"
            f"Example: 'Earlier, Otter lost 3 games in a row, chat roasted him, and he blamed the RNG.'\n\n"
            f"Combined summary:"
        )

        try:
            client = ollama.AsyncClient()
            response = await client.generate(model=OLLAMA_MODEL, prompt=prompt)
            ancient_summary = response['response'].strip()
            
            # Clean up
            if ancient_summary.lower().startswith("combined summary:"):
                ancient_summary = ancient_summary[17:].strip()
            if ancient_summary.startswith('"') and ancient_summary.endswith('"'):
                ancient_summary = ancient_summary[1:-1]
            
            if ancient_summary and len(ancient_summary) > 15:
                with store.lock:
                    store.narrative_log = store.narrative_log[5:]
                    store.archive_ancient_history(ancient_summary)
                print(f"📜 [History] Archived: {ancient_summary[:60]}...")
        except Exception as e:
            print(f"❌ [Compressor] Ancient compression error: {e}")
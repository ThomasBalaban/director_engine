# Save as: director_engine/context_compression.py
import time
import ollama
from context_store import ContextStore, EventItem
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

        # 2. [REQ 10] Ancient History Compression (Narrative Log -> Ancient History Log)
        # Run this less frequently, e.g., every 5 minutes
        if now - self.last_ancient_compression_time >= 300: 
            self.last_ancient_compression_time = now
            await self._compress_ancient(store)
            
    async def _compress_recent(self, store: ContextStore):
        layers = store.get_all_events_for_summary()
        background_events = layers['background']
        if not background_events and len(layers['recent']) > 5:
             background_events = layers['recent']

        if not background_events: return
            
        context_text = "\n".join([f"- [{e.source.name}] {e.text}" for e in background_events])
        prompt = f"Summarize these stream events into one narrative sentence:\n{context_text}\nNarrative Sentence:"

        try:
            client = ollama.AsyncClient()
            response = await client.generate(model=OLLAMA_MODEL, prompt=prompt)
            narrative = response['response'].strip()
            if narrative:
                store.add_narrative_segment(narrative)
        except Exception as e:
            print(f"❌ [Compressor] Error: {e}")

    async def _compress_ancient(self, store: ContextStore):
        """
        Takes the oldest chunk of the narrative log and compresses it into a single 'Ancient' summary.
        """
        with store.lock:
            # Trigger only if we have a significant backlog (e.g., > 10 segments)
            if len(store.narrative_log) < 10:
                return

            # Take the oldest 5 segments
            chunk_to_compress = store.narrative_log[:5]
            
        print(f"📚 [Compressor] Compressing {len(chunk_to_compress)} narrative segments into Ancient History...")
        
        context_text = "\n".join([f"- {seg}" for seg in chunk_to_compress])
        prompt = f"Condense these 5 events into a single, high-level historical summary sentence:\n{context_text}\nSummary:"

        try:
            client = ollama.AsyncClient()
            response = await client.generate(model=OLLAMA_MODEL, prompt=prompt)
            ancient_summary = response['response'].strip()
            
            if ancient_summary:
                with store.lock:
                    # Remove the processed segments
                    store.narrative_log = store.narrative_log[5:]
                    store.archive_ancient_history(ancient_summary)
        except Exception as e:
            print(f"❌ [Compressor] Ancient compression error: {e}")
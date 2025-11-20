# Save as: director_engine/context_compression.py
import time
import ollama
from context_store import ContextStore, EventItem
from config import OLLAMA_MODEL, COMPRESSION_INTERVAL, InputSource

class ContextCompressor:
    def __init__(self):
        # [FIX] Set to 0 so it runs on the first available cycle
        self.last_compression_time = 0 
        
    async def run_compression_cycle(self, store: ContextStore):
        now = time.time()
        if now - self.last_compression_time < COMPRESSION_INTERVAL:
            return
            
        self.last_compression_time = now
        
        layers = store.get_all_events_for_summary()
        # We combine recent and background to ensure we have *something* to summarize on startup
        background_events = layers['background']
        
        # If background is empty (start up), peek at 'recent' if it has enough items
        if not background_events and len(layers['recent']) > 5:
             background_events = layers['recent']

        if not background_events:
            return
            
        print(f"📚 [Compressor] Compressing {len(background_events)} events...")
        
        context_text = "\n".join([f"- [{e.source.name}] {e.text}" for e in background_events])
        
        prompt = f"""
Review these events from the last few minutes of a livestream:
{context_text}

Synthesize them into ONE single narrative sentence describing what happened. 
Focus on the user's actions, major conversation topics, or key game events. 
Ignore noise.

Example: "The user struggled with the water temple boss while chatting about pizza toppings."
Narrative Sentence:
"""

        try:
            client = ollama.AsyncClient()
            response = await client.generate(model=OLLAMA_MODEL, prompt=prompt)
            narrative = response['response'].strip()
            
            if narrative:
                store.add_narrative_segment(narrative)
                
        except Exception as e:
            print(f"❌ [Compressor] Error: {e}")
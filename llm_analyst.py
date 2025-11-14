# Save as: director_engine/llm_analyst.py
import ollama
import json
import httpx
# --- THIS IS THE FIX ---
from config import (
    OLLAMA_MODEL, 
    OLLAMA_HOST, 
    NAMI_INTERJECT_URL, 
    INTERJECTION_THRESHOLD, 
    InputSource # <--- THIS IMPORT WAS MISSING
)
# ----------------------
from context_store import ContextStore, EventItem
from typing import List

http_client: httpx.AsyncClient | None = None

try:
    ollama_client = ollama.Client(host=OLLAMA_HOST)
    ollama_client.list()
    print(f"[Analyst] Successfully connected to Ollama at {OLLAMA_HOST}")
except Exception as e:
    print(f"[Analyst] ERROR: Could not connect to Ollama at {OLLAMA_HOST}. Is it running?")
    ollama_client = None


def build_analysis_prompt(text: str) -> str:
    """Builds the prompt for the analyst LLM to force JSON output."""
    return f"""
You are an event analyzer. Your only job is to rate the "interestingness" of the
following event for a streamer's AI assistant.
Respond ONLY with a single, valid JSON object: {{"score": <a_float_from_0.0_to_1.0>}}

Event: "{text}"
"""

def parse_llm_response(response_text: str) -> float | None:
    """Safely parses the JSON string response from the LLM."""
    try:
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start == -1 or end == 0:
            raise json.JSONDecodeError("No JSON object found", response_text, 0)
            
        json_str = response_text[start:end]
        data = json.loads(json_str)
        score = data.get("score")
        if isinstance(score, (float, int)):
            return max(0.0, min(float(score), 1.0)) # Clamp score between 0 and 1
    except json.JSONDecodeError:
        print(f"[Analyst] LLM response was not valid JSON: {response_text}")
    except Exception as e:
        print(f"[Analyst] Error parsing LLM response: {e}")
    return None


async def analyze_and_update_event(event: EventItem, store: ContextStore):
    """
    (Background Task) Calls Ollama to get an "intelligent" score and updates the store.
    """
    if not ollama_client: return

    prompt = build_analysis_prompt(event.text)
    
    try:
        # print(f"[Analyst] Analyzing event {event.id} (Sieve score: {event.score:.2f})...")
        response = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            options={"temperature": 0.2},
            format='json'
        )
        response_text = response['message']['content']
        new_score = parse_llm_response(response_text)
        
        if new_score is not None:
            if new_score > event.score:
                print(f"[Analyst] ✅ Score UPDATED: {event.id} from {event.score:.2f} -> {new_score:.2f}")
                store.update_event_score(event.id, new_score)
                
                if new_score >= INTERJECTION_THRESHOLD:
                    print(f"[Analyst] Tier 2 Interjection! LLM score {new_score:.2f} >= {INTERJECTION_THRESHOLD}")
                    await trigger_nami_interjection(event, new_score)
        
    except Exception as e:
        print(f"[Analyst] ERROR: Ollama call failed for event {event.id}: {e}")

async def trigger_nami_interjection(event: EventItem, score: float) -> bool:
    """Proactively sends a high-priority "interject" event to Nami (Brain 2)."""
    global http_client
    if not http_client:
        print("[Director] FAILED to send interjection: HTTP client not initialized.")
        return False
    try:
        interject_payload = {
            "content": event.text,
            "priority": 1.0 - score,
            "source_info": {"source": f"DIRECTOR_{event.source.name}", "use_tts": True, **event.metadata}
        }
        response = await http_client.post(NAMI_INTERJECT_URL, json=interject_payload, timeout=2.0)
        if response.status_code == 200:
            print(f"[Director] Nami's Funnel accepted interjection.")
            return True
        else:
            print(f"[Director] Nami's Funnel rejected interjection. Status: {response.status_code}")
            return False
    except Exception as e:
        print(f"[Director] FAILED to send interjection: {e}")
        return False

# --- (build_summary_prompt is modified to use InputSource) ---
def build_summary_prompt(events: List[EventItem]) -> str:
    """Builds a prompt for the LLM to summarize recent events."""
    if not events:
        return ""
        
    prompt = "You are a situation summarizer. Watched events are listed below in chronological order. Create a single, concise, 10-word-max summary of the *current* situation. Focus on the user's main activity. If nothing is happening, say 'The user is idle.'\n\n[EVENTS]\n"
    
    event_lines = []
    for event in events:
        if event.source in [InputSource.MICROPHONE, InputSource.DIRECT_MICROPHONE]:
            source = "User"
        elif event.source in [InputSource.TWITCH_CHAT, InputSource.TWITCH_MENTION]:
            source = "Twitch"
        elif event.source == InputSource.BOT_TWITCH_REPLY:
            source = "Nami"
        elif event.source == InputSource.AMBIENT_AUDIO:
            source = "DesktopAudio"
        elif event.source == InputSource.VISUAL_CHANGE:
            source = "Screen"
        else:
            source = "Other"
        event_lines.append(f"{source}: {event.text}")
        
    unique_lines = []
    seen = set()
    for line in reversed(event_lines): # Keep the *newest* unique lines
        if line not in seen:
            unique_lines.append(line)
            seen.add(line)
    
    # Use last 15 unique events
    prompt_context = "\n".join(reversed(unique_lines[-15:]))
    if not prompt_context:
        prompt_context = "No events detected."
        
    prompt += prompt_context
    prompt += "\n\n[SUMMARY (10 words max)]\n"
    
    # Return both the context (for the UI) and the full prompt (for the LLM)
    return prompt_context, prompt

async def generate_summary(store: ContextStore):
    """(Background Task) Gets all events, calls Ollama, and updates the summary."""
    if not ollama_client:
        print("[Analyst] Cannot generate summary: Ollama client not available.")
        return

    events = store.get_all_events_for_summary()
    raw_context, full_prompt = build_summary_prompt(events)
    
    if not full_prompt:
        store.set_summary("No events to summarize.", "No events detected.")
        return

    try:
        response = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': full_prompt}],
            options={"temperature": 0.0} # Be factual
        )
        summary_text = response['message']['content'].strip().replace('"', '')
        
        if "summary:" in summary_text.lower():
            summary_text = summary_text.split(":", 1)[-1].strip()
            
        # --- MODIFIED: Store both the summary and the raw context ---
        store.set_summary(summary_text, raw_context)
        
    except Exception as e:
        print(f"[Analyst] ERROR: Ollama summary call failed: {e}")
        store.set_summary("Error generating summary.", raw_context)

# --- (Client create/close functions are unchanged) ---
async def create_http_client():
    global http_client
    if http_client is None:
        http_client = httpx.AsyncClient()
        print("[Director] HTTP client created.")

async def close_http_client():
    global http_client
    if http_client:
        await http_client.aclose()
        http_client = None
        print("[Director] HTTP client closed.")
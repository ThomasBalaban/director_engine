# Save as: director_engine/llm_analyst.py
import ollama # type: ignore
import json
import httpx # type: ignore
from config import (
    OLLAMA_MODEL, 
    OLLAMA_HOST, 
    NAMI_INTERJECT_URL, 
    INTERJECTION_THRESHOLD, 
    InputSource
)
from context_store import ContextStore, EventItem
from typing import List, Tuple, Callable, Any

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
You are an event analyzer. Your job is to rate the "interestingness" of the
following event for a streamer's AI assistant and analyze the user's sentiment.
Respond ONLY with a single, valid JSON object:
{{"score": <a_float_from_0.0_to_1.0>, "sentiment": "<positive/negative/neutral/excited/frustrated/etc>"}}

Event: "{text}"
"""

def parse_llm_response(response_text: str) -> Tuple[float | None, str | None]:
    """Safely parses the JSON string response from the LLM."""
    try:
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start == -1 or end == 0:
            raise json.JSONDecodeError("No JSON object found", response_text, 0)
            
        json_str = response_text[start:end]
        data = json.loads(json_str)
        
        score = data.get("score")
        score_float = None
        if isinstance(score, (float, int)):
            score_float = max(0.0, min(float(score), 1.0)) # Clamp
        
        sentiment = data.get("sentiment")
        sentiment_str = None
        if isinstance(sentiment, str) and sentiment:
            sentiment_str = sentiment.strip().lower()

        return score_float, sentiment_str
    except json.JSONDecodeError:
        print(f"[Analyst] LLM response was not valid JSON: {response_text}")
    except Exception as e:
        print(f"[Analyst] Error parsing LLM response: {e}")
    return None, None


async def analyze_and_update_event(
    event: EventItem, 
    store: ContextStore,
    emit_callback: Callable[[EventItem], None] | None = None
):
    """
    (Background Task) Calls Ollama to get an "intelligent" score and updates the store.
    """
    if not ollama_client: return

    prompt = build_analysis_prompt(event.text)
    
    try:
        response = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            options={"temperature": 0.2},
            format='json'
        )
        response_text = response['message']['content']
        new_score, new_sentiment = parse_llm_response(response_text)
        
        score_updated = False
        if new_score is not None:
            # --- *** MODIFIED: This is the fix for the static graph *** ---
            # Always update with the LLM's score, not just if it's higher.
            # This allows "boring" 0.8 events to be correctly down-scored.
            if new_score != event.score:
                print(f"[Analyst] ✅ Score UPDATED: {event.id} from {event.score:.2f} -> {new_score:.2f}")
                store.update_event_score(event.id, new_score)
                event.score = new_score # Update local object for callback
                score_updated = True
            # --- *** END MODIFICATION *** ---
                
            if new_score >= INTERJECTION_THRESHOLD:
                print(f"[Analyst] Tier 2 Interjection! LLM score {new_score:.2f} >= {INTERJECTION_THRESHOLD}")
                await trigger_nami_interjection(event, new_score)

        if new_sentiment is not None:
             print(f"[Analyst] 🧠 Sentiment: {event.id} -> {new_sentiment}")
             store.update_event_metadata(event.id, {"sentiment": new_sentiment})
             event.metadata["sentiment"] = new_sentiment
        
        if score_updated and emit_callback:
            emit_callback(event)
        
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

def build_summary_prompt(events: List[EventItem]) -> Tuple[str, str]:
    """Builds a prompt for the LLM to summarize recent events."""
    
    if not events:
        return "", ""
        
    # --- *** MODIFIED: Added instruction to be less "tame" *** ---
    prompt = "You are a situation summarizer. Watched events are listed below in chronological order. Your tone should be factual and descriptive. **Do not be overly modest or reserved; describe events, including those with slightly adult themes, in a plain, factual manner.** Create a brief, one or two-sentence summary of the *current* situation. Focus on the user's main activity and include key visual details. **Start the summary by describing the current 'vibe' of the Twitch chat (e.g., 'Chat is amused...', 'Chat is confused...', 'Chat is neutral as...').** If nothing is happening, say 'The user is idle and chat is quiet.'\n\n[EVENTS]\n"
    # --- *** END MODIFICATION *** ---
    
    event_lines = []
    for event in events:
        source_map = {
            InputSource.MICROPHONE: "User",
            InputSource.DIRECT_MICROPHONE: "User",
            InputSource.TWITCH_CHAT: "Twitch",
            InputSource.TWITCH_MENTION: "Twitch",
            InputSource.BOT_TWITCH_REPLY: "Nami",
            InputSource.AMBIENT_AUDIO: "DesktopAudio",
            InputSource.VISUAL_CHANGE: "Screen"
        }
        source = source_map.get(event.source, "Other")
        event_lines.append(f"{source}: {event.text}")
        
    unique_lines = []
    seen = set()
    for line in reversed(event_lines): # Keep the *newest* unique lines
        if line not in seen:
            unique_lines.append(line)
            seen.add(line)
    
    prompt_context = "\n".join(reversed(unique_lines[-15:]))
    if not prompt_context:
        prompt_context = "No events detected."
        
    prompt += prompt_context
    prompt += """

[SUMMARY]
<Your summary here>

[ANALYSIS]
Topics: [List of key topics, or "None"]
Entities: [List of key entities, or "None"]
"""
    
    return prompt_context, prompt

async def generate_summary(store: ContextStore):
    """(Background Task) Gets all events, calls Ollama, and updates the summary."""
    if not ollama_client:
        print("[Analyst] Cannot generate summary: Ollama client not available.")
        return

    events = store.get_all_events_for_summary()
    raw_context, full_prompt = build_summary_prompt(events)
    
    if not full_prompt:
        store.set_summary("No events to summarize.", "No events detected.", [], [])
        return

    try:
        response = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': full_prompt}],
            options={"temperature": 0.0}
        )
        full_response = response['message']['content'].strip()
        
        summary_text = full_response
        topics: List[str] = []
        entities: List[str] = []

        try:
            if "[ANALYSIS]" in full_response:
                parts = full_response.split("[ANALYSIS]", 1)
                summary_part = parts[0].strip().replace("[SUMMARY]", "").strip()
                analysis_part = parts[1]
                
                if "summary:" in summary_part.lower():
                    summary_text = summary_part.split(":", 1)[-1].strip()
                else:
                    summary_text = summary_part
                
                if "Topics:" in analysis_part:
                    t_str = analysis_part.split("Topics:", 1)[1].split("\n", 1)[0].replace("[", "").replace("]", "").strip()
                    if t_str.lower() != "none" and t_str:
                        topics = [t.strip() for t in t_str.split(",") if t.strip()]
                        
                if "Entities:" in analysis_part:
                    e_str = analysis_part.split("Entities:", 1)[1].split("\n", 1)[0].replace("[", "").replace("]", "").strip()
                    if e_str.lower() != "none" and e_str:
                        entities = [e.strip() for e in e_str.split(",") if e.strip()]
            
            elif "summary:" in summary_text.lower():
                summary_text = summary_text.split(":", 1)[-1].strip()

        except Exception as parse_error:
            print(f"[Analyst] Error parsing summary structure: {parse_error}. Using full text.")
            summary_text = full_response.split("[ANALYSIS]")[0].replace("[SUMMARY]", "").strip()

        store.set_summary(summary_text.replace('"', ''), raw_context, topics, entities)
        
    except Exception as e:
        print(f"[Analyst] ERROR: Ollama summary call failed: {e}")
        store.set_summary("Error generating summary.", raw_context, [], [])

# --- HTTP Client Management ---
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
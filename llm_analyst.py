# Save as: director_engine/llm_analyst.py
import ollama # type: ignore
import json
import httpx # type: ignore
from config import (
    OLLAMA_MODEL, 
    OLLAMA_HOST, 
    NAMI_INTERJECT_URL, 
    INTERJECTION_THRESHOLD, 
    InputSource,
    MEMORY_THRESHOLD
)
from context_store import ContextStore, EventItem
from user_profile_manager import UserProfileManager
from typing import List, Tuple, Callable, Any, Optional

# Global Async Clients
http_client: httpx.AsyncClient | None = None
ollama_client: ollama.AsyncClient | None = None

async def create_http_client():
    global http_client, ollama_client
    if http_client is None:
        http_client = httpx.AsyncClient()
    
    # Initialize Async Ollama Client
    if ollama_client is None:
        try:
            ollama_client = ollama.AsyncClient(host=OLLAMA_HOST)
            # Test connection (optional, might throw if server down)
            # await ollama_client.list() 
            print(f"[Analyst] ✅ Async Ollama Client connected at {OLLAMA_HOST}")
        except Exception as e:
            print(f"[Analyst] ❌ Failed to connect to Ollama: {e}")

async def close_http_client():
    global http_client
    if http_client:
        await http_client.aclose()
        http_client = None

def build_analysis_prompt(text: str, username: str = None) -> str:
    """Builds the prompt for the analyst LLM."""
    user_instruction = ""
    if username:
        user_instruction = f"4. If the user '{username}' reveals a NEW permanent fact about themselves (e.g., 'I live in Texas', 'I hate spiders'), extract it as a string. If not, return an empty list."

    return f"""
You are an event analyzer. Rate the event: "{text}"

1. Rate interestingness (0.0 to 1.0).
2. Determine sentiment (positive, negative, neutral, excited, frustrated, scared).
3. Write a brief (one sentence) synopsis of what happened for long-term memory.
{user_instruction}

Respond ONLY with a single, valid JSON object:
{{
  "score": <float_0.0_to_1.0>,
  "sentiment": "<string>",
  "summary": "<short_synopsis_of_event>",
  "user_facts": ["<fact1>", "<fact2>"]
}}
"""

def parse_llm_response(response_text: str) -> Tuple[float | None, str | None, str | None, List[str]]:
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
            score_float = max(0.0, min(float(score), 1.0))
        
        sentiment = data.get("sentiment")
        sentiment_str = sentiment.strip().lower() if (isinstance(sentiment, str) and sentiment) else None

        summary = data.get("summary")
        summary_str = summary.strip() if (isinstance(summary, str) and summary) else None
        
        facts = data.get("user_facts", [])
        if not isinstance(facts, list): facts = []

        return score_float, sentiment_str, summary_str, facts

    except Exception as e:
        print(f"[Analyst] Error parsing LLM response: {e}")
        return None, None, None, []


async def analyze_and_update_event(
    event: EventItem, 
    store: ContextStore,
    profile_manager: UserProfileManager,
    emit_callback: Callable[[EventItem], None] | None = None
):
    """
    (Background Task) Calls Ollama asynchronously to get score, sentiment, memory, and facts.
    """
    if not ollama_client: 
        print("[Analyst] Ollama client not initialized.")
        return

    # Check if this event is from a specific user for fact extraction
    username = event.metadata.get('username')
    target_user = username if event.source in [InputSource.DIRECT_MICROPHONE, InputSource.TWITCH_MENTION] else None

    prompt = build_analysis_prompt(event.text, target_user)
    
    try:
        # --- ASYNC CALL (Fixes the freezing) ---
        response = await ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            options={"temperature": 0.2},
            format='json'
        )
        response_text = response['message']['content']
        
        # Parse fields
        new_score, new_sentiment, summary_str, new_facts = parse_llm_response(response_text)
        
        score_updated = False
        promoted = False
        profile_updated = False

        # 1. Update Score
        if new_score is not None:
            if new_score != event.score:
                store.update_event_score(event.id, new_score)
                event.score = new_score 
                score_updated = True
            
            # 2. Memory Promotion Check
            if new_score >= MEMORY_THRESHOLD:
                store.promote_to_memory(event, summary_text=summary_str)
                promoted = True

            if new_score >= INTERJECTION_THRESHOLD:
                await trigger_nami_interjection(event, new_score)

        # 3. Update Sentiment & Mood
        if new_sentiment:
             store.update_event_metadata(event.id, {"sentiment": new_sentiment})
             event.metadata["sentiment"] = new_sentiment
             store.update_mood(new_sentiment)

        # 4. Update User Profile (if facts found)
        if target_user and new_facts:
            print(f"[Analyst] 📝 Found new facts for {target_user}: {new_facts}")
            profile_manager.update_profile(target_user, {'new_facts': new_facts})
            
            updated_profile = profile_manager.get_profile(target_user)
            store.set_active_user(updated_profile)
            profile_updated = True

        # 5. Fire Callback (Updates UI)
        if (score_updated or promoted or profile_updated) and emit_callback:
            emit_callback(event)
        
    except Exception as e:
        print(f"[Analyst] ERROR: Ollama call failed for event {event.id}: {e}")

async def trigger_nami_interjection(event: EventItem, score: float) -> bool:
    global http_client
    if not http_client: return False
    try:
        interject_payload = {
            "content": event.text,
            "priority": 1.0 - score,
            "source_info": {"source": f"DIRECTOR_{event.source.name}", "use_tts": True, **event.metadata}
        }
        response = await http_client.post(NAMI_INTERJECT_URL, json=interject_payload, timeout=2.0)
        return response.status_code == 200
    except Exception as e:
        print(f"[Director] FAILED to send interjection: {e}")
        return False

def build_summary_prompt(events: List[EventItem]) -> Tuple[str, str]:
    if not events: return "", ""
    
    event_lines = []
    for event in events:
        source_map = {
            InputSource.MICROPHONE: "User", InputSource.DIRECT_MICROPHONE: "User",
            InputSource.TWITCH_CHAT: "Twitch", InputSource.TWITCH_MENTION: "Twitch",
            InputSource.BOT_TWITCH_REPLY: "Nami", InputSource.AMBIENT_AUDIO: "DesktopAudio",
            InputSource.VISUAL_CHANGE: "Screen"
        }
        source = source_map.get(event.source, "Other")
        event_lines.append(f"{source}: {event.text}")
        
    unique_lines = list(dict.fromkeys(reversed(event_lines)))[-15:] # Dedupe keep order
    prompt_context = "\n".join(reversed(unique_lines)) or "No events detected."
        
    prompt = f"""
You are a situation summarizer and predictor. Watched events are listed below.
[EVENTS]
{prompt_context}

1. Summarize the current situation in 1-2 sentences. Start by describing the 'vibe'.
2. PREDICT what the user might do or ask next, or what topic is likely to come up.

Respond in this format:
[SUMMARY]
<summary text>

[PREDICTION]
<your short prediction>

[ANALYSIS]
Topics: [list]
Entities: [list]
"""
    return prompt_context, prompt

async def generate_summary(store: ContextStore):
    if not ollama_client: return

    events = store.get_all_events_for_summary()
    raw_context, full_prompt = build_summary_prompt(events)
    if not full_prompt:
        store.set_summary("No events.", "No events detected.", [], [], "None")
        return

    try:
        # --- ASYNC CALL ---
        response = await ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': full_prompt}],
            options={"temperature": 0.3}
        )
        full_response = response['message']['content'].strip()
        
        summary_text = full_response
        prediction_text = "None"
        topics = []
        entities = []

        try:
            if "[SUMMARY]" in full_response:
                parts = full_response.split("[SUMMARY]")
                remaining = parts[1]
                if "[PREDICTION]" in remaining:
                    sum_parts = remaining.split("[PREDICTION]")
                    summary_text = sum_parts[0].strip()
                    remaining = sum_parts[1]
                    if "[ANALYSIS]" in remaining:
                        pred_parts = remaining.split("[ANALYSIS]")
                        prediction_text = pred_parts[0].strip()
            
            summary_text = summary_text.replace('"', '')
            prediction_text = prediction_text.replace('"', '')

        except:
            summary_text = full_response

        store.set_summary(summary_text, raw_context, topics, entities, prediction_text)
        
    except Exception as e:
        print(f"[Analyst] Summary error: {e}")
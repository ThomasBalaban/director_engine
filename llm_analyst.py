# Save as: director_engine/llm_analyst.py
import ollama
import json
import httpx
from config import (
    OLLAMA_MODEL, OLLAMA_HOST, NAMI_INTERJECT_URL, 
    INTERJECTION_THRESHOLD, InputSource, MEMORY_THRESHOLD
)
from context_store import ContextStore, EventItem
from user_profile_manager import UserProfileManager
from scoring import EventScore
from typing import List, Tuple, Callable, Any, Optional, Dict

# Global Async Clients
http_client: httpx.AsyncClient | None = None
ollama_client: ollama.AsyncClient | None = None

async def create_http_client():
    global http_client, ollama_client
    if http_client is None:
        http_client = httpx.AsyncClient()
    if ollama_client is None:
        try:
            ollama_client = ollama.AsyncClient(host=OLLAMA_HOST)
            print(f"[Analyst] ✅ Async Ollama Client connected at {OLLAMA_HOST}")
        except Exception as e:
            print(f"[Analyst] ❌ Failed to connect to Ollama: {e}")

async def close_http_client():
    global http_client
    if http_client:
        await http_client.aclose()
        http_client = None

def build_analysis_prompt(text: str, username: str = None) -> str:
    user_instruction = ""
    if username:
        user_instruction = f"4. If the user '{username}' reveals a NEW permanent fact about themselves, extract it as a string."

    return f"""
Analyze this streaming event: "{text}"

1. Rate on a scale of 0.0 to 1.0:
   - Interestingness (General value)
   - Urgency (Need for immediate response)
   - Conversational Value (Potential to spark dialogue)
   - Emotional Intensity (Strength of emotion)
   - Topic Relevance (Connection to ongoing themes)
2. Determine sentiment (one word: positive, negative, neutral, excited, frustrated, scared, horny, tired).
3. Write a 1-sentence synopsis for long-term memory.
{user_instruction}

Respond ONLY with this JSON structure:
{{
  "scores": {{
    "interestingness": <float>,
    "urgency": <float>,
    "conversational_value": <float>,
    "emotional_intensity": <float>,
    "topic_relevance": <float>
  }},
  "sentiment": "<string>",
  "summary": "<string>",
  "user_facts": ["<fact1>"]
}}
"""

def parse_llm_response(response_text: str) -> Tuple[EventScore | None, str | None, str | None, List[str]]:
    try:
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start == -1 or end == 0:
            raise json.JSONDecodeError("No JSON object found", response_text, 0)
            
        json_str = response_text[start:end]
        data = json.loads(json_str)
        
        scores_data = data.get("scores", {})
        
        # Create EventScore object
        event_score = EventScore(
            interestingness=float(scores_data.get("interestingness", 0.0)),
            urgency=float(scores_data.get("urgency", 0.0)),
            conversational_value=float(scores_data.get("conversational_value", 0.0)),
            emotional_intensity=float(scores_data.get("emotional_intensity", 0.0)),
            topic_relevance=float(scores_data.get("topic_relevance", 0.0))
        )
        
        sentiment = data.get("sentiment")
        sentiment_str = sentiment.strip().lower() if (isinstance(sentiment, str) and sentiment) else None
        summary = data.get("summary")
        summary_str = summary.strip() if (isinstance(summary, str) and summary) else None
        facts = data.get("user_facts", [])
        if not isinstance(facts, list): facts = []

        return event_score, sentiment_str, summary_str, facts

    except Exception as e:
        print(f"[Analyst] Error parsing LLM response: {e}")
        return None, None, None, []

async def analyze_and_update_event(
    event: EventItem, 
    store: ContextStore,
    profile_manager: UserProfileManager,
    emit_callback: Callable[[EventItem], None] | None = None
):
    if not ollama_client: return

    username = event.metadata.get('username')
    target_user = username if event.source in [InputSource.DIRECT_MICROPHONE, InputSource.TWITCH_MENTION] else None

    prompt = build_analysis_prompt(event.text, target_user)
    
    try:
        response = await ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            options={"temperature": 0.2},
            format='json'
        )
        
        new_score, new_sentiment, summary_str, new_facts = parse_llm_response(response['message']['content'])
        
        score_updated = False
        promoted = False
        profile_updated = False

        if new_score:
            # Check if significantly different from heuristic score
            if abs(new_score.interestingness - event.score.interestingness) > 0.1:
                store.update_event_score(event.id, new_score)
                event.score = new_score
                score_updated = True
            
            if new_score.interestingness >= MEMORY_THRESHOLD:
                store.promote_to_memory(event, summary_text=summary_str)
                promoted = True

            if new_score.urgency >= INTERJECTION_THRESHOLD:
                await trigger_nami_interjection(event, new_score.urgency)

        if new_sentiment:
             store.update_event_metadata(event.id, {"sentiment": new_sentiment})
             event.metadata["sentiment"] = new_sentiment
             store.update_mood(new_sentiment)

        if target_user and new_facts:
            print(f"[Analyst] 📝 Found new facts for {target_user}: {new_facts}")
            profile_manager.update_profile(target_user, {'new_facts': new_facts})
            updated_profile = profile_manager.get_profile(target_user)
            store.set_active_user(updated_profile)
            profile_updated = True

        if (score_updated or promoted or profile_updated) and emit_callback:
            emit_callback(event)
        
    except Exception as e:
        print(f"[Analyst] ERROR: Ollama call failed for event {event.id}: {e}")

async def trigger_nami_interjection(event: EventItem, urgency_score: float) -> bool:
    global http_client
    if not http_client: return False
    try:
        interject_payload = {
            "content": event.text,
            "priority": 1.0 - urgency_score, # Convert score to priority (lower is better in funnel)
            "source_info": {"source": f"DIRECTOR_{event.source.name}", "use_tts": True, **event.metadata}
        }
        response = await http_client.post(NAMI_INTERJECT_URL, json=interject_payload, timeout=2.0)
        return response.status_code == 200
    except Exception as e:
        print(f"[Director] FAILED to send interjection: {e}")
        return False

def build_summary_prompt(layers: Dict[str, List[EventItem]]) -> Tuple[str, str]:
    # Construct Layered Context String
    
    def format_layer(events):
        lines = []
        for e in events:
            source_map = {
                InputSource.MICROPHONE: "User", InputSource.DIRECT_MICROPHONE: "User",
                InputSource.TWITCH_CHAT: "Twitch", InputSource.TWITCH_MENTION: "Twitch",
                InputSource.BOT_TWITCH_REPLY: "Nami", InputSource.AMBIENT_AUDIO: "Audio",
                InputSource.VISUAL_CHANGE: "Vision"
            }
            src = source_map.get(e.source, "Other")
            lines.append(f"- [{src}] {e.text}")
        return "\n".join(lines)

    immediate_txt = format_layer(layers['immediate']) or "None"
    recent_txt = format_layer(layers['recent']) or "None"
    background_txt = format_layer(layers['background'][-5:]) or "None" # Limit background to last 5 for now

    prompt_context = f"""
[IMMEDIATE EVENTS (Last 10s)]
{immediate_txt}

[RECENT CONTEXT (Last 30s)]
{recent_txt}

[BACKGROUND CONTEXT (Earlier)]
{background_txt}
"""
        
    prompt = f"""
You are a situation summarizer.
{prompt_context}

1. Summarize the CURRENT situation (1-2 sentences). Start with the 'vibe'.
2. PREDICT user intent or next topic.

Respond:
[SUMMARY]
<text>

[PREDICTION]
<text>

[ANALYSIS]
Topics: []
Entities: []
"""
    return prompt_context, prompt

async def generate_summary(store: ContextStore):
    if not ollama_client: return

    layers = store.get_all_events_for_summary()
    raw_context, full_prompt = build_summary_prompt(layers)
    
    try:
        response = await ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': full_prompt}],
            options={"temperature": 0.3}
        )
        full_response = response['message']['content'].strip()
        
        # Basic parsing (same as before)
        summary_text = full_response
        prediction_text = "None"
        
        if "[SUMMARY]" in full_response:
            parts = full_response.split("[SUMMARY]")
            remaining = parts[1]
            if "[PREDICTION]" in remaining:
                sum_parts = remaining.split("[PREDICTION]")
                summary_text = sum_parts[0].strip()
                prediction_text = sum_parts[1].split("[ANALYSIS]")[0].strip()

        store.set_summary(summary_text, raw_context, [], [], prediction_text)
        
    except Exception as e:
        print(f"[Analyst] Summary error: {e}")
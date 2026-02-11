# Save as: director_engine/services/llm_analyst.py
import ollama
import json
import httpx
import asyncio
from config import (
    OLLAMA_MODEL, OLLAMA_HOST, NAMI_INTERJECT_URL, 
    INTERJECTION_THRESHOLD, InputSource,
    ConversationState, FlowState, UserIntent 
)
from context.context_store import ContextStore, EventItem
from context.user_profile_manager import UserProfileManager
from scoring import EventScore
from typing import List, Tuple, Callable, Any, Optional, Dict

# Global Async Clients
http_client: httpx.AsyncClient | None = None
ollama_client: ollama.AsyncClient | None = None

# --- Context Inference State (non-blocking) ---
_context_inference_running = False
_last_inferred_result: Optional[Dict[str, str]] = None

# --- LOWERED MEMORY THRESHOLD ---
# Events with this score or higher get promoted to long-term memory
MEMORY_PROMOTION_THRESHOLD = 0.70  # Lowered from 0.85

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

# --- Thought Generation for Internal Monologue ---
async def generate_thought(prompt_text: str) -> Optional[str]:
    """Generates a short, quirky thought based on the behavior engine's prompt."""
    if not ollama_client: return None
    
    full_prompt = (
        f"You are Nami's internal monologue (a chaotic, confident AI). "
        f"Generate a single short sentence based on this thought prompt: '{prompt_text}'.\n"
        f"Make it sound like a random shower thought, a conspiracy theory, or a sudden realization. "
        f"Do not ask questions. Be confident but weird."
    )
    
    try:
        response = await ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': full_prompt}],
            options={"temperature": 0.8, "num_predict": 50}
        )
        return response['message']['content'].strip().strip('"')
    except Exception as e:
        print(f"[Analyst] Thought generation error: {e}")
        return None

# --- NON-BLOCKING Context Inference using Ollama ---
async def _do_context_inference(store: ContextStore) -> Optional[Dict[str, str]]:
    """
    Internal function that actually performs the inference.
    Called as a background task - never blocks the main loop.
    """
    global _context_inference_running, _last_inferred_result
    
    if not ollama_client: 
        return None
    
    try:
        layers = store.get_all_events_for_summary()
        
        # Gather recent visual and audio context
        recent_visuals = [e.text for e in layers['immediate'] + layers['recent'] 
                          if e.source == InputSource.VISUAL_CHANGE][-5:]
        recent_audio = [e.text for e in layers['immediate'] + layers['recent'] 
                        if e.source == InputSource.AMBIENT_AUDIO][-3:]
        recent_speech = [e.text for e in layers['immediate'] + layers['recent'] 
                         if e.source in [InputSource.MICROPHONE, InputSource.DIRECT_MICROPHONE]][-3:]
        
        if not recent_visuals and not recent_audio:
            return None
        
        context_block = ""
        if recent_visuals:
            context_block += f"VISUALS: {' | '.join(recent_visuals[:3])}\n"
        if recent_audio:
            context_block += f"AUDIO: {' | '.join(recent_audio[:2])}\n"
        if recent_speech:
            context_block += f"SPEECH: {' | '.join(recent_speech[:2])}\n"
        
        prompt = f"""Based on this stream data, identify what game or activity is being shown.

{context_block}

Respond ONLY with this JSON (no other text):
{{
  "game": "<game name or 'Unknown' or 'Just Chatting'>",
  "context": "<max 120 char description of what's happening>"
}}

Rules:
- Game should be the actual game title if identifiable
- Context should be SHORT (under 120 chars)
- Context should describe current activity (e.g. "Exploring haunted asylum, looking for ghost evidence")
- If unsure, say "Unknown" for game
"""
        
        # Use asyncio.wait_for to add a timeout so it never hangs
        response = await asyncio.wait_for(
            ollama_client.chat(
                model=OLLAMA_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                options={"temperature": 0.3, "num_predict": 100}
            ),
            timeout=10.0  # 10 second timeout
        )
        
        result_text = response['message']['content'].strip()
        
        # Parse JSON
        start = result_text.find('{')
        end = result_text.rfind('}') + 1
        if start == -1 or end == 0:
            return None
            
        data = json.loads(result_text[start:end])
        
        game = data.get('game', 'Unknown')
        context = data.get('context', '')
        
        # Enforce 120 char limit
        if len(context) > 120:
            context = context[:117] + "..."
        
        result = {"game": game, "context": context}
        _last_inferred_result = result
        return result
        
    except asyncio.TimeoutError:
        print(f"[Analyst] ⏱️ Context inference timed out (non-blocking, continuing)")
        return None
    except Exception as e:
        print(f"[Analyst] Context inference error: {e}")
        return None
    finally:
        _context_inference_running = False


def start_context_inference_task(store: ContextStore, callback: Callable[[Dict[str, str]], None] = None):
    """
    Starts a background task for context inference.
    Non-blocking - returns immediately. 
    Calls the callback with results when done (if provided).
    """
    global _context_inference_running
    
    # Don't start if already running
    if _context_inference_running:
        return
    
    _context_inference_running = True
    
    async def _task():
        result = await _do_context_inference(store)
        if result and callback:
            callback(result)
    
    # Fire and forget - create task but don't await it
    asyncio.create_task(_task())

def build_analysis_prompt(text: str, username: str = None) -> str:
    user_instruction = ""
    if username:
       user_instruction = (
            f"4. Check if '{username}' explicitly reveals a concrete bio detail (e.g., age, job, pet, location, hobby). "
            f"STRICTLY FORBIDDEN: Facts about the stream itself, 'testing', 'existing', or 'chatting'. "
            f"If the fact is trivial, return an empty list."
        )

    return f"""
Analyze this streaming event: "{text}"

1. Rate on a scale of 0.0 to 1.0:
   - Interestingness (General value - how noteworthy is this?)
   - Urgency (Need for immediate response)
   - Conversational Value (Potential to spark dialogue or be referenced later)
   - Emotional Intensity (Strength of emotion displayed)
   - Topic Relevance (Connection to ongoing themes)

2. Determine sentiment (one word: positive, negative, neutral, excited, frustrated, scared, horny, tired).

3. Write a 1-sentence synopsis for long-term memory. Make it SPECIFIC and MEMORABLE.
   Good: "Otter rage-quit after getting hit by a blue shell right before the finish line"
   Bad: "The user reacted to a game event"
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
  "summary": "<string - specific and memorable>",
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
            if abs(new_score.interestingness - event.score.interestingness) > 0.1:
                store.update_event_score(event.id, new_score)
                event.score = new_score
                score_updated = True
            
            # --- LOWERED THRESHOLD for memory promotion ---
            # Also consider conversational_value for memory worthiness
            should_promote = (
                new_score.interestingness >= MEMORY_PROMOTION_THRESHOLD or
                new_score.conversational_value >= 0.75 or
                new_score.emotional_intensity >= 0.8
            )
            
            if should_promote:
                store.promote_to_memory(event, summary_text=summary_str)
                promoted = True
                print(f"💾 [Analyst] Promoted to memory: {summary_str[:50] if summary_str else event.text[:50]}...")

            if new_score.urgency >= INTERJECTION_THRESHOLD:
                is_interrupt = event.metadata.get('interrupt_priority', False)
                await trigger_nami_interjection(event, new_score.urgency, is_interrupt=is_interrupt)

        if new_sentiment:
             store.update_event_metadata(event.id, {"sentiment": new_sentiment})
             event.metadata["sentiment"] = new_sentiment
             store.update_mood(new_sentiment)

        if target_user and new_facts:
            profile_manager.update_profile(target_user, {'new_facts': new_facts})
            updated_profile = profile_manager.get_profile(target_user)
            store.set_active_user(updated_profile)
            profile_updated = True

        if (score_updated or promoted or profile_updated) and emit_callback:
            emit_callback(event)
        
    except Exception as e:
        print(f"[Analyst] ERROR: Ollama call failed for event {event.id}: {e}")

async def trigger_nami_interjection(event: EventItem, urgency_score: float, is_interrupt: bool = False) -> bool:
    """
    Send an interjection to Nami.
    
    If is_interrupt=True, this is a direct address that should interrupt 
    whatever Nami is currently saying.
    """
    global http_client
    if not http_client: return False
    try:
        interject_payload = {
            "content": event.text,
            "priority": 0.0 if is_interrupt else (1.0 - urgency_score),  # 0.0 = highest priority
            "source_info": {
                "source": f"DIRECTOR_{event.source.name}", 
                "use_tts": True,
                "is_interrupt": is_interrupt,  # NEW: Signal to Nami that this interrupts
                "is_direct_address": event.metadata.get('is_direct_address', False),
                **{k: v for k, v in event.metadata.items() if k not in ['is_direct_address']}
            }
        }
        
        if is_interrupt:
            print(f"🛑 [Analyst] Sending INTERRUPT interjection: {event.text[:50]}...")
        
        response = await http_client.post(NAMI_INTERJECT_URL, json=interject_payload, timeout=2.0)
        return response.status_code == 200
    except Exception as e:
        print(f"[Director] FAILED to send interjection: {e}")
        return False

def build_summary_prompt(layers: Dict[str, List[EventItem]]) -> Tuple[str, str]:
    def format_layer(events):
        lines = []
        for e in events:
            source_map = {
                InputSource.MICROPHONE: "User", InputSource.DIRECT_MICROPHONE: "User",
                InputSource.TWITCH_CHAT: "Twitch", InputSource.TWITCH_MENTION: "Twitch",
                InputSource.BOT_TWITCH_REPLY: "Nami", InputSource.AMBIENT_AUDIO: "Audio",
                InputSource.VISUAL_CHANGE: "Vision", InputSource.SYSTEM_PATTERN: "Insight"
            }
            src = source_map.get(e.source, "Other")
            lines.append(f"- [{src}] {e.text}")
        return "\n".join(lines)

    immediate_txt = format_layer(layers['immediate']) or "None"
    recent_txt = format_layer(layers['recent']) or "None"
    background_txt = format_layer(layers['background'][-5:]) or "None" 

    prompt_context = f"""
[IMMEDIATE EVENTS (Last 10s)]
{immediate_txt}

[RECENT CONTEXT (Last 30s)]
{recent_txt}

[BACKGROUND CONTEXT (Earlier)]
{background_txt}
"""
    
    conv_states = ", ".join([s.name for s in ConversationState])
    flow_states = ", ".join([s.name for s in FlowState])
    intent_states = ", ".join([s.name for s in UserIntent])
    
    prompt = f"""
You are a situation summarizer.
{prompt_context}

1. Summarize the CURRENT situation (1-2 sentences). Start with the 'vibe'.
2. PREDICT what might happen next.
3. CLASSIFY the current state.

Respond using this format EXACTLY:
[SUMMARY]
<summary text>

[PREDICTION]
<prediction text>

[CLASSIFICATION]
State: <{conv_states}>
Flow: <{flow_states}>
Intent: <{intent_states}>
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
        
        summary_text = full_response
        prediction_text = "None"
        
        # Parse Summary & Prediction
        if "[SUMMARY]" in full_response:
            parts = full_response.split("[SUMMARY]")
            if len(parts) > 1:
                remainder = parts[1]
                if "[PREDICTION]" in remainder:
                    split_pred = remainder.split("[PREDICTION]")
                    summary_text = split_pred[0].strip()
                    remainder = split_pred[1]
                    
                    if "[CLASSIFICATION]" in remainder:
                        split_class = remainder.split("[CLASSIFICATION]")
                        prediction_text = split_class[0].strip()
                        class_block = split_class[1].strip()
                        
                        # Parse Classification Block
                        lines = class_block.split('\n')
                        for line in lines:
                            line = line.strip().upper()
                            if line.startswith("STATE:"):
                                try:
                                    val = line.split(":")[1].strip()
                                    store.set_conversation_state(ConversationState[val])
                                except: pass
                            elif line.startswith("FLOW:"):
                                try:
                                    val = line.split(":")[1].strip()
                                    store.set_flow_state(FlowState[val])
                                except: pass
                            elif line.startswith("INTENT:"):
                                try:
                                    val = line.split(":")[1].strip()
                                    store.set_user_intent(UserIntent[val])
                                except: pass

        store.set_summary(summary_text, raw_context, [], [], prediction_text)
        
    except Exception as e:
        print(f"[Analyst] Summary error: {e}")
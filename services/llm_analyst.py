# Save as: director_engine/services/llm_analyst.py
"""
LLM Analyst — Uses Ollama for event analysis, scoring, and summarization.

NOTE: trigger_nami_interjection now routes through the Prompt Service
instead of POSTing directly to Nami.
"""

import ollama
import json
import httpx
import asyncio
import google.generativeai as genai  # type: ignore
from google.generativeai.types import HarmCategory, HarmBlockThreshold  # type: ignore
from config import (
    OLLAMA_MODEL, OLLAMA_HOST,
    OLLAMA_TIMEOUT_THOUGHT, OLLAMA_TIMEOUT_ANALYZE,
    OLLAMA_TIMEOUT_SUMMARY, OLLAMA_TIMEOUT_CTX_INFER,
    INTERJECTION_THRESHOLD, InputSource,
    ConversationState, FlowState, UserIntent,
    OWNER_STREAMER_ID, GEMINI_API_KEY
)
from context.context_store import ContextStore, EventItem
from context.user_profile_manager import UserProfileManager
from scoring import EventScore
from services.ollama_gate import get_ollama_gate
from diagnostics import log_error
from typing import List, Tuple, Callable, Any, Optional, Dict

# Global Async Clients
http_client: httpx.AsyncClient | None = None
ollama_client: ollama.AsyncClient | None = None

# Gemini model for the situation summarizer. Kept separate from Ollama because
# the summary path needs JSON-structured output and strict grounding — Ollama
# was producing hallucinations and template fixations under sparse input.
gemini_summary_model: Optional[Any] = None

NO_ACTIVITY_SUMMARY = "(No current activity observed.)"
UNAVAILABLE_SUMMARY = "(Summary unavailable.)"

# --- Context Inference State ---
_context_inference_running = False
_last_inferred_result: Optional[Dict[str, str]] = None

MEMORY_PROMOTION_THRESHOLD = 0.70

# --- Concurrency control for analyze_and_update_event ---
# Each call holds the event loop while doing JSON parse / regex / store
# mutation after the ollama response returns. Without a bound, vision events
# (one every ~2s) stack up faster than ollama can drain them, starving the
# asyncio loop and causing asyncio.sleep(1.0) to overrun by ~10s.
#
# MAX_CONCURRENT: in-flight analyses. ollama serves one inference at a time
#   anyway, so >2 here just adds GIL ping-pong for sync post-processing.
# MAX_PENDING:    hard cap on queue depth. Beyond this, drop the newest call
#   with a clear log line. Dropped events skip analysis but still hit the
#   store via heuristic scoring upstream.
_ANALYZE_MAX_CONCURRENT = 2
_ANALYZE_MAX_PENDING = 4
_analyze_sem: Optional[asyncio.Semaphore] = None
_analyze_pending = 0
_analyze_dropped_total = 0


def _get_analyze_sem() -> asyncio.Semaphore:
    """Lazy init so the Semaphore binds to the running event loop."""
    global _analyze_sem
    if _analyze_sem is None:
        _analyze_sem = asyncio.Semaphore(_ANALYZE_MAX_CONCURRENT)
    return _analyze_sem

async def create_http_client():
    global http_client, ollama_client, gemini_summary_model
    if http_client is None:
        http_client = httpx.AsyncClient()
    if ollama_client is None:
        try:
            ollama_client = ollama.AsyncClient(host=OLLAMA_HOST)
            print(f"[Analyst] ✅ Async Ollama Client connected at {OLLAMA_HOST}")
        except Exception as e:
            print(f"[Analyst] ❌ Failed to connect to Ollama: {e}")
    if gemini_summary_model is None and GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            gemini_summary_model = genai.GenerativeModel(
                'gemini-2.0-flash',
                safety_settings=safety_settings,
            )
            print("[Analyst] ✅ Gemini summary model initialized (gemini-2.0-flash)")
        except Exception as e:
            print(f"[Analyst] ❌ Failed to init Gemini summary model: {e}")

async def close_http_client():
    global http_client
    if http_client:
        await http_client.aclose()
        http_client = None


# --- Thought Generation ---

async def generate_thought(prompt_text: str, stream_context: str = "", watching_context: str = "") -> Optional[str]:
    """
    Generates a spontaneous thought for Nami to say during silence.

    Args:
        prompt_text:       The specific prompt/topic to riff on (e.g. "A weird theory about fire")
        stream_context:    Short description of what's currently happening on screen
                           (e.g. "HORROR_TENSION - Otter is creeping through a dark hallway")
        watching_context:  Pre-built sentence describing the watching relationship
                           (e.g. "You are watching PeepingOtter stream" or
                                 "You and PeepingOtter are watching xQc together")
    """
    if not ollama_client:
        return None

    # Build the context block — only include what we have
    context_lines = []
    if watching_context:
        context_lines.append(watching_context)
    if stream_context:
        context_lines.append(f"Current situation: {stream_context}")

    context_block = "\n".join(context_lines)
    if context_block:
        context_block = f"\n{context_block}\n"

    full_prompt = (
        f"You are PeepingNami, often referred to as Nami. "
        f"You are PeepingOtter's personal AI companion.{context_block}\n"
        f"You just had a random thought. It could be any of these:\n"
        f"- A weird or unhinged observation about what is currently happening\n"
        f"- A question you suddenly became curious about\n"
        f"- An opinion or hot take on something that just happened\n"
        f"- Something completely random and chaotic\n"
        f"- Teasing or prodding PeepingOtter about something\n\n"
        f"Thought prompt: {prompt_text}\n\n"
        f"Rules:\n"
        f"- Say ONE sentence only. Do not explain it.\n"
        f"- Direct it at PeepingOtter, not at a crowd.\n"
        f"- Base it on what is actually happening in the current situation above.\n"
        f"- Do NOT reference anime, fictional universes, games, or any media "
        f"unless it is explicitly visible in the current situation.\n"
        f"- Do not start with 'I think' or 'I feel'.\n"
    )

    try:
        async with get_ollama_gate():
            response = await asyncio.wait_for(
                ollama_client.chat(
                    model=OLLAMA_MODEL,
                    messages=[{'role': 'user', 'content': full_prompt}],
                    options={"temperature": 0.8, "num_predict": 60}
                ),
                timeout=OLLAMA_TIMEOUT_THOUGHT,
            )
        return response['message']['content'].strip().strip('"')
    except asyncio.TimeoutError:
        print(f"⏱️  [Analyst] generate_thought timed out after {OLLAMA_TIMEOUT_THOUGHT}s")
        log_error("analyst.generate_thought", "ollama chat timeout",
                  timeout_s=OLLAMA_TIMEOUT_THOUGHT)
        # Notify core_logic so it can flip into backoff mode after enough timeouts.
        # Lazy import avoids circular dependency at module-load time.
        try:
            import core_logic as _cl
            _cl._note_thought_timeout()
        except Exception:
            pass
        return None
    except Exception as e:
        kind = type(e).__name__
        detail = str(e) or repr(e)
        print(f"[Analyst] Thought generation error ({kind}): {detail}")
        log_error("analyst.generate_thought", "ollama chat error", exc=e)
        return None


# --- NON-BLOCKING Context Inference ---
async def _do_context_inference(store: ContextStore) -> Optional[Dict[str, str]]:
    global _context_inference_running, _last_inferred_result

    if not ollama_client:
        return None

    try:
        layers = store.get_all_events_for_summary()

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
- Context should describe current activity
- If unsure, say "Unknown" for game
"""

        async with get_ollama_gate():
            response = await asyncio.wait_for(
                ollama_client.chat(
                    model=OLLAMA_MODEL,
                    messages=[{'role': 'user', 'content': prompt}],
                    options={"temperature": 0.3, "num_predict": 100}
                ),
                timeout=OLLAMA_TIMEOUT_CTX_INFER
            )

        result_text = response['message']['content'].strip()

        start = result_text.find('{')
        end = result_text.rfind('}') + 1
        if start == -1 or end == 0:
            return None

        data = json.loads(result_text[start:end])

        game = data.get('game', 'Unknown')
        context = data.get('context', '')

        if len(context) > 120:
            context = context[:117] + "..."

        result = {"game": game, "context": context}
        _last_inferred_result = result
        return result

    except asyncio.TimeoutError:
        print(f"[Analyst] ⏱️ Context inference timed out")
        log_error("analyst.context_inference", "ollama chat timeout",
                  timeout_s=OLLAMA_TIMEOUT_CTX_INFER)
        return None
    except Exception as e:
        print(f"[Analyst] Context inference error: {e}")
        log_error("analyst.context_inference", "ollama chat error", exc=e)
        return None
    finally:
        _context_inference_running = False


def start_context_inference_task(store: ContextStore, callback: Callable[[Dict[str, str]], None] = None):
    global _context_inference_running

    if _context_inference_running:
        return

    _context_inference_running = True

    async def _task():
        result = await _do_context_inference(store)
        if result and callback:
            callback(result)

    asyncio.create_task(_task())


# --- Analysis ---

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

    global _analyze_pending, _analyze_dropped_total

    # Backpressure: drop if already saturated. Prevents create_task callers
    # from piling up unbounded when ollama can't keep up with vision events.
    if _analyze_pending >= _ANALYZE_MAX_PENDING:
        _analyze_dropped_total += 1
        # Log every drop initially, then sample (mod 10) to avoid log spam during sustained load
        if _analyze_dropped_total <= 5 or _analyze_dropped_total % 10 == 0:
            print(
                f"⚠️  [Analyst] DROPPED analysis for event {event.id} "
                f"({_analyze_pending} pending, {_analyze_dropped_total} total dropped) "
                f"— backpressure active"
            )
        return

    _analyze_pending += 1
    try:
        async with _get_analyze_sem():
            await _analyze_and_update_event_inner(event, store, profile_manager, emit_callback)
    finally:
        _analyze_pending -= 1


async def _analyze_and_update_event_inner(
    event: EventItem,
    store: ContextStore,
    profile_manager: UserProfileManager,
    emit_callback: Callable[[EventItem], None] | None = None
):
    username = event.metadata.get('username')
    target_user = username if event.source in [InputSource.DIRECT_MICROPHONE, InputSource.TWITCH_MENTION] else None

    prompt = build_analysis_prompt(event.text, target_user)

    try:
        async with get_ollama_gate():
            response = await asyncio.wait_for(
                ollama_client.chat(
                    model=OLLAMA_MODEL,
                    messages=[{'role': 'user', 'content': prompt}],
                    options={"temperature": 0.2},
                    format='json'
                ),
                timeout=OLLAMA_TIMEOUT_ANALYZE,
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

            should_promote = (
                new_score.interestingness >= MEMORY_PROMOTION_THRESHOLD or
                new_score.conversational_value >= 0.75 or
                new_score.emotional_intensity >= 0.8
            )

            if should_promote:
                store.promote_to_memory(event, summary_text=summary_str)
                promoted = True

            if new_score.urgency >= INTERJECTION_THRESHOLD:
                is_interrupt = event.metadata.get('interrupt_priority', False)
                if not event.metadata.get('is_direct_address'):
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

    except asyncio.TimeoutError:
        print(f"⏱️  [Analyst] analyze_and_update_event timed out after {OLLAMA_TIMEOUT_ANALYZE}s for event {event.id}")
        log_error("analyst.analyze_and_update_event", "ollama chat timeout",
                  timeout_s=OLLAMA_TIMEOUT_ANALYZE, event_id=event.id)
    except Exception as e:
        kind = type(e).__name__
        detail = str(e) or repr(e)
        print(f"[Analyst] ERROR ({kind}): Ollama call failed for event {event.id}: {detail}")
        log_error("analyst.analyze_and_update_event", "ollama chat error",
                  exc=e, event_id=event.id)


async def trigger_nami_interjection(event: EventItem, urgency_score: float, is_interrupt: bool = False) -> bool:
    import services.prompt_client as prompt_client

    result = await prompt_client.request_speech(
        trigger=f"urgency_{event.source.name}",
        content=event.text,
        priority=0.0 if is_interrupt else (1.0 - urgency_score),
        source=f"DIRECTOR_{event.source.name}",
        is_interrupt=is_interrupt,
        event_id=event.id,
        metadata={
            'is_direct_address': event.metadata.get('is_direct_address', False),
            **{k: v for k, v in event.metadata.items() if k not in ['is_direct_address']}
        }
    )

    return result.get("delivered", False)


# --- Summary Generation ---
#
# Switched from Ollama to Gemini 2.0 Flash for the situation summarizer.
# Rationale:
#  - JSON-structured output (response_mime_type) eliminates the text-parsing
#    failure mode that previously dropped classifications when the model
#    formatted its output slightly differently.
#  - Strict grounding rules in the prompt + low temperature reduce the
#    hallucination problem documented in notes.md (2026-05-17, 2026-05-21).
#  - Bot replies (BOT_TWITCH_REPLY) are dropped from input so Nami's own
#    past quips can't bias the next summary's framing — that was the
#    template-fixation loop ("oxygen dropping faster than viewership"
#    surviving even after oxygen recovered).
#  - If the event layers are completely empty (which can now happen in
#    reply_only mode or when vision/audio services aren't running), the
#    summarizer short-circuits to a fixed "no activity" string WITHOUT
#    calling the LLM at all — zero hallucination risk on empty input.

# Sources whose contents are world observations (safe to summarize).
# Excludes BOT_TWITCH_REPLY: summarizing Nami's own past commentary creates
# a feedback loop where templates self-reinforce.
_SUMMARY_INPUT_SOURCES = {
    InputSource.MICROPHONE,
    InputSource.DIRECT_MICROPHONE,
    InputSource.TWITCH_CHAT,
    InputSource.TWITCH_MENTION,
    InputSource.AMBIENT_AUDIO,
    InputSource.VISUAL_CHANGE,
    InputSource.SYSTEM_PATTERN,
}


def _filter_for_summary(events: List[EventItem]) -> List[EventItem]:
    return [e for e in events if e.source in _SUMMARY_INPUT_SOURCES]


def build_summary_prompt(layers: Dict[str, List[EventItem]]) -> Tuple[str, str, int]:
    """
    Returns (raw_context_for_storage, llm_prompt, total_event_count).

    total_event_count is 0 when every layer is empty after filtering — the
    caller uses this to short-circuit before any LLM call.
    """
    source_map = {
        InputSource.MICROPHONE: "User", InputSource.DIRECT_MICROPHONE: "User",
        InputSource.TWITCH_CHAT: "Twitch", InputSource.TWITCH_MENTION: "Twitch",
        InputSource.AMBIENT_AUDIO: "Audio", InputSource.VISUAL_CHANGE: "Vision",
        InputSource.SYSTEM_PATTERN: "Insight",
    }

    def format_layer(events: List[EventItem]) -> str:
        if not events:
            return "(empty)"
        return "\n".join(f"- [{source_map.get(e.source, 'Other')}] {e.text}" for e in events)

    immediate = _filter_for_summary(layers.get('immediate', []))
    recent = _filter_for_summary(layers.get('recent', []))
    background = _filter_for_summary(layers.get('background', []))[-5:]

    total = len(immediate) + len(recent) + len(background)

    raw_context = f"""[IMMEDIATE EVENTS (Last 10s)]
{format_layer(immediate)}

[RECENT CONTEXT (Last 30s)]
{format_layer(recent)}

[BACKGROUND CONTEXT (Earlier)]
{format_layer(background)}
"""

    prompt = f"""You are a situation summarizer for a Twitch streaming AI.

Your job is to describe what is CURRENTLY observable, grounded strictly in the
events listed below. You are not a writer, a comedian, or a storyteller.

EVIDENCE:
{raw_context}

HARD RULES — these override any creative instinct:
1. Use ONLY information present in the events above. Do not infer entities,
   characters, objects, locations, emotions, or events that are not stated.
2. IMMEDIATE EVENTS are ground truth for the current state. If IMMEDIATE
   contradicts BACKGROUND, IMMEDIATE wins (e.g. if BACKGROUND said oxygen
   was low and IMMEDIATE says oxygen is 75%, the current state is 75%).
3. BACKGROUND is historical context only. Never lead a summary with it.
   Never use it as the "current" situation.
4. If IMMEDIATE is "(empty)", the current state is unknown. Say so plainly
   ("No new activity in the last 10 seconds; last observed: ...") rather than
   inventing detail.
5. If ALL THREE layers are "(empty)", set summary to exactly:
   "(No current activity observed.)"
6. Do not echo or extend any prior commentary. You have not been shown
   Nami's previous replies for a reason — do not invent them either.
7. No metaphors, no jokes, no "vibe" descriptions. Describe what is on screen
   or in the audio, factually. Other components handle the personality layer.

Produce one JSON object with these fields:
- summary: 1-2 sentences. Plain factual description of current state.
- prediction: 1 sentence. What might happen next, based only on evidence above.
  Use "Unknown" if there is no basis for a prediction.
- conversation_state: one of [{", ".join(s.name for s in ConversationState)}]
- flow_state: one of [{", ".join(s.name for s in FlowState)}]
- user_intent: one of [{", ".join(s.name for s in UserIntent)}]
"""
    return raw_context, prompt, total


_SUMMARY_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "prediction": {"type": "string"},
        "conversation_state": {
            "type": "string",
            "enum": [s.name for s in ConversationState],
        },
        "flow_state": {
            "type": "string",
            "enum": [s.name for s in FlowState],
        },
        "user_intent": {
            "type": "string",
            "enum": [s.name for s in UserIntent],
        },
    },
    "required": ["summary", "prediction", "conversation_state", "flow_state", "user_intent"],
}


async def generate_summary(store: ContextStore):
    layers = store.get_all_events_for_summary()
    raw_context, full_prompt, total_events = build_summary_prompt(layers)

    # Hard short-circuit: no events, no LLM call, no hallucination possible.
    if total_events == 0:
        store.set_summary(NO_ACTIVITY_SUMMARY, raw_context, [], [], "Unknown")
        return

    if gemini_summary_model is None:
        # Gemini not initialized (no API key, init failed). Leave the previous
        # summary in place rather than risk an Ollama hallucination here.
        print("[Analyst] ⚠️  Gemini summary model unavailable — skipping summary cycle")
        return

    try:
        response = await asyncio.wait_for(
            gemini_summary_model.generate_content_async(
                full_prompt,
                generation_config={
                    "temperature": 0.1,
                    "response_mime_type": "application/json",
                    "response_schema": _SUMMARY_RESPONSE_SCHEMA,
                },
            ),
            timeout=OLLAMA_TIMEOUT_SUMMARY,
        )
    except asyncio.TimeoutError:
        print(f"⏱️  [Analyst] generate_summary timed out after {OLLAMA_TIMEOUT_SUMMARY}s")
        log_error("analyst.generate_summary", "gemini timeout", timeout_s=OLLAMA_TIMEOUT_SUMMARY)
        return
    except Exception as e:
        kind = type(e).__name__
        detail = str(e) or repr(e)
        print(f"[Analyst] Summary error ({kind}): {detail}")
        log_error("analyst.generate_summary", "gemini error", exc=e)
        return

    try:
        data = json.loads(response.text)
    except (json.JSONDecodeError, AttributeError, ValueError) as e:
        # Schema enforcement should make this rare, but guard anyway. Write
        # the unavailable marker so the UI shows something honest instead of
        # leaving a stale summary that looks current.
        print(f"[Analyst] Summary JSON parse failed: {e}")
        store.set_summary(UNAVAILABLE_SUMMARY, raw_context, [], [], "Unknown")
        return

    summary_text = (data.get("summary") or UNAVAILABLE_SUMMARY).strip()
    prediction_text = (data.get("prediction") or "Unknown").strip()

    conv_state = data.get("conversation_state")
    if conv_state in ConversationState.__members__:
        store.set_conversation_state(ConversationState[conv_state])

    flow_state = data.get("flow_state")
    if flow_state in FlowState.__members__:
        store.set_flow_state(FlowState[flow_state])

    user_intent = data.get("user_intent")
    if user_intent in UserIntent.__members__:
        store.set_user_intent(UserIntent[user_intent])

    store.set_summary(summary_text, raw_context, [], [], prediction_text)
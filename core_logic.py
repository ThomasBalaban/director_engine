# Save as: director_engine/core_logic.py
import asyncio
import time
from typing import Dict, Any, Optional
import config
import services.llm_analyst as llm_analyst
from scoring import calculate_event_score, EventScore
from context.context_store import EventItem
from config import InputSource
import shared 

# --- AI CONTEXT INFERENCE STATE ---
last_context_inference_time = 0
CONTEXT_INFERENCE_INTERVAL = 45.0  # Check every 45 seconds (non-blocking anyway)
last_inferred_game = None
last_inferred_context = None

def _handle_context_inference_result(result: Dict[str, str]):
    """
    Callback for when context inference completes.
    This runs in the main event loop but doesn't block anything.
    """
    global last_inferred_game, last_inferred_context
    
    if not result:
        return
        
    new_game = result.get('game', 'Unknown')
    new_context = result.get('context', '')
    
    # Check if we have a meaningful update
    game_changed = new_game != last_inferred_game and new_game != 'Unknown'
    context_changed = new_context != last_inferred_context and new_context
    
    if game_changed or context_changed:
        print(f"🤖 [AI Context] Inferred: {new_game} | {new_context[:50]}...")
        
        # Update if not locked
        if not shared.is_context_locked() and context_changed:
            if shared.set_manual_context(new_context, from_ai=True):
                last_inferred_context = new_context
        
        # Emit the suggestion to UI (even if locked, UI can show it)
        shared.emit_ai_context_suggestion(
            streamer=None,  # Don't suggest streamer changes for now
            context=new_context if context_changed else None
        )
        
        if game_changed:
            last_inferred_game = new_game

# --- EVENT PROCESSOR ---
async def process_engine_event(source: config.InputSource, text: str, metadata: Dict[str, Any] = {}, username: Optional[str] = None):
    if source in [config.InputSource.DIRECT_MICROPHONE, config.InputSource.TWITCH_MENTION]:
        shared.clear_user_awaiting()
    
    # 1. UI Emit
    if source == config.InputSource.VISUAL_CHANGE:
        shared.emit_vision_context(text)
    elif source in [config.InputSource.MICROPHONE, config.InputSource.DIRECT_MICROPHONE]:
        shared.emit_spoken_word_context(text)
    elif source == config.InputSource.AMBIENT_AUDIO:
        shared.emit_audio_context(text, is_partial=metadata.get("is_partial", False))
    elif source in [config.InputSource.TWITCH_CHAT, config.InputSource.TWITCH_MENTION]:
        shared.emit_twitch_message(username or "Chat", text)
    
    # 2. User Profile Update
    if username:
        profile = shared.profile_manager.get_profile(username)
        shared.store.set_active_user(profile)

    # 3. Handle Bot Self-Reply
    if source == config.InputSource.BOT_TWITCH_REPLY:
        zero_score = EventScore()
        shared.store.add_event(source, text, metadata, zero_score)
        shared.emit_twitch_message(username or "Nami", text)
        shared.behavior_engine.register_bot_action(shared.store, text)
        shared.energy_system.spend(config.ENERGY_COST_REPLY)
        return
    
    # 4. Scoring & Storage
    heuristic_score: EventScore = calculate_event_score(source, metadata, config.SOURCE_WEIGHTS)
    event = shared.store.add_event(source, text, metadata, heuristic_score)
    shared.emit_event_scored(event)
    
    # 5. Debt Check
    if source in [config.InputSource.MICROPHONE, config.InputSource.DIRECT_MICROPHONE]:
         shared.behavior_engine.check_debt_resolution(shared.store, text)

    # 6. Event Bundling
    bundle_event_created = False
    if source in [config.InputSource.DIRECT_MICROPHONE, config.InputSource.MICROPHONE] and heuristic_score.interestingness >= 0.6:
        shared.store.set_pending_speech(event)
    elif source not in [config.InputSource.DIRECT_MICROPHONE, config.InputSource.MICROPHONE] and heuristic_score.interestingness >= 0.7:
        pending_speech = shared.store.get_and_clear_pending_speech(max_age_seconds=3.0)
        if pending_speech:
            bundle_text = f"User reacted with '{pending_speech.text}' to: '{event.text}'"
            bundle_metadata = {**metadata, "is_bundle": True, "speech_text": pending_speech.text, "event_text": event.text}
            bundle_score = EventScore(interestingness=1.0, urgency=0.9, conversational_value=1.0, topic_relevance=1.0)
            bundle_event = shared.store.add_event(event.source, bundle_text, bundle_metadata, bundle_score)
            shared.emit_event_scored(bundle_event)
            asyncio.create_task(llm_analyst.analyze_and_update_event(
                bundle_event, shared.store, shared.profile_manager, handle_analysis_complete
            ))
            bundle_event_created = True

    # 7. Attention & Analysis
    if not bundle_event_created:
        attended_event = shared.behavior_engine.direct_attention(shared.store, [event])
        if attended_event:
            if heuristic_score.interestingness >= config.OLLAMA_TRIGGER_THRESHOLD:
                asyncio.create_task(llm_analyst.analyze_and_update_event(
                    event, shared.store, shared.profile_manager, handle_analysis_complete
                ))
            elif heuristic_score.urgency >= shared.adaptive_ctrl.current_threshold:
                 if shared.energy_system.can_afford(config.ENERGY_COST_INTERJECTION):
                     asyncio.create_task(llm_analyst.analyze_and_update_event(
                        event, shared.store, shared.profile_manager, handle_analysis_complete
                    ))

def handle_analysis_complete(event: EventItem):
    shared.emit_event_scored(event)

# --- TICKERS ---
async def reflex_ticker():
    while not shared.server_ready: await asyncio.sleep(0.1)
    print("✅ Reflex ticker starting (High Frequency)")
    await shared.speech_dispatcher.initialize()
    
    while True:
        try:
            shared.behavior_engine.update_goal(shared.store)
            chat_vel, energy_level = shared.store.get_activity_metrics()
            shared.adaptive_ctrl.update(chat_vel, energy_level)
            
            directive = shared.decision_engine.generate_directive(shared.store, shared.behavior_engine, shared.adaptive_ctrl, shared.energy_system)
            shared.store.set_directive(directive)
            
            # Use the new check that respects user-response state
            if not shared.should_suppress_idle():
                thought_text = await shared.behavior_engine.check_internal_monologue(shared.store)
                if thought_text:
                    print(f"💡 [Reflex] Thought: {thought_text}")
                    thought_event = shared.store.add_event(
                        config.InputSource.INTERNAL_THOUGHT, thought_text,
                        {"type": "shower_thought", "goal": "fill_silence"},
                        EventScore(interestingness=0.95, conversational_value=1.0, urgency=0.8)
                    )
                    shared.emit_event_scored(thought_event)

                speech_decision = shared.speech_dispatcher.evaluate(shared.store, shared.behavior_engine, shared.energy_system, directive)
                if speech_decision:
                    print(f"🎤 [Reflex] Trigger: {speech_decision.reason}")
                    await shared.speech_dispatcher.dispatch(speech_decision, shared.energy_system)
                    
                callback_text = shared.behavior_engine.check_callbacks(shared.store)
                if callback_text:
                    cb_event = shared.store.add_event(
                        config.InputSource.INTERNAL_THOUGHT, callback_text,
                        {"type": "callback", "goal": "context_continuity"},
                        EventScore(interestingness=0.7, conversational_value=0.8)
                    )
                    shared.emit_event_scored(cb_event)
            else:
                if shared.awaiting_user_response:
                    pass  # Silently wait for user to respond
                    
        except Exception as e:
            print(f"⚠️ [Reflex] Error: {e}")
        await asyncio.sleep(1.0)


def _build_memory_query(store, summary_data: Dict[str, Any]) -> str:
    """
    Build a rich query for semantic memory retrieval.
    Combines recent speech, visual context, and topics for better matching.
    """
    query_parts = []
    
    # 1. Recent speech (what the user/streamer actually said)
    layers = store.get_all_events_for_summary()
    recent_speech = [
        e.text for e in layers['immediate'] + layers['recent'] 
        if e.source in [InputSource.MICROPHONE, InputSource.DIRECT_MICROPHONE]
    ][-3:]  # Last 3 speech events
    
    if recent_speech:
        query_parts.extend(recent_speech)
    
    # 2. Recent visual keywords (what's on screen)
    recent_visual = [
        e.text for e in layers['immediate'] 
        if e.source == InputSource.VISUAL_CHANGE
    ][-1:]  # Most recent visual
    
    if recent_visual:
        # Take just the first 100 chars of visual to avoid noise
        query_parts.append(recent_visual[0][:100])
    
    # 3. Current topics from summary
    topics = summary_data.get('topics', [])
    if topics:
        query_parts.extend(topics[:3])
    
    # 4. Fallback to summary if nothing else
    if not query_parts:
        summary = summary_data.get('summary', '')
        if summary:
            query_parts.append(summary)
    
    # Combine into a single query string
    query = " ".join(query_parts)
    
    # Limit length to avoid issues
    if len(query) > 500:
        query = query[:500]
    
    return query

        
async def summary_ticker():
    global last_context_inference_time
    
    while not shared.server_ready: await asyncio.sleep(0.1)
    print("✅ Summary ticker starting (Low Frequency)")
    await asyncio.sleep(5) 
    
    while True:
        try:
            await llm_analyst.generate_summary(shared.store)
            shared.adaptive_ctrl.process_feedback(shared.store) 
            shared.scene_manager.update_scene(shared.store)
            
            patterns = shared.correlation_engine.correlate(shared.store)
            for pat in patterns:
                sys_event = shared.store.add_event(config.InputSource.SYSTEM_PATTERN, pat['text'], pat['metadata'], pat['score'])
                shared.emit_event_scored(sys_event)

            shared.memory_optimizer.decay_memories(shared.store)
            await shared.context_compressor.run_compression_cycle(shared.store)

            summary_data = shared.store.get_summary_data()
            
            # --- IMPROVED: Build a richer query for memory retrieval ---
            current_query = _build_memory_query(shared.store, summary_data)

            # --- DEBUG: Log memory retrieval ---
            print(f"🧠 [Memory] Query: '{current_query[:60]}...' " if len(current_query) > 60 else f"🧠 [Memory] Query: '{current_query}'")
            print(f"🧠 [Memory] Total memories in store: {len(shared.store.all_memories)}")

            smart_memories = shared.memory_optimizer.retrieve_relevant_memories(shared.store, current_query, limit=5)
            
            print(f"🧠 [Memory] Retrieved {len(smart_memories)} relevant memories")
            for i, m in enumerate(smart_memories[:3]):
                content = m.memory_text or m.text
                print(f"   {i+1}. [{m.source.name}] {content[:50]}... (score: {m.score.interestingness:.2f})")

            memories_list = [
                {
                    "source": m.source.name, 
                    "text": m.memory_text or m.text, 
                    "score": round(m.score.interestingness, 2), 
                    "type": "memory"
                } for m in smart_memories
            ]

            # Add narrative history entries
            if shared.store.narrative_log:
                for i, story in enumerate(reversed(shared.store.narrative_log[-3:])):
                    memories_list.insert(i, {
                        "source": "NARRATIVE_HISTORY", 
                        "text": f"Earlier: {story}", 
                        "score": 1.0, 
                        "type": "narrative"
                    })

            chat_vel, energy_level = shared.store.get_activity_metrics()
            
            # --- NON-BLOCKING AI CONTEXT INFERENCE ---
            now = time.time()
            if now - last_context_inference_time >= CONTEXT_INFERENCE_INTERVAL:
                last_context_inference_time = now
                
                if not shared.is_context_locked():
                    llm_analyst.start_context_inference_task(
                        shared.store, 
                        callback=_handle_context_inference_result
                    )
            
            shared.emit_director_state(
                summary=summary_data['summary'], raw_context=summary_data['raw_context'],
                prediction=summary_data['prediction'], mood=summary_data['mood'],
                conversation_state=summary_data['conversation_state'], flow_state=summary_data['flow'],
                user_intent=summary_data['intent'], active_user=shared.store.active_user_profile,
                memories=memories_list, directive=summary_data['directive'].to_dict() if summary_data['directive'] else None,
                adaptive_state={
                    "threshold": round(shared.adaptive_ctrl.current_threshold, 2), "state": shared.adaptive_ctrl.state_label,
                    "chat_velocity": round(chat_vel, 1), "energy": round(energy_level, 2),
                    "social_battery": shared.energy_system.get_status(),
                    "current_goal": shared.behavior_engine.current_goal.name, "current_scene": summary_data['scene']
                }
            )

            stale_event = shared.store.get_stale_event_for_analysis()
            if stale_event:
                asyncio.create_task(llm_analyst.analyze_and_update_event(stale_event, shared.store, shared.profile_manager, handle_analysis_complete))
        except Exception as e:
            print(f"[Director] Error in summary ticker: {e}")
            import traceback
            traceback.print_exc()
        await asyncio.sleep(config.SUMMARY_INTERVAL_SECONDS)
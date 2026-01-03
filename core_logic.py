# Save as: director_engine/core_logic.py
import asyncio
import time
from typing import Dict, Any, Optional
import config
import llm_analyst
from scoring import calculate_event_score, EventScore
from context_store import EventItem
import shared 

# --- EVENT PROCESSOR ---
async def process_engine_event(source: config.InputSource, text: str, metadata: Dict[str, Any] = {}, username: Optional[str] = None):
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
        except Exception as e:
            print(f"⚠️ [Reflex] Error: {e}")
        await asyncio.sleep(1.0)

async def summary_ticker():
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
            current_query = summary_data.get('summary', "") or " ".join(summary_data.get('topics', []))

            smart_memories = shared.memory_optimizer.retrieve_relevant_memories(shared.store, current_query, limit=5)
            memories_list = [{"source": m.source.name, "text": m.memory_text or m.text, "score": round(m.score.interestingness, 2), "type": "memory"} for m in smart_memories]

            if shared.store.narrative_log:
                for i, story in enumerate(reversed(shared.store.narrative_log[-3:])):
                    memories_list.insert(i, {"source": "NARRATIVE_HISTORY", "text": f"Previously: {story}", "score": 1.0, "type": "narrative"})

            chat_vel, energy_level = shared.store.get_activity_metrics()
            
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
        await asyncio.sleep(config.SUMMARY_INTERVAL_SECONDS)
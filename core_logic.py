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
from typing import Dict, Any
from config import SceneType, UserIntent, ConversationState, InputSource


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
    
    # =====================================================
    # INTERRUPT CHECK: Only TWO cases can interrupt Nami:
    #   1. Mic input (DIRECT_MICROPHONE) - user speaking directly
    #   2. "peepingotter" says "nami" in Twitch chat
    # All other sources are normal (no interrupt capability).
    # =====================================================
    is_direct_address = source == config.InputSource.DIRECT_MICROPHONE
    
    if source == config.InputSource.TWITCH_MENTION:
        mention_username = (username or metadata.get('username', '')).lower()
        mention_text = text.lower()
        if mention_username == 'peepingotter' and 'nami' in mention_text:
            is_direct_address = True
    
    if is_direct_address:
        shared.clear_user_awaiting()
        
        # INTERRUPT: If Nami is speaking, cut her off immediately (ONE-SHOT)
        if shared.is_nami_speaking():
            interrupt_reason = f"direct_{'mic' if source == config.InputSource.DIRECT_MICROPHONE else 'mention'}"
            was_interrupted = shared.interrupt_nami(reason=interrupt_reason)
            if was_interrupted:
                print(f"🛑 [CoreLogic] Nami interrupted for direct address: {text[:50]}...")
                await asyncio.sleep(0.05)
    
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

    # Track conversation threads
    if source in [config.InputSource.MICROPHONE, config.InputSource.DIRECT_MICROPHONE]:
        detected_topic = metadata.get('topic')
        importance = metadata.get('importance', 0.5)
        
        shared.store.thread_manager.track_user_statement(
            text=text,
            detected_topic=detected_topic,
            importance=importance
        )

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
    
    # BOOST: Direct addresses get maximum scores to ensure they trigger interjection
    if is_direct_address:
        heuristic_score.interestingness = max(heuristic_score.interestingness, 0.95)
        heuristic_score.urgency = max(heuristic_score.urgency, 0.95)
        heuristic_score.conversational_value = max(heuristic_score.conversational_value, 0.95)
        metadata['is_direct_address'] = True
        metadata['interrupt_priority'] = True
    
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
        # DIRECT ADDRESSES: Always analyze immediately, skip attention gating
        if is_direct_address:
            print(f"🎯 [CoreLogic] Direct address - fast-tracking to analysis: {text[:50]}...")
            asyncio.create_task(llm_analyst.analyze_and_update_event(
                event, shared.store, shared.profile_manager, handle_analysis_complete
            ))
            # Also trigger immediate interjection (don't wait for LLM analysis)
            await llm_analyst.trigger_nami_interjection(event, 1.0, is_interrupt=True)
        else:
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


def build_smart_memory_query(store, summary_data: Dict[str, Any]) -> str:
    """
    Build a SMART query for semantic memory retrieval.
    Adapts based on:
    - Current scene type (different memory priorities)
    - User intent (what kind of memories matter now)  
    - Conversation state (support vs entertainment)
    
    IMPROVEMENTS:
    - Scene-aware keyword injection
    - Intent-driven memory focus
    - Reduced noise from irrelevant context
    - Priority weighting for different contexts
    """
    query_parts = []
    priority_keywords = []
    
    # === SCENE-AWARE MEMORY FOCUS ===
    scene = store.current_scene
    
    if scene == SceneType.HORROR_TENSION:
        # Prioritize past scary moments, reactions to fear
        priority_keywords.extend(["scared", "tension", "jumpscare", "afraid", "creepy"])
        print(f"🧠 [Memory Query] Horror scene - prioritizing fear-related memories")
    
    elif scene == SceneType.COMBAT_HIGH:
        # Prioritize victories, defeats, skill patterns
        priority_keywords.extend(["won", "died", "combat", "fight", "boss", "victory", "defeat"])
        print(f"🧠 [Memory Query] Combat scene - prioritizing performance memories")
    
    elif scene == SceneType.COMEDY_MOMENT:
        # Prioritize past funny moments, memes
        priority_keywords.extend(["funny", "laugh", "joke", "meme", "chat said"])
        print(f"🧠 [Memory Query] Comedy scene - prioritizing humor memories")
    
    elif scene == SceneType.EXPLORATION:
        # Prioritize discoveries, locations
        priority_keywords.extend(["found", "discovered", "location", "area", "new"])
        print(f"🧠 [Memory Query] Exploration scene - prioritizing discovery memories")
    
    elif scene == SceneType.MENUING:
        # Prioritize past decisions, choices made
        priority_keywords.extend(["chose", "selected", "menu", "inventory", "equipped"])
    
    # === INTENT-AWARE ADDITIONS ===
    intent = store.current_intent

    if intent == UserIntent.VALIDATION:
        # Pull memories of past praise/criticism
        validation_keywords = ["good job", "well done", "nice", "amazing", "great"]
        priority_keywords.extend(validation_keywords)
        # Ensure validation keywords always appear in query
        query_parts.insert(0, "praise validation " + " ".join(validation_keywords[:3]))
        print(f"🧠 [Memory Query] User seeking validation - adding praise memories")

    elif intent == UserIntent.HELP_SEEKING:
        # Pull memories of solutions, tips, strategies
        priority_keywords.extend(["how to", "solution", "fix", "try", "strategy", "worked"])
        print(f"🧠 [Memory Query] User seeking help - adding solution memories")
    
    elif intent == UserIntent.PROVOKING:
        # Pull memories of past roasts, banter
        priority_keywords.extend(["roasted", "skill issue", "trash", "bad", "terrible"])
        print(f"🧠 [Memory Query] User provoking - adding banter memories")
    
    elif intent == UserIntent.INFO_SEEKING:
        # Pull memories with facts, information
        priority_keywords.extend(["learned", "discovered", "found out", "fact"])
    
    # === CONVERSATION STATE CONTEXT ===
    conv_state = store.current_conversation_state
    
    if conv_state == ConversationState.FRUSTRATED:
        # Pull memories of past frustrations to validate/empathize
        priority_keywords.extend(["frustrated", "angry", "rage", "annoyed", "difficult"])
        print(f"🧠 [Memory Query] User frustrated - adding empathy context")
    
    elif conv_state == ConversationState.CELEBRATORY:
        # Pull memories of past wins
        priority_keywords.extend(["celebrate", "won", "success", "finally", "beat"])
        print(f"🧠 [Memory Query] Celebration mode - adding victory memories")
    
    elif conv_state == ConversationState.STORYTELLING:
        # Pull related narrative memories
        priority_keywords.extend(["story", "told", "happened", "remember when"])
    
    # Add priority keywords first (weighted higher in semantic search)
    if priority_keywords:
        # Deduplicate and take top 5
        unique_keywords = list(dict.fromkeys(priority_keywords))[:5]
        query_parts.append(" ".join(unique_keywords))
    
    # === IMMEDIATE CONTEXT (Always include) ===
    layers = store.get_all_events_for_summary()
    
    # Recent speech - what user JUST said (most important)
    recent_speech = [
        e.text for e in layers['immediate'] + layers['recent'][:2]
        if e.source in [InputSource.MICROPHONE, InputSource.DIRECT_MICROPHONE]
    ][-2:]  # Last 2 only (more focused than 3)
    
    if recent_speech:
        query_parts.extend(recent_speech)
        print(f"🧠 [Memory Query] Including recent speech: {len(recent_speech)} items")
    
    # Current visual context (if relevant to scene)
    # Skip visuals during menuing/downtime to reduce noise
    if scene not in [SceneType.TECHNICAL_DOWNTIME, SceneType.MENUING]:
        recent_visual = [
            e.text for e in layers['immediate']
            if e.source == InputSource.VISUAL_CHANGE
        ][-1:]  # Only most recent
        
        if recent_visual:
            # Extract just key nouns/entities (first 60 chars to avoid bloat)
            visual_snippet = recent_visual[0][:60]
            query_parts.append(visual_snippet)
    
    # === TOPICS & ENTITIES (Lower priority) ===
    # Only add if we don't have enough context yet
    if len(query_parts) < 3:
        topics = summary_data.get('topics', [])
        if topics:
            query_parts.extend(topics[:2])
        
        entities = summary_data.get('entities', [])
        if entities:
            query_parts.extend(entities[:2])
    
    # === FALLBACK ===
    if not query_parts:
        # Absolute fallback - use summary
        summary = summary_data.get('summary', '')
        if summary:
            query_parts.append(summary[:100])
        else:
            # Last resort - generic query
            query_parts.append("recent events gameplay")
    
    # Combine into query string
    query = " ".join(query_parts)
    
    # Limit length to prevent token bloat
    if len(query) > 500:
        query = query[:500]
    
    # Debug output
    print(f"🧠 [Memory Query] Final query ({len(query)} chars): '{query[:80]}...'")
    
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
            current_query = build_smart_memory_query(shared.store, summary_data)

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
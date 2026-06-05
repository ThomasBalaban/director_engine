# Save as: director_engine/core_logic.py
"""
Core Logic — The brain's main processing loop.

NOTE: All speaking state checks have been removed. The brain fires
speech requests freely via prompt_client. The Prompt Service (port 8001)
decides what actually reaches Nami.
"""

import asyncio
import time
from typing import Dict, Any, Optional
import config
import services.llm_analyst as llm_analyst
import services.prompt_client as prompt_client
from scoring import calculate_event_score, EventScore
from context.context_store import EventItem
from config import InputSource, SceneType, UserIntent, ConversationState
import shared


# --- AI CONTEXT INFERENCE STATE ---
last_context_inference_time = 0
CONTEXT_INFERENCE_INTERVAL = 45.0
last_inferred_game = None
last_inferred_context = None


def _handle_context_inference_result(result: Dict[str, str]):
    global last_inferred_game, last_inferred_context
    
    if not result:
        return
        
    new_game = result.get('game', 'Unknown')
    new_context = result.get('context', '')
    
    game_changed = new_game != last_inferred_game and new_game != 'Unknown'
    context_changed = new_context != last_inferred_context and new_context
    
    if game_changed or context_changed:
        print(f"🤖 [AI Context] Inferred: {new_game} | {new_context[:50]}...")
        
        if not shared.is_context_locked() and context_changed:
            if shared.set_manual_context(new_context, from_ai=True):
                last_inferred_context = new_context
        
        shared.emit_ai_context_suggestion(
            streamer=None,
            context=new_context if context_changed else None
        )
        
        if game_changed:
            last_inferred_game = new_game


# ── Per-source rate limiting ─────────────────────────────────────────────────
# Vision events arrive at ~30/min sustained, with bursts. Each one triggers
# scoring → store mutation → emit → maybe analyze. Under load that's enough
# to add measurable latency to the asyncio loop. Drop excess vision events
# at the front door so only ~1 every 2s reaches the heavy path.
#
# High-priority sources (mic, direct mentions, twitch interactions) are
# NEVER rate-limited — they're rare and conversation-critical.
_VISION_MIN_INTERVAL_S = 2.0
_last_vision_event_at = 0.0
_vision_events_dropped_total = 0


def _should_accept_event(source: config.InputSource) -> bool:
    """Front-door rate limit. Returns False to drop the event entirely."""
    global _last_vision_event_at, _vision_events_dropped_total
    if source != config.InputSource.VISUAL_CHANGE:
        return True
    now = _time.monotonic()
    if now - _last_vision_event_at < _VISION_MIN_INTERVAL_S:
        _vision_events_dropped_total += 1
        if _vision_events_dropped_total <= 5 or _vision_events_dropped_total % 50 == 0:
            print(
                f"⚠️  [CoreLogic] DROPPED vision event — rate limit "
                f"({_vision_events_dropped_total} total dropped)"
            )
        return False
    _last_vision_event_at = now
    return True


# --- EVENT PROCESSOR ---
async def process_engine_event(
    source: config.InputSource,
    text: str,
    metadata: Dict[str, Any] = {},
    username: Optional[str] = None,
):
    """
    Process an incoming event from any source.

    For direct addresses (mic with "nami", or handler twitch mention),
    we send an interrupt request to the prompt service. We don't check
    local speaking state — the prompt service handles that.
    """
    # Front-door rate limiting (vision only — see _should_accept_event)
    if not _should_accept_event(source):
        return

    # Yield to the loop — gives socket.io PINGs and other tasks a chance.
    await asyncio.sleep(0)
    # --- DETECT DIRECT ADDRESS ---
    is_direct_address = source == config.InputSource.DIRECT_MICROPHONE
    
    if source == config.InputSource.TWITCH_MENTION:
        # twitch_service already gated this on the (nami|peepingnami) regex,
        # so the source tag itself is sufficient — every chat user who @s her
        # counts as a direct address.
        is_direct_address = True
    
    if is_direct_address:
        # Tell prompt service: user spoke, clear awaiting state
        asyncio.create_task(prompt_client.notify_user_responded())

    
    # 2. User Profile Update
    if username:
        profile = shared.profile_manager.get_profile(username)
        shared.store.set_active_user(profile)
    elif source in (config.InputSource.MICROPHONE, config.InputSource.DIRECT_MICROPHONE):
        profile = shared.profile_manager.get_profile(shared.get_current_streamer())
        shared.store.set_active_user(profile)

    # 3. Track conversation threads
    if source in [config.InputSource.MICROPHONE, config.InputSource.DIRECT_MICROPHONE]:
        detected_topic = metadata.get('topic')
        importance = metadata.get('importance', 0.5)
        
        shared.store.thread_manager.track_user_statement(
            text=text,
            detected_topic=detected_topic,
            importance=importance
        )

    # 4. Handle Bot Self-Reply
    if source == config.InputSource.BOT_TWITCH_REPLY:
        zero_score = EventScore()
        shared.store.add_event(source, text, metadata, zero_score)
        shared.behavior_engine.register_bot_action(shared.store, text)
        shared.energy_system.spend(config.ENERGY_COST_REPLY)
        return
    
    # 5. Scoring & Storage
    heuristic_score: EventScore = calculate_event_score(source, metadata, config.SOURCE_WEIGHTS)
    
    # BOOST: Direct addresses get maximum scores
    if is_direct_address:
        heuristic_score.interestingness = max(heuristic_score.interestingness, 0.95)
        heuristic_score.urgency = max(heuristic_score.urgency, 0.95)
        heuristic_score.conversational_value = max(heuristic_score.conversational_value, 0.95)
        metadata['is_direct_address'] = True
        metadata['interrupt_priority'] = True
    
    event = shared.store.add_event(source, text, metadata, heuristic_score)
    shared.emit_event_scored(event)
    
    # 6. Debt Check
    if source in [config.InputSource.MICROPHONE, config.InputSource.DIRECT_MICROPHONE]:
        shared.behavior_engine.check_debt_resolution(shared.store, text)

    # 7. Event Bundling
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

    # 8. Attention & Analysis
    if not bundle_event_created:
        if is_direct_address:
            # Direct address: fast-track to analysis + send interrupt to prompt service
            print(f"🎯 [CoreLogic] Direct address - fast-tracking: {text[:50]}...")
            asyncio.create_task(llm_analyst.analyze_and_update_event(
                event, shared.store, shared.profile_manager, handle_analysis_complete
            ))
            # Send as interrupt — prompt service will gate it
            asyncio.create_task(prompt_client.send_interrupt(
                content=event.text,
                source=f"DIRECTOR_{event.source.name}",
                trigger=f"direct_{'mic' if source == config.InputSource.DIRECT_MICROPHONE else 'mention'}",
                event_id=event.id,
                metadata={
                    'is_direct_address': True,
                    **{k: v for k, v in event.metadata.items() if k != 'is_direct_address'}
                }
            ))
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

# ── Freeze-diagnostic instrumentation ─────────────────────────────────────────
# Updated as the reflex_ticker progresses. heartbeat_ticker reads this to
# report which step the reflex loop was on when (if) it stops making progress.
import time as _time

_reflex_state = {
    "step": "boot",
    "step_started_at": _time.monotonic(),
    "iteration": 0,
    # Tracked here for the file_heartbeat diagnostic + backoff logic
    "last_speech_trigger_at": 0.0,
    "last_iter_actual_duration_s": 0.0,
    "last_iter_loop_drift_s": 0.0,
    "recent_thought_timeouts": 0,
    "recent_thought_timeouts_window_start": _time.monotonic(),
    "backoff_active_until": 0.0,
}


def _mark_reflex_step(name: str) -> None:
    _reflex_state["step"] = name
    _reflex_state["step_started_at"] = _time.monotonic()


# ── Reflex-loop self-protection knobs ────────────────────────────────────────
# Tunable thresholds. All times in seconds.
MIN_SPEECH_TRIGGER_INTERVAL_S = 4.0       # cap director-side speech rate
THOUGHT_TIMEOUT_WINDOW_S       = 30.0     # rolling window for timeout counting
THOUGHT_TIMEOUT_BACKOFF_THRESHOLD = 3     # N timeouts in window → backoff
BACKOFF_DURATION_S             = 20.0     # how long to back off when overloaded
LOOP_DRIFT_BACKOFF_THRESHOLD_S = 3.0      # if sleep(1.0) takes 3s+, loop is starved


def _note_thought_timeout() -> None:
    """Track a generate_thought timeout in the rolling window."""
    now = _time.monotonic()
    win_start = _reflex_state["recent_thought_timeouts_window_start"]
    if now - win_start > THOUGHT_TIMEOUT_WINDOW_S:
        _reflex_state["recent_thought_timeouts"] = 0
        _reflex_state["recent_thought_timeouts_window_start"] = now
    _reflex_state["recent_thought_timeouts"] += 1
    if _reflex_state["recent_thought_timeouts"] >= THOUGHT_TIMEOUT_BACKOFF_THRESHOLD:
        _reflex_state["backoff_active_until"] = now + BACKOFF_DURATION_S
        print(
            f"🛑 [Reflex] Ollama overloaded ({_reflex_state['recent_thought_timeouts']} timeouts "
            f"in {THOUGHT_TIMEOUT_WINDOW_S:.0f}s) — backing off speech triggers for {BACKOFF_DURATION_S:.0f}s"
        )


def _is_in_backoff() -> bool:
    return _time.monotonic() < _reflex_state["backoff_active_until"]


async def reflex_ticker():
    """
    High-frequency tick. Generates thoughts and speech decisions.

    The brain generates freely — the prompt service gates delivery.
    No speaking-state checks here.
    """
    while not shared.server_ready:
        await asyncio.sleep(0.1)
    print("✅ Reflex ticker starting (High Frequency)")

    # Initialize prompt client
    await prompt_client.initialize()

    while True:
        iter_start = _time.monotonic()
        _reflex_state["iteration"] += 1
        try:
            # Yield to the loop early — gives the socket.io PING handler and
            # other hot tasks a chance to run before we hog CPU on sync work.
            await asyncio.sleep(0)

            _mark_reflex_step("update_host_state")
            shared.store.update_host_state()

            _mark_reflex_step("update_goal")
            shared.behavior_engine.update_goal(shared.store)

            _mark_reflex_step("get_activity_metrics")
            chat_vel, energy_level = shared.store.get_activity_metrics()
            shared.adaptive_ctrl.update(chat_vel, energy_level)

            _mark_reflex_step("generate_directive")
            directive = shared.decision_engine.generate_directive(
                shared.store, shared.behavior_engine, shared.adaptive_ctrl, shared.energy_system
            )
            shared.store.set_directive(directive)

            # Spawn (fire-and-forget) monologue generation. This does NOT
            # block on Ollama anymore — the background task adds the event
            # to the store directly when the LLM responds.
            _mark_reflex_step("kick_monologue")
            await shared.behavior_engine.check_internal_monologue(shared.store)

            # Another yield before potentially CPU-heavy speech dispatcher work
            await asyncio.sleep(0)

            # ── Speech dispatcher gating ─────────────────────────────────────
            # Three gates added to prevent the trigger flood:
            #   1. Backoff mode (Ollama overloaded) — skip entirely
            #   2. Min interval since last trigger — director-side cooldown
            #   3. Loop drift detection — if the loop is starved, don't pile on
            now = _time.monotonic()
            time_since_last_trigger = now - _reflex_state["last_speech_trigger_at"]
            should_evaluate_speech = (
                not _is_in_backoff()
                and time_since_last_trigger >= MIN_SPEECH_TRIGGER_INTERVAL_S
                and _reflex_state["last_iter_loop_drift_s"] < LOOP_DRIFT_BACKOFF_THRESHOLD_S
            )

            speech_decision = None
            if should_evaluate_speech:
                _mark_reflex_step("evaluate_speech")
                speech_decision = shared.speech_dispatcher.evaluate(
                    shared.store, shared.behavior_engine, shared.energy_system, directive
                )

            if speech_decision:
                print(f"🎤 [Reflex] Trigger: {speech_decision.reason}")
                _reflex_state["last_speech_trigger_at"] = now
                _mark_reflex_step("spend_energy")
                # Spend energy on our side
                shared.energy_system.spend(config.ENERGY_COST_INTERJECTION)
                _mark_reflex_step("dispatch_speech_task")
                # Send to prompt service (it decides whether to deliver)
                asyncio.create_task(prompt_client.request_speech(
                    trigger=speech_decision.reason,
                    content=speech_decision.content,
                    priority=speech_decision.priority,
                    source=speech_decision.source_info.get('source', 'DIRECTOR'),
                    event_id=speech_decision.source_info.get('event_id'),
                    metadata=speech_decision.source_info,
                ))

            # Check callbacks
            _mark_reflex_step("check_callbacks")
            callback_text = shared.behavior_engine.check_callbacks(shared.store)
            if callback_text:
                _mark_reflex_step("emit_callback_event")
                cb_event = shared.store.add_event(
                    config.InputSource.INTERNAL_THOUGHT, callback_text,
                    {"type": "callback", "goal": "context_continuity"},
                    EventScore(interestingness=0.7, conversational_value=0.8)
                )
                shared.emit_event_scored(cb_event)

            _mark_reflex_step("idle_sleep")

        except Exception as e:
            kind = type(e).__name__
            detail = str(e) or repr(e)
            print(f"⚠️ [Reflex] Error ({kind}) at step={_reflex_state['step']}: {detail}")

        # ── Loop drift measurement ────────────────────────────────────────────
        # Compare wall-clock elapsed against the 1.0s sleep target. If drift
        # exceeds the threshold, the loop is starved (sync work running too
        # long elsewhere). Next iteration's dispatcher will see this and skip.
        sleep_target = 1.0
        before_sleep = _time.monotonic()
        await asyncio.sleep(sleep_target)
        after_sleep = _time.monotonic()
        actual_sleep = after_sleep - before_sleep
        iter_total = after_sleep - iter_start
        _reflex_state["last_iter_actual_duration_s"] = round(iter_total, 3)
        _reflex_state["last_iter_loop_drift_s"] = round(actual_sleep - sleep_target, 3)


async def heartbeat_ticker():
    """
    Independent asyncio task that proves the event loop is alive.

    Stuck-detection is now based on iteration progression, NOT step age
    (step_age gets reset every _mark_reflex_step call and is misleading).

    - "stuck" = iteration counter hasn't advanced for > STUCK_THRESHOLD_S.
    - "recovered" only fires when iteration actually moves forward.

    Interpretation:
    - Heartbeat keeps logging while reflex iteration is frozen → loop is
      alive, the main task is stuck on a specific await (named in 'step').
    - Heartbeat ALSO stops → event loop itself is blocked. Compare against
      the thread watchdog: if that one keeps going, the asyncio loop is
      wedged; if both stop, the whole Python process is wedged.
    """
    STUCK_THRESHOLD_S = 10.0

    while not shared.server_ready:
        await asyncio.sleep(0.5)
    print("✅ Heartbeat starting (diagnostic, every 5s)")

    last_iteration = -1
    last_iteration_change_at = _time.monotonic()
    was_stuck = False
    stuck_warnings_emitted = 0

    while True:
        await asyncio.sleep(5.0)
        now = _time.monotonic()
        step = _reflex_state["step"]
        iteration = _reflex_state["iteration"]

        progressed = iteration != last_iteration
        if progressed:
            last_iteration = iteration
            last_iteration_change_at = now

        time_since_progress = now - last_iteration_change_at
        is_stuck = time_since_progress > STUCK_THRESHOLD_S

        if is_stuck:
            stuck_warnings_emitted += 1
            was_stuck = True
            print(
                f"🚨 [Heartbeat] reflex STUCK on step='{step}' — iteration={iteration} "
                f"has not advanced for {time_since_progress:.1f}s "
                f"(warning #{stuck_warnings_emitted}) | loop alive, main task hung"
            )
        else:
            if was_stuck and progressed:
                print(
                    f"✅ [Heartbeat] reflex RECOVERED — iteration advanced to {iteration} "
                    f"after {stuck_warnings_emitted} stuck warning(s)"
                )
                was_stuck = False
                stuck_warnings_emitted = 0
            print(f"💓 [Heartbeat] alive | reflex_step='{step}' iter={iteration}")


def _thread_watchdog_loop():
    """
    OS-thread watchdog. Does NOT use asyncio — runs in its own daemon thread,
    so it logs even if the entire asyncio event loop is blocked by a
    synchronous call.

    Diagnostic interpretation:
    - Asyncio heartbeat stops, this keeps logging → event loop is wedged
      by a sync call somewhere. Need run_in_executor for the suspect call.
    - Both stop → the whole Python process is stuck (OOM, GIL held by C
      extension, signal handler deadlock, swap thrash, etc).
    """
    import threading as _threading
    print(f"✅ Thread watchdog starting (diagnostic, every 10s, thread={_threading.current_thread().name})")
    while True:
        _time.sleep(10.0)
        step = _reflex_state["step"]
        iteration = _reflex_state["iteration"]
        print(f"🧵 [ThreadWatchdog] alive | reflex_step='{step}' iter={iteration}")


def start_thread_watchdog() -> None:
    """Start the thread-based watchdog. Called from main during boot."""
    import threading as _threading
    t = _threading.Thread(target=_thread_watchdog_loop, name="thread-watchdog", daemon=True)
    t.start()


def build_smart_memory_query(store, summary_data: Dict[str, Any]) -> str:
    """
    Build a SMART query for semantic memory retrieval.
    (Unchanged from original — pure brain logic)
    """
    query_parts = []
    priority_keywords = []
    
    scene = store.current_scene
    
    if scene == SceneType.HORROR_TENSION:
        priority_keywords.extend(["scared", "tension", "jumpscare", "afraid", "creepy"])
    elif scene == SceneType.COMBAT_HIGH:
        priority_keywords.extend(["won", "died", "combat", "fight", "boss", "victory", "defeat"])
    elif scene == SceneType.COMEDY_MOMENT:
        priority_keywords.extend(["funny", "laugh", "joke", "meme", "chat said"])
    elif scene == SceneType.EXPLORATION:
        priority_keywords.extend(["found", "discovered", "location", "area", "new"])
    elif scene == SceneType.MENUING:
        priority_keywords.extend(["chose", "selected", "menu", "inventory", "equipped"])
    
    intent = store.current_intent

    if intent == UserIntent.VALIDATION:
        validation_keywords = ["good job", "well done", "nice", "amazing", "great"]
        priority_keywords.extend(validation_keywords)
        query_parts.insert(0, "praise validation " + " ".join(validation_keywords[:3]))
    elif intent == UserIntent.HELP_SEEKING:
        priority_keywords.extend(["how to", "solution", "fix", "try", "strategy", "worked"])
    elif intent == UserIntent.PROVOKING:
        priority_keywords.extend(["roasted", "skill issue", "trash", "bad", "terrible"])
    elif intent == UserIntent.INFO_SEEKING:
        priority_keywords.extend(["learned", "discovered", "found out", "fact"])
    
    conv_state = store.current_conversation_state
    
    if conv_state == ConversationState.FRUSTRATED:
        priority_keywords.extend(["frustrated", "angry", "rage", "annoyed", "difficult"])
    elif conv_state == ConversationState.CELEBRATORY:
        priority_keywords.extend(["celebrate", "won", "success", "finally", "beat"])
    elif conv_state == ConversationState.STORYTELLING:
        priority_keywords.extend(["story", "told", "happened", "remember when"])
    
    if priority_keywords:
        unique_keywords = list(dict.fromkeys(priority_keywords))[:5]
        query_parts.append(" ".join(unique_keywords))
    
    layers = store.get_all_events_for_summary()
    
    recent_speech = [
        e.text for e in layers['immediate'] + layers['recent'][:2]
        if e.source in [InputSource.MICROPHONE, InputSource.DIRECT_MICROPHONE]
    ][-2:]
    
    if recent_speech:
        query_parts.extend(recent_speech)
    
    if scene not in [SceneType.TECHNICAL_DOWNTIME, SceneType.MENUING]:
        recent_visual = [
            e.text for e in layers['immediate']
            if e.source == InputSource.VISUAL_CHANGE
        ][-1:]
        
        if recent_visual:
            visual_snippet = recent_visual[0][:60]
            query_parts.append(visual_snippet)
    
    if len(query_parts) < 3:
        topics = summary_data.get('topics', [])
        if topics:
            query_parts.extend(topics[:2])
        entities = summary_data.get('entities', [])
        if entities:
            query_parts.extend(entities[:2])
    
    if not query_parts:
        summary = summary_data.get('summary', '')
        if summary:
            query_parts.append(summary[:100])
        else:
            query_parts.append("recent events gameplay")
    
    query = " ".join(query_parts)
    if len(query) > 500:
        query = query[:500]
    
    print(f"🧠 [Memory Query] Final query ({len(query)} chars): '{query[:80]}...'")
    return query

     
async def summary_ticker():
    global last_context_inference_time
    
    while not shared.server_ready:
        await asyncio.sleep(0.1)
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

            # --- DELETED LOCAL MEMORY DECAY & RETRIEVAL ---
            # Decay is now handled by the Memory Service microservice!
            
            await shared.context_compressor.run_compression_cycle(shared.store)
            summary_data = shared.store.get_summary_data()

            print(f"🧠 [Memory] Pending saves to Hub: {len(getattr(shared.store, 'pending_memories_to_save', []))}")

            # Prepare narrative history for the UI dashboard
            # (The actual long-term memories are fetched via the /memory_stats endpoint)
            memories_list = []
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
                asyncio.create_task(llm_analyst.analyze_and_update_event(
                    stale_event, shared.store, shared.profile_manager, handle_analysis_complete
                ))
        except Exception as e:
            print(f"[Director] Error in summary ticker: {e}")
            import traceback
            traceback.print_exc()
        await asyncio.sleep(config.SUMMARY_INTERVAL_SECONDS)
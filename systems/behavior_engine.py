# Save as: director_engine/systems/behavior_engine.py
import time
import random
import asyncio
from typing import List, Optional
from config import (
    BotGoal, InputSource, ConversationState, FlowState, SceneType,
    CURIOSITY_INTERVAL, CALLBACK_INTERVAL, OWNER_STREAMER_ID
)
from context.context_store import ContextStore, EventItem
from scoring import EventScore
from context.user_profile_manager import UserProfileManager
import services.llm_analyst as llm_analyst


class BehaviorEngine:
    def __init__(self):
        self.current_goal = BotGoal.OBSERVE
        self.last_curiosity_check = time.time()
        self.last_callback_check = time.time()
        self.attention_lock_duration = 5.0
        # Tracks whether a fire-and-forget generate_thought is currently in
        # flight. Prevents queueing duplicate tasks when Ollama is slow.
        self._monologue_task_inflight = False

    def update_goal(self, store: ContextStore):
        # Scene override: if Otter is quiet and game is busy, lock OBSERVE.
        # HOST_LOW_ENERGY does NOT override — we want the existing
        # INVESTIGATE/TROLL chaos to fire when there's room to fill.
        if store.current_scene == SceneType.HOST_FOCUSED_QUIET:
            self.current_goal = BotGoal.OBSERVE
            return

        chat_vel, stream_energy = store.get_activity_metrics()
        state = store.current_conversation_state
        if state == ConversationState.FRUSTRATED: self.current_goal = BotGoal.SUPPORT
        elif state == ConversationState.CELEBRATORY: self.current_goal = BotGoal.ENTERTAIN
        elif stream_energy > 0.8: self.current_goal = BotGoal.ENTERTAIN
        elif state == ConversationState.IDLE and chat_vel < 2:
            self.current_goal = random.choice([BotGoal.INVESTIGATE, BotGoal.TROLL])
        else: self.current_goal = BotGoal.OBSERVE

    def direct_attention(self, store: ContextStore, events: List[EventItem]) -> Optional[EventItem]:
        if not events: return None
        now = time.time()
        focus = store.focus_state

        if self.current_goal == BotGoal.SUPPORT:
            candidates = [e for e in events if e.source in [InputSource.MICROPHONE, InputSource.DIRECT_MICROPHONE]]
        elif self.current_goal == BotGoal.ENTERTAIN:
            candidates = [e for e in events if e.source in [InputSource.VISUAL_CHANGE, InputSource.SYSTEM_PATTERN]]
        elif self.current_goal == BotGoal.INVESTIGATE:
            candidates = [e for e in events if e.source == InputSource.DIRECT_MICROPHONE]
        else:
            candidates = events

        if not candidates: candidates = events
        candidate = max(candidates, key=lambda x: x.score.interestingness)
        candidate_score = candidate.score.interestingness

        if focus.target_event_id and now < focus.locked_until:
            switch_cost = 0.3
            if candidate.id != focus.target_event_id:
                if candidate_score < (focus.strength + switch_cost):
                    return None

        store.focus_state.target_event_id = candidate.id
        store.focus_state.strength = candidate_score
        store.focus_state.locked_until = now + self.attention_lock_duration

        print(f"🔒 [Attention] Locked on {candidate.source.name} (Score: {candidate_score:.2f})")
        return candidate

    def register_bot_action(self, store: ContextStore, text: str):
        action_type = "support"
        if "?" in text:
            action_type = "question"
            store.add_debt(text)
        elif any(x in text.lower() for x in ["haha", "lol", "lmao", "roast"]):
            action_type = "joke"
        elif "!" in text:
            action_type = "roast"

        store.log_bot_action(action_type, text)

    def check_debt_resolution(self, store: ContextStore, user_input: str) -> Optional[str]:
        if store.conversation_debt:
            store.resolve_debt()
            return None
        now = time.time()
        for debt in store.conversation_debt:
            if now - debt.timestamp > 60.0:
                store.conversation_debt.remove(debt)
                return f"Wait, you never told me: {debt.text}"
        return None

    async def check_internal_monologue(self, store: ContextStore) -> None:
        """
        Decide whether to generate a spontaneous thought, and if so, kick off
        the LLM call as a fire-and-forget background task.

        Returns None always — the spawned task adds the resulting event to
        the store directly. This decouples reflex_ticker's iteration rate
        from Ollama latency: if Ollama takes 13s, the reflex loop still
        runs at 1Hz; the thought just appears in the store ~13s later.

        Dedup: at most ONE monologue task in flight at a time.
        """
        now = time.time()

        # Scene-aware gating:
        #   HOST_FOCUSED_QUIET → don't pull focus; suppress idle thoughts entirely.
        #   HOST_LOW_ENERGY    → fill the dead air more aggressively (½ interval).
        if store.current_scene == SceneType.HOST_FOCUSED_QUIET:
            return None
        interval = CURIOSITY_INTERVAL
        if store.current_scene == SceneType.HOST_LOW_ENERGY:
            interval = CURIOSITY_INTERVAL * 0.5

        if now - self.last_curiosity_check < interval:
            return None

        # Already generating one — don't queue another
        if self._monologue_task_inflight:
            return None

        import shared
        chat_vel, _ = store.get_activity_metrics()

        should_ramble = (
            store.current_flow != FlowState.DOMINATED or
            chat_vel < 5.0
        )

        if not should_ramble:
            return None

        self.last_curiosity_check = now

        # --- BUILD WATCHING CONTEXT ---
        current_streamer = shared.get_current_streamer()

        if not current_streamer:
            # No streamer selected — generic companion context
            watching_context = "You are PeepingOtter's personal AI companion."
        elif current_streamer.lower() == OWNER_STREAMER_ID.lower():
            # Watching Otter himself stream
            watching_context = (
                f"You are PeepingOtter's personal AI companion. "
                f"Right now you are watching him stream."
            )
        else:
            # Watching a third-party stream alongside Otter
            watching_context = (
                f"You are PeepingOtter's personal AI companion. "
                f"Right now you and PeepingOtter are watching {current_streamer} together."
            )

        # --- BUILD STREAM CONTEXT ---
        # Combine scene type and current summary so Llama knows what's on screen.
        # Keep it short — this is grounding, not a full prompt.
        scene_name = store.current_scene.name.replace("_", " ").title()
        summary = store.current_summary or ""

        if summary and summary != "Just starting up.":
            stream_context = f"{scene_name} - {summary}"
        else:
            stream_context = scene_name

        # Truncate if summary is unusually long
        if len(stream_context) > 200:
            stream_context = stream_context[:197] + "..."

        # --- PICK TOPIC ---
        topic = "the current situation"
        if store.current_topics:
            topic = random.choice(store.current_topics)
        elif store.current_entities:
            topic = f"the {random.choice(store.current_entities)}"

        # Fire and forget — does NOT block this iteration
        self._monologue_task_inflight = True
        asyncio.create_task(self._run_monologue_generation(
            topic, stream_context, watching_context, scene_name, store
        ))
        return None

    async def _run_monologue_generation(
        self,
        topic: str,
        stream_context: str,
        watching_context: str,
        scene_name: str,
        store: ContextStore,
    ) -> None:
        """
        Background task that actually calls Ollama for the monologue.
        Decoupled from reflex_ticker so Ollama latency doesn't block the loop.
        """
        import shared
        import config as _cfg

        try:
            print(f"💭 [Monologue] Generating thought | topic: {topic} | scene: {scene_name}")

            thought = await llm_analyst.generate_thought(
                prompt_text=f"A weird or funny observation about {topic}",
                stream_context=stream_context,
                watching_context=watching_context,
            )

            if thought:
                print(f"💡 [Reflex] Thought: {thought}")
                thought_event = store.add_event(
                    _cfg.InputSource.INTERNAL_THOUGHT, thought,
                    {"type": "shower_thought", "goal": "fill_silence"},
                    EventScore(interestingness=0.95, conversational_value=1.0, urgency=0.8)
                )
                shared.emit_event_scored(thought_event)
        except Exception as e:
            kind = type(e).__name__
            detail = str(e) or repr(e)
            print(f"⚠️ [Monologue] Background generation failed ({kind}): {detail}")
        finally:
            self._monologue_task_inflight = False

    # Alias for backwards compatibility
    check_curiosity = check_internal_monologue

    def check_callbacks(self, store: ContextStore) -> Optional[str]:
        now = time.time()
        if now - self.last_callback_check < CALLBACK_INTERVAL: return None
        debt_prompt = self.check_debt_resolution(store, "")
        if debt_prompt:
            self.last_callback_check = now
            return debt_prompt
        self.last_callback_check = now
        if store.narrative_log and store.current_conversation_state in [ConversationState.IDLE, ConversationState.ENGAGED]:
            if len(store.narrative_log) >= 3:
                past_event = store.narrative_log[-3]
                return f"Recall: Remember when this happened? '{past_event}'"
        return None
# Save as: director_engine/services/speech_dispatcher.py
"""
The Speech Dispatcher — Decides WHAT Nami should say and WHY.

This module only handles the DECISION. Delivery is handled by the
Prompt Service via prompt_client.

No speaking state tracking, no HTTP client, no cooldowns here.
The prompt service gates everything.
"""

import time
from typing import Optional, Dict, Any
from dataclasses import dataclass

from config import (
    ENERGY_COST_INTERJECTION,
    InputSource,
    BotGoal,
    FlowState,
    ConversationState,
)
from context.context_store import ContextStore, EventItem
from systems.decision_engine import Directive
from systems.energy_system import EnergySystem
from systems.behavior_engine import BehaviorEngine


@dataclass
class SpeechDecision:
    should_speak: bool
    reason: str
    content: str
    priority: float
    source_info: Dict[str, Any]


class SpeechDispatcher:
    def __init__(self):
        pass

    def evaluate(
        self,
        store: ContextStore,
        behavior: BehaviorEngine,
        energy: EnergySystem,
        directive: Optional[Directive],
    ) -> Optional[SpeechDecision]:
        """
        Evaluate whether Nami should speak right now.
        
        Only checks BRAIN-SIDE gates (energy, flow state).
        Speaking state / cooldowns are handled by the prompt service.
        """
        # Gate 1: Energy
        if not energy.can_afford(ENERGY_COST_INTERJECTION):
            return None

        # Gate 2: Don't interrupt if user is dominating
        if store.current_flow == FlowState.DOMINATED:
            return None

        # All brain-side gates passed — look for trigger events
        return self._find_speech_trigger(store, behavior, directive)

    def _find_speech_trigger(
        self,
        store: ContextStore,
        behavior: BehaviorEngine,
        directive: Optional[Directive],
    ) -> Optional[SpeechDecision]:
        """
        Find something worth reacting to.

        Priority order:
        1. System patterns (skill issues, victories, memes)
        2. Internal thoughts (dead air fillers)
        3. High-interest visual/audio events
        """
        layers = store.get_all_events_for_summary()
        immediate = layers["immediate"]
        recent = layers["recent"]

        # --- Priority 1: System Patterns (always react) ---
        patterns = [e for e in immediate if e.source == InputSource.SYSTEM_PATTERN]
        for event in patterns:
            pattern_type = event.metadata.get("type", "")

            if pattern_type == "skill_issue":
                return SpeechDecision(
                    should_speak=True,
                    reason="Skill Issue Detected",
                    content="React to the user's failure. Say 'skill issue' or roast them.",
                    priority=0.1,
                    source_info={
                        "source": "DIRECTOR_SKILL_ISSUE",
                        "use_tts": True,
                        "event_id": event.id,
                    },
                )

            if pattern_type == "pattern_victory":
                return SpeechDecision(
                    should_speak=True,
                    reason="Victory Detected",
                    content="Celebrate the user's win! Be hype!",
                    priority=0.2,
                    source_info={
                        "source": "DIRECTOR_VICTORY",
                        "use_tts": True,
                        "event_id": event.id,
                    },
                )

            if pattern_type == "pattern_meme":
                visual_ref = event.metadata.get("visual_ref", "something funny")
                return SpeechDecision(
                    should_speak=True,
                    reason="Meme Moment",
                    content=f"React to this funny moment: {visual_ref}",
                    priority=0.3,
                    source_info={
                        "source": "DIRECTOR_MEME",
                        "use_tts": True,
                        "event_id": event.id,
                    },
                )

            if pattern_type == "pattern_void":
                return SpeechDecision(
                    should_speak=True,
                    reason="Dead Air",
                    content="Fill the awkward silence. Say something random or provocative.",
                    priority=0.5,
                    source_info={
                        "source": "DIRECTOR_DEAD_AIR",
                        "use_tts": True,
                        "event_id": event.id,
                    },
                )

            if pattern_type == "fixation":
                entity = event.metadata.get("entity", "thing")
                return SpeechDecision(
                    should_speak=True,
                    reason="Visual Fixation",
                    content=f"You keep seeing a {entity}. Comment on it obsessively.",
                    priority=0.4,
                    source_info={
                        "source": "DIRECTOR_FIXATION",
                        "use_tts": True,
                        "event_id": event.id,
                    },
                )

        # --- Priority 2: Internal Thoughts ---
        # IMPORTANT: Do NOT pass the raw thought text as `content` — that makes
        # it land verbatim as "USER INPUT" in the final prompt, causing Nami to
        # respond to her own generated text and creating topic feedback loops
        # (e.g. One Piece theories snowballing when watching a Phas stream).
        #
        # The thought is already stored in the event log and will appear in the
        # background chat section of the context block. The trigger content
        # should be a generic instruction so the LLM uses the *context* (which
        # includes the thought) to inform what it says, rather than treating the
        # thought as a direct user message to respond to.
        thoughts = [e for e in immediate if e.source == InputSource.INTERNAL_THOUGHT]
        for event in thoughts:
            # Derive a short, safe instruction from the thought's metadata
            thought_goal = event.metadata.get("goal", "fill_silence")
            thought_type = event.metadata.get("type", "shower_thought")

            if thought_type == "callback":
                instruction = "Reference something that happened earlier in the stream naturally."
            elif thought_goal == "fill_silence":
                instruction = "Fill the silence — say something weird, random, or provocative based on what you've been seeing."
            else:
                instruction = "React to the current situation based on what you know."

            return SpeechDecision(
                should_speak=True,
                reason="Internal Thought",
                content=instruction,
                priority=0.6,
                source_info={
                    "source": "DIRECTOR_THOUGHT",
                    "use_tts": True,
                    "event_id": event.id,
                },
            )

        # --- Priority 3: Low-Threshold Events ---
        interesting_events = [
            e
            for e in immediate + recent[:3]
            if e.source in [InputSource.VISUAL_CHANGE, InputSource.AMBIENT_AUDIO]
            and e.score.interestingness >= 0.25
        ]

        if interesting_events:
            best = max(interesting_events, key=lambda x: x.score.interestingness)

            action = "comment on"
            if behavior.current_goal == BotGoal.TROLL:
                action = "roast"
            elif behavior.current_goal == BotGoal.OBSERVE:
                action = "notice"

            return SpeechDecision(
                should_speak=True,
                reason=f"Passive Observation ({best.source.name})",
                content=f"You notice: {best.text}. React to it - {action} this.",
                priority=0.7,
                source_info={
                    "source": f"DIRECTOR_{best.source.name}",
                    "use_tts": True,
                    "event_id": best.id,
                },
            )

        return None
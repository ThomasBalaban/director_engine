# Save as: director_engine/services/speech_dispatcher.py
"""
The Speech Dispatcher - Decides when Nami should speak proactively.

This module bridges the Director's "thinking" with Nami's "speaking".
It monitors the stream state and pushes interjections to Nami when appropriate.

NEW: Now checks if Nami is currently speaking before dispatching.
"""

import time
import httpx
from typing import Optional, Dict, Any
from dataclasses import dataclass

from config import (
    NAMI_INTERJECT_URL, 
    ENERGY_COST_INTERJECTION,
    InputSource,
    BotGoal,
    FlowState,
    ConversationState
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
        self.last_speech_time = 0
        self.min_speech_interval = 3.0  # Minimum between dispatches
        self.post_response_cooldown = 10.0  # NEW: Cooldown after Nami responds to user
        self.last_user_response_time = 0  # NEW: Track when Nami last responded to user
        self.http_client: Optional[httpx.AsyncClient] = None
        self.reacted_event_ids: set = set()
        self.max_tracked_events = 50
    
    async def initialize(self):
        """Initialize the HTTP client for sending interjections."""
        if self.http_client is None:
            self.http_client = httpx.AsyncClient()
            print("✅ [SpeechDispatcher] HTTP client initialized")
    
    async def close(self):
        """Close the HTTP client."""
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None
            print("✅ [SpeechDispatcher] HTTP client closed")
    
    def register_user_response(self):
        """Call this when Nami responds to a direct user interaction."""
        self.last_user_response_time = time.time()
        print(f"🎯 [SpeechDispatcher] User response registered - cooldown active for {self.post_response_cooldown}s")
    
    def evaluate(
        self,
        store: ContextStore,
        behavior: BehaviorEngine,
        energy: EnergySystem,
        directive: Optional[Directive]
    ) -> Optional[SpeechDecision]:
        """
        Evaluate whether Nami should speak right now.
        """
        import shared
        if shared.is_nami_speaking():
            return None
        
        now = time.time()
        
        # Check cooldown from last dispatch
        time_since_last = now - self.last_speech_time
        if time_since_last < self.min_speech_interval:
            return None
        
        # NEW: Check post-response cooldown (don't interrupt after responding to user)
        time_since_user_response = now - self.last_user_response_time
        if time_since_user_response < self.post_response_cooldown:
            # Still in cooldown after responding to user
            return None
        
        # Check energy
        if not energy.can_afford(ENERGY_COST_INTERJECTION):
            return None
        
        # Check flow - Don't interrupt if user is dominating
        if store.current_flow == FlowState.DOMINATED:
            return None
        
        # Look for trigger events
        decision = self._find_speech_trigger(store, behavior, directive)
        
        return decision
    
    def _find_speech_trigger(
        self,
        store: ContextStore,
        behavior: BehaviorEngine,
        directive: Optional[Directive]
    ) -> Optional[SpeechDecision]:
        """
        Find something worth reacting to.
        
        Priority order:
        1. System patterns (skill issues, victories, memes)
        2. Internal thoughts (dead air fillers)
        3. High-interest visual/audio events
        """
        
        # Get recent events we haven't reacted to
        layers = store.get_all_events_for_summary()
        immediate = layers['immediate']
        recent = layers['recent']
        
        # --- Priority 1: System Patterns (always react) ---
        patterns = [e for e in immediate if e.source == InputSource.SYSTEM_PATTERN]
        for event in patterns:
            if event.id in self.reacted_event_ids:
                continue
                
            pattern_type = event.metadata.get('type', '')
            
            # Skill issue - ALWAYS roast
            if pattern_type == 'skill_issue':
                return SpeechDecision(
                    should_speak=True,
                    reason="Skill Issue Detected",
                    content="React to the user's failure. Say 'skill issue' or roast them.",
                    priority=0.1,  # High priority (lower number)
                    source_info={
                        'source': 'DIRECTOR_SKILL_ISSUE',
                        'use_tts': True,
                        'event_id': event.id
                    }
                )
            
            # Victory - Celebrate
            if pattern_type == 'pattern_victory':
                return SpeechDecision(
                    should_speak=True,
                    reason="Victory Detected",
                    content="Celebrate the user's win! Be hype!",
                    priority=0.2,
                    source_info={
                        'source': 'DIRECTOR_VICTORY',
                        'use_tts': True,
                        'event_id': event.id
                    }
                )
            
            # Meme moment - React
            if pattern_type == 'pattern_meme':
                visual_ref = event.metadata.get('visual_ref', 'something funny')
                return SpeechDecision(
                    should_speak=True,
                    reason="Meme Moment",
                    content=f"React to this funny moment: {visual_ref}",
                    priority=0.3,
                    source_info={
                        'source': 'DIRECTOR_MEME',
                        'use_tts': True,
                        'event_id': event.id
                    }
                )
            
            # Dead air - Fill silence
            if pattern_type == 'pattern_void':
                return SpeechDecision(
                    should_speak=True,
                    reason="Dead Air",
                    content="Fill the awkward silence. Say something random or provocative.",
                    priority=0.5,
                    source_info={
                        'source': 'DIRECTOR_DEAD_AIR',
                        'use_tts': True,
                        'event_id': event.id
                    }
                )
            
            # Fixation (Gymbag effect)
            if pattern_type == 'fixation':
                entity = event.metadata.get('entity', 'thing')
                return SpeechDecision(
                    should_speak=True,
                    reason="Visual Fixation",
                    content=f"You keep seeing a {entity}. Comment on it obsessively.",
                    priority=0.4,
                    source_info={
                        'source': 'DIRECTOR_FIXATION',
                        'use_tts': True,
                        'event_id': event.id
                    }
                )
        
        # --- Priority 2: Internal Thoughts ---
        thoughts = [e for e in immediate if e.source == InputSource.INTERNAL_THOUGHT]
        for event in thoughts:
            if event.id in self.reacted_event_ids:
                continue
            
            return SpeechDecision(
                should_speak=True,
                reason="Internal Thought",
                content=event.text,  # The thought itself is the content
                priority=0.6,
                source_info={
                    'source': 'DIRECTOR_THOUGHT',
                    'use_tts': True,
                    'event_id': event.id
                }
            )
        
        # --- Priority 3: Low-Threshold Events ---
        # React to anything even remotely interesting (> 0.4)
        interesting_events = [
            e for e in immediate + recent[:3]
            if e.source in [InputSource.VISUAL_CHANGE, InputSource.AMBIENT_AUDIO]
            and e.score.interestingness >= 0.25  # Increased sensitivity
            and e.id not in self.reacted_event_ids
        ]
        
        if interesting_events:
            best = max(interesting_events, key=lambda x: x.score.interestingness)
            
            # Dynamic action based on goal
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
                    'source': f'DIRECTOR_{best.source.name}',
                    'use_tts': True,
                    'event_id': best.id
                }
            )
        
        return None
    
    async def dispatch(
        self,
        decision: SpeechDecision,
        energy: EnergySystem
    ) -> bool:
        """
        Send the speech request to Nami's interjection endpoint.
        """
        # --- NEW: Double-check speech state before sending ---
        import shared
        if shared.is_nami_speaking():
            print(f"🔇 [SpeechDispatcher] Blocked dispatch - Nami is still speaking")
            return False
        
        if not self.http_client:
            await self.initialize()
        
        # Spend energy
        if not energy.spend(ENERGY_COST_INTERJECTION):
            print(f"⚡ [SpeechDispatcher] Not enough energy to speak")
            return False
        
        # Build payload
        payload = {
            "content": decision.content,
            "priority": decision.priority,
            "source_info": decision.source_info
        }
        
        try:
            print(f"🎤 [SpeechDispatcher] Pushing to Nami: {decision.reason}")
            print(f"   Content: {decision.content[:60]}...")
            
            response = await self.http_client.post(
                NAMI_INTERJECT_URL,
                json=payload,
                timeout=2.0
            )
            
            if response.status_code == 200:
                self.last_speech_time = time.time()
                
                # Track that we reacted to this event
                event_id = decision.source_info.get('event_id')
                if event_id:
                    self.reacted_event_ids.add(event_id)
                    # Prevent memory leak
                    if len(self.reacted_event_ids) > self.max_tracked_events:
                        # Remove oldest (arbitrary, but works)
                        self.reacted_event_ids.pop()
                
                print(f"✅ [SpeechDispatcher] Nami accepted the interjection")
                return True
            else:
                print(f"❌ [SpeechDispatcher] Nami rejected: {response.status_code}")
                return False
                
        except httpx.ConnectError:
            print(f"❌ [SpeechDispatcher] Cannot reach Nami at {NAMI_INTERJECT_URL}")
            return False
        except Exception as e:
            print(f"❌ [SpeechDispatcher] Error: {e}")
            return False
    
    def set_speech_interval(self, seconds: float):
        """Adjust how often Nami can speak proactively."""
        self.min_speech_interval = max(1.0, seconds)
        print(f"🎤 [SpeechDispatcher] Speech interval set to {self.min_speech_interval}s")
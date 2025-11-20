# Save as: director_engine/prompt_constructor.py
from typing import List, Dict, Any, Optional
from config import ConversationState, FlowState, InputSource
from context_store import ContextStore, EventItem
from decision_engine import Directive

class PromptConstructor:
    """
    The Storyteller.
    Takes the raw mathematical state of the engine and turns it into 
    a natural language prompt for the LLM.
    """
    
    def construct_context_block(self, 
                              store: ContextStore, 
                              directive: Directive, 
                              memories: List[EventItem]) -> str:
        """
        Builds the dynamic context block to be injected into the LLM's immediate awareness.
        """
        parts = []

        # 1. The "Now" (Scene & Vibe)
        parts.append(self._format_scene_context(store))
        
        # 2. The "Orders" (Directive)
        parts.append(self._format_directive(directive))
        
        # 3. The "User" (Profile & Relationship)
        if store.active_user_profile:
            parts.append(self._format_user_context(store.active_user_profile))
            
        # 4. The "Past" (Relevant Memories & Narrative)
        parts.append(self._format_memories(memories, store.narrative_log))
        
        # 5. The "Flow" (Recent Events)
        # We get the recent events directly from the store
        layers = store.get_all_events_for_summary()
        # Filter for high relevance or recency (last 15s + highly interesting recent)
        active_events = layers['immediate'] + [
            e for e in layers['recent'] 
            if e.score.interestingness > 0.4 or e.source in [InputSource.TWITCH_MENTION, InputSource.DIRECT_MICROPHONE]
        ]
        # Sort by time
        active_events.sort(key=lambda x: x.timestamp)
        
        parts.append(self._format_recent_events(active_events))

        return "\n\n".join(parts)

    def _format_scene_context(self, store: ContextStore) -> str:
        """Describes the current environment."""
        mood_str = f"Current Mood: {store.current_mood} ({store.emotional_momentum})"
        scene_str = f"Scene: {store.current_scene.name}"
        flow_str = f"Conversation Flow: {store.current_flow.name}"
        
        return f"### CURRENT SITUATION\n{scene_str}\n{mood_str}\n{flow_str}\nSummary: {store.current_summary}"

    def _format_directive(self, directive: Directive) -> str:
        """Formats the marching orders."""
        if not directive:
            return ""
            
        constraints = ""
        if directive.constraints:
            constraints = "\nConstraints: " + ", ".join(directive.constraints)
            
        return (
            f"### INSTRUCTION (Top Priority)\n"
            f"Goal: {directive.objective}\n"
            f"Tone: {directive.tone}\n"
            f"Action: {directive.suggested_action}"
            f"{constraints}"
        )

    def _format_user_context(self, profile: Dict[str, Any]) -> str:
        """Formats active user info."""
        facts = [f"- {f['content']}" for f in profile.get('facts', [])[-5:]]
        facts_str = "\n".join(facts) if facts else "No known facts."
        
        return (
            f"### ACTIVE USER: {profile['username']}\n"
            f"Relationship: {profile['relationship']['tier']} (Affinity: {profile['relationship']['affinity']}%)\n"
            f"Known Facts:\n{facts_str}"
        )

    def _format_memories(self, memories: List[EventItem], narrative_log: List[str]) -> str:
        """Formats long-term and relevant memories."""
        if not memories and not narrative_log:
            return ""
            
        text = "### RELEVANT CONTEXT"
        
        # 1. Narrative History (Mid-term)
        if narrative_log:
            text += "\n[Previously...]\n" + "\n".join([f"- {entry}" for entry in narrative_log[-3:]])
            
        # 2. Associative Memories (Long-term)
        if memories:
            text += "\n[Related Memories]\n"
            for mem in memories:
                # Clean up the text to be concise
                content = mem.memory_text or mem.text
                text += f"- (Recall) {content}\n"
                
        return text

    def _format_recent_events(self, events: List[EventItem]) -> str:
        """Formats the raw stream of consciousness."""
        if not events:
            return "### RECENT EVENTS\n(Silence...)"
            
        lines = []
        for e in events:
            # Format logic: [Source] Text
            # Visuals get special icon
            prefix = f"[{e.source.name}]"
            if e.source == InputSource.VISUAL_CHANGE:
                prefix = "[👁️ VISION]"
            elif e.source == InputSource.DIRECT_MICROPHONE:
                prefix = "[🎤 USER SAID]"
            elif e.source == InputSource.TWITCH_MENTION:
                prefix = f"[💬 {e.metadata.get('username', 'Chat')}]"
            
            lines.append(f"{prefix} {e.text}")
            
        return "### IMMEDIATE STREAM (Last 30s)\n" + "\n".join(lines)
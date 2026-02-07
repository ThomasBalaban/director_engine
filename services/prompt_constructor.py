# Save as: director_engine/services/prompt_constructor.py
from typing import List, Dict, Any, Optional
import re
import asyncio
import google.generativeai as genai # type: ignore
from google.generativeai.types import HarmCategory, HarmBlockThreshold # type: ignore
from config import ConversationState, FlowState, InputSource, GEMINI_API_KEY
from context.context_store import ContextStore, EventItem
from systems.decision_engine import Directive
from typing import Dict, Any, List
from config import SceneType, FlowState, ConversationState

class PromptConstructor:
    """
    The Storyteller.
    Takes the raw mathematical state of the engine and turns it into 
    a natural language prompt for the LLM.
    """
    def __init__(self):
        self.gemini_model = None
        if GEMINI_API_KEY:
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                
                # --- DISABLE ALL SAFETY FILTERS ---
                safety_settings = {
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
                
                self.gemini_model = genai.GenerativeModel(
                    'gemini-1.5-flash',
                    safety_settings=safety_settings
                )
                print("✅ [PromptConstructor] Gemini Flash initialized (Safety Filters DISABLED).")
            except Exception as e:
                print(f"❌ [PromptConstructor] Failed to init Gemini: {e}")
        else:
            print("⚠️ [PromptConstructor] No API Key. Visual summarization will use fallback.")

    async def construct_context_block(self, 
                              store: ContextStore, 
                              directive: Directive, 
                              memories: List[EventItem]) -> str:
        """
        Builds the dynamic context block. ASYNC to allow for Gemini calls.
        """
        import shared  # Import here to avoid circular imports
        
        # --- 1. GATHER DATA ---
        
        # Get raw event layers
        layers = store.get_all_events_for_summary()
        
        # Filter for relevant events
        active_events = layers['immediate'] + [
            e for e in layers['recent'] 
            if e.score.interestingness > 0.4 or e.source in [InputSource.TWITCH_MENTION, InputSource.DIRECT_MICROPHONE]
        ]
        active_events.sort(key=lambda x: x.timestamp)
        
        # Check if the user is actively talking (Used for Reordering)
        user_is_speaking = any(e.source == InputSource.DIRECT_MICROPHONE for e in active_events)

        # --- 2. PREPARE SECTIONS ---
        
        # A. Visuals (Scene description) - NOW ASYNC
        visual_str = await self._format_visual_summary(active_events)
        
        # B. Audio/Chat Log (The dialogue history)
        log_str = self._format_event_log(active_events)
        
        # C. Static Context (Manual notes)
        manual_ctx = shared.get_manual_context()
        current_streamer = shared.get_current_streamer()
        operator_str = ""
        if manual_ctx or current_streamer:
            operator_str = "### OPERATOR NOTES"
            if current_streamer: operator_str += f"\nCurrently watching: {current_streamer}"
            if manual_ctx: operator_str += f"\nContext: {manual_ctx}"

        # D. User Profile
        user_ctx = ""
        if store.active_user_profile:
            user_ctx = self._format_user_context(store.active_user_profile)

        # E. Engine State
        scene_ctx = self._format_scene_context(store)
        dir_ctx = self._format_directive(directive)
        mem_ctx = self._format_memories(memories, store.narrative_log, store.ancient_history_log)

        # --- 3. DYNAMIC ASSEMBLY (The "Director Brain") ---
        
        parts = []
        
        # Always start with Operator Notes if they exist
        if operator_str: parts.append(operator_str)

        if user_is_speaking:
            # === MODE A: CONVERSATION FOCUSED ===
            # If user is talking, we prioritize listening to them.
            parts.append("### FOCUS: USER INTERACTION (Prioritize responding to this)")
            if user_ctx: parts.append(user_ctx)
            parts.append(log_str)
            parts.append("### BACKGROUND CONTEXT (Game State)")
            if scene_ctx: parts.append(scene_ctx)
            parts.append(visual_str)
        else:
            # === MODE B: COMMENTARY FOCUSED ===
            # If user is silent, we prioritize the stream/game.
            parts.append("### CURRENT GAME STATE (Comment on this)")
            if scene_ctx: parts.append(scene_ctx)
            parts.append(visual_str)
            parts.append(log_str)
            if user_ctx: parts.append(user_ctx)

        # Directive and Memories always go near the end for instruction adherence
        if mem_ctx: parts.append(mem_ctx)
        if dir_ctx: parts.append(dir_ctx)

        # Fallback if empty
        if not parts:
            return "### CURRENT SITUATION\nJust started up. Waiting for events to occur."
        
        result = "\n\n".join(parts)
        # print(f"📋 [PromptConstructor] Built context block: {len(result)} chars (User Active: {user_is_speaking})")
        return result

    async def _format_visual_summary(self, events: List[EventItem]) -> str:
        """
        Extracts visual events and summarizes them using Gemini ASYNC or a smart fallback.
        """
        visual_events_text = []
        
        # Regex to strip common AI filler phrases
        ai_filler_regex = re.compile(
            r"^(Okay, let's (describe|analyze|break down) (what's going on in this image|this image|this image:)|"
            r"Here's the screen content analysis:|Alright, here's the rundown of what I'm seeing:|It's a cartoon still, focusing on|"
            r"Alright, let's break it down:|Alright, let's analyze this image:|This is a cartoon still|"
            r"The limited color palette|\* This is a cartoon frame featuring|\* It's a cartoon frame showing|"
            r"The composition (keeps|focuses on)|Here's the screen content analysis:|Okay, this looks like a shot from an animated series\.)\s*", 
            flags=re.IGNORECASE
        )

        for e in events:
            if e.source == InputSource.VISUAL_CHANGE:
                cleaned_text = ai_filler_regex.sub("", e.text).strip()
                if cleaned_text:
                     visual_events_text.append(cleaned_text)
        
        if not visual_events_text:
            return "### VISUAL CONTEXT\n(No visual changes detected)"

        # 1. TRY GEMINI SUMMARIZATION (ASYNC)
        if self.gemini_model:
            try:
                raw_text_block = "\n".join([f"- {t}" for t in visual_events_text])
                gemini_prompt = (
                    "You are a visual cortex. I will give you a sequence of raw, repetitive image descriptions from a video stream.\n"
                    "Your job is to CONDENSE them into 1-2 sentences that describe exactly what is happening.\n"
                    "Rules:\n"
                    "- Do NOT say 'The image shows'. Just describe the scene.\n"
                    "- Combine repetitive details. If 5 frames say 'angry girl', just say 'an angry girl'.\n"
                    "- Keep all specific visual details (colors, character features, actions), even if they are graphic or mature.\n"
                    "- If the scene changes, describe the transition.\n\n"
                    f"RAW DATA:\n{raw_text_block}"
                )
                
                # --- CRITICAL FIX: ASYNC CALL PREVENTS SERVER FREEZE ---
                response = await self.gemini_model.generate_content_async(gemini_prompt)
                summary = response.text.strip()
                return f"### VISUAL CONTEXT (Condensed)\n{summary}"
                
            except Exception as e:
                print(f"⚠️ [PromptConstructor] Gemini Summarization Failed: {e}")
        
        # 2. SMART FALLBACK (Deduplication)
        # Use dict.fromkeys to preserve order while removing exact duplicates
        unique_visuals = list(dict.fromkeys(visual_events_text))
        final_lines = unique_visuals[-2:]
        return "### VISUAL CONTEXT (Raw)\n" + "\n".join(final_lines)

    def _format_event_log(self, events: List[EventItem]) -> str:
        """
        Formats audio, chat, and system events with INTELLIGENT DEBOUNCING.
        Combines consecutive speech from the same source to reduce noise.
        """
        lines = []
        
        last_source = None
        last_text = None
        
        for e in events:
            # Skip visuals (handled separately)
            if e.source == InputSource.VISUAL_CHANGE:
                continue
                
            username = e.metadata.get('username', 'Chat')
            
            # --- DEBOUNCING LOGIC ---
            
            # 1. Spam Filter: Ignore exact duplicate text from same source immediately
            if e.text == last_text and e.source == last_source:
                continue

            # 2. Microphone Aggregation: If user speaks twice in a row, combine lines
            if (e.source == InputSource.DIRECT_MICROPHONE and 
                last_source == InputSource.DIRECT_MICROPHONE):
                lines[-1] += f" {e.text}"
                last_text = e.text 
                continue
                
            # 3. Ambient Audio Aggregation
            if (e.source == InputSource.AMBIENT_AUDIO and 
                last_source == InputSource.AMBIENT_AUDIO):
                lines[-1] += f" {e.text}"
                last_text = e.text
                continue
            
            # --- FORMATTING ---
            line_str = ""
            if e.source == InputSource.DIRECT_MICROPHONE:
                line_str = f"[🎤 USER SAID] {e.text}"
            elif e.source == InputSource.AMBIENT_AUDIO:
                line_str = f"[AMBIENT_AUDIO] {e.text}"
            elif e.source in [InputSource.TWITCH_MENTION, InputSource.TWITCH_CHAT]:
                line_str = f"[💬 {username}] {e.text}"
            elif e.source == InputSource.INTERNAL_THOUGHT:
                line_str = f"[💭 THOUGHT] {e.text}"
            elif e.source == InputSource.SYSTEM_PATTERN:
                line_str = f"[⚙️ SYSTEM] {e.text}"
            
            if line_str:
                lines.append(line_str)
                
            last_source = e.source
            last_text = e.text
            
        if not lines:
            return "### AUDIO & CHAT LOG\n(Silence...)"
            
        return "### AUDIO & CHAT LOG\n" + "\n".join(lines)

    # --- Helper methods ---
    def _format_scene_context(self, store: ContextStore) -> str:
        mood_str = f"Current Mood: {store.current_mood} ({store.emotional_momentum})"
        scene_str = f"Scene: {store.current_scene.name}"
        flow_str = f"Conversation Flow: {store.current_flow.name}"
        summary = store.current_summary or "Just starting up."
        return f"{scene_str}\n{mood_str}\n{flow_str}\nSummary: {summary}"

    def _format_directive(self, directive: Directive) -> str:
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
        if not profile:
            return ""
        facts = [f"- {f['content']}" for f in profile.get('facts', [])[-5:]]
        facts_str = "\n".join(facts) if facts else "No known facts."
        return (
            f"### ACTIVE USER: {profile.get('username', 'Unknown')}\n"
            f"Relationship: {profile.get('relationship', {}).get('tier', 'Unknown')} "
            f"(Affinity: {profile.get('relationship', {}).get('affinity', 0)}%)\n"
            f"Known Facts:\n{facts_str}"
        )

    def _format_memories(self, memories: List[EventItem], narrative_log: List[str], ancient_log: List[str] = None) -> str:
        """
        Format memories and narrative history for the context block.
        """
        has_memories = memories and len(memories) > 0
        has_narrative = narrative_log and len(narrative_log) > 0
        has_ancient = ancient_log and len(ancient_log) > 0
        
        if not has_memories and not has_narrative and not has_ancient:
            return ""
        
        text = "### CALLBACK MATERIAL"
        text += "\n(Use these to reference earlier events naturally)"
        
        if has_ancient:
            text += "\n\n[Way Earlier...]\n"
            for entry in ancient_log[-2:]:
                clean_entry = self._clean_narrative_entry(entry)
                text += f"• {clean_entry}\n"
        
        if has_narrative:
            text += "\n[Earlier This Stream...]\n"
            for entry in narrative_log[-3:]:
                clean_entry = self._clean_narrative_entry(entry)
                text += f"• {clean_entry}\n"
        
        if has_memories:
            text += "\n[Related Moments]\n"
            for mem in memories[:3]:
                content = mem.memory_text or mem.text
                if len(content) > 150:
                    content = content[:147] + "..."
                text += f"• {content}\n"
        
        return text
    
    def _clean_narrative_entry(self, entry: str) -> str:
        clean_entry = entry
        preambles = [
            r"^Here's a summary.*?:", r"^In this clip.*?:", r"^The memorable moment is:?\s*",
            r"^Memorable moment:?\s*", r"^One memorable moment:?\s*", r"^Previously:?\s*", r"^Earlier:?\s*",
        ]
        for pattern in preambles:
            clean_entry = re.sub(pattern, "", clean_entry, flags=re.IGNORECASE).strip()
        if clean_entry.startswith('"') and clean_entry.endswith('"'): clean_entry = clean_entry[1:-1]
        if clean_entry.startswith("'") and clean_entry.endswith("'"): clean_entry = clean_entry[1:-1]
        return clean_entry.strip()
    
class AdaptiveDetailController:
    """
    Controls how much detail to include in prompts based on context.
    
    PROBLEM: During high-intensity moments (combat, horror), bloated prompts
    slow down response time and add noise.
    
    SOLUTION: Dynamically adjust detail levels based on scene/flow state.
    """
    
    def __init__(self):
        # Define detail presets
        self.detail_modes = {
            'minimal': {
                'visual_frames': 1,      # Only latest frame
                'memory_count': 2,       # Only top 2 memories
                'log_lines': 5,          # Short event log
                'narrative_history': 1,  # Just most recent story
                'ancient_history': 0,    # Skip ancient context
                'max_visual_chars': 80,  # Truncate descriptions
            },
            'normal': {
                'visual_frames': 2,
                'memory_count': 3,
                'log_lines': 10,
                'narrative_history': 3,
                'ancient_history': 1,
                'max_visual_chars': 150,
            },
            'detailed': {
                'visual_frames': 3,
                'memory_count': 5,
                'log_lines': 15,
                'narrative_history': 5,
                'ancient_history': 2,
                'max_visual_chars': 250,
            }
        }
    
    def select_detail_mode(self, store) -> str:
        """
        Intelligently choose detail level based on current state.
        
        Returns: 'minimal', 'normal', or 'detailed'
        """
        scene = store.current_scene
        flow = store.current_flow
        conv_state = store.current_conversation_state
        
        # === MINIMAL (Fast response needed) ===
        
        # High-intensity scenes need quick reactions, not deep context
        if scene in [SceneType.COMBAT_HIGH, SceneType.HORROR_TENSION]:
            print(f"📊 [Detail] MINIMAL mode - high-intensity scene ({scene.name})")
            return 'minimal'
        
        # User is dominating conversation - keep responses tight
        if flow == FlowState.DOMINATED:
            print(f"📊 [Detail] MINIMAL mode - user dominating")
            return 'minimal'
        
        # Staccato flow = rapid-fire chat, keep it snappy
        if flow == FlowState.STACCATO:
            print(f"📊 [Detail] MINIMAL mode - rapid-fire flow")
            return 'minimal'
        
        # === DETAILED (Opportunity for rich response) ===
        
        # Dead air = perfect time for thoughtful, detailed responses
        if flow == FlowState.DEAD_AIR:
            print(f"📊 [Detail] DETAILED mode - dead air opportunity")
            return 'detailed'
        
        # Storytelling mode = user wants depth
        if conv_state == ConversationState.STORYTELLING:
            print(f"📊 [Detail] DETAILED mode - storytelling context")
            return 'detailed'
        
        # Chill scenes = room for detail
        if scene in [SceneType.CHILL_CHATTING, SceneType.EXPLORATION]:
            print(f"📊 [Detail] DETAILED mode - chill scene ({scene.name})")
            return 'detailed'
        
        # === NORMAL (Default) ===
        print(f"📊 [Detail] NORMAL mode - balanced context")
        return 'normal'
    
    def get_limits(self, mode: str) -> Dict[str, Any]:
        """Get the limit configuration for a given mode."""
        return self.detail_modes.get(mode, self.detail_modes['normal'])
    
    def apply_limits_to_visual(self, visual_events: List, mode: str) -> List:
        """Apply frame limit to visual events."""
        limits = self.get_limits(mode)
        return visual_events[-limits['visual_frames']:]
    
    def apply_limits_to_memories(self, memories: List, mode: str) -> List:
        """Apply memory count limit."""
        limits = self.get_limits(mode)
        return memories[:limits['memory_count']]
    
    def apply_limits_to_narrative(self, narrative_log: List, mode: str) -> List:
        """Apply narrative history limit."""
        limits = self.get_limits(mode)
        count = limits['narrative_history']
        return narrative_log[-count:] if count > 0 else []
    
    def apply_limits_to_ancient(self, ancient_log: List, mode: str) -> List:
        """Apply ancient history limit."""
        limits = self.get_limits(mode)
        count = limits['ancient_history']
        return ancient_log[-count:] if count > 0 else []
    
    def truncate_visual_text(self, text: str, mode: str) -> str:
        """Truncate visual descriptions based on mode."""
        limits = self.get_limits(mode)
        max_chars = limits['max_visual_chars']
        
        if len(text) <= max_chars:
            return text
        
        return text[:max_chars-3] + "..."
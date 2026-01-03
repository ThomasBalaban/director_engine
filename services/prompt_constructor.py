# Save as: director_engine/prompt_constructor.py
from typing import List, Dict, Any, Optional
import re
import google.generativeai as genai # type: ignore
from google.generativeai.types import HarmCategory, HarmBlockThreshold # type: ignore
from config import ConversationState, FlowState, InputSource, GEMINI_API_KEY
from context.context_store import ContextStore, EventItem
from systems.decision_engine import Directive

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
                
                # --- NEW: DISABLE ALL SAFETY FILTERS ---
                # We need the raw, unfiltered truth of what is on screen
                # so Nami can decide how to react to it herself.
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
        layers = store.get_all_events_for_summary()
        
        # Filter for high relevance or recency
        active_events = layers['immediate'] + [
            e for e in layers['recent'] 
            if e.score.interestingness > 0.4 or e.source in [InputSource.TWITCH_MENTION, InputSource.DIRECT_MICROPHONE]
        ]
        active_events.sort(key=lambda x: x.timestamp)
        
        # AWAIT the formatting
        recent_events_str = await self._format_recent_events(active_events)
        parts.append(recent_events_str)

        return "\n\n".join(parts)

    async def _format_recent_events(self, events: List[EventItem]) -> str:
        """
        Formats the raw stream. 
        Uses Gemini to condense visual hallucinations if available.
        """
        if not events:
            return "### IMMEDIATE STREAM (Last 30s)\n(Silence...)"
            
        visual_events_text = []
        other_events_lines = []
        
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
                
            elif e.source == InputSource.DIRECT_MICROPHONE:
                other_events_lines.append(f"[🎤 USER SAID] {e.text}")
            elif e.source == InputSource.AMBIENT_AUDIO:
                other_events_lines.append(f"[AMBIENT_AUDIO] {e.text}")
            elif e.source == InputSource.TWITCH_MENTION:
                other_events_lines.append(f"[💬 {e.metadata.get('username', 'Chat')}] {e.text}")
            elif e.source == InputSource.TWITCH_CHAT:
                other_events_lines.append(f"[💬 {e.metadata.get('username', 'Chat')}] {e.text}")
            elif e.source == InputSource.INTERNAL_THOUGHT:
                other_events_lines.append(f"[💭 THOUGHT] {e.text}")
            elif e.source == InputSource.SYSTEM_PATTERN:
                other_events_lines.append(f"[⚙️ SYSTEM] {e.text}")

        all_lines = []
        
        # 1. INTELLIGENT VISUAL SUMMARY
        if visual_events_text:
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
                    
                    response = self.gemini_model.generate_content(gemini_prompt)
                    summary = response.text.strip()
                    
                    all_lines.append(f"### VISUAL CONTEXT (Condensed)\n{summary}")
                    
                except Exception as e:
                    print(f"⚠️ [PromptConstructor] Gemini Summarization Failed: {e}")
                    all_lines.append("### VISUAL CONTEXT (Raw)\n" + "\n".join(visual_events_text[:3]))
            else:
                all_lines.append("### VISUAL CONTEXT (Raw)\n" + "\n".join(visual_events_text))

        # 2. Append non-visual events
        if other_events_lines:
            all_lines.append("### AUDIO & CHAT LOG")
            all_lines.extend(other_events_lines)
        
        return "\n\n".join(all_lines)

    # --- Helper methods ---
    def _format_scene_context(self, store: ContextStore) -> str:
        mood_str = f"Current Mood: {store.current_mood} ({store.emotional_momentum})"
        scene_str = f"Scene: {store.current_scene.name}"
        flow_str = f"Conversation Flow: {store.current_flow.name}"
        return f"### CURRENT SITUATION\n{scene_str}\n{mood_str}\n{flow_str}\nSummary: {store.current_summary}"

    def _format_directive(self, directive: Directive) -> str:
        if not directive: return ""
        constraints = ""
        if directive.constraints: constraints = "\nConstraints: " + ", ".join(directive.constraints)
        return (
            f"### INSTRUCTION (Top Priority)\n"
            f"Goal: {directive.objective}\n"
            f"Tone: {directive.tone}\n"
            f"Action: {directive.suggested_action}"
            f"{constraints}"
        )

    def _format_user_context(self, profile: Dict[str, Any]) -> str:
        facts = [f"- {f['content']}" for f in profile.get('facts', [])[-5:]]
        facts_str = "\n".join(facts) if facts else "No known facts."
        return (
            f"### ACTIVE USER: {profile['username']}\n"
            f"Relationship: {profile['relationship']['tier']} (Affinity: {profile['relationship']['affinity']}%)\n"
            f"Known Facts:\n{facts_str}"
        )

    def _format_memories(self, memories: List[EventItem], narrative_log: List[str]) -> str:
        if not memories and not narrative_log: return ""
        text = "### RELEVANT CONTEXT"
        if narrative_log:
            text += "\n[Previously...]\n"
            for entry in narrative_log[-3:]:
                clean_entry = re.sub(r"^(Here's a summary.*?|In this clip.*?):", "", entry, flags=re.IGNORECASE).strip()
                text += f"- {clean_entry}\n"
        if memories:
            text += "\n[Related Memories]\n"
            for mem in memories:
                content = mem.memory_text or mem.text
                text += f"- (Recall) {content}\n"
        return text
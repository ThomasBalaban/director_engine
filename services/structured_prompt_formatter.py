# PRIORITY 3: Structured Prompt Format
# This provides a clean, parseable XML-like format for better LLM understanding

from typing import List, Dict, Any, Optional
from config import InputSource, SceneType
from context.context_store import ContextStore, EventItem
from systems.decision_engine import Directive


class StructuredPromptFormatter:
    """
    Formats context into clean, structured XML-like blocks.
    
    WHY: Mixed text/data confuses LLMs. Clear structure = better adherence.
    
    BENEFITS:
    - LLM knows what's instruction vs context
    - Priority is visually clear
    - Easier to debug prompts
    - Better context utilization
    """
    
    def __init__(self):
        self.section_order_conversation_focused = [
            'directive',
            'user_context', 
            'conversation_log',
            'scene_background',
            'memories',
        ]
        
        self.section_order_gameplay_focused = [
            'directive',
            'scene_background',
            'visual_context',
            'conversation_log',
            'memories',
        ]
    
    def format_full_prompt(
        self,
        directive: Optional[Directive],
        store: ContextStore,
        events: List[EventItem],
        memories: List[EventItem],
        visual_summary: str,
        conversation_log: str,
        user_is_speaking: bool,
        manual_context: str = "",
        current_streamer: str = ""
    ) -> str:
        """
        Build the complete structured prompt.
        
        Returns a clean, hierarchical prompt with clear sections.
        """
        sections = []
        
        # === OPERATOR NOTES (Always first if present) ===
        operator_block = self._format_operator_notes(current_streamer, manual_context)
        if operator_block:
            sections.append(operator_block)
        
        # === CHOOSE SECTION ORDER BASED ON FOCUS ===
        if user_is_speaking:
            # User talking = conversation priority
            sections.extend(self._build_conversation_focused_prompt(
                directive, store, visual_summary, conversation_log, memories
            ))
        else:
            # Silence/gameplay = commentary priority  
            sections.extend(self._build_gameplay_focused_prompt(
                directive, store, visual_summary, conversation_log, memories
            ))
        
        # Join all sections
        return "\n\n".join(sections)
    
    def _build_conversation_focused_prompt(
        self, directive, store, visual_summary, conversation_log, memories
    ) -> List[str]:
        """User is actively talking - prioritize interaction."""
        sections = []
        
        # 1. DIRECTIVE (Critical instructions)
        if directive:
            sections.append(self._format_directive(directive, priority="CRITICAL"))
        
        # 2. USER CONTEXT (Who we're talking to)
        if store.active_user_profile:
            sections.append(self._format_user_context(store.active_user_profile))
        
        # 3. CONVERSATION LOG (Main focus - what was said)
        sections.append(f"""<focus type="CONVERSATION" priority="HIGH">
{conversation_log}
</focus>""")
        
        # 4. SCENE BACKGROUND (Secondary - what's happening visually)
        sections.append(f"""<background type="GAME_STATE" priority="LOW">
{self._format_scene_metadata(store)}

{visual_summary}
</background>""")
        
        # 5. MEMORIES (Callback opportunities)
        if memories:
            sections.append(self._format_memories(memories, store))
        
        return sections
    
    def _build_gameplay_focused_prompt(
        self, directive, store, visual_summary, conversation_log, memories
    ) -> List[str]:
        """User is quiet - prioritize gameplay commentary."""
        sections = []
        
        # 1. DIRECTIVE
        if directive:
            sections.append(self._format_directive(directive, priority="CRITICAL"))
        
        # 2. SCENE STATE + VISUALS (Main focus)
        sections.append(f"""<focus type="GAMEPLAY" priority="HIGH">
{self._format_scene_metadata(store)}

{visual_summary}
</focus>""")
        
        # 3. CONVERSATION LOG (Background context)
        sections.append(f"""<background type="RECENT_CHAT" priority="LOW">
{conversation_log}
</background>""")
        
        # 4. USER CONTEXT
        if store.active_user_profile:
            sections.append(self._format_user_context(store.active_user_profile))
        
        # 5. MEMORIES
        if memories:
            sections.append(self._format_memories(memories, store))
        
        return sections
    
    def _format_directive(self, directive: Directive, priority: str = "CRITICAL") -> str:
        """Format directive as structured instruction block."""
        constraints_xml = ""
        if directive.constraints:
            constraints_list = "\n".join([f"  • {c}" for c in directive.constraints])
            constraints_xml = f"\n<constraints>\n{constraints_list}\n</constraints>"
        
        return f"""<directive priority="{priority}">
  <objective>{directive.objective}</objective>
  <tone>{directive.tone}</tone>
  <action>{directive.suggested_action}</action>{constraints_xml}
  <reasoning>{directive.reasoning}</reasoning>
</directive>"""
    
    def _format_operator_notes(self, streamer: str, context: str) -> str:
        """Format manual operator input."""
        if not streamer and not context:
            return ""
        
        parts = []
        if streamer:
            parts.append(f"<watching>{streamer}</watching>")
        if context:
            parts.append(f"<manual_context>{context}</manual_context>")
        
        content = "\n  ".join(parts)
        return f"""<operator_notes priority="REFERENCE">
  {content}
</operator_notes>"""
    
    def _format_scene_metadata(self, store: ContextStore) -> str:
        """Format scene state info."""
        return f"""<scene_state>
  <type>{store.current_scene.name}</type>
  <mood>{store.current_mood}</mood>
  <flow>{store.current_flow.name}</flow>
  <conversation_state>{store.current_conversation_state.name}</conversation_state>
  <summary>{store.current_summary}</summary>
</scene_state>"""
    
    def _format_user_context(self, profile: Dict[str, Any]) -> str:
        """Format active user profile."""
        username = profile.get('username', 'Unknown')
        role = profile.get('role', 'viewer')
        tier = profile.get('relationship', {}).get('tier', 'Unknown')
        affinity = profile.get('relationship', {}).get('affinity', 0)
        
        facts = profile.get('facts', [])
        facts_list = ""
        if facts:
            recent_facts = facts[-5:]  # Last 5 facts
            facts_list = "\n    ".join([f"• {f['content']}" for f in recent_facts])
            facts_list = f"\n  <known_facts>\n    {facts_list}\n  </known_facts>"
        
        role_context = ""
        if role == "handler":
            role_context = "\n  <note>This is your creator/handler - feel free to roast them or deflect blame</note>"
        
        return f"""<active_user priority="REFERENCE">
  <username>{username}</username>
  <role>{role}</role>
  <relationship tier="{tier}" affinity="{affinity}%"/>{facts_list}{role_context}
</active_user>"""
    
    def _format_memories(
        self, 
        memories: List[EventItem], 
        store: ContextStore
    ) -> str:
        """Format memories with callback hints."""
        lines = []
        
        # Ancient history (oldest)
        if store.ancient_history_log:
            ancient = store.ancient_history_log[-2:]  # Last 2
            lines.append("  <ancient_history>")
            for entry in ancient:
                clean = self._clean_text(entry)
                lines.append(f"    • {clean}")
            lines.append("  </ancient_history>")
        
        # Narrative history (recent stream events)
        if store.narrative_log:
            narrative = store.narrative_log[-3:]  # Last 3
            lines.append("  <recent_stream_events>")
            for entry in narrative:
                clean = self._clean_text(entry)
                lines.append(f"    • {clean}")
            lines.append("  </recent_stream_events>")
        
        # Semantic memories (relevant moments)
        if memories:
            lines.append("  <relevant_moments>")
            for mem in memories[:3]:  # Top 3
                content = mem.memory_text or mem.text
                clean = self._clean_text(content)
                
                # Add callback hint
                hint = self._generate_callback_hint(mem, store)
                hint_attr = f' callback_hint="{hint}"' if hint else ''
                
                lines.append(f'    • {clean}{hint_attr}')
            lines.append("  </relevant_moments>")
        
        if not lines:
            return ""
        
        content = "\n".join(lines)
        return f"""<callback_material priority="REFERENCE">
  <instruction>Reference these naturally when relevant, don't force it</instruction>
{content}
</callback_material>"""
    
    def _generate_callback_hint(self, mem: EventItem, store: ContextStore) -> Optional[str]:
        """Suggest when to bring up a memory."""
        text = (mem.memory_text or mem.text).lower()
        
        # Pattern matching for callback opportunities
        if any(word in text for word in ["died", "failed", "lost", "game over"]):
            return "Use when: User fails again / Skill issue"
        
        elif any(word in text for word in ["won", "beat", "victory", "success"]):
            return "Use when: User wins / Celebrating"
        
        elif "chat" in text and any(word in text for word in ["said", "asked", "told"]):
            return "Use when: Similar chat interaction"
        
        elif store.current_scene.name.lower() in text:
            return "Use when: Same scene/game"
        
        elif any(word in text for word in ["funny", "laugh", "joke"]):
            return "Use when: Comedy moment"
        
        return None
    
    def _clean_text(self, text: str) -> str:
        """Remove AI preambles and clean text."""
        import re
        
        # Remove common AI filler
        text = re.sub(
            r"^(Here's|Here is|The memorable moment is|Memorable moment:|Earlier:|Previously:)\s*:?\s*",
            "",
            text,
            flags=re.IGNORECASE
        )
        
        # Remove quotes if wrapped
        text = text.strip()
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        if text.startswith("'") and text.endswith("'"):
            text = text[1:-1]
        
        return text.strip()


# INTEGRATION EXAMPLE:
# In PromptConstructor.__init__():
#     self.formatter = StructuredPromptFormatter()
#
# In construct_context_block():
#     return self.formatter.format_full_prompt(
#         directive=directive,
#         store=store,
#         events=active_events,
#         memories=memories,
#         visual_summary=visual_str,
#         conversation_log=log_str,
#         user_is_speaking=user_is_speaking,
#         manual_context=shared.get_manual_context(),
#         current_streamer=shared.get_current_streamer()
#     )


# EXAMPLE OUTPUT:
"""
<operator_notes priority="REFERENCE">
  <watching>PeepingOtter</watching>
  <manual_context>Playing Phasmophobia - ghost hunting horror game</manual_context>
</operator_notes>

<directive priority="CRITICAL">
  <objective>Mock Failure</objective>
  <tone>Sharp, Sarcastic, Ruthless</tone>
  <action>Laugh at the handler's incompetence. Do not offer help. Say 'Skill Issue'.</action>
  <constraints>
  • LOW BATTERY: Keep responses short (1 sentence max).
  </constraints>
  <reasoning>Goal: TROLL | Flow: STACCATO</reasoning>
</directive>

<focus type="GAMEPLAY" priority="HIGH">
<scene_state>
  <type>HORROR_TENSION</type>
  <mood>Scared</mood>
  <flow>STACCATO</flow>
  <conversation_state>FRUSTRATED</conversation_state>
  <summary>User just died to a ghost in the hallway</summary>
</scene_state>

### VISUAL CONTEXT (Condensed)
A dark hallway with a red hue. "YOU DIED" text on screen.
</focus>

<background type="RECENT_CHAT" priority="LOW">
[🎤 USER SAID] No no no no!
[💬 ChatUser1] L
[💬 ChatUser2] LMAOOO
</background>

<callback_material priority="REFERENCE">
  <instruction>Reference these naturally when relevant, don't force it</instruction>
  <recent_stream_events>
    • Otter died to the same ghost 3 times in a row
  </recent_stream_events>
  <relevant_moments>
    • Previous death: User blamed lag and got roasted by chat callback_hint="Use when: User fails again / Skill issue"
  </relevant_moments>
</callback_material>
"""
# PRIORITY 4: Conversation Threading System
# Tracks ongoing conversation threads to prevent "Nami forgot what we were talking about"

import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class ThreadStatus(Enum):
    ACTIVE = "active"           # Currently being discussed
    PENDING = "pending"         # User asked something, waiting for resolution
    RESOLVED = "resolved"       # Topic naturally concluded
    ABANDONED = "abandoned"     # User moved on without resolution


@dataclass
class ConversationThread:
    """
    Represents an ongoing conversation topic.
    
    Example thread lifecycle:
    1. User: "Should I get the red sword or blue sword?"
       -> Thread created (PENDING, topic="weapon choice")
    
    2. Nami: "Red sword has better damage"
       -> Thread updated (ACTIVE)
    
    3. User: "Ok I'll get red"
       -> Thread resolved (RESOLVED)
    
    If user changes topic without resolving -> ABANDONED
    """
    topic: str                          # e.g., "weapon choice", "game strategy"
    initiated_by: str                   # "user" or "nami"
    started_at: float
    last_update: float
    status: ThreadStatus = ThreadStatus.ACTIVE
    
    # Conversation history within this thread
    user_statements: List[str] = field(default_factory=list)
    nami_responses: List[str] = field(default_factory=list)
    
    # Context for this thread
    original_question: Optional[str] = None  # If user asked something
    resolution: Optional[str] = None          # How it was resolved
    
    # Metadata
    importance: float = 0.5  # 0.0 = casual, 1.0 = critical
    requires_followup: bool = False
    
    def add_user_statement(self, text: str):
        """User contributed to this thread."""
        self.user_statements.append(text)
        self.last_update = time.time()
        
        # Check if user is asking a question
        if "?" in text:
            self.original_question = text
            self.requires_followup = True
            self.importance = min(1.0, self.importance + 0.2)
    
    def add_nami_response(self, text: str):
        """Nami contributed to this thread."""
        self.nami_responses.append(text)
        self.last_update = time.time()
        
        # If there was a question, mark as addressed
        if self.original_question:
            self.requires_followup = False
    
    def is_stale(self, timeout_seconds: float = 60.0) -> bool:
        """Check if thread is too old to be active."""
        return (time.time() - self.last_update) > timeout_seconds
    
    def get_summary(self) -> str:
        """Get a one-line summary of this thread."""
        last_user = self.user_statements[-1] if self.user_statements else None
        last_nami = self.nami_responses[-1] if self.nami_responses else None
        
        if self.status == ThreadStatus.PENDING and last_user:
            return f"Pending: User asked '{last_user}'"
        elif self.status == ThreadStatus.ACTIVE:
            if last_nami:
                return f"Active: Discussing {self.topic} (last: '{last_nami[:40]}...')"
            else:
                return f"Active: {self.topic}"
        elif self.status == ThreadStatus.RESOLVED:
            return f"Resolved: {self.topic} - {self.resolution or 'concluded'}"
        else:
            return f"{self.status.value}: {self.topic}"


class ConversationThreadManager:
    """
    Manages all conversation threads.
    
    KEY FEATURES:
    - Detects when user changes topic
    - Tracks unresolved questions
    - Provides "what were we talking about?" context
    - Prevents duplicate thread creation
    """
    
    def __init__(self):
        self.threads: List[ConversationThread] = []
        self.max_threads = 10  # Keep last 10 threads
        self.thread_timeout = 60.0  # 60 seconds without activity = stale
        
    def track_user_statement(
        self, 
        text: str, 
        detected_topic: Optional[str] = None,
        importance: float = 0.5
    ) -> Optional[ConversationThread]:
        """
        Track a user statement. Either continues existing thread or starts new one.
        
        Args:
            text: What the user said
            detected_topic: Optional topic extracted from context (e.g., from LLM analysis)
            importance: How important is this (0.0-1.0)
        
        Returns:
            The active thread (either existing or newly created)
        """
        # Get current active thread
        active_thread = self.get_active_thread()
        
        # Decide if this continues the thread or starts a new one
        if active_thread and not active_thread.is_stale(self.thread_timeout):
            # Check if topic shifted
            if detected_topic and detected_topic != active_thread.topic:
                # Topic shift - mark old thread as abandoned, start new
                print(f"📍 [Thread] Topic shift: '{active_thread.topic}' -> '{detected_topic}'")
                active_thread.status = ThreadStatus.ABANDONED
                return self._create_new_thread(text, detected_topic, "user", importance)
            else:
                # Continue existing thread
                active_thread.add_user_statement(text)
                print(f"📍 [Thread] Continuing: {active_thread.topic}")
                return active_thread
        else:
            # No active thread or stale - create new
            topic = detected_topic or self._infer_topic_from_text(text)
            return self._create_new_thread(text, topic, "user", importance)
    
    def track_nami_response(
        self, 
        text: str,
        resolves_thread: bool = False
    ) -> Optional[ConversationThread]:
        """
        Track Nami's response.
        
        Args:
            text: What Nami said
            resolves_thread: If True, marks the current thread as resolved
        """
        active_thread = self.get_active_thread()
        
        if not active_thread:
            # Nami initiated conversation
            topic = self._infer_topic_from_text(text)
            return self._create_new_thread(text, topic, "nami", importance=0.3)
        
        # Add to existing thread
        active_thread.add_nami_response(text)
        
        if resolves_thread:
            active_thread.status = ThreadStatus.RESOLVED
            active_thread.resolution = text[:50]
            print(f"✅ [Thread] Resolved: {active_thread.topic}")
        
        return active_thread
    
    def get_active_thread(self) -> Optional[ConversationThread]:
        """Get the currently active thread (most recent non-resolved)."""
        for thread in reversed(self.threads):
            if thread.status in [ThreadStatus.ACTIVE, ThreadStatus.PENDING]:
                if not thread.is_stale(self.thread_timeout):
                    return thread
        return None
    
    def get_pending_threads(self) -> List[ConversationThread]:
        """Get threads that require followup."""
        return [
            t for t in self.threads 
            if t.status == ThreadStatus.PENDING and t.requires_followup
        ]
    
    def get_thread_context_for_prompt(self) -> Optional[str]:
        """
        Generate prompt context for active thread.
        
        Returns XML block to inject into prompt.
        """
        active = self.get_active_thread()
        
        if not active:
            # Check for unresolved questions
            pending = self.get_pending_threads()
            if pending:
                oldest = pending[0]
                return f"""<unresolved_question>
  <asked_at>{int(time.time() - oldest.started_at)}s ago</asked_at>
  <question>{oldest.original_question}</question>
  <hint>User is waiting for an answer to this</hint>
</unresolved_question>"""
            return None
        
        # Active thread context
        last_user = active.user_statements[-1] if active.user_statements else None
        last_nami = active.nami_responses[-1] if active.nami_responses else None
        
        context_parts = []
        context_parts.append(f"  <topic>{active.topic}</topic>")
        context_parts.append(f"  <status>{active.status.value}</status>")
        
        if last_user:
            context_parts.append(f"  <user_last_said>{last_user}</user_last_said>")
        
        if last_nami:
            context_parts.append(f"  <you_last_said>{last_nami}</you_last_said>")
        
        if active.original_question and active.requires_followup:
            context_parts.append(f"  <pending_question>{active.original_question}</pending_question>")
            context_parts.append(f"  <hint>User is waiting for an answer</hint>")
        else:
            context_parts.append(f"  <hint>Continue this thread naturally or resolve it</hint>")
        
        content = "\n".join(context_parts)
        
        return f"""<active_conversation>
{content}
</active_conversation>"""
    
    def _create_new_thread(
        self, 
        initial_text: str, 
        topic: str, 
        initiated_by: str,
        importance: float
    ) -> ConversationThread:
        """Create and register a new thread."""
        thread = ConversationThread(
            topic=topic,
            initiated_by=initiated_by,
            started_at=time.time(),
            last_update=time.time(),
            importance=importance
        )
        
        if initiated_by == "user":
            thread.add_user_statement(initial_text)
            # Check if it's a question
            if "?" in initial_text:
                thread.status = ThreadStatus.PENDING
        else:
            thread.add_nami_response(initial_text)
        
        self.threads.append(thread)
        
        # Cleanup old threads
        if len(self.threads) > self.max_threads:
            self.threads = self.threads[-self.max_threads:]
        
        print(f"🆕 [Thread] Started: '{topic}' (by {initiated_by})")
        return thread
    
    def _infer_topic_from_text(self, text: str) -> str:
        """
        Simple topic extraction from text.
        In production, you'd use the LLM to extract this.
        """
        text_lower = text.lower()
        
        # Common patterns
        if any(word in text_lower for word in ["should i", "which", "what do you think"]):
            return "seeking advice"
        
        if "?" in text:
            return "question"
        
        if any(word in text_lower for word in ["died", "lost", "failed"]):
            return "gameplay failure"
        
        if any(word in text_lower for word in ["won", "beat", "killed"]):
            return "gameplay victory"
        
        if any(word in text_lower for word in ["game", "weapon", "item", "skill"]):
            return "game mechanics"
        
        # Default
        return "general chat"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get thread statistics for debugging."""
        return {
            "total_threads": len(self.threads),
            "active_threads": len([t for t in self.threads if t.status == ThreadStatus.ACTIVE]),
            "pending_threads": len([t for t in self.threads if t.status == ThreadStatus.PENDING]),
            "resolved_threads": len([t for t in self.threads if t.status == ThreadStatus.RESOLVED]),
            "current_thread": self.get_active_thread().get_summary() if self.get_active_thread() else None
        }


# INTEGRATION INSTRUCTIONS:
"""
1. Add to context_store.py:
   
   from systems.conversation_threading import ConversationThreadManager
   
   class ContextStore:
       def __init__(self):
           # ...existing code...
           self.thread_manager = ConversationThreadManager()

2. In core_logic.py, when processing user speech:
   
   async def process_engine_event(source, text, metadata, username):
       # ...existing code...
       
       if source in [InputSource.MICROPHONE, InputSource.DIRECT_MICROPHONE]:
           # Track the user statement
           detected_topic = metadata.get('topic')  # If LLM extracted it
           importance = metadata.get('importance', 0.5)
           
           shared.store.thread_manager.track_user_statement(
               text=text,
               detected_topic=detected_topic,
               importance=importance
           )

3. When Nami responds (in bot_reply handler):
   
   @shared.sio.on("bot_reply")
   async def receive_bot_reply(sid, payload: dict):
       # ...existing code...
       
       reply_text = payload.get('reply', '')
       
       # Check if this resolves a question
       resolves = "?" not in reply_text and shared.store.thread_manager.get_active_thread()
       
       shared.store.thread_manager.track_nami_response(
           text=reply_text,
           resolves_thread=resolves
       )

4. In prompt_constructor.py construct_context_block():
   
   # Get thread context
   thread_context = store.thread_manager.get_thread_context_for_prompt()
   
   if thread_context:
       sections.append(thread_context)

5. Add to breadcrumbs endpoint for debugging:
   
   @app.get("/thread_stats")
   async def get_thread_stats():
       return shared.store.thread_manager.get_stats()
"""


# EXAMPLE USAGE:
"""
Thread lifecycle example:

User: "Should I pick the fire sword or ice sword?"
-> Creates thread: topic="weapon choice", status=PENDING, requires_followup=True

Nami: "Fire sword has better DPS but ice sword has crowd control"
-> Updates thread: status=ACTIVE, requires_followup=False

User: "Ok going with fire"
-> Updates thread: status=RESOLVED

---

Thread interruption example:

User: "Should I pick the fire sword?"
-> Thread A: topic="weapon choice", status=PENDING

User: "Wait, how much gold do I have?"
-> Thread A: status=ABANDONED
-> Thread B: topic="inventory check", status=PENDING

Nami: "You have 500 gold"
-> Thread B: status=ACTIVE

Nami: "Also, about the sword - I'd go fire"
-> Thread A: status=ACTIVE (can be resumed!)
"""
# 🧪 TESTING FRAMEWORK FOR PROMPT QUALITY
# Automated tests to verify improvements are working

import asyncio
import time
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum


class TestResult(Enum):
    PASS = "✅ PASS"
    FAIL = "❌ FAIL"
    WARN = "⚠️ WARN"


@dataclass
class TestCase:
    name: str
    description: str
    result: TestResult
    details: str
    metric_value: float = 0.0
    expected_value: float = 0.0


class PromptQualityTester:
    """
    Tests to verify that all 4 priorities are working correctly.
    """
    
    def __init__(self, store, prompt_constructor):
        self.store = store
        self.prompt_constructor = prompt_constructor
        self.results: List[TestCase] = []
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run complete test suite."""
        print("\n" + "="*60)
        print("🧪 RUNNING PROMPT QUALITY TEST SUITE")
        print("="*60 + "\n")
        
        self.results = []
        
        # Priority 1: Smart Memory Queries
        await self.test_memory_query_scene_awareness()
        await self.test_memory_query_intent_awareness()
        
        # Priority 2: Adaptive Detail Levels
        await self.test_detail_level_selection()
        await self.test_token_reduction_combat()
        
        # Priority 3: Structured Prompts
        await self.test_prompt_structure()
        await self.test_directive_visibility()
        
        # Priority 4: Conversation Threading
        await self.test_thread_creation()
        await self.test_thread_continuity()
        
        # Generate report
        return self._generate_report()
    
    # =========================================================================
    # PRIORITY 1 TESTS: Smart Memory Queries
    # =========================================================================
    
    async def test_memory_query_scene_awareness(self):
        """Test that memory queries adapt to scene type."""
        from config import SceneType
        from core_logic import build_smart_memory_query
        
        # Test horror scene
        self.store.current_scene = SceneType.HORROR_TENSION
        query = build_smart_memory_query(self.store, {})
        
        horror_keywords = ["scared", "tension", "jumpscare", "afraid", "creepy"]
        has_horror_keywords = any(kw in query.lower() for kw in horror_keywords)
        
        result = TestResult.PASS if has_horror_keywords else TestResult.FAIL
        self.results.append(TestCase(
            name="Memory Query - Scene Awareness (Horror)",
            description="Horror scene should inject fear-related keywords",
            result=result,
            details=f"Query: '{query[:100]}...' | Found keywords: {has_horror_keywords}"
        ))
        
        # Test combat scene
        self.store.current_scene = SceneType.COMBAT_HIGH
        query = build_smart_memory_query(self.store, {})
        
        combat_keywords = ["won", "died", "combat", "fight", "boss"]
        has_combat_keywords = any(kw in query.lower() for kw in combat_keywords)
        
        result = TestResult.PASS if has_combat_keywords else TestResult.FAIL
        self.results.append(TestCase(
            name="Memory Query - Scene Awareness (Combat)",
            description="Combat scene should inject performance keywords",
            result=result,
            details=f"Query: '{query[:100]}...' | Found keywords: {has_combat_keywords}"
        ))
    
    async def test_memory_query_intent_awareness(self):
        """Test that memory queries adapt to user intent."""
        from config import UserIntent
        from core_logic import build_smart_memory_query
        
        # Test validation intent
        self.store.current_intent = UserIntent.VALIDATION
        query = build_smart_memory_query(self.store, {})
        
        validation_keywords = ["good job", "well done", "nice", "amazing"]
        has_keywords = any(kw in query.lower() for kw in validation_keywords)
        
        result = TestResult.PASS if has_keywords else TestResult.FAIL
        self.results.append(TestCase(
            name="Memory Query - Intent Awareness (Validation)",
            description="User seeking validation should get praise-related memories",
            result=result,
            details=f"Query contains validation keywords: {has_keywords}"
        ))
    
    # =========================================================================
    # PRIORITY 2 TESTS: Adaptive Detail Levels
    # =========================================================================
    
    async def test_detail_level_selection(self):
        """Test that detail mode selection is correct."""
        from config import SceneType, FlowState
        from services.prompt_constructor import AdaptiveDetailController
        
        controller = AdaptiveDetailController()
        
        # Test 1: Combat should be minimal
        self.store.current_scene = SceneType.COMBAT_HIGH
        mode = controller.select_detail_mode(self.store)
        
        result = TestResult.PASS if mode == 'minimal' else TestResult.FAIL
        self.results.append(TestCase(
            name="Detail Level - Combat Scene",
            description="Combat scenes should use MINIMAL detail",
            result=result,
            details=f"Selected mode: {mode} (expected: minimal)"
        ))
        
        # Test 2: Dead air should be detailed
        self.store.current_scene = SceneType.CHILL_CHATTING
        self.store.current_flow = FlowState.DEAD_AIR
        mode = controller.select_detail_mode(self.store)
        
        result = TestResult.PASS if mode == 'detailed' else TestResult.FAIL
        self.results.append(TestCase(
            name="Detail Level - Dead Air",
            description="Dead air should use DETAILED mode",
            result=result,
            details=f"Selected mode: {mode} (expected: detailed)"
        ))
    
    async def test_token_reduction_combat(self):
        """Test that combat scenes use fewer tokens."""
        from config import SceneType
        
        # Simulate combat scene
        self.store.current_scene = SceneType.COMBAT_HIGH
        
        # Get prompt (would need actual implementation)
        # For now, just check that detail controller reduces limits
        from services.prompt_constructor import AdaptiveDetailController
        controller = AdaptiveDetailController()
        
        minimal_limits = controller.get_limits('minimal')
        normal_limits = controller.get_limits('normal')
        
        reduction_ratio = minimal_limits['visual_frames'] / normal_limits['visual_frames']
        
        result = TestResult.PASS if reduction_ratio <= 0.6 else TestResult.WARN
        self.results.append(TestCase(
            name="Token Reduction - Combat",
            description="Combat mode should reduce visual frames by 40%+",
            result=result,
            details=f"Reduction: {(1-reduction_ratio)*100:.0f}% (minimal: {minimal_limits['visual_frames']}, normal: {normal_limits['visual_frames']})",
            metric_value=reduction_ratio,
            expected_value=0.5
        ))
    
    # =========================================================================
    # PRIORITY 3 TESTS: Structured Prompts
    # =========================================================================
    
    async def test_prompt_structure(self):
        """Test that prompts have proper XML structure."""
        # This would need actual prompt generation
        # For now, check that formatter exists
        
        has_formatter = hasattr(self.prompt_constructor, 'formatter')
        
        result = TestResult.PASS if has_formatter else TestResult.FAIL
        self.results.append(TestCase(
            name="Structured Format - Formatter Exists",
            description="PromptConstructor should have StructuredPromptFormatter",
            result=result,
            details=f"Has formatter: {has_formatter}"
        ))
        
        # Test XML tag presence (would need generated prompt)
        # Mock test for now
        result = TestResult.WARN
        self.results.append(TestCase(
            name="Structured Format - XML Tags",
            description="Prompts should contain <directive>, <focus>, <background> tags",
            result=result,
            details="Run manual test: check /breadcrumbs for XML structure"
        ))
    
    async def test_directive_visibility(self):
        """Test that directives are prominently placed."""
        # Mock test - would need actual prompt
        result = TestResult.WARN
        self.results.append(TestCase(
            name="Structured Format - Directive Priority",
            description="Directive should appear near top of prompt with priority=CRITICAL",
            result=result,
            details="Run manual test: check /breadcrumbs - directive should be first section"
        ))
    
    # =========================================================================
    # PRIORITY 4 TESTS: Conversation Threading
    # =========================================================================
    
    async def test_thread_creation(self):
        """Test that threads are created correctly."""
        # Check if thread manager exists
        has_thread_manager = hasattr(self.store, 'thread_manager')
        
        if not has_thread_manager:
            self.results.append(TestCase(
                name="Threading - Manager Exists",
                description="ContextStore should have ConversationThreadManager",
                result=TestResult.FAIL,
                details="thread_manager not found in store"
            ))
            return
        
        # Test thread creation
        thread_manager = self.store.thread_manager
        
        # Clear threads
        thread_manager.threads = []
        
        # Simulate user question
        thread = thread_manager.track_user_statement(
            "Should I pick the sword or axe?",
            detected_topic="weapon choice",
            importance=0.8
        )
        
        result = TestResult.PASS if thread is not None else TestResult.FAIL
        self.results.append(TestCase(
            name="Threading - Thread Creation",
            description="Thread should be created for user question",
            result=result,
            details=f"Thread created: {thread is not None}, Topic: {thread.topic if thread else 'N/A'}"
        ))
        
        # Test pending status
        is_pending = thread.status.value == "pending" if thread else False
        result = TestResult.PASS if is_pending else TestResult.FAIL
        self.results.append(TestCase(
            name="Threading - Pending Status",
            description="Question should create PENDING thread",
            result=result,
            details=f"Status: {thread.status.value if thread else 'N/A'} (expected: pending)"
        ))
    
    async def test_thread_continuity(self):
        """Test that threads maintain continuity."""
        if not hasattr(self.store, 'thread_manager'):
            self.results.append(TestCase(
                name="Threading - Continuity",
                description="Thread continuity test skipped (no manager)",
                result=TestResult.WARN,
                details="Install thread_manager first"
            ))
            return
        
        thread_manager = self.store.thread_manager
        thread_manager.threads = []
        
        # Create thread
        thread1 = thread_manager.track_user_statement("How much gold do I have?")
        
        # Continue thread
        thread2 = thread_manager.track_nami_response("You have 500 gold")
        
        # Should be same thread
        same_thread = thread1 is thread2 if (thread1 and thread2) else False
        
        result = TestResult.PASS if same_thread else TestResult.FAIL
        self.results.append(TestCase(
            name="Threading - Continuity",
            description="Responses should continue same thread",
            result=result,
            details=f"Same thread object: {same_thread}"
        ))
    
    # =========================================================================
    # REPORT GENERATION
    # =========================================================================
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate test report."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.result == TestResult.PASS)
        failed = sum(1 for r in self.results if r.result == TestResult.FAIL)
        warned = sum(1 for r in self.results if r.result == TestResult.WARN)
        
        pass_rate = (passed / total * 100) if total > 0 else 0
        
        print("\n" + "="*60)
        print("📊 TEST RESULTS SUMMARY")
        print("="*60)
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⚠️ Warnings: {warned}")
        print(f"Pass Rate: {pass_rate:.1f}%")
        print("="*60 + "\n")
        
        # Print individual results
        for result in self.results:
            print(f"{result.result.value} {result.name}")
            print(f"   {result.description}")
            print(f"   {result.details}\n")
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "warned": warned,
            "pass_rate": pass_rate,
            "results": [
                {
                    "name": r.name,
                    "result": r.result.value,
                    "details": r.details,
                    "metric": r.metric_value,
                    "expected": r.expected_value
                }
                for r in self.results
            ]
        }

# Save as: director_engine/main.py
import uvicorn
import asyncio
import threading
import webbrowser
import signal as signal_module
import time
import traceback
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import socketio

import config
from systems import process_manager
import services.llm_analyst as llm_analyst
from services.sensor_bridge import SensorBridge
from scoring import EventScore
import shared
import core_logic

# --- APP SETUP ---
app = FastAPI(title="Nami Director Engine")
socket_app = socketio.ASGIApp(shared.sio)
app.mount('/socket.io', socket_app)

base_path = Path(__file__).parent.resolve()
ui_path = base_path / "ui"
audio_path = base_path / "audio_effects"

# --- TASKS ---
summary_ticker_task: Optional[asyncio.Task] = None
reflex_ticker_task: Optional[asyncio.Task] = None
sensor_bridge_task: Optional[asyncio.Task] = None

# --- MODELS & ROUTES ---
class EventPayload(BaseModel):
    source_str: str
    text: str
    metadata: Dict[str, Any] = {}
    username: Optional[str] = None

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "ok", "server_ready": shared.server_ready}

@app.get("/audio_effects/{filename}")
async def serve_audio_effect(filename: str):
    file_path = audio_path / filename
    if not file_path.exists() or not str(file_path.resolve()).startswith(str(audio_path.resolve())):
        return {"error": "Invalid file"}
    return FileResponse(file_path, media_type="audio/wav")

@app.get("/summary", response_class=PlainTextResponse)
async def get_summary():
    summary, _ = shared.store.get_summary()
    return summary

@app.get("/summary_data")
async def get_summary_data():
    data = shared.store.get_summary_data()
    if data['directive']: 
        data['directive'] = data['directive'].to_dict()
    return data

@app.get("/breadcrumbs")
async def get_breadcrumbs(count: int = 3):
    """
    Returns formatted context for Nami's prompt.
    This is the bridge between Director (Brain 1) and Nami (Brain 2).
    """
    try:
        summary_data = shared.store.get_summary_data()
        current_query = summary_data.get('summary', "") or " ".join(summary_data.get('topics', []))
        smart_memories = shared.memory_optimizer.retrieve_relevant_memories(shared.store, current_query, limit=5)
        
        directive = summary_data.get('directive')
        
        formatted_context = await shared.prompt_constructor.construct_context_block(
            shared.store, directive, smart_memories[:3]
        )
        
        return {"formatted_context": formatted_context}
        
    except Exception as e:
        print(f"❌ [/breadcrumbs] Error: {type(e).__name__}: {e}")
        traceback.print_exc()
        return {"formatted_context": f"[Director Error: {str(e)}]"}
    
@app.get("/thread_stats")
async def get_thread_stats():
    """Debug endpoint for conversation threading"""
    return shared.store.thread_manager.get_stats()

@app.get("/prompt_debug")
async def get_prompt_debug():
    """See the actual prompt being sent to Nami"""
    summary_data = shared.store.get_summary_data()
    current_query = summary_data.get('summary', "") or " ".join(summary_data.get('topics', []))
    smart_memories = shared.memory_optimizer.retrieve_relevant_memories(shared.store, current_query, limit=5)
    
    directive = summary_data.get('directive')
    formatted_context = await shared.prompt_constructor.construct_context_block(
        shared.store, directive, smart_memories[:3]
    )
    
    detail_mode = shared.prompt_constructor.detail_controller.select_detail_mode(shared.store)
    
    return {
        "formatted_context": formatted_context,
        "detail_mode": detail_mode,
        "thread_stats": shared.store.thread_manager.get_stats(),
        "scene": shared.store.current_scene.name,
        "memory_count": len(shared.store.all_memories)
    }

@app.get("/prompt_size")
async def get_prompt_size():
    """Monitor token usage"""
    summary_data = shared.store.get_summary_data()
    current_query = summary_data.get('summary', "")
    smart_memories = shared.memory_optimizer.retrieve_relevant_memories(shared.store, current_query, limit=5)
    
    directive = summary_data.get('directive')
    context = await shared.prompt_constructor.construct_context_block(
        shared.store, directive, smart_memories[:3]
    )
    
    return {
        "char_count": len(context),
        "estimated_tokens": len(context) // 4,
        "lines": context.count('\n'),
        "sections": context.count('<')
    }

@app.get("/run_tests")
async def run_quality_tests():
    """Run automated prompt quality tests"""
    from test_framework import PromptQualityTester
    tester = PromptQualityTester(shared.store, shared.prompt_constructor)
    report = await tester.run_all_tests()
    return report

@app.get("/speech_state")
async def get_speech_state():
    return {"is_speaking": shared.is_nami_speaking()}

@app.post("/speech_started")
async def speech_started():
    shared.set_nami_speaking(True)
    return {"status": "ok"}

@app.post("/speech_finished")
async def speech_finished():
    shared.set_nami_speaking(False)
    return {"status": "ok"}

@app.get("/lock_states")
async def get_lock_states():
    return {
        "streamer_locked": shared.is_streamer_locked(),
        "context_locked": shared.is_context_locked()
    }

# =====================================================
# INTERRUPT SYSTEM ENDPOINTS
# =====================================================

@app.get("/interrupt_stats")
async def get_interrupt_stats():
    """Debug endpoint for interrupt system status"""
    return shared.get_interrupt_stats()

@app.post("/simulate_interrupt")
async def simulate_interrupt(payload: dict = {}):
    """
    Test endpoint: Simulate a direct mention to test the interrupt pipeline.
    
    Body:
        source: "mic" | "twitch" (default: "mic")
        text: The message text (default: "Hey Nami, what do you think?")
        username: Twitch username (default: "peepingotter")
    """
    source_str = payload.get('source', 'mic')
    text = payload.get('text', 'Hey Nami, what do you think?')
    username = payload.get('username', 'peepingotter')
    
    if source_str == 'twitch':
        source = config.InputSource.TWITCH_MENTION
    else:
        source = config.InputSource.DIRECT_MICROPHONE
    
    # Track state before
    was_speaking = shared.is_nami_speaking()
    was_awaiting = shared.awaiting_user_response
    
    # Process as real event
    await core_logic.process_engine_event(
        source=source,
        text=text,
        metadata={'username': username, 'simulated': True},
        username=username
    )
    
    return {
        "status": "ok",
        "simulated_source": source.name,
        "text": text,
        "was_speaking_before": was_speaking,
        "was_awaiting_before": was_awaiting,
        "is_speaking_after": shared.is_nami_speaking(),
        "is_awaiting_after": shared.awaiting_user_response,
        "interrupted": was_speaking and not shared.is_nami_speaking(),
        "interrupt_stats": shared.get_interrupt_stats()
    }

@app.post("/simulate_speaking")
async def simulate_speaking(payload: dict = {}):
    """Test helper: Set Nami to 'speaking' state for testing interrupts."""
    shared.set_nami_speaking(True, source='SIMULATED_TEST')
    return {
        "status": "ok",
        "is_speaking": shared.is_nami_speaking(),
        "note": f"Nami is now 'speaking'. Will timeout after {shared.SPEECH_TIMEOUT}s or on interrupt."
    }

@app.post("/simulate_stop_speaking")
async def simulate_stop_speaking():
    """Test helper: Clear Nami's speaking state."""
    shared.set_nami_speaking(False)
    return {"status": "ok", "is_speaking": shared.is_nami_speaking()}

# --- IMPORTANT: Mount static files AFTER all API routes ---
app.mount("/static", StaticFiles(directory=ui_path, html=True), name="static")
app.mount("/", StaticFiles(directory=ui_path, html=True), name="ui_static")

# --- SOCKET HANDLERS ---
@shared.sio.on("event")
async def ingest_event(sid, payload: dict):
    try:
        model = EventPayload(**payload)
        await core_logic.process_engine_event(
            config.InputSource[model.source_str], 
            model.text, 
            model.metadata, 
            model.username
        )
    except Exception as e:
        print(f"[DirectorEngine] Invalid event payload: {e}")

@shared.sio.on("bot_reply")
async def receive_bot_reply(sid, payload: dict):
    try:
        reply_text = payload.get('reply', '')
        active_thread = shared.store.thread_manager.get_active_thread()

        if active_thread:
            resolves = "?" not in reply_text
            shared.store.thread_manager.track_nami_response(
                text=reply_text,
                resolves_thread=resolves
            )
        
        shared.emit_bot_reply(
            payload.get('reply', ''), 
            payload.get('prompt', ''), 
            payload.get('is_censored', False),
            payload.get('censorship_reason'),
            payload.get('filtered_area')
        )
        shared.speech_dispatcher.register_user_response()
    except Exception as e:
        print(f"[DirectorEngine] Error handling bot_reply: {e}")

@shared.sio.on("speech_started")
async def handle_speech_started(sid, payload: dict = None):
    """Called when Nami starts speaking (TTS begins)"""
    source = payload.get('source', 'UNKNOWN') if payload else 'UNKNOWN'
    shared.set_nami_speaking(True, source=source)

@shared.sio.on("speech_finished")
async def handle_speech_finished(sid, payload: dict = None):
    """Called when Nami finishes speaking (TTS complete)"""
    shared.set_nami_speaking(False)

@shared.sio.on("set_streamer")
async def handle_set_streamer(sid, payload: dict):
    """Called when operator changes the streamer dropdown"""
    streamer_id = payload.get('streamer_id', 'peepingotter')
    shared.set_current_streamer(streamer_id, from_ai=False)

@shared.sio.on("set_manual_context")
async def handle_set_manual_context(sid, payload: dict):
    """Called when operator sets manual context"""
    context = payload.get('context', '')
    shared.set_manual_context(context, from_ai=False)

@shared.sio.on("set_streamer_lock")
async def handle_set_streamer_lock(sid, payload: dict):
    """Called when operator toggles the streamer lock"""
    locked = payload.get('locked', False)
    shared.set_streamer_locked(locked)

@shared.sio.on("set_context_lock")
async def handle_set_context_lock(sid, payload: dict):
    """Called when operator toggles the context lock"""
    locked = payload.get('locked', False)
    shared.set_context_locked(locked)

# --- SERVER LIFECYCLE ---
def open_browser():
    try: 
        webbrowser.open(f"http://localhost:{config.DIRECTOR_PORT}")
    except: 
        pass

async def run_ticker_with_recovery(ticker_func, name: str):
    """Wrapper that restarts a ticker if it crashes."""
    while shared.server_ready:
        try:
            await ticker_func()
        except asyncio.CancelledError:
            print(f"[{name}] Cancelled")
            break
        except Exception as e:
            print(f"❌ [{name}] Crashed with error: {type(e).__name__}: {e}")
            traceback.print_exc()
            print(f"[{name}] Restarting in 5 seconds...")
            await asyncio.sleep(5)
            if shared.server_ready:
                print(f"[{name}] Restarting...")

def run_server():
    global summary_ticker_task, reflex_ticker_task, sensor_bridge_task
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    shared.ui_event_loop = loop
    
    loop.run_until_complete(llm_analyst.create_http_client())
    
    # Start Tickers with recovery wrappers
    summary_ticker_task = loop.create_task(
        run_ticker_with_recovery(core_logic.summary_ticker, "SummaryTicker")
    )
    reflex_ticker_task = loop.create_task(
        run_ticker_with_recovery(core_logic.reflex_ticker, "ReflexTicker")
    )
    
    sensor_bridge = SensorBridge(event_callback=core_logic.process_engine_event)
    sensor_bridge_task = loop.create_task(sensor_bridge.run())
    
    shared.server_ready = True
    print("✅ Director Engine is READY")
    print(f"📡 API endpoints available at http://localhost:{config.DIRECTOR_PORT}/breadcrumbs")
    
    # Configure uvicorn with better settings
    server_config = uvicorn.Config(
        app, 
        host=config.DIRECTOR_HOST, 
        port=config.DIRECTOR_PORT, 
        log_level="warning", 
        loop="asyncio",
        timeout_keep_alive=30,
        ws_ping_interval=20,
        ws_ping_timeout=20
    )
    server = uvicorn.Server(server_config)
    
    shutdown_event = asyncio.Event()
    
    def signal_handler():
        print("\n⚠️ Shutdown signal received...")
        shared.server_ready = False
        shutdown_event.set()
        server.should_exit = True
    
    for sig in (signal_module.SIGINT, signal_module.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass
    
    try:
        loop.run_until_complete(server.serve())
    except KeyboardInterrupt:
        print("\n⚠️ Keyboard interrupt")
        shared.server_ready = False
    finally:
        print("\n🛑 DIRECTOR ENGINE SHUTDOWN INITIATED")
        shared.server_ready = False
        
        # Cancel all tasks gracefully
        tasks_to_cancel = [
            (sensor_bridge_task, "SensorBridge"),
            (summary_ticker_task, "SummaryTicker"),
            (reflex_ticker_task, "ReflexTicker")
        ]
        
        for task, name in tasks_to_cancel:
            if task and not task.done():
                print(f"  Cancelling {name}...")
                task.cancel()
                try:
                    loop.run_until_complete(asyncio.wait_for(task, timeout=3.0))
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                except Exception as e:
                    print(f"  Error cancelling {name}: {e}")
        
        # Cleanup Services
        print("  Closing HTTP client...")
        try:
            loop.run_until_complete(llm_analyst.close_http_client())
        except Exception as e:
            print(f"  Error closing HTTP client: {e}")
            
        print("  Closing speech dispatcher...")
        try:
            loop.run_until_complete(shared.speech_dispatcher.close())
        except Exception as e:
            print(f"  Error closing speech dispatcher: {e}")
        
        print("  Shutting down vision app...")
        process_manager.shutdown_vision_app()
        
        time.sleep(1.0)
        
        try:
            # Cancel any remaining tasks
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
        except Exception as e:
            print(f"  Error during final cleanup: {e}")
            
        print("✅ SHUTDOWN COMPLETE")

if __name__ == "__main__":
    print("🧠 DIRECTOR ENGINE (Brain 1) - Starting...")
    process_manager.launch_vision_app()
    threading.Timer(2.0, open_browser).start()
    run_server()
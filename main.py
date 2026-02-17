# Save as: director_engine/main.py
"""
Director Engine — The Brain.

NOTE: Speech state endpoints (/speech_started, /speech_finished, /interrupt_*)
have been moved to the Prompt Service (port 8001). Nami now reports her TTS
state to the prompt service, not here.
"""

import uvicorn
import asyncio
import threading
import signal as signal_module
import time
import traceback
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import socketio

import config
from systems import process_manager
import services.llm_analyst as llm_analyst
import services.prompt_client as prompt_client
from services.sensor_bridge import SensorBridge
from scoring import EventScore
import shared
import core_logic

# --- APP SETUP ---
app = FastAPI(title="Nami Director Engine")
socket_app = socketio.ASGIApp(shared.sio)
app.mount('/socket.io', socket_app)

base_path = Path(__file__).parent.resolve()

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
    return {"status": "ok", "server_ready": shared.server_ready}

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
    Bridge between Director (Brain 1) and Nami (Brain 2).
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
    return shared.store.thread_manager.get_stats()

@app.get("/prompt_debug")
async def get_prompt_debug():
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

@app.get("/lock_states")
async def get_lock_states():
    return {
        "streamer_locked": shared.is_streamer_locked(),
        "context_locked": shared.is_context_locked()
    }

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
        
        # Tell prompt service: Nami just responded to user
        asyncio.create_task(prompt_client.notify_bot_response())

    except Exception as e:
        print(f"[DirectorEngine] Error handling bot_reply: {e}")

@shared.sio.on("set_streamer")
async def handle_set_streamer(sid, payload: dict):
    streamer_id = payload.get('streamer_id', 'peepingotter')
    shared.set_current_streamer(streamer_id, from_ai=False)

@shared.sio.on("set_manual_context")
async def handle_set_manual_context(sid, payload: dict):
    context = payload.get('context', '')
    shared.set_manual_context(context, from_ai=False)

@shared.sio.on("set_streamer_lock")
async def handle_set_streamer_lock(sid, payload: dict):
    locked = payload.get('locked', False)
    shared.set_streamer_locked(locked)

@shared.sio.on("set_context_lock")
async def handle_set_context_lock(sid, payload: dict):
    locked = payload.get('locked', False)
    shared.set_context_locked(locked)


async def run_ticker_with_recovery(ticker_func, name: str):
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
    
    # Start Tickers
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
    print(f"📡 Prompt Service expected at {config.PROMPT_SERVICE_URL}")
    
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
            pass
    
    try:
        loop.run_until_complete(server.serve())
    except KeyboardInterrupt:
        print("\n⚠️ Keyboard interrupt")
        shared.server_ready = False
    finally:
        print("\n🛑 DIRECTOR ENGINE SHUTDOWN INITIATED")
        shared.server_ready = False
        
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
        print("  Closing HTTP clients...")
        try:
            loop.run_until_complete(llm_analyst.close_http_client())
        except Exception as e:
            print(f"  Error closing HTTP client: {e}")
            
        print("  Closing prompt client...")
        try:
            loop.run_until_complete(prompt_client.close())
        except Exception as e:
            print(f"  Error closing prompt client: {e}")
        
        print("  Shutting down vision app...")
        process_manager.shutdown_vision_app()
        
        time.sleep(1.0)
        
        try:
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
    run_server()
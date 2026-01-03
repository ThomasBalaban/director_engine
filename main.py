# Save as: director_engine/main.py
import uvicorn
import asyncio
import threading
import webbrowser
import signal as signal_module
import time
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
import shared
import core_logic

# --- APP SETUP ---
app = FastAPI(title="Nami Director Engine")
socket_app = socketio.ASGIApp(shared.sio)
app.mount('/socket.io', socket_app)

base_path = Path(__file__).parent.resolve()
ui_path = base_path / "ui"
audio_path = base_path / "audio_effects"
app.mount("/static", StaticFiles(directory=ui_path, html=True), name="static")
app.mount("/", StaticFiles(directory=ui_path, html=True), name="ui_static")

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
    if data['directive']: data['directive'] = data['directive'].to_dict()
    return data

@app.get("/breadcrumbs")
async def get_breadcrumbs(count: int = 3):
    summary_data = shared.store.get_summary_data()
    current_query = summary_data.get('summary', "") or " ".join(summary_data.get('topics', []))
    smart_memories = shared.memory_optimizer.retrieve_relevant_memories(shared.store, current_query, limit=5)
    formatted_context = await shared.prompt_constructor.construct_context_block(
        shared.store, summary_data.get('directive'), smart_memories[:3]
    )
    return {"formatted_context": formatted_context}

# --- SOCKET HANDLERS ---
@shared.sio.on("event")
async def ingest_event(sid, payload: dict):
    try:
        model = EventPayload(**payload)
        await core_logic.process_engine_event(config.InputSource[model.source_str], model.text, model.metadata, model.username)
    except Exception as e:
        print(f"Invalid event payload: {e}")

@shared.sio.on("bot_reply")
async def receive_bot_reply(sid, payload: dict):
    shared.emit_bot_reply(payload.get('reply', ''), payload.get('prompt', ''), payload.get('is_censored', False))

# --- SERVER LIFECYCLE ---
def open_browser():
    try: webbrowser.open(f"http://localhost:{config.DIRECTOR_PORT}")
    except: pass

def run_server():
    global summary_ticker_task, reflex_ticker_task, sensor_bridge_task
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    shared.ui_event_loop = loop
    
    loop.run_until_complete(llm_analyst.create_http_client())
    
    # Start Tickers
    summary_ticker_task = loop.create_task(core_logic.summary_ticker())
    reflex_ticker_task = loop.create_task(core_logic.reflex_ticker())
    
    sensor_bridge = SensorBridge(event_callback=core_logic.process_engine_event)
    sensor_bridge_task = loop.create_task(sensor_bridge.run())
    
    shared.server_ready = True
    print("✅ Director Engine is READY")
    
    server = uvicorn.Server(uvicorn.Config(app, host=config.DIRECTOR_HOST, port=config.DIRECTOR_PORT, log_level="warning", loop="asyncio"))
    
    def signal_handler():
        print("\n⚠️ Shutdown signal received...")
        server.should_exit = True
    
    for sig in (signal_module.SIGINT, signal_module.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)
    
    try:
        loop.run_until_complete(server.serve())
    except KeyboardInterrupt:
        print("\n⚠️ Keyboard interrupt")
    finally:
        print("\n🛑 DIRECTOR ENGINE SHUTDOWN INITIATED")
        shared.server_ready = False
        
        # Cleanup Tasks
        for task in [sensor_bridge_task, summary_ticker_task, reflex_ticker_task]:
            if task and not task.done():
                task.cancel()
                try: loop.run_until_complete(asyncio.wait_for(task, timeout=2.0))
                except: pass
        
        # Cleanup Services
        try: loop.run_until_complete(llm_analyst.close_http_client())
        except: pass
        try: loop.run_until_complete(shared.speech_dispatcher.close())
        except: pass
        
        process_manager.shutdown_vision_app()
        time.sleep(1.0)
        try: loop.close()
        except: pass
        print("✅ SHUTDOWN COMPLETE")

if __name__ == "__main__":
    print("🧠 DIRECTOR ENGINE (Brain 1) - Starting...")
    process_manager.launch_vision_app()
    threading.Timer(2.0, open_browser).start()
    run_server()
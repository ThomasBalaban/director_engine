# thomasbalaban/director_engine/main.py
import uvicorn
import asyncio
import threading
import signal as signal_module
import time
import traceback
import re
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional

import config
from systems import process_manager
import services.llm_analyst as llm_analyst
import services.prompt_client as prompt_client
from scoring import EventScore
import shared
import core_logic

app = FastAPI(title="Nami Director Engine")

# --- HUB HANDLERS (Brain intake) ---

@shared.sio.on("connect")
async def on_hub_connect():
    print("✅ [Director] Connected to Hub")

@shared.sio.on("vision_context")
async def handle_vision_context(payload: dict):
    """Processes vision updates relayed from the Monitor via Hub."""
    content = payload.get("content", "").strip()
    if not content or content.lower() in ["[silence]", "none"]: 
        return

    print(f"👁️ [Director] vision_context received: {len(content)} chars")

    # Re-use parsing logic (XML support)
    xml_pattern = r"<(\w+)>([\s\S]*?)<\/\1>"
    matches = list(re.finditer(xml_pattern, content))
    
    if matches:
        for match in matches:
            tag = match.group(1).lower()
            text = match.group(2).strip()
            source = config.InputSource.VISUAL_CHANGE
            if tag == "audio_context": source = config.InputSource.AMBIENT_AUDIO
            
            await core_logic.process_engine_event(
                source, text, {"xml_tag": tag, "type": "hub_vision"}
            )
    else:
        # Fallback to plain narrative
        await core_logic.process_engine_event(
            config.InputSource.VISUAL_CHANGE, 
            content, 
            {"type": "hub_vision_plain"}
        )

@shared.sio.on("audio_context")
async def handle_audio_context(payload: dict):
    """Processes transcripts relayed from the Monitor via Hub."""
    text = payload.get("text", "").strip()
    source_str = payload.get("source", "desktop")
    if not text: return

    if source_str == "microphone":
        source = config.InputSource.DIRECT_MICROPHONE if "nami" in text.lower() else config.InputSource.MICROPHONE
    else:
        source = config.InputSource.AMBIENT_AUDIO

    await core_logic.process_engine_event(
        source, text, payload.get("metadata", {"type": "hub_audio"})
    )

@shared.sio.on("event")
async def ingest_event(payload: dict):
    try:
        model = EventPayload(**payload)
        await core_logic.process_engine_event(
            config.InputSource[model.source_str],
            model.text,
            model.metadata,
            model.username
        )
    except Exception as e: print(f"[DirectorEngine] Invalid event payload: {e}")

@shared.sio.on("bot_reply")
async def receive_bot_reply(payload: dict):
    try:
        reply_text = payload.get('reply', '')
        active_thread = shared.store.thread_manager.get_active_thread()
        if active_thread:
            resolves = "?" not in reply_text
            shared.store.thread_manager.track_nami_response(text=reply_text, resolves_thread=resolves)
        shared.emit_bot_reply(reply_text, payload.get('prompt', ''), payload.get('is_censored', False))
        asyncio.create_task(prompt_client.notify_bot_response())
    except Exception as e: print(f"[DirectorEngine] Error handling bot_reply: {e}")

# ... other set_* handlers remain the same as standard Socket.IO client implementation ...

async def connect_to_hub():
    while shared.server_ready:
        if not shared.sio.connected:
            try:
                await shared.sio.connect(config.HUB_URL, transports=['websocket', 'polling'])
            except Exception as e:
                print(f"⚠️ Hub failed: {e}")
                await asyncio.sleep(5)
                continue
        await asyncio.sleep(2)

def run_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    shared.ui_event_loop = loop

    loop.run_until_complete(llm_analyst.create_http_client())
    loop.create_task(connect_to_hub())
    loop.create_task(core_logic.summary_ticker())
    loop.create_task(core_logic.reflex_ticker())

    shared.server_ready = True
    print("✅ Director Engine is READY")

    server_config = uvicorn.Config(app, host=config.DIRECTOR_HOST, port=config.DIRECTOR_PORT, log_level="warning")
    server = uvicorn.Server(server_config)
    try:
        loop.run_until_complete(server.serve())
    finally:
        loop.close()

if __name__ == "__main__":
    run_server()
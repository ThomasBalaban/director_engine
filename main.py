# director_engine/main.py
import uvicorn
import asyncio
import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import config
import shared
import core_logic
import services.llm_analyst as llm_analyst
from services.sensor_bridge import SensorBridge

app = FastAPI(title="Nami Director Engine")

# Initialize the SensorBridge so it hooks up to the shared.sio Hub connection
sensor_bridge = SensorBridge()
pending_memory_requests = {}

# --- Request Models ---

class ContextRequest(BaseModel):
    trigger: Optional[str] = None
    event_id: Optional[str] = None
    metadata: Optional[dict] = {}


# --- Hub Connection ---

async def connect_to_hub():
    """Background task to maintain connection to the Central Hub."""
    while shared.server_ready:
        if not shared.sio.connected:
            try:
                print(f"🔌 [Director Engine] Connecting to Hub at {config.HUB_URL}...")
                await shared.sio.connect(config.HUB_URL, transports=['websocket', 'polling'])
            except Exception as e:
                print(f"⚠️ [Director Engine] Hub connection failed: {e}. Retrying in 5s...")
                await asyncio.sleep(5)
                continue
        await asyncio.sleep(2)


# --- Endpoints ---

@app.get("/health")
async def health():
    """The UI Launcher queries this to check if the Director is online."""
    return {
        "status": "ok",
        "service": "director_engine",
        "hub_connected": shared.sio.connected
    }


@app.post("/context")
async def get_context(payload: ContextRequest = ContextRequest()):
    """
    Called by the prompt service when it needs to build a full prompt.

    Returns the complete structured context block assembled from:
    - Current visual/audio/chat events in the context store
    - Semantic memory retrieval (hybrid scored)
    - Current directive from the decision engine
    - Narrative log, ancient history, thread state
    - Active user profile

    The prompt service passes optional trigger metadata so the director
    can tailor the context to what triggered the speech request.
    """
    directive = shared.store.current_directive

    # Build smart query for semantic memory retrieval
    summary_data = shared.store.get_summary_data()
    memory_query = core_logic.build_smart_memory_query(shared.store, summary_data)

    # --- NEW: Flush pending memories to the Hub before we query ---
    if hasattr(shared.store, 'pending_memories_to_save') and shared.store.pending_memories_to_save:
        for mem in shared.store.pending_memories_to_save:
            await shared.sio.emit('save_memory', mem)
        shared.store.pending_memories_to_save.clear()

    # --- NEW: Fetch memories from the microservice instead of locally ---
    smart_memories = await fetch_memories_for_prompt(memory_query, limit=5)

    try:
        context_block = await shared.prompt_constructor.construct_context_block(
            store=shared.store,
            directive=directive,
            memories=smart_memories
        )
    except Exception as e:
        print(f"❌ [Director /context] Failed to build context: {e}")
        import traceback
        traceback.print_exc()
        context_block = f"[Context build error: {e}]"

    return {
        "context": context_block,
        "mood": shared.store.current_mood,
        "scene": shared.store.current_scene.name,
        "flow": shared.store.current_flow.name,
        "conversation_state": shared.store.current_conversation_state.name,
        "summary": summary_data.get("summary", ""),
        "directive": directive.to_dict() if directive else None,
        "active_user": shared.store.active_user_profile,
        "manual_context": shared.get_manual_context(),
        "current_streamer": shared.get_current_streamer(),
    }


@app.get("/breadcrumbs")
async def breadcrumbs(count: int = 5):
    """
    Lightweight endpoint for the prompt service to poll recent high-interest events
    without triggering a full context build.

    Useful for quick checks (e.g., 'has anything important happened recently?')
    before deciding whether to call /context.
    """
    return shared.store.get_breadcrumbs(count=count)


@app.get("/thread_stats")
async def thread_stats():
    """Debug endpoint to inspect conversation thread state."""
    return shared.store.thread_manager.get_stats()


@app.get("/memory_stats")
async def memory_stats():
    """Debug endpoint to inspect memory store state."""
    
    # 1. Ask the microservice for the top 5 memories. 
    # (Sending an empty query triggers it to just return the highest importance ones)
    top_memories = await fetch_memories_for_prompt("", limit=5)
    
    with shared.store.lock:
        return {
            "total_memories": "Managed by Memory Service",
            "narrative_log_count": len(shared.store.narrative_log),
            "ancient_history_count": len(shared.store.ancient_history_log),
            "pending_saves": len(getattr(shared.store, 'pending_memories_to_save', [])),
            
            # 2. Format them exactly how the UI expects them!
            "top_memories": [
                {
                    "text": (m.get('memory_text') or m.get('text', ''))[:80],
                    "score": round(m.get('importance', 0), 3),
                    "source": "memory_service",
                }
                for m in top_memories
            ]
        }


@shared.sio.on('memory_results')
def on_memory_results(data):
    """Catches the reply from the memory_service."""
    req_id = data.get('request_id')
    if req_id in pending_memory_requests:
        # Resolve the future with the returned memories
        pending_memory_requests[req_id].set_result(data.get('memories', []))

async def fetch_memories_for_prompt(query_text: str, limit: int = 5) -> list:
    """Helper to ask the memory service for context and await the reply."""
    req_id = str(uuid.uuid4())
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    pending_memory_requests[req_id] = future
    
    # Send the request to the Hub
    await shared.sio.emit('query_memories', {
        'request_id': req_id, 
        'query': query_text, 
        'limit': limit
    })
    
    try:
        # Wait up to 2.5 seconds for the memory_service to reply
        results = await asyncio.wait_for(future, timeout=2.5)
        return results
    except asyncio.TimeoutError:
        print("⚠️ [Memory] Query timed out. Memory service might be offline.")
        return []
    finally:
        pending_memory_requests.pop(req_id, None)

@app.get("/store_stats")
async def store_stats():
    """Debug endpoint to inspect event store layer sizes."""
    layers = shared.store.get_all_events_for_summary()
    return {
        "immediate": len(layers["immediate"]),
        "recent": len(layers["recent"]),
        "background": len(layers["background"]),
        "narrative_log": len(shared.store.narrative_log),
        "ancient_history": len(shared.store.ancient_history_log),
        "current_mood": shared.store.current_mood,
        "current_scene": shared.store.current_scene.name,
        "current_flow": shared.store.current_flow.name,
        "conversation_state": shared.store.current_conversation_state.name,
    }


# --- Server Boot ---

def run_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    shared.ui_event_loop = loop

    # Boot up background tasks
    loop.run_until_complete(llm_analyst.create_http_client())
    loop.create_task(connect_to_hub())
    loop.create_task(core_logic.summary_ticker())
    loop.create_task(core_logic.reflex_ticker())

    shared.server_ready = True
    print("✅ Director Engine is READY")
    print(f"   /context      → Full structured context for prompt service")
    print(f"   /breadcrumbs  → Lightweight recent event poll")
    print(f"   /health       → Status check")
    print(f"   /store_stats  → Event layer debug")
    print(f"   /memory_stats → Memory debug")
    print(f"   /thread_stats → Conversation thread debug")

    # Start the HTTP server required by the Nami Launcher UI
    server_config = uvicorn.Config(
        app,
        host=config.DIRECTOR_HOST,
        port=config.DIRECTOR_PORT,
        log_level="warning"
    )
    server = uvicorn.Server(server_config)
    try:
        loop.run_until_complete(server.serve())
    finally:
        loop.close()


if __name__ == "__main__":
    run_server()
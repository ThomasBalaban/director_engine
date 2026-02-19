# director_engine/main.py
import uvicorn
import asyncio
from fastapi import FastAPI
import config
import shared
import core_logic
import services.llm_analyst as llm_analyst
from services.sensor_bridge import SensorBridge

app = FastAPI(title="Nami Director Engine")

# Initialize the SensorBridge so it hooks up to the shared.sio Hub connection
sensor_bridge = SensorBridge()

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

@app.get("/health")
async def health():
    """The UI Launcher queries this to check if the Director is online."""
    return {"status": "ok", "service": "director_engine", "hub_connected": shared.sio.connected}

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

    # Start the HTTP server required by the Nami Launcher UI
    server_config = uvicorn.Config(app, host=config.DIRECTOR_HOST, port=config.DIRECTOR_PORT, log_level="warning")
    server = uvicorn.Server(server_config)
    try:
        loop.run_until_complete(server.serve())
    finally:
        loop.close()

if __name__ == "__main__":
    run_server()
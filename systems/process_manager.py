# Save as: director_engine/process_manager.py
import subprocess
import os
import sys
import signal as signal_module
import atexit
from pathlib import Path
from typing import Optional

# Global handle for the child process
vision_process: Optional[subprocess.Popen] = None

def get_conda_python_path(env_name):
    """Finds the direct python executable for a conda env."""
    home = Path.home()
    possible_paths = [
        home / "miniconda3" / "envs" / env_name / "bin" / "python",
        home / "anaconda3" / "envs" / env_name / "bin" / "python",
        home / "opt" / "miniconda3" / "envs" / env_name / "bin" / "python",
    ]
    for p in possible_paths:
        if p.exists(): return str(p)
    return "python"

def launch_vision_app():
    """Attempts to launch the sibling 'desktop_mon_gemini' app."""
    global vision_process
    print("\n👁️ [Launcher] Attempting to start Vision Subsystem...")
    
    current_dir = Path(__file__).parent.resolve()
    workspace_root = current_dir.parent.parent
    vision_app_path = workspace_root / "desktop_mon_gemini"
    
    if not vision_app_path.exists():
        vision_app_path = workspace_root / "desktop_monitor_gemini"
        if not vision_app_path.exists():
            print(f"⚠️ [Launcher] Vision app not found at {workspace_root}")
            return

    python_exe = get_conda_python_path("gemini-screen-watcher")
    cmd = [python_exe, "main.py"]

    try:
        # Create a process group so we can kill the whole tree if needed
        vision_process = subprocess.Popen(
            cmd, 
            cwd=vision_app_path,
            preexec_fn=os.setsid 
        )
        print(f"✅ [Launcher] Vision Subsystem started (PID: {vision_process.pid})")
    except Exception as e:
        print(f"❌ [Launcher] Failed to start vision app: {e}")

def shutdown_vision_app():
    """Kills the vision subsystem child process securely."""
    global vision_process
    if vision_process:
        print(f"🛑 [Shutdown] Terminating Vision Subsystem (PID: {vision_process.pid})...")
        try:
            os.killpg(os.getpgid(vision_process.pid), signal_module.SIGTERM)
            vision_process.wait(timeout=3)
            print("✅ Vision Subsystem stopped.")
        except Exception as e:
            print(f"⚠️ [Shutdown] Error killing process group: {e}")
            try:
                vision_process.kill()
            except: pass
        vision_process = None

atexit.register(shutdown_vision_app)
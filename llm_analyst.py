# director_engine/llm_analyst.py
import ollama
import json
import httpx
from config import OLLAMA_MODEL, OLLAMA_HOST, NAMI_INTERJECT_URL, INTERJECTION_THRESHOLD
from context_store import ContextStore, EventItem

# Initialize a persistent client for Ollama
try:
    ollama_client = ollama.Client(host=OLLAMA_HOST)
    # Test connection
    ollama_client.list()
    print(f"[Analyst] Successfully connected to Ollama at {OLLAMA_HOST}")
except Exception as e:
    print(f"[Analyst] ERROR: Could not connect to Ollama at {OLLAMA_HOST}. Is it running?")
    print(f"[Analyst] {e}")
    ollama_client = None

# A persistent client for making HTTP calls (e.g., to trigger Nami)
http_client = httpx.AsyncClient()


def build_analysis_prompt(text: str) -> str:
    """Builds the prompt for the analyst LLM to force JSON output."""
    return f"""
You are an event analyzer. Your only job is to rate the "interestingness" of the
following event for a streamer's AI assistant.
Respond ONLY with a single, valid JSON object: {{"score": <a_float_from_0.0_to_1.0>}}

Event: "{text}"
"""

def parse_llm_response(response_text: str) -> float | None:
    """Safely parses the JSON string response from the LLM."""
    try:
        data = json.loads(response_text)
        score = data.get("score")
        if isinstance(score, (float, int)):
            return max(0.0, min(float(score), 1.0)) # Clamp score between 0 and 1
    except json.JSONDecodeError:
        print(f"[Analyst] LLM response was not valid JSON: {response_text}")
    except Exception as e:
        print(f"[Analyst] Error parsing LLM response: {e}")
    return None


async def analyze_and_update_event(event: EventItem, store: ContextStore):
    """
    This is the main background task.
    It calls the Ollama LLM to get an "intelligent" score and updates the store.
    """
    if not ollama_client:
        print("[Analyst] Cannot analyze: Ollama client not available.")
        return

    prompt = build_analysis_prompt(event.text)
    
    try:
        print(f"[Analyst] Analyzing event {event.id} (Sieve score: {event.score:.2f})...")
        response = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            options={"temperature": 0.2},
            format='json'  # Force JSON output from Ollama
        )
        
        response_text = response['message']['content']
        new_score = parse_llm_response(response_text)
        
        if new_score is not None:
            # Successfully got a new score from the LLM
            if new_score > event.score:
                print(f"[Analyst] ✅ Score UPDATED: {event.id} from {event.score:.2f} -> {new_score:.2f}")
                store.update_event_score(event.id, new_score)
                
                # Check if this NEW score triggers an interjection
                if new_score >= INTERJECTION_THRESHOLD:
                    print(f"[Analyst] Tier 2 Interjection! LLM score {new_score:.2f} >= {INTERJECTION_THRESHOLD}")
                    await trigger_nami_interjection(event, new_score)
            else:
                print(f"[Analyst] LLM score not used (new: {new_score:.2f}, old: {event.score:.2f})")
        else:
            print(f"[Analyst] ❌ LLM analysis FAILED for event {event.id}.")
            
    except Exception as e:
        print(f"[Analyst] ERROR: Ollama call failed for event {event.id}: {e}")


async def trigger_nami_interjection(event: EventItem, score: float) -> bool:
    """
    Proactively sends a high-priority "interject" event to Nami's
    Input Funnel (Brain 2).
    """
    try:
        interject_payload = {
            "content": event.text,
            "priority": 1.0 - score,  # Higher score = lower priority number
            "source_info": {
                "source": f"DIRECTOR_{event.source.name}",
                "use_tts": True,
                **event.metadata
            }
        }
        
        # We will implement the receiving end of this in Nami later
        response = await http_client.post(NAMI_INTERJECT_URL, json=interject_payload, timeout=2.0)
        
        if response.status_code == 200:
            print(f"[Director] Nami's Funnel accepted interjection.")
            return True
        else:
            print(f"[Director] Nami's Funnel rejected interjection. Status: {response.status_code}")
            return False
            
    except httpx.ConnectError:
        print(f"[Director] FAILED to connect to Nami's Funnel at {NAMI_INTERJECT_URL}")
        return False
    except Exception as e:
        print(f"[Director] FAILED to send interjection: {e}")
        return False

async def close_http_client():
    """Closes the persistent HTTP client."""
    await http_client.aclose()
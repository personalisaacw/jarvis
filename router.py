from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response
import httpx
import json
import time

# Import Semantic Router
from semantic_router import Route, SemanticRouter
from semantic_router.encoders import HuggingFaceEncoder

app = FastAPI()

# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_URL = "http://127.0.0.1:11434"
BASE_MODEL = "qwen3:4b"
EXPOSED_MODEL = "jarvis-assistant"
QUICK_NUM_CTX = 4096
THINKING_NUM_CTX = 8192

# ============================================================
# SEMANTIC ROUTER INITIALIZATION
# ============================================================

print("[INIT] Loading Semantic Router on CPU...")
encoder = HuggingFaceEncoder(name="sentence-transformers/all-MiniLM-L6-v2")

quick_route = Route(
    name="quick",
    utterances=[
        "what time is it",
        "how tall is the eiffel tower",
        "give me a quick summary of hashmaps",
        "turn off the living room lights",
        "who is the president of france",
        "translate hello to spanish"
    ]
)

thinking_route = Route(
    name="think",
    utterances=[
        "design a scalable architecture for a react app",
        "analyze this concept and explain the pros and cons",
        "plan a detailed itinerary for my trip to japan",
        "walk me through the logic of a neural network",
        "refactor this code and explain the changes"
    ]
)

code_route = Route(
    name="code",
    utterances=[
        "write a python script to list files",
        "implement a new feature in my project",
        "fix the bug in router.py",
        "build a flask api",
        "refactor the authentication logic",
        "write a javascript function to sort an array",
        "create a new react component",
        "write code for sorting an array",
        "develop a coding solution",
        "help me refactor some functions",
        "write a code script",
        "add a feature to the codebase"
    ]
)

semantic_intent_router = SemanticRouter(
    encoder=encoder, 
    routes=[quick_route, thinking_route, code_route],
    auto_sync="local",
    aggregation="max"
)

# ============================================================
# AUDIT LOGGER HELPERS
# ============================================================

def get_last_user_message(data: dict) -> str:
    messages = data.get("messages", [])
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        return str(content) if not isinstance(content, str) else content
    return ""

def determine_intent(text: str):
    """Returns a tuple of (boolean_should_think, raw_route_object)"""
    route = semantic_intent_router(text)
    if route.name == "think":
        return True, route
    return False, route

def print_audit_box(title: str, content: str):
    """Prints a clean, boxed layout in the terminal for debugging."""
    print(f"\n┌── {title} " + "─" * (75 - len(title)))
    for line in str(content).split('\n'):
        print(f"│ {line}")
    print("└" + "─" * 79)

# ============================================================
# MODEL LIST
# ============================================================

@app.get("/api/tags")
async def list_models():
    now = int(time.time())
    return {
        "models": [{
            "name": EXPOSED_MODEL,
            "model": EXPOSED_MODEL,
            "modified_at": now,
            "size": 0,
            "digest": "",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": "qwen3",
                "families": ["qwen3"],
                "parameter_size": "4B",
                "quantization_level": "Q4_K_M",
            },
        }]
    }

# ============================================================
# MAIN PROXY
# ============================================================

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def proxy(request: Request, path: str):
    start_time = time.perf_counter()
    url = f"{OLLAMA_URL}/{path}"
    body = await request.body()
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)

    client = httpx.AsyncClient(timeout=None)

    try:
        # Intercept Chat / Generate requests
        if body and path in ["api/chat", "api/generate"]:
            
            print_audit_box("1. INCOMING REQUEST FROM ANYTHINGLLM", f"Endpoint: {path}\nRaw Body Snippet: {body.decode('utf-8')[:300]}...")
            
            try:
                data = json.loads(body)
                
                # 2. Extract Text
                text = get_last_user_message(data) if path == "api/chat" else str(data.get("prompt", ""))
                print_audit_box("2. EXTRACTED PROMPT", text)
                
                # 3. Analyze Intent
                should_think, route_info = determine_intent(text)
                
                print_audit_box(
                    "3. SEMANTIC ROUTER DECISION", 
                    f"Matched Route: {route_info.name if route_info.name else 'None (Fell back to QUICK)'}\n"
                    f"Action: Setting think={should_think}\n"
                )
                
                if route_info.name == "code":
                    task_file = "C:\\OllamaThinkRouter\\opencode_task.md"
                    with open(task_file, "w", encoding="utf-8") as f:
                        f.write(f"# Opencode Task\n\n- **Prompt**: {text}\n- **Created**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\nThis task was automatically routed from router.py proxy. Opencode, please analyze and implement this task.")
                    
                    print_audit_box("ROUTED TO OPENCODE", f"Task written to {task_file}")
                    
                    msg = f"[Opencode Router] This coding task has been successfully routed to Opencode for execution. Please check the 'opencode_task.md' file in your workspace."
                    
                    async def custom_stream():
                        if path == "api/chat":
                            yield json.dumps({"message": {"role": "assistant", "content": msg}, "done": True}).encode("utf-8") + b"\n"
                        else:
                            yield json.dumps({"response": msg, "done": True}).encode("utf-8") + b"\n"
                    
                    return StreamingResponse(
                        custom_stream(),
                        status_code=200,
                        media_type="application/x-ndjson",
                    )
                
                # 4. Modify Payload
                data["model"] = BASE_MODEL
                data["think"] = should_think
                data["stream"] = True
                
                options = data.get("options", {})
                options["num_ctx"] = THINKING_NUM_CTX if should_think else QUICK_NUM_CTX
                data["options"] = options
                
                body = json.dumps(data).encode("utf-8")
                
                print_audit_box("4. MODIFIED PAYLOAD SENT TO OLLAMA", json.dumps(data, indent=2))
                
            except Exception as e:
                print(f"[JARVIS Audit] Payload modification error: {e}")

        # Execute Request (Stream)
        if request.method == "POST" and path in ["api/chat", "api/generate"]:
            req = client.build_request("POST", url, headers=headers, content=body, params=request.query_params)
            response = await client.send(req, stream=True)

            async def stream_generator():
                stream_start = time.perf_counter()
                first_chunk = True
                chunk_count = 0
                
                try:
                    async for chunk in response.aiter_bytes():
                        if not chunk: continue
                        chunk_count += 1
                        
                        if first_chunk:
                            ttft = time.perf_counter() - stream_start
                            print_audit_box("5. OLLAMA RESPONSE STARTED", f"Time to First Token (TTFT): {ttft:.3f}s")
                            first_chunk = False
                            
                        # Note: We aren't printing the raw chunks here to avoid flooding your terminal,
                        # but we still count them to track stream health.
                            
                        yield chunk
                finally:
                    total_time = time.perf_counter() - stream_start
                    print_audit_box("6. STREAM COMPLETE", f"Total Time: {total_time:.3f}s\nTotal Chunks Received: {chunk_count}")
                    await response.aclose()
                    await client.aclose()

            return StreamingResponse(
                stream_generator(),
                status_code=response.status_code,
                media_type="application/x-ndjson",
            )

        # Non-streaming fallback
        response = await client.request(request.method, url, headers=headers, content=body, params=request.query_params)
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers={k: v for k, v in response.headers.items() if k.lower() not in ["content-length", "transfer-encoding", "content-encoding", "connection"]}
        )

    except Exception as e:
        print(f"[JARVIS Audit] Proxy error: {e}")
        await client.aclose()
        raise
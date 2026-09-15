import re
import time
import io
import wave
import threading
import subprocess
import uvicorn
import sounddevice as sd
import numpy as np
import speech_recognition as sr
from faster_whisper import WhisperModel
from semantic_router import Route, SemanticRouter
from semantic_router.encoders import HuggingFaceEncoder
from piper.voice import PiperVoice
import ollama

# ============================================================
# 1. CONFIGURATION & FUTURE-PROOF MODEL REGISTRY
# ============================================================

# To test another model, just change this string!
ACTIVE_MODEL = "llama3.2:3b"

# Tailored system prompts for voice output
SYSTEM_PROMPTS = {
    "quick": (
        "You are JARVIS, an ambient AI home assistant. "
        "Provide a direct, concise answer in 1 or 2 spoken sentences. "
        "Do not use markdown, lists, bullet points, or greetings."
    ),
    "think": (
        "You are JARVIS, an analytical AI assistant. "
        "For this complex query, talk through your thought process and reasoning "
        "step-by-step in natural, conversational sentences suitable for speech, "
        "then provide your final recommendation. Do not use markdown formatting or code blocks."
    ),
    "code": ""
}

# ============================================================
# 2. INITIALIZE AUDIO & ROUTING ENGINES (CPU)
# ============================================================
print("Loading Whisper STT on CPU...")
stt_model = WhisperModel("base.en", device="cpu", compute_type="int8")

print("Loading Semantic Router on CPU...")
encoder = HuggingFaceEncoder(name="sentence-transformers/all-MiniLM-L6-v2")

quick_route = Route(
    name="jarvis_quick", # Changed to bypass stale cache
    utterances=[
        "what time is it", 
        "who is the president of france",
        "turn off the living room lights",
        "what is the weather like",
        "define a hashmap in one sentence"
    ]
)

thinking_route = Route(
    name="jarvis_think", # Changed to bypass stale cache
    score_threshold=0.45,
    utterances=[
        "design a scalable architecture", 
        "analyze this concept and explain the trade-offs",
        "plan a detailed travel itinerary",
        "explain the logic of backpropagation",
        "compare relational and document databases",
        "walk me through the steps to solve this",
        "break down how this works",
        "what is the deep reasoning behind this"
    ]
)

code_route = Route(
    name="jarvis_code",
    score_threshold=0.45,
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
        "add a feature to the codebase",
        "code a feature",
        "help me write code",
        "write a readme file for my repo",
        "create a new branch and commit the changes",
        "push the newest code to our repository"
    ]
)

router = SemanticRouter(encoder=encoder, routes=[quick_route, thinking_route, code_route], auto_sync="local", aggregation="max")
tts_lock = threading.Lock()

def run_opencode_with_diff(prompt: str):
    """Launches OpenCode to implement the coding task."""
    print("[OpenCode] Running coding task...")
    proc = subprocess.Popen(
        ["opencode", "--prompt", prompt, "--auto"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    proc.wait()
    print("[OpenCode] OpenCode finished executing.")
    speak("Coding task completed. Review git diffs on the dashboard.")

print("Loading Piper TTS Voice into Memory...")
piper_voice = PiperVoice.load("en_US-lessac-medium.onnx")

# ============================================================
# 3. HELPERS & ROUTING AUDIT
# ============================================================
def clean_for_speech(text: str) -> str:
    """Removes markdown symbols like **, *, #, and ` so Piper speaks cleanly."""
    text = re.sub(r'[*#_`~>]', '', text)
    return text.strip()

def get_raw_audit_scores(clean_text: str) -> dict:
    """Manually calculates the exact cosine similarity BEFORE routing occurs."""
    try:
        # 1. Convert the incoming text into a mathematical vector
        query_vec = np.array(encoder([clean_text])[0])
        scores = {}
        
        # 2. Calculate the math against our three routes
        for r in [quick_route, thinking_route, code_route]:
            route_vecs = np.array(encoder(r.utterances))
            
            # Cosine similarity formula
            norms = np.linalg.norm(route_vecs, axis=1) * np.linalg.norm(query_vec)
            sims = np.dot(route_vecs, query_vec) / norms
            
            # Grab the highest scoring utterance in the route
            scores[r.name] = float(np.max(sims))
            
        return scores
    except Exception as e:
        print(f"│  * (Audit Math Error: {e})")
        return {}

def determine_intent(text: str):
    """
    Evaluates semantic intent, prints a detailed math audit box, 
    and returns (is_thinking_task, mode_key).
    """
    # 1. Normalize text (remove punctuation, lowercase) to maximize match accuracy
    clean_text = re.sub(r'[^\w\s]', '', text).lower()
    
    # 2. Print the Audit Box Header
    print("\n┌── [SEMANTIC ROUTER AUDIT] " + "─" * 30)
    print(f"│ Spoken:  '{text}'")
    print(f"│ Cleaned: '{clean_text}'")
    print(f"├─ [Pre-Routing Calculated Scores]")
    
    # 3. Fetch raw scores manually BEFORE the library applies thresholds
    raw_scores = get_raw_audit_scores(clean_text)
    if raw_scores:
        for r_name, score in raw_scores.items():
            print(f"│  * Route '{r_name}': {score:.4f}")
    else:
        print(f"│  * (Could not compute raw scores)")
            
    print(f"├─ [Final Decision]")
    
    # 4. Now send it to the router library for the final decision
    route = router(clean_text)
    
    # 5. Safely handle the route object (it will be None if below threshold)
    matched_name = getattr(route, 'name', None)
    
    if matched_name:
        print(f"│  Selected Route: {matched_name}")
    else:
        print(f"│  Selected Route: NONE (Defaulting to QUICK)")
        
    print("└" + "─" * 58 + "\n")
    
    if matched_name == "jarvis_think":
        return True, "think"
    if matched_name == "jarvis_code":
        return False, "code"
    return False, "quick"

def listen_for_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\nListening...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.listen(source)
        
        with open("temp.wav", "wb") as f:
            f.write(audio.get_wav_data())
        return "temp.wav"

def speak(text):
    """In-memory TTS playback: synthesizes to RAM and writes directly to sound card."""
    cleaned = clean_for_speech(text)
    if not cleaned:
        return
        
    with tts_lock:
        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wav_file:
            piper_voice.synthesize_wav(cleaned, wav_file)
            
        wav_io.seek(0)
        with wave.open(wav_io, 'rb') as wav_file:
            raw_audio = wav_file.readframes(wav_file.getnframes())
            int_data = np.frombuffer(raw_audio, dtype=np.int16)
            
        stream = sd.OutputStream(
            samplerate=piper_voice.config.sample_rate, 
            channels=1, 
            dtype='int16'
        )
        stream.start()
        stream.write(int_data)
        stream.stop()
        stream.close()

# ============================================================
# 4. THE MAIN PIPELINE
# ============================================================
def main_loop():
    speak(f"JARVIS online. Active model is {ACTIVE_MODEL}.")
    
    # Start Code Review UI server in background
    def start_ui():
        print("[JARVIS] Starting Code Review UI server on http://127.0.0.1:8000")
        uvicorn.run("ui_server:app", host="127.0.0.1", port=8000, log_level="error")
    
    threading.Thread(target=start_ui, daemon=True).start()
    
    while True:
        try:
            audio_file = listen_for_command()
            t0 = time.perf_counter()
            
            # STT
            segments, _ = stt_model.transcribe(audio_file, beam_size=5)
            transcript = "".join([segment.text for segment in segments]).strip()
            
            if not transcript:
                continue
                
            print(f"\nYou: {transcript}")

            # Intent Classification
            is_thinking_task, mode_key = determine_intent(transcript)
            print(f"[Router: {mode_key.upper()} | Model: {ACTIVE_MODEL}]")

            # Dynamic System Prompt Selection
            messages = [
                {"role": "system", "content": SYSTEM_PROMPTS[mode_key]},
                {"role": "user", "content": transcript}
            ]

            print("JARVIS: ", end="", flush=True)
            
            if mode_key == "code":
                threading.Thread(target=run_opencode_with_diff, args=(transcript,)).start()
                continue
            
            # Standard Ollama Streaming
            t_req = time.perf_counter()
            stream = ollama.chat(
                model=ACTIVE_MODEL,
                messages=messages,
                stream=True
            )
            
            sentence_buffer = ""
            first_token = True
            
            for chunk in stream:
                content = chunk.get('message', {}).get('content', '')
                if content:
                    if first_token:
                        ttft = time.perf_counter() - t_req
                        first_token = False

                    print(content, end='', flush=True)
                    sentence_buffer += content
                    
                    # Sentence boundary trigger for instant speech streaming
                    if any(punc in content for punc in ['.', '!', '?', '\n']):
                        threading.Thread(target=speak, args=(sentence_buffer,)).start()
                        sentence_buffer = ""
            
            # Handle final trailing sentence
            if sentence_buffer.strip():
                threading.Thread(target=speak, args=(sentence_buffer,)).start()
                
            print()
            
        except KeyboardInterrupt:
            print("\nShutting down JARVIS.")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main_loop()
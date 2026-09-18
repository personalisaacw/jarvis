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

from adapters.vector_store import FaissAdapter
from adapters.embeddings import HuggingFaceAdapter
from use_cases.routing import RouteCommandUseCase, LearnFromFeedbackUseCase
from domain.entities import Intent

print("Loading New Router Architecture...")
vector_store = FaissAdapter()
embedding_engine = HuggingFaceAdapter()
route_use_case = RouteCommandUseCase(vector_store, embedding_engine, fallback_threshold=0.30)
feedback_use_case = LearnFromFeedbackUseCase(vector_store, embedding_engine)

tts_lock = threading.Lock()

def run_agy_with_diff(prompt: str):
    """Launches Antigravity CLI, waits for it to finish, calculates diffs, and saves them."""
    import diff_engine
    print("[Antigravity] Starting coding task and backing up files...")
    diff_engine.backup_files()
    
    import shutil
    agy_path = shutil.which("agy")
    if not agy_path:
        print("[Error] agy CLI not found in PATH.")
        return
        
    print("[Antigravity] Running Antigravity CLI...")
    proc = subprocess.Popen(
        [agy_path, "--prompt", prompt, "--dangerously-skip-permissions"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    proc.wait()  # Wait for Antigravity to finish
    print("[Antigravity] Antigravity finished executing.")
    
    diffs = diff_engine.compute_diffs()
    diff_engine.save_diffs(diffs)
    
    print(f"[Antigravity] Done. {len(diffs)} file(s) changed.")
    speak(f"Coding task completed. You can review the changes on your code review dashboard.")

print("Loading Piper TTS Voice into Memory...")
piper_voice = PiperVoice.load("en_US-lessac-medium.onnx")

# ============================================================
# 3. HELPERS & ROUTING AUDIT
# ============================================================
def clean_for_speech(text: str) -> str:
    """Removes markdown symbols like **, *, #, and ` so Piper speaks cleanly."""
    text = re.sub(r'[*#_`~>]', '', text)
    return text.strip()

def determine_intent(text: str):
    """
    Evaluates semantic intent using the new Vector DB UseCase.
    """
    clean_text = re.sub(r'[^\w\s]', '', text).lower()
    
    print("\n┌── [VECTOR DB ROUTER AUDIT] " + "─" * 30)
    print(f"│ Spoken:  '{text}'")
    print(f"│ Cleaned: '{clean_text}'")
    
    result = route_use_case.execute(clean_text)
    
    print(f"├─ [Final Decision]")
    print(f"│  Selected Route: {result.intent.value.upper()}")
    print(f"│  Confidence: {result.confidence_score:.4f}")
    print(f"│  Clarification Needed: {result.requires_clarification}")
    print("└" + "─" * 58 + "\n")
    
    return result


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
            route_result = determine_intent(transcript)
            mode_key = route_result.intent.value
            
            if route_result.requires_clarification:
                speak("I'm not completely sure. Should I write code for this, or just think about it?")
                print("\n[Waiting for clarification...]")
                clarification_audio = listen_for_command()
                cl_segments, _ = stt_model.transcribe(clarification_audio, beam_size=5)
                clarification = "".join([s.text for s in cl_segments]).strip().lower()
                
                if "code" in clarification:
                    mode_key = "code"
                elif "think" in clarification:
                    mode_key = "think"
                else:
                    mode_key = "quick"
                
                print(f"[Learned new mapping: {mode_key.upper()}]")
                feedback_use_case.execute(transcript, mode_key)
                
            print(f"[Router: {mode_key.upper()} | Model: {ACTIVE_MODEL}]")

            # Dynamic System Prompt Selection
            messages = [
                {"role": "system", "content": SYSTEM_PROMPTS[mode_key]},
                {"role": "user", "content": transcript}
            ]

            print("JARVIS: ", end="", flush=True)
            
            if mode_key == "code":
                threading.Thread(target=run_agy_with_diff, args=(transcript,)).start()
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
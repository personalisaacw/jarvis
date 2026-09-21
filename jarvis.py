import os
from dotenv import load_dotenv

load_dotenv()

import shutil
import re
import time
import io
import wave
import queue
import collections
import threading
import subprocess
import uvicorn
import sounddevice as sd
import numpy as np
import onnxruntime as ort
import speech_recognition as sr
from faster_whisper import WhisperModel
from semantic_router import Route, SemanticRouter
from semantic_router.encoders import HuggingFaceEncoder
from piper.voice import PiperVoice
import ollama
from typing import Optional
from agent_adapter import AgentManager, GroqTerminalParser, RegexFallbackParser

# ============================================================
# SILERO VAD INITIALIZATION (CPU - ONNX)
# ============================================================
print("Loading Silero VAD on CPU...")
vad_options = ort.SessionOptions()
vad_options.inter_op_num_threads = 1
vad_options.intra_op_num_threads = 1

vad_session = ort.InferenceSession(
    "silero_vad.onnx", 
    sess_options=vad_options, 
    providers=["CPUExecutionProvider"]
)

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

# coding vocabulary
CODING_VOCAB_PROMPT = (
    "A software development session discussing README, PyTorch, "
    "FastAPI, GitHub repo, Docker, Kubernetes, JSON, YAML, git diff, refactor."
)

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

ANTIGRAVITY_CODING_MODEL = "gemini-3.1-pro-high"
agent_manager = None

def run_antigravity_coding_task(prompt: str):
    """Launches Antigravity CLI via AgentManager."""
    print(f"\n[Antigravity CLI] Running coding task via AgentManager...")
    print(f"[Antigravity CLI] Prompt: '{prompt}'")
    threading.Thread(target=speak, args=("I am on it. Directing the coding task to Antigravity.",), daemon=True).start()
    project_dir = os.path.dirname(os.path.abspath(__file__))
    if agent_manager:
        agent_manager.start_session("antigravity", prompt=prompt, cwd=project_dir)

# Maintain backwards compatibility
run_opencode_with_diff = run_antigravity_coding_task


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


speaking_condition = threading.Condition()
speaking_threads_count = 0

def get_preferred_microphone():
    """Finds Yeti or AirPods microphone, otherwise falls back to default."""
    try:
        devices = sd.query_devices()
        yeti_id = None
        airpods_id = None
        
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                name = dev['name'].lower()
                if 'yeti' in name:
                    yeti_id = i
                    break  # Highest priority
                if 'isaac' in name and 'airpod' in name and airpods_id is None:
                    airpods_id = i
                    
        if yeti_id is not None:
            print(f"[Audio] Selected Yeti microphone (Device {yeti_id})")
            return yeti_id
        if airpods_id is not None:
            print(f"[Audio] Selected AirPods microphone (Device {airpods_id})")
            return airpods_id
    except Exception as e:
        print(f"[Audio] Error finding preferred mic: {e}")
        
    print("[Audio] Using default system microphone")
    return None

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import audioop

import msvcrt

def listen_for_command(allow_keyboard: bool = False):
    """
    Listens using Silero VAD running locally on ONNX Runtime.
    Tolerates cognitive pauses and outputs 16kHz WAV for Whisper.
    Captures at native mic sample rate and resamples in real-time to avoid driver distortion.
    Uses a 64-sample rolling context buffer required by Silero VAD v5.
    """
    with speaking_condition:
        while speaking_threads_count > 0:
            speaking_condition.wait()
            
    device_id = get_preferred_microphone()
    
    # Get native sample rate for the selected device
    if device_id is not None:
        device_info = sd.query_devices(device_id, 'input')
        native_sr = int(device_info['default_samplerate'])
    else:
        native_sr = int(sd.query_devices(sd.default.device[0], 'input')['default_samplerate'])
        
    TARGET_SR = 16000
    CHUNK_DURATION = 0.032  # 32ms
    NATIVE_CHUNK = int(native_sr * CHUNK_DURATION)
    CONTEXT_SIZE = 64  # Rolling context window required by Silero VAD v5
    
    SPEECH_PROB_THRESHOLD = 0.5
    PAUSE_TOLERANCE_SEC = 1.6  # Silence duration to conclude speaker is finished
    PRE_BUFFER_CHUNKS = 12     # Preserves ~0.38s of audio before voice triggers

    # Silero recurrent LSTM hidden states
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros((CONTEXT_SIZE,), dtype=np.float32)
    sample_rate_tensor = np.array([TARGET_SR], dtype=np.int64)

    audio_q = queue.Queue()

    def mic_callback(indata, frames, time_info, status):
        audio_q.put(indata.copy())

    print(f"\nListening on Device {device_id} at {native_sr}Hz (Silero VAD)...")
    if allow_keyboard:
        print("[Press 1 for Quick, 2 for Think, 3 for Code]")
    
    pre_speech_ring = collections.deque(maxlen=PRE_BUFFER_CHUNKS)
    voiced_chunks = []
    is_speaking = False
    silence_start_time = None
    state_ratecv = None
    
    # Speed A: Streaming STT overlapping
    live_transcript = ""
    last_transcribe_len = 0
    stt_lock = threading.Lock()
    
    def live_transcribe_worker():
        nonlocal live_transcript, last_transcribe_len
        while True:
            time.sleep(1.0)
            with stt_lock:
                if not is_speaking and silence_start_time and (time.perf_counter() - silence_start_time) > PAUSE_TOLERANCE_SEC:
                    break
                current_len = len(voiced_chunks)
            
            if current_len > last_transcribe_len and current_len > 10:
                # Take snapshot and transcribe
                with stt_lock:
                    snapshot = list(voiced_chunks)
                audio_np = np.concatenate(snapshot, axis=0).astype(np.float32) / 32768.0
                try:
                    segments, _ = stt_model.transcribe(audio_np, beam_size=1)
                    with stt_lock:
                        live_transcript = "".join([s.text for s in segments]).strip()
                        last_transcribe_len = current_len
                except Exception:
                    pass

    transcribe_thread = threading.Thread(target=live_transcribe_worker, daemon=True)

    with sd.InputStream(device=device_id, samplerate=native_sr, channels=1, dtype='int16', 
                        blocksize=NATIVE_CHUNK, callback=mic_callback):
        while True:
            chunk_int16 = audio_q.get()
            
            if allow_keyboard and msvcrt.kbhit():
                char = msvcrt.getch().decode('utf-8', errors='ignore')
                if char in ['1', '2', '3']:
                    return "keyboard", char
            
            if tts_lock.locked():
                # Prevent the assistant from hearing its own TTS output
                pre_speech_ring.clear()
                continue

            # Resample to 16kHz for VAD and Whisper
            if native_sr != TARGET_SR:
                chunk_bytes, state_ratecv = audioop.ratecv(chunk_int16.tobytes(), 2, 1, native_sr, TARGET_SR, state_ratecv)
                chunk_16k = np.frombuffer(chunk_bytes, dtype=np.int16)
            else:
                chunk_16k = chunk_int16.flatten()
                
            chunk_f32 = chunk_16k.astype(np.float32) / 32768.0

            # Prepend 64-sample rolling context (required by Silero VAD v5)
            input_with_context = np.concatenate([context, chunk_f32])
            context = chunk_f32[-CONTEXT_SIZE:]
            
            chunk_flat = input_with_context.reshape(1, -1)

            # Silero VAD forward pass (< 1 ms on CPU)
            ort_inputs = {
                'input': chunk_flat,
                'state': state,
                'sr': sample_rate_tensor
            }
            ort_outs = vad_session.run(None, ort_inputs)
            speech_prob = ort_outs[0][0][0]
            state = ort_outs[1]

            if speech_prob >= SPEECH_PROB_THRESHOLD:
                with stt_lock:
                    if not is_speaking:
                        is_speaking = True
                        if not transcribe_thread.is_alive():
                            transcribe_thread.start()
                        # Prepend buffered audio so first syllable is never lost
                        voiced_chunks.extend(pre_speech_ring)
                        pre_speech_ring.clear()
                    
                    voiced_chunks.append(chunk_16k)
                    silence_start_time = None  # Reset silence timer while talking
            else:
                with stt_lock:
                    if is_speaking:
                        voiced_chunks.append(chunk_16k)
                        if silence_start_time is None:
                            silence_start_time = time.perf_counter()
                        elif time.perf_counter() - silence_start_time >= PAUSE_TOLERANCE_SEC:
                            # User took a full 1.6s pause after speaking: turn is complete
                            break
                    else:
                        pre_speech_ring.append(chunk_16k)

    # Convert collected 16-bit PCM chunks and export to temp.wav
    with stt_lock:
        all_audio = np.concatenate(voiced_chunks, axis=0)

    with wave.open("temp.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SR)
        wf.writeframes(all_audio.tobytes())

    return "temp.wav", live_transcript

def speak(text):
    """In-memory TTS playback: synthesizes to RAM and writes directly to sound card."""
    cleaned = clean_for_speech(text)
    if not cleaned:
        return
        
    global speaking_threads_count
    with speaking_condition:
        speaking_threads_count += 1
        
    try:
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
    finally:
        with speaking_condition:
            speaking_threads_count -= 1
            if speaking_threads_count == 0:
                speaking_condition.notify_all()

def main_loop():
    global agent_manager
    agent_manager = AgentManager(on_speech=speak)
    parser_type = "GroqCloud LPU" if isinstance(agent_manager.parser, GroqTerminalParser) else "Regex Fallback"
    print(f"[Agent Adapter] Active Parser: {parser_type}")
    speak(f"JARVIS online. Active model is {ACTIVE_MODEL}.")
    
    # Start Code Review Gateway server in background (REST + WebSocket for mobile)
    def start_gateway():
        from fastapi import FastAPI
        from verifier.adapters.gateway.api_router import create_review_router, create_mobile_app_router

        gateway_app = FastAPI(title="JARVIS Code Review Gateway")

        # The coordinator is created per-session inside AgentManager,
        # but we need a reference for the API router. We use a lazy proxy.
        class CoordinatorProxy:
            @property
            def active_session(self):
                if agent_manager and agent_manager.review_coordinator:
                    return agent_manager.review_coordinator.active_session
                return None
            def accept_hunk(self, hunk_id):
                if agent_manager and agent_manager.review_coordinator:
                    agent_manager.review_coordinator.accept_hunk(hunk_id)
            def reject_hunk(self, hunk_id):
                if agent_manager and agent_manager.review_coordinator:
                    agent_manager.review_coordinator.reject_hunk(hunk_id)
            def accept_all(self):
                if agent_manager and agent_manager.review_coordinator:
                    agent_manager.review_coordinator.accept_all()
            def reject_all(self):
                if agent_manager and agent_manager.review_coordinator:
                    agent_manager.review_coordinator.reject_all()
            def skip_hunk(self):
                if agent_manager and agent_manager.review_coordinator:
                    agent_manager.review_coordinator.skip_hunk()
            def explain_hunk(self, hunk_id):
                if agent_manager and agent_manager.review_coordinator:
                    agent_manager.review_coordinator.explain_hunk(hunk_id)

        proxy = CoordinatorProxy()
        review_router = create_review_router(proxy, agent_manager.mobile_presenter)
        mobile_router = create_mobile_app_router()
        gateway_app.include_router(review_router)
        gateway_app.include_router(mobile_router)

        print("[JARVIS] Starting Code Review Gateway on http://127.0.0.1:8000")
        uvicorn.run(gateway_app, host="127.0.0.1", port=8000, log_level="error")
    
    threading.Thread(target=start_gateway, daemon=True).start()
    
    while True:
        try:
            audio_file, live_transcript = listen_for_command()
            t0 = time.perf_counter()
            
            # Optimistic Acknowledgment (UX A)
            def play_chime():
                try:
                    fs = 44100
                    duration = 0.15
                    t = np.linspace(0, duration, int(fs * duration), False)
                    note = np.sin(880.0 * t * 2 * np.pi) * np.exp(-5 * t) * 0.1
                    sd.play((note * 32767).astype(np.int16), samplerate=fs, blocking=False)
                except:
                    pass
            threading.Thread(target=play_chime, daemon=True).start()
            
            # STT - Use live transcript if available to save time
            if live_transcript.strip():
                transcript = live_transcript.strip()
            else:
                segments, _ = stt_model.transcribe(audio_file, beam_size=5)
                transcript = "".join([segment.text for segment in segments]).strip()
            
            if not transcript:
                continue
                
            print(f"\nYou: {transcript}")

            # 1. Route voice to diff review if a review session is active
            if agent_manager and agent_manager.is_reviewing:
                print(f"[Diff Review] Routing voice to review grammar: '{transcript}'")
                agent_manager.send_input(transcript)
                continue

            # 2. Route voice input directly to active agent session if running
            if agent_manager and agent_manager.has_active_session():
                active_cli = agent_manager.active_session.cli_name
                print(f"[Active Agent Session: {active_cli}] Piping voice input to stdin: '{transcript}'")
                agent_manager.send_input(transcript)
                continue

            # 3. Manual Route Confirmation (Training Mode)
            # determine_intent is called just to log the current vector DB state
            determine_intent(transcript)
            
            speak("Route to quick, think, or code?")
            print("\n[Waiting for route confirmation (Say route name or press 1, 2, or 3)...]")
            
            audio_or_kb, live_transcript = listen_for_command(allow_keyboard=True)
            
            mode_key = None
            if audio_or_kb == "keyboard":
                if live_transcript == '1':
                    mode_key = "quick"
                elif live_transcript == '2':
                    mode_key = "think"
                elif live_transcript == '3':
                    mode_key = "code"
            else:
                if live_transcript.strip():
                    clarification = live_transcript.strip().lower()
                else:
                    cl_segments, _ = stt_model.transcribe(audio_or_kb, beam_size=5)
                    clarification = "".join([s.text for s in cl_segments]).strip().lower()
                
                if "code" in clarification or "3" in clarification:
                    mode_key = "code"
                elif "think" in clarification or "2" in clarification:
                    mode_key = "think"
                else:
                    mode_key = "quick"
            
            if mode_key:
                print(f"[Learned new mapping: {mode_key.upper()}]")
                feedback_use_case.execute(transcript, mode_key)
            else:
                mode_key = "quick"
            print(f"[Router: {mode_key.upper()} | Model: {ACTIVE_MODEL}]")

            if mode_key == "code":
                run_antigravity_coding_task(transcript)
                continue

            # Dynamic System Prompt Selection
            messages = [
                {"role": "system", "content": SYSTEM_PROMPTS[mode_key]},
                {"role": "user", "content": transcript}
            ]

            print("JARVIS: ", end="", flush=True)

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
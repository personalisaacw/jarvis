import os
import pytest
import numpy as np
import onnxruntime as ort

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "silero_vad.onnx")

@pytest.fixture(scope="session")
def vad_session():
    """Provides a shared ONNX InferenceSession for Silero VAD."""
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Silero VAD ONNX model file not found at {MODEL_PATH}")
    
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    return ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=["CPUExecutionProvider"])

@pytest.fixture
def initial_vad_state():
    """Returns a fresh zeroed state tensor required by Silero VAD LSTM [2, 1, 128]."""
    return np.zeros((2, 1, 128), dtype=np.float32)

@pytest.fixture
def initial_context():
    """Returns a fresh 64-sample rolling context window required by Silero VAD v5."""
    return np.zeros(64, dtype=np.float32)

@pytest.fixture
def sample_rate_tensor():
    """Returns 16000Hz sample rate tensor expected by ONNX model."""
    return np.array(16000, dtype=np.int64)

@pytest.fixture
def synthetic_speech_chunk():
    """Generates a 512-sample synthetic audio chunk with a 440Hz sine wave."""
    t = np.linspace(0, 0.032, 512, endpoint=False)
    sine = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5
    return sine

@pytest.fixture
def synthetic_silence_chunk():
    """Generates a 512-sample silent audio chunk."""
    return np.zeros(512, dtype=np.float32)

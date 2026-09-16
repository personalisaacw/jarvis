import numpy as np
import pytest
import warnings
from unittest.mock import patch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from jarvis import get_preferred_microphone

def test_get_preferred_microphone_yeti_priority():
    """Verify that Yeti microphone takes highest priority when available."""
    mock_devices = [
        {'name': 'Microsoft Sound Mapper - Input', 'max_input_channels': 2},
        {'name': 'Headset (Isaac\'s Airpods - Find My)', 'max_input_channels': 1},
        {'name': 'Microphone (Yeti Classic)', 'max_input_channels': 1}
    ]

    with patch('sounddevice.query_devices', return_value=mock_devices):
        dev_id = get_preferred_microphone()
        assert dev_id == 2  # Index of Yeti Classic

def test_get_preferred_microphone_airpods_fallback():
    """Verify that AirPods microphone is selected when Yeti is not plugged in."""
    mock_devices = [
        {'name': 'Microsoft Sound Mapper - Input', 'max_input_channels': 2},
        {'name': 'Réseau de microphones (Intel Smart Sound)', 'max_input_channels': 2},
        {'name': 'Headset (Isaac\'s Airpods - Find My)', 'max_input_channels': 1}
    ]

    with patch('sounddevice.query_devices', return_value=mock_devices):
        dev_id = get_preferred_microphone()
        assert dev_id == 2  # Index of AirPods

def test_get_preferred_microphone_default_fallback():
    """Verify fallback to default system microphone when neither Yeti nor AirPods present."""
    mock_devices = [
        {'name': 'Microsoft Sound Mapper - Input', 'max_input_channels': 2},
        {'name': 'Internal Microphone', 'max_input_channels': 1}
    ]

    with patch('sounddevice.query_devices', return_value=mock_devices):
        dev_id = get_preferred_microphone()
        assert dev_id is None  # None triggers default mic fallback

def test_audioop_resampling_44k_to_16k():
    """Verify real-time audioop.ratecv resampling from 44.1kHz to 16kHz."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import audioop

    native_sr = 44100
    target_sr = 16000
    chunk_duration = 0.032  # 32ms
    native_chunk_size = int(native_sr * chunk_duration)  # 1411

    # Generate synthetic 44.1kHz int16 audio
    t = np.linspace(0, chunk_duration, native_chunk_size, endpoint=False)
    audio_44k = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)

    chunk_bytes, state_ratecv = audioop.ratecv(audio_44k.tobytes(), 2, 1, native_sr, target_sr, None)
    audio_16k = np.frombuffer(chunk_bytes, dtype=np.int16)

    # 32ms at 16kHz is 512 samples
    assert len(audio_16k) == 512

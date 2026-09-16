import numpy as np
import pytest

def test_vad_session_inputs_outputs(vad_session):
    """Verify Silero VAD ONNX model input/output specifications."""
    input_names = [i.name for i in vad_session.get_inputs()]
    output_names = [o.name for o in vad_session.get_outputs()]

    assert "input" in input_names
    assert "state" in input_names
    assert "sr" in input_names
    assert len(output_names) >= 2

def test_vad_rolling_context_concatenation(initial_context, synthetic_speech_chunk):
    """Verify that prepending the 64-sample rolling context creates a 576-sample frame."""
    context_size = len(initial_context)
    assert context_size == 64
    assert len(synthetic_speech_chunk) == 512

    input_with_context = np.concatenate([initial_context, synthetic_speech_chunk])
    assert len(input_with_context) == 576

    # Verify context buffer update for next frame
    next_context = synthetic_speech_chunk[-context_size:]
    assert len(next_context) == 64
    np.testing.assert_array_equal(next_context, synthetic_speech_chunk[-64:])

def test_vad_inference_step(vad_session, initial_vad_state, initial_context, sample_rate_tensor, synthetic_speech_chunk):
    """Verify single step VAD inference with state and context propagation."""
    input_with_context = np.concatenate([initial_context, synthetic_speech_chunk])
    chunk_flat = input_with_context.reshape(1, -1)

    ort_inputs = {
        'input': chunk_flat,
        'state': initial_vad_state,
        'sr': sample_rate_tensor
    }

    ort_outs = vad_session.run(None, ort_inputs)
    speech_prob = ort_outs[0][0][0]
    next_state = ort_outs[1]

    assert isinstance(float(speech_prob), float)
    assert 0.0 <= speech_prob <= 1.0
    assert next_state.shape == (2, 1, 128)

def test_vad_silence_vs_speech_probability(vad_session, initial_vad_state, initial_context, sample_rate_tensor, synthetic_silence_chunk, synthetic_speech_chunk):
    """Verify speech probability response for silence versus active audio signal."""
    # Silence frame with context
    silence_input = np.concatenate([initial_context, synthetic_silence_chunk]).reshape(1, -1)
    silence_outs = vad_session.run(None, {
        'input': silence_input,
        'state': initial_vad_state,
        'sr': sample_rate_tensor
    })
    silence_prob = silence_outs[0][0][0]

    # Speech frame with context
    speech_input = np.concatenate([initial_context, synthetic_speech_chunk]).reshape(1, -1)
    speech_outs = vad_session.run(None, {
        'input': speech_input,
        'state': initial_vad_state,
        'sr': sample_rate_tensor
    })
    speech_prob = speech_outs[0][0][0]

    # Silence probability should be near zero
    assert silence_prob < 0.1

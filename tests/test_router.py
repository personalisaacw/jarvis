import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from jarvis import determine_intent

def test_determine_intent_think_mode():
    """Verify that deep thinking queries route to think mode."""
    is_thinking, mode = determine_intent("design a scalable architecture for microservices")
    assert is_thinking is True
    assert mode == "think"

def test_determine_intent_code_mode():
    """Verify that coding queries route to code mode."""
    is_thinking, mode = determine_intent("write a python script to list files")
    assert is_thinking is False
    assert mode == "code"

def test_determine_intent_quick_default():
    """Verify that general queries default to quick mode."""
    is_thinking, mode = determine_intent("what time is it in Tokyo")
    assert is_thinking is False
    assert mode == "quick"

"""Tests for ReviewVoiceGrammar in verifier.adapters.input.voice_grammar."""

import pytest
from unittest.mock import MagicMock
from verifier.adapters.input.voice_grammar import ReviewVoiceGrammar
from verifier.ports.inbound import IReviewCommandPort


@pytest.fixture
def mock_coordinator():
    coordinator = MagicMock(spec=IReviewCommandPort)
    session = MagicMock()
    hunk = MagicMock()
    hunk.hunk_id = "hunk_test_42"
    session.current_hunk = hunk
    coordinator.active_session = session
    return coordinator


@pytest.fixture
def grammar(mock_coordinator):
    return ReviewVoiceGrammar(mock_coordinator)


def test_classify_accept_patterns(grammar):
    for phrase in [
        "accept",
        "yes",
        "looks good",
        "keep this",
        "approve",
        "keep it",
        "good",
        "lgtm",
    ]:
        assert grammar.classify(phrase) == "accept"
        assert grammar.classify(phrase.upper()) == "accept"


def test_classify_reject_patterns(grammar):
    for phrase in [
        "reject",
        "no",
        "discard",
        "undo",
        "revert",
        "remove",
        "delete this",
        "bad",
    ]:
        assert grammar.classify(phrase) == "reject"
        assert grammar.classify(phrase.upper()) == "reject"


def test_classify_accept_all_patterns(grammar):
    for phrase in [
        "accept all",
        "approve everything",
        "keep all",
        "accept everything",
        "approve all",
    ]:
        assert grammar.classify(phrase) == "accept_all"
        assert grammar.classify(phrase.upper()) == "accept_all"


def test_classify_reject_all_patterns(grammar):
    for phrase in [
        "reject all",
        "discard all",
        "revert everything",
        "reject everything",
        "discard everything",
    ]:
        assert grammar.classify(phrase) == "reject_all"
        assert grammar.classify(phrase.upper()) == "reject_all"


def test_classify_explain_patterns(grammar):
    for phrase in [
        "explain this",
        "why did you change this",
        "what does this do",
        "explain",
        "why",
    ]:
        assert grammar.classify(phrase) == "explain"
        assert grammar.classify(phrase.upper()) == "explain"


def test_classify_skip_patterns(grammar):
    for phrase in [
        "next",
        "skip",
        "pass",
        "move on",
    ]:
        assert grammar.classify(phrase) == "skip"
        assert grammar.classify(phrase.upper()) == "skip"


def test_classify_back_patterns(grammar):
    for phrase in [
        "previous",
        "back",
        "go back",
    ]:
        assert grammar.classify(phrase) == "back"
        assert grammar.classify(phrase.upper()) == "back"


def test_classify_precedence(grammar):
    # ACCEPT_ALL before ACCEPT
    assert grammar.classify("accept all") == "accept_all"
    assert grammar.classify("please accept all changes") == "accept_all"
    assert grammar.classify("accept") == "accept"
    assert grammar.classify("please accept") == "accept"

    # REJECT_ALL before REJECT
    assert grammar.classify("reject all") == "reject_all"
    assert grammar.classify("revert everything") == "reject_all"
    assert grammar.classify("revert") == "reject"


def test_classify_unrecognized_or_empty(grammar):
    assert grammar.classify("") is None
    assert grammar.classify("   ") is None
    assert grammar.classify(None) is None
    assert grammar.classify("tell me a story") is None
    assert grammar.classify("nobody") is None
    assert grammar.classify("compass") is None
    assert grammar.classify("feedback") is None
    assert grammar.classify("unacceptable") is None


def test_handle_voice_command_accept(grammar, mock_coordinator):
    grammar.handle_voice_command("looks good")
    mock_coordinator.accept_hunk.assert_called_once_with("hunk_test_42")


def test_handle_voice_command_reject(grammar, mock_coordinator):
    grammar.handle_voice_command("delete this")
    mock_coordinator.reject_hunk.assert_called_once_with("hunk_test_42")


def test_handle_voice_command_accept_all(grammar, mock_coordinator):
    grammar.handle_voice_command("accept all")
    mock_coordinator.accept_all.assert_called_once()


def test_handle_voice_command_reject_all(grammar, mock_coordinator):
    grammar.handle_voice_command("revert everything")
    mock_coordinator.reject_all.assert_called_once()


def test_handle_voice_command_explain(grammar, mock_coordinator):
    grammar.handle_voice_command("why did you change this")
    mock_coordinator.explain_hunk.assert_called_once_with("hunk_test_42")


def test_handle_voice_command_skip(grammar, mock_coordinator):
    grammar.handle_voice_command("move on")
    mock_coordinator.skip_hunk.assert_called_once()


def test_handle_voice_command_back_with_previous(grammar, mock_coordinator):
    mock_coordinator.previous_hunk = MagicMock()
    grammar.handle_voice_command("go back")
    mock_coordinator.previous_hunk.assert_called_once()


def test_handle_voice_command_unrecognized(grammar, mock_coordinator):
    grammar.handle_voice_command("what is your name")
    mock_coordinator.accept_hunk.assert_not_called()
    mock_coordinator.reject_hunk.assert_not_called()
    mock_coordinator.accept_all.assert_not_called()
    mock_coordinator.reject_all.assert_not_called()

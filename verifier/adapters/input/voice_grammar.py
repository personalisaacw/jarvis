"""Voice grammar parser for the sequential diff review subsystem.

Classifies spoken voice transcripts during code review into specific review
commands (accept, reject, accept_all, reject_all, explain, skip, back) and
dispatches them to the ReviewSessionCoordinator.
"""

from typing import List, Optional
import re

try:
    from ...ports.inbound import IVoiceInputPort, IReviewCommandPort
except (ImportError, ValueError):
    from verifier.ports.inbound import IVoiceInputPort, IReviewCommandPort


class ReviewVoiceGrammar(IVoiceInputPort):
    """Voice input adapter that translates spoken utterances into review commands.

    Implements IVoiceInputPort to classify incoming voice transcripts against
    established review grammar patterns and execute the corresponding action
    on the review session coordinator.
    """

    ACCEPT: List[str] = [
        "accept",
        "yes",
        "looks good",
        "keep this",
        "approve",
        "keep it",
        "good",
        "lgtm",
    ]

    REJECT: List[str] = [
        "reject",
        "no",
        "discard",
        "undo",
        "revert",
        "remove",
        "delete this",
        "bad",
    ]

    ACCEPT_ALL: List[str] = [
        "accept all",
        "approve everything",
        "keep all",
        "accept everything",
        "approve all",
    ]

    REJECT_ALL: List[str] = [
        "reject all",
        "discard all",
        "revert everything",
        "reject everything",
        "discard everything",
    ]

    EXPLAIN: List[str] = [
        "explain this",
        "why did you change this",
        "what does this do",
        "explain",
        "why",
    ]

    SKIP: List[str] = [
        "next",
        "skip",
        "pass",
        "move on",
    ]

    BACK: List[str] = [
        "previous",
        "back",
        "go back",
    ]

    def __init__(self, coordinator: Optional[IReviewCommandPort] = None) -> None:
        """Initialize the voice grammar parser.

        Args:
            coordinator: Coordinator implementing IReviewCommandPort to receive
                parsed review commands.
        """
        self.coordinator = coordinator

    @classmethod
    def classify(cls, transcript: str) -> Optional[str]:
        """Classify a voice transcript into an action name without executing it.

        Normalization and evaluation precedence:
        1. Checks ACCEPT_ALL and REJECT_ALL first (longer matches before shorter).
        2. Checks EXPLAIN, SKIP, BACK.
        3. Checks ACCEPT and REJECT.

        Args:
            transcript: Spoken voice transcript string.

        Returns:
            Action name ('accept', 'reject', 'accept_all', 'reject_all',
            'explain', 'skip', 'back') or None if unrecognized.
        """
        if not transcript or not isinstance(transcript, str):
            return None

        normalized = transcript.lower().strip()
        if not normalized:
            return None

        # Clean trailing and leading punctuation for exact matching
        clean = re.sub(r"^[^\w]+|[^\w]+$", "", normalized)

        # Precedence order as specified:
        # 1. ACCEPT_ALL, REJECT_ALL
        # 2. EXPLAIN, SKIP, BACK
        # 3. ACCEPT, REJECT
        categories = [
            ("accept_all", cls.ACCEPT_ALL),
            ("reject_all", cls.REJECT_ALL),
            ("explain", cls.EXPLAIN),
            ("skip", cls.SKIP),
            ("back", cls.BACK),
            ("accept", cls.ACCEPT),
            ("reject", cls.REJECT),
        ]

        # First pass: check for exact match against normalized text or punctuation-stripped text
        for action, patterns in categories:
            for pattern in patterns:
                p_norm = pattern.lower().strip()
                if normalized == p_norm or (clean and clean == p_norm):
                    return action

        # Second pass: check word boundary pattern matching in normalized transcript
        # Patterns within each category are sorted by length descending so longer phrases match first
        for action, patterns in categories:
            sorted_patterns = sorted(patterns, key=len, reverse=True)
            for pattern in sorted_patterns:
                p_norm = pattern.lower().strip()
                regex = r"(?:^|\b)" + re.escape(p_norm) + r"(?:\b|$)"
                if re.search(regex, normalized):
                    return action

        return None

    def handle_voice_command(self, transcript: str) -> None:
        """Parse spoken transcript and route it to the corresponding review command.

        Normalizes the transcript, classifies the intended command, and executes
        the matching method on the coordinator. For hunk-level accept and reject,
        retrieves the current hunk ID from the coordinator's active session.

        Args:
            transcript: Spoken voice input transcript.
        """
        action = self.classify(transcript)
        if not action:
            print(f"[VoiceGrammar] Unrecognized voice command: '{transcript}'")
            return

        if not self.coordinator:
            print(f"[VoiceGrammar] Cannot execute '{action}': No coordinator configured.")
            return

        try:
            if action == "accept_all":
                self.coordinator.accept_all()
            elif action == "reject_all":
                self.coordinator.reject_all()
            elif action == "explain":
                session = getattr(self.coordinator, "active_session", None)
                if session and session.current_hunk:
                    self.coordinator.explain_hunk(session.current_hunk.hunk_id)
                elif hasattr(self.coordinator, "explain"):
                    self.coordinator.explain()
                else:
                    print("[VoiceGrammar] Cannot explain hunk: No active hunk or session available.")
            elif action == "skip":
                if hasattr(self.coordinator, "skip_hunk"):
                    self.coordinator.skip_hunk()
                elif hasattr(self.coordinator, "skip"):
                    self.coordinator.skip()
                else:
                    print("[VoiceGrammar] Skip command is not supported by coordinator.")
            elif action == "back":
                if hasattr(self.coordinator, "previous_hunk"):
                    self.coordinator.previous_hunk()
                elif hasattr(self.coordinator, "back_hunk"):
                    self.coordinator.back_hunk()
                elif hasattr(self.coordinator, "go_back"):
                    self.coordinator.go_back()
                elif hasattr(self.coordinator, "back"):
                    self.coordinator.back()
                elif hasattr(self.coordinator, "previous"):
                    self.coordinator.previous()
                else:
                    print("[VoiceGrammar] Back navigation is not supported by coordinator.")
            elif action == "accept":
                session = getattr(self.coordinator, "active_session", None)
                if session and session.current_hunk:
                    self.coordinator.accept_hunk(session.current_hunk.hunk_id)
                else:
                    print("[VoiceGrammar] Cannot accept hunk: No active hunk or session available.")
            elif action == "reject":
                session = getattr(self.coordinator, "active_session", None)
                if session and session.current_hunk:
                    self.coordinator.reject_hunk(session.current_hunk.hunk_id)
                else:
                    print("[VoiceGrammar] Cannot reject hunk: No active hunk or session available.")
        except Exception as e:
            print(f"[VoiceGrammar] Error executing voice command '{action}': {e}")

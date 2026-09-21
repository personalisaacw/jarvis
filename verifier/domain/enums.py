from enum import Enum

class HunkStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SKIPPED = "skipped"

class ChangeType(str, Enum):
    MODIFIED = "modified"
    ADDED = "added"
    DELETED = "deleted"
    RENAMED = "renamed"

class SessionState(str, Enum):
    INITIALIZING = "initializing"
    AGENT_RUNNING = "agent_running"
    REVIEW_ACTIVE = "review_active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class InputModality(str, Enum):
    VOICE = "voice"
    KEYBOARD = "keyboard"
    TOUCH = "touch"
    MOUSE = "mouse"

class DeviceType(str, Enum):
    DESKTOP = "desktop"
    MOBILE_IOS = "mobile_ios"
    MOBILE_OTHER = "mobile_other"

import time
from abc import ABC, abstractmethod


class IClock(ABC):
    @abstractmethod
    def now(self) -> float:
        """Current unix time in seconds."""


class SystemClock(IClock):
    def now(self) -> float:
        return time.time()

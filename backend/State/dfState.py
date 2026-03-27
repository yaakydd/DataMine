# State/dfState.py

from __future__ import annotations
import threading
import time
from typing import Optional
import pandas as pd


class UserSession:
    """
    Holds the DataFrame and metadata for a single user session.
    
    Attributes:
        df:           The user's current working DataFrame (None until upload).
        last_active:  Unix timestamp of the last request — used for TTL eviction.
        user_id:      The owner's ID, stored here for logging/debugging convenience.
    """
    def __init__(self, user_id: str):
        self.df:          Optional[pd.DataFrame] = None
        self.last_active: float                  = time.time()
        self.user_id:     str                    = user_id

    def touch(self) -> None:
        """Refresh the last-active timestamp on every request."""
        self.last_active = time.time()


class MultiUserDatasetState:
    """
    Thread-safe, per-user DataFrame store with automatic TTL eviction.

    Design decisions:
        - One UserSession per user_id, keyed in a plain dict.
        - A threading.Lock() guards all reads and writes so concurrent
          requests from different users don't race on the dict.
        - Sessions older than TTL_SECONDS are evicted on every new
          request (lazy eviction) — no background thread needed.

    TTL_SECONDS = 3600 (1 hour)
        After 1 hour of inactivity the session is dropped and the
        DataFrame is released from memory. Adjust to your needs —
        shorter for large datasets, longer for interactive sessions.
    """

    TTL_SECONDS = 3600  # 1 hour of inactivity before a session expires

    def __init__(self):
        self._sessions: dict[str, UserSession] = {}
        self._lock:     threading.Lock          = threading.Lock()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _evict_expired(self) -> None:
        """
        Removes sessions that have been idle longer than TTL_SECONDS.
        Called internally on every public method — no background thread needed.
        Runs inside the caller's lock, so it's always safe.
        """
        now     = time.time()
        expired = [uid for uid, s in self._sessions.items()
                   if now - s.last_active > self.TTL_SECONDS]
        for uid in expired:
            del self._sessions[uid]

    def _get_or_create(self, user_id: str) -> UserSession:
        """
        Returns the existing session for user_id, or creates a new one.
        Assumes the caller holds self._lock.
        """
        if user_id not in self._sessions:
            self._sessions[user_id] = UserSession(user_id)
        return self._sessions[user_id]

    # ── Public API ────────────────────────────────────────────────────────────

    def get_df(self, user_id: str) -> Optional[pd.DataFrame]:
        """
        Returns the DataFrame for this user, or None if not yet uploaded.
        Refreshes the session's last-active timestamp.
        """
        with self._lock:
            self._evict_expired()
            if user_id not in self._sessions:
                return None
            session = self._sessions[user_id]
            session.touch()
            return session.df

    def set_df(self, user_id: str, df: pd.DataFrame) -> None:
        """
        Stores or replaces the DataFrame for this user.
        Creates a new session automatically if one doesn't exist.
        """
        with self._lock:
            self._evict_expired()
            session    = self._get_or_create(user_id)
            session.df = df
            session.touch()

    def clear_session(self, user_id: str) -> None:
        """
        Explicitly removes a user's session (e.g. on logout or new upload).
        Safe to call even if the session doesn't exist.
        """
        with self._lock:
            self._sessions.pop(user_id, None)

    def session_exists(self, user_id: str) -> bool:
        """Returns True if the user has an active, non-expired session."""
        with self._lock:
            self._evict_expired()
            return user_id in self._sessions

    @property
    def active_session_count(self) -> int:
        """How many sessions are currently alive — useful for monitoring."""
        with self._lock:
            self._evict_expired()
            return len(self._sessions)


# Single instance imported everywhere — same import pattern as before,
# just replace dataset_state.df = x  →  dataset_state.set_df(user_id, x)
dataset_state = MultiUserDatasetState()
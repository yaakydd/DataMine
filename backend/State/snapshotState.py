"""
State/snapshotState.py

Per-user snapshot store. Each user gets their own isolated 
SnapshotStore instance, keyed by user_id.

Usage in any task endpoint:
    from State.snapshotState import snapshot_store

    snapshot_store.save(user_id, "Task 2 — filled age with median", df)
    dataset_state.set_df(user_id, cleaned_df)
"""

from __future__ import annotations
import datetime
import threading
import time
from typing import Optional
import pandas as pd


# ── SnapshotEntry ─────────────────────────────────────────────────────────────
# Unchanged from your original — it's already well-designed.

class SnapshotEntry:
    """
    One saved state of the DataFrame for one user.
    Stores a full .copy() so later mutations don't corrupt it.
    """
    def __init__(self, label: str, df: pd.DataFrame, index: int):
        self.label:     str               = label
        self.df:        pd.DataFrame      = df.copy()
        self.timestamp: datetime.datetime = datetime.datetime.now()
        self.shape:     tuple             = df.shape
        self.index:     int               = index

    def summary(self) -> dict:
        return {
            "index":     self.index,
            "label":     self.label,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "rows":      self.shape[0],
            "cols":      self.shape[1],
        }


# ── UserSnapshotHistory ───────────────────────────────────────────────────────
# This replaces what SnapshotStore used to be — one per user, not global.

class UserSnapshotHistory:
    """
    The snapshot history for a SINGLE user.
    Equivalent to your original SnapshotStore, but scoped to one user.
    
    This is never imported directly — MultiUserSnapshotStore manages 
    one of these per user and exposes them through the public API.
    """
    MAX_SNAPSHOTS = 20

    def __init__(self):
        self._history:    list[SnapshotEntry] = []
        self._next_index: int                 = 0

    def save(self, label: str, df: pd.DataFrame) -> SnapshotEntry:
        entry = SnapshotEntry(label=label, df=df, index=self._next_index)
        self._history.append(entry)
        self._next_index += 1
        if len(self._history) > self.MAX_SNAPSHOTS:
            self._history.pop(0)
        return entry

    def restore(self, index: int) -> Optional[pd.DataFrame]:
        entry = next((e for e in self._history if e.index == index), None)
        if entry is None:
            return None
        position = self._history.index(entry)
        self._history = self._history[:position + 1]
        return entry.df.copy()

    def list_summaries(self) -> list:
        return [e.summary() for e in reversed(self._history)]

    def latest(self) -> Optional[SnapshotEntry]:
        return self._history[-1] if self._history else None

    def clear(self) -> None:
        self._history    = []
        self._next_index = 0

    @property
    def count(self) -> int:
        return len(self._history)


# ── MultiUserSnapshotStore ────────────────────────────────────────────────────

class MultiUserSnapshotStore:
    """
    Thread-safe registry of per-user snapshot histories.

    Every public method accepts user_id as its first argument.
    Internally it routes to that user's UserSnapshotHistory,
    creating one automatically if this is the user's first operation.

    TTL_SECONDS matches dfState so sessions expire together.
    Eviction is lazy — checked on every public call, no background thread.
    """

    TTL_SECONDS = 3600  # must match MultiUserDatasetState.TTL_SECONDS

    def __init__(self):
        # Maps user_id → (UserSnapshotHistory, last_active_timestamp)
        self._stores:     dict[str, tuple[UserSnapshotHistory, float]] = {}
        self._lock:       threading.Lock = threading.Lock()

    # ── Internal helpers ──────────────────────────────────────────────────

    def _evict_expired(self) -> None:
        """Drop histories for users who haven't been active within TTL."""
        now     = time.time()
        expired = [uid for uid, (_, last) in self._stores.items()
                   if now - last > self.TTL_SECONDS]
        for uid in expired:
            del self._stores[uid]

    def _get_or_create(self, user_id: str) -> UserSnapshotHistory:
        """
        Returns the UserSnapshotHistory for user_id.
        Creates and registers a new one if this user has no history yet.
        Refreshes the last-active timestamp.
        Assumes the caller holds self._lock.
        """
        if user_id not in self._stores:
            self._stores[user_id] = (UserSnapshotHistory(), time.time())
        history, _ = self._stores[user_id]
        # Refresh the timestamp
        self._stores[user_id] = (history, time.time())
        return history

    # ── Public API ────────────────────────────────────────────────────────

    def save(self, user_id: str, label: str, df: pd.DataFrame) -> SnapshotEntry:
        """
        Saves a snapshot of df for this user BEFORE a write operation.

        Call pattern in every task endpoint:
            snapshot_store.save(user_id, "Task 1 — renamed columns", df)
            dataset_state.set_df(user_id, cleaned_df)
        """
        with self._lock:
            self._evict_expired()
            return self._get_or_create(user_id).save(label, df)

    def restore(self, user_id: str, index: int) -> Optional[pd.DataFrame]:
        """Restores this user's DataFrame to the snapshot at index."""
        with self._lock:
            self._evict_expired()
            if user_id not in self._stores:
                return None
            return self._get_or_create(user_id).restore(index)

    def list_summaries(self, user_id: str) -> list:
        """Returns this user's snapshot history, newest first."""
        with self._lock:
            self._evict_expired()
            if user_id not in self._stores:
                return []
            return self._get_or_create(user_id).list_summaries()

    def latest(self, user_id: str) -> Optional[SnapshotEntry]:
        """Returns this user's most recent snapshot, or None."""
        with self._lock:
            self._evict_expired()
            if user_id not in self._stores:
                return None
            return self._get_or_create(user_id).latest()

    def clear(self, user_id: str) -> None:
        """
        Clears snapshot history for this user only.
        Called on new file upload — doesn't touch other users' histories.
        """
        with self._lock:
            if user_id in self._stores:
                history, _ = self._stores[user_id]
                history.clear()

    def count(self, user_id: str) -> int:
        """Returns the number of snapshots for this user."""
        with self._lock:
            if user_id not in self._stores:
                return 0
            history, _ = self._stores[user_id]
            return history.count

    @property
    def MAX_SNAPSHOTS(self) -> int:
        return UserSnapshotHistory.MAX_SNAPSHOTS


# Single instance — same import pattern as before
snapshot_store = MultiUserSnapshotStore()
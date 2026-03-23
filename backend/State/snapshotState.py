"""
snapshot/snapshotState.py

The global snapshot store. Keeps a capped history of DataFrame copies
so the user can roll back to any previous state.

Import and use in every task endpoint that writes to dataset_state.df:

    from snapshot.snapshotState import snapshot_store

    # Before writing:
    snapshot_store.save("Task 2 — filled age with median")
    dataset_state.df = df

That single line before every save is all any task needs to do.
The rest is handled here.
"""

from __future__ import annotations

import datetime
from typing import Optional
import pandas as pd


class SnapshotEntry:
    """
    One saved state of the DataFrame.

    Attributes:
        label:      Plain-English description of what is about to change.
                    Written BEFORE the change — so it reads "filled age
                    with median" not "after filling age with median".
        df:         A full copy of the DataFrame at the moment of saving.
        timestamp:  When the snapshot was taken.
        shape:      The shape at the time of saving — shown in the UI
                    without needing to load the full DataFrame.
        index:      Position in the history list (0 = oldest, n = newest).
    """
    def __init__(self, label: str, df: pd.DataFrame, index: int):
        self.label:     str               = label
        self.df:        pd.DataFrame      = df.copy()   # full independent copy
        self.timestamp: datetime.datetime = datetime.datetime.now()
        self.shape:     tuple             = df.shape
        self.index:     int               = index

    def summary(self) -> dict:
        """
        Returns a lightweight summary safe for JSON serialisation.
        Does NOT include the DataFrame itself — only metadata.
        """
        return {
            "index":     self.index,
            "label":     self.label,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "rows":      self.shape[0],
            "cols":      self.shape[1],
        }


class SnapshotStore:
    """
    Holds the history of snapshots in a fixed-size list.

    Design decisions:

    MAX_SNAPSHOTS = 20
        Capped to prevent memory overflow on large datasets.
        When the cap is reached, the oldest snapshot is dropped.
        20 gives the user plenty of undo depth for a normal cleaning
        session without holding 20 copies of a 500MB DataFrame.

    .save() takes a snapshot BEFORE the change.
        This is the correct mental model — "here is what the data
        looked like before this operation". Rolling back restores
        exactly that state.

    .restore(index) restores by snapshot index.
        Snapshots taken AFTER the restored point are discarded —
        the history is linear, not branching.
    """

    MAX_SNAPSHOTS = 20

    def __init__(self):
        self._history:      list[SnapshotEntry] = []
        self._next_index:   int                 = 0

    def save(self, label: str, df: pd.DataFrame) -> SnapshotEntry:
        """
        Saves a snapshot of df before a write operation.

        Call this BEFORE setting dataset_state.df = df.

        Args:
            label:  Human-readable description of the operation about
                    to be performed. e.g. "Task 2 — filled age with median"
            df:     The CURRENT DataFrame — the state before the change.

        Returns:
            The SnapshotEntry that was saved.
        """
        entry = SnapshotEntry(label=label, df=df, index=self._next_index)
        self._history.append(entry)
        self._next_index += 1

        # Drop oldest if over the cap
        if len(self._history) > self.MAX_SNAPSHOTS:
            self._history.pop(0)

        return entry

    def restore(self, index: int) -> Optional[pd.DataFrame]:
        """
        Restores the DataFrame to the state saved at the given index.

        Snapshots taken AFTER this index are discarded so the history
        stays linear — you cannot have a branch where two different
        operations both follow from the same restore point.

        Args:
            index: The snapshot index from .list_summaries()

        Returns:
            A copy of the restored DataFrame, or None if index not found.
        """
        # Find the entry with this index
        entry = next((e for e in self._history if e.index == index), None)

        if entry is None:
            return None

        # Discard all snapshots taken after this one
        position = self._history.index(entry)
        self._history = self._history[:position + 1]

        return entry.df.copy()

    def list_summaries(self) -> list:
        """
        Returns a list of snapshot summaries (no DataFrames — metadata only).
        Ordered newest first so the most recent state is at the top.
        """
        return [e.summary() for e in reversed(self._history)]

    def latest(self) -> Optional[SnapshotEntry]:
        """
        Returns the most recently saved snapshot, or None if empty.
        """
        return self._history[-1] if self._history else None

    def clear(self) -> None:
        """
        Clears all snapshots. Called when a new file is uploaded
        so old history from a previous dataset doesn't persist.
        """
        self._history    = []
        self._next_index = 0

    @property
    def count(self) -> int:
        return len(self._history)


# Single instance imported everywhere — same pattern as dfState
snapshot_store = SnapshotStore()
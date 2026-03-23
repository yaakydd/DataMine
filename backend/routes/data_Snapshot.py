"""
data_cleaning/snapshot_router.py

API endpoints for the undo/snapshot system.

Register in main.py:
    from data_cleaning.snapshot_router import snapshot_router
    app.include_router(snapshot_router, prefix="/api")
"""

from fastapi import HTTPException, APIRouter
from pydantic import BaseModel
from routes.dfState import dataset_state
from snapshot.snapshotState import snapshot_store

snapshot_router = APIRouter()


def require_df():
    if dataset_state.df is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset loaded."
        )
    return dataset_state.df


# =============================================================================
# SNAPSHOT ENDPOINTS
# =============================================================================

@snapshot_router.get("/snapshots")
async def list_snapshots():
    """
    Returns the full snapshot history, newest first.

    Each entry contains:
        index:      the snapshot's unique identifier — use this to restore
        label:      plain-English description of what changed
        timestamp:  when the snapshot was taken
        rows/cols:  the DataFrame shape at that point

    The frontend uses this to render the undo history sidebar —
    a list of labelled checkpoints the user can click to restore.

    Note: the CURRENT state of the DataFrame is not a snapshot.
    Snapshots are states BEFORE each change. To see the current
    shape, call GET /task_info or any of the info endpoints.
    """
    summaries = snapshot_store.list_summaries()

    return {
        "total_snapshots":  len(summaries),
        "max_snapshots":    snapshot_store.MAX_SNAPSHOTS,
        "snapshots":        summaries,
        "message": (
            "No snapshots yet. Snapshots are created automatically "
            "each time you apply a cleaning operation."
            if not summaries else
            f"{len(summaries)} snapshot(s) available. "
            f"Use POST /api/snapshots/restore to roll back to any of them."
        ),
    }


class RestorePayload(BaseModel):
    """
    index: the snapshot index from GET /api/snapshots
           Snapshots taken after this index will be discarded.

    Example:
        {"index": 3}
    """
    index: int


@snapshot_router.post("/snapshots/restore")
async def restore_snapshot(payload: RestorePayload):
    """
    Restores the DataFrame to the state saved at the given snapshot index.

    What happens:
        1. The snapshot at payload.index is found in history.
        2. The DataFrame is restored to that saved state.
        3. All snapshots taken AFTER that index are discarded — history
           stays linear. You cannot branch from a restored state.
        4. dataset_state.df is updated to the restored DataFrame.

    After restoring, all subsequent cleaning operations will start from
    the restored state — as if the operations after that snapshot
    never happened.

    The response includes the restored shape and label so the user
    can confirm they restored the right checkpoint.
    """
    restored_df = snapshot_store.restore(payload.index)

    if restored_df is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Snapshot with index {payload.index} not found. "
                "Call GET /api/snapshots to see available indices."
            )
        )

    # Find the label of the restored snapshot for the confirmation message
    # After .restore() the last entry in history IS the restored snapshot
    latest = snapshot_store.latest()
    label  = latest.label if latest else "unknown"

    dataset_state.df = restored_df

    return {
        "success":         True,
        "restored_index":  payload.index,
        "restored_label":  label,
        "restored_shape":  list(restored_df.shape),
        "snapshots_remaining": snapshot_store.count,
        "message": (
            f"Dataset restored to the state before: '{label}'. "
            f"Shape: {restored_df.shape[0]} rows x {restored_df.shape[1]} columns. "
            f"{snapshot_store.count} snapshot(s) remain in history."
        ),
    }


@snapshot_router.delete("/snapshots/clear")
async def clear_snapshots():
    """
    Clears all snapshots from memory.

    Called automatically when a new file is uploaded so history
    from a previous dataset does not carry over.

    Can also be called manually if the user wants to reset their
    undo history without re-uploading the file.
    """
    count_before = snapshot_store.count
    snapshot_store.clear()

    return {
        "success":         True,
        "snapshots_cleared": count_before,
        "message":         f"Cleared {count_before} snapshot(s) from memory.",
    }


@snapshot_router.get("/snapshots/latest")
async def get_latest_snapshot():
    """
    Returns the most recent snapshot summary without restoring anything.

    Useful for the frontend to show the user what the last saved
    state was, so they know what 'undo' would roll back to.
    """
    latest = snapshot_store.latest()

    if latest is None:
        return {
            "has_snapshot": False,
            "message":      "No snapshots saved yet.",
        }

    return {
        "has_snapshot": True,
        "latest":       latest.summary(),
    }
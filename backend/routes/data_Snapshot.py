"""
data_cleaning/snapshot_router.py

All endpoints now require X-User-ID header so each user
only sees and modifies their own snapshot history.
"""

from fastapi import HTTPException, APIRouter, Depends, Header
from pydantic import BaseModel
from State.dfState import dataset_state
from State.snapshotState import snapshot_store

snapshot_router = APIRouter()


# ── Shared dependencies ───────────────────────────────────────────────────────

async def get_user_id(x_user_id: str = Header(...)) -> str:
    """
    Reads the X-User-ID header from every request.
    FastAPI injects this automatically via Depends().
    Returns 422 automatically if the header is missing.
    """
    if not x_user_id or not x_user_id.strip():
        raise HTTPException(status_code=400, detail="X-User-ID header is required.")
    return x_user_id.strip()


async def require_df(user_id: str = Depends(get_user_id)):
    """
    Confirms this user has a loaded DataFrame before any operation runs.
    Use as a dependency on any endpoint that reads or modifies data.
    """
    df = dataset_state.get_df(user_id)
    if df is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset loaded. Upload a file first via POST /api/dataset_info."
        )
    return df


# ── Endpoints ─────────────────────────────────────────────────────────────────

@snapshot_router.get("/snapshots")
async def list_snapshots(user_id: str = Depends(get_user_id)):
    summaries = snapshot_store.list_summaries(user_id)
    return {
        "total_snapshots": len(summaries),
        "max_snapshots":   snapshot_store.MAX_SNAPSHOTS,
        "snapshots":       summaries,
        "message": (
            "No snapshots yet. Apply a cleaning operation to create one."
            if not summaries else
            f"{len(summaries)} snapshot(s) available."
        ),
    }


class RestorePayload(BaseModel):
    index: int


@snapshot_router.post("/snapshots/restore")
async def restore_snapshot(
    payload: RestorePayload,
    user_id: str = Depends(get_user_id)
):
    restored_df = snapshot_store.restore(user_id, payload.index)

    if restored_df is None:
        raise HTTPException(
            status_code=404,
            detail=f"Snapshot {payload.index} not found for your session."
        )

    latest = snapshot_store.latest(user_id)
    label  = latest.label if latest else "unknown"

    # Write the restored df back to this user's session
    dataset_state.set_df(user_id, restored_df)

    return {
        "success":             True,
        "restored_index":      payload.index,
        "restored_label":      label,
        "restored_shape":      list(restored_df.shape),
        "snapshots_remaining": snapshot_store.count(user_id),
        "message": (
            f"Restored to state before: '{label}'. "
            f"Shape: {restored_df.shape[0]} rows × {restored_df.shape[1]} cols."
        ),
    }


@snapshot_router.delete("/snapshots/clear")
async def clear_snapshots(user_id: str = Depends(get_user_id)):
    count_before = snapshot_store.count(user_id)
    snapshot_store.clear(user_id)
    return {
        "success":           True,
        "snapshots_cleared": count_before,
        "message":           f"Cleared {count_before} snapshot(s).",
    }


@snapshot_router.get("/snapshots/latest")
async def get_latest_snapshot(user_id: str = Depends(get_user_id)):
    latest = snapshot_store.latest(user_id)
    if latest is None:
        return {"has_snapshot": False, "message": "No snapshots saved yet."}
    return {"has_snapshot": True, "latest": latest.summary()}
```

---

## The Full Picture of What Changed
```
BEFORE                          AFTER
──────────────────────────────────────────────────────
snapshot_store._history         snapshot_store._stores[user_id]._history
snapshot_store.save(label, df)  snapshot_store.save(user_id, label, df)
snapshot_store.restore(index)   snapshot_store.restore(user_id, index)
snapshot_store.count            snapshot_store.count(user_id)
dataset_state.df = x            dataset_state.set_df(user_id, x)
dataset_state.df                dataset_state.get_df(user_id)
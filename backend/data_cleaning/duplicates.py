# =============================================================================
# duplicates.py
#
# This file handles Task 3 of DataMine: detecting and removing duplicate rows
# in the uploaded dataset.
#
# It defines 2 endpoints:
#
#   GET  /task3/duplicates_info  → scan the DataFrame for full and partial
#                                  duplicate rows, return counts, percentages,
#                                  sample rows, and xAI explanations for every
#                                  fix strategy. Never modifies the data.
#
#   POST /task3/fix_duplicates   → apply the user's chosen removal strategy:
#                                  keep_first, keep_last, drop_all, do_nothing.
#                                  Can target all columns or a specific subset.
# =============================================================================


# HTTPException → return a clean error + HTTP status code instead of a Python crash
# APIRouter     → groups all Task 3 endpoints under one object registered in main.py
from fastapi import HTTPException, APIRouter

# BaseModel → Pydantic base class that auto-validates incoming JSON request bodies
from pydantic import BaseModel

# Shared singleton holding the current DataFrame in memory across all routers
from State.dfState import dataset_state

# Each function returns a plain-English explanation string for the xAI panel.
# Keeping explanation text out of this file means wording can change
# without touching cleaning logic.
from xAI.duplicates_explainer import (
    explain_duplicates_overview,              # top-level summary of what was found
    explain_duplicates_if_you_keep_first,     # what keeping the first occurrence means
    explain_duplicates_if_you_keep_last,      # what keeping the last occurrence means
    explain_duplicates_if_you_drop_all,       # what dropping every copy means
    explain_duplicates_if_you_do_nothing,     # what leaving duplicates in place means
    explain_subset_duplicates,                # what it means when a specific column has duplicates
)

import pandas as pd   # main data manipulation library
import numpy as np    # used for numpy type detection in safe_json()

# Optional → marks fields that don't have to be provided in the request body
# List    → type hint for the subset column list
from typing import Optional, List

# Snapshot store saves a copy of the DataFrame before every write.
# Powers the undo/rollback system — up to 20 snapshots are kept.
from State.snapshotState import snapshot_store

# All Task 3 endpoints are attached to this router.
# main.py registers it with: app.include_router(task3, prefix="/api")
task3 = APIRouter()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def require_df() -> pd.DataFrame:
    """
    Called at the start of every endpoint to guard against missing uploads.

    If no file has been uploaded, dataset_state.df is None and any pandas
    operation would crash with a confusing AttributeError deep in a library.
    This catches it early and returns a clear HTTP 400 instead.
    """
    if dataset_state.df is None:
        raise HTTPException(
            status_code=400,   # 400 = Bad Request — the client called this before uploading
            detail="No dataset loaded. Please upload a file first via POST /api/dataset_info"
        )
    return dataset_state.df


def safe_json(obj):
    """
    Recursively converts numpy and pandas types to plain Python equivalents
    so FastAPI's JSON serialiser never encounters a type it doesn't know.

    Why this is needed:
        When we pull rows out of a DataFrame with .to_dict(), the values are
        still numpy types — np.int64, np.float64, np.bool_, etc.
        Python's built-in json module (which FastAPI uses) does not know
        how to handle these and will raise a TypeError.
        safe_json() walks the entire object and converts every numpy type
        to its plain Python equivalent before we return it.

    Handles:
        dict       → recurse into each value
        list       → recurse into each item
        np.integer → int()
        np.floating → float(), or None if the value is NaN or Inf
                      (JSON has no representation for NaN/Inf)
        np.bool_   → bool()
        np.ndarray → convert to list first, then recurse
        pd.NA / pd.NaT / float('nan') → None
                      (JSON null is the correct representation of missing)
        anything else → return as-is (already a plain Python type)
    """
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}   # recurse into dict values
    if isinstance(obj, list):
        return [safe_json(v) for v in obj]                 # recurse into list items
    if isinstance(obj, np.integer):
        return int(obj)         # np.int64(5) → 5
    if isinstance(obj, np.floating):
        # JSON cannot represent NaN or Infinity — convert them to null instead
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)        # np.bool_(True) → True
    if isinstance(obj, np.ndarray):
        return safe_json(obj.tolist())   # convert array to list, then recurse
    try:
        # pd.isna() catches pd.NA, pd.NaT, float('nan'), and None all at once
        if pd.isna(obj):
            return None         # JSON null for any missing value
    except (TypeError, ValueError):
        # pd.isna() raises on some types (e.g. lists) — ignore and fall through
        pass
    return obj   # already a plain Python type — return unchanged


# =============================================================================
# TASK 3 — HANDLING DUPLICATES
#
# Two endpoints:
#
#   GET  /task3/duplicates_info
#        Counts full duplicate rows and returns a report with counts,
#        percentages, sample rows, and all xAI explanations for every
#        fix option. Also checks for subset duplicates (partial matches
#        on specific columns the user might care about).
#        Never modifies the data.
#
#   POST /task3/fix_duplicates
#        Applies the user's chosen strategy:
#          keep_first  — keep first occurrence, drop the rest
#          keep_last   — keep last occurrence, drop the rest
#          drop_all    — drop every row that has any duplicate
#          do_nothing  — leave the dataset unchanged
#        Can also operate on a subset of columns instead of full rows.
# =============================================================================

@task3.get("/task3/duplicates_info")
async def get_duplicates_info():
    """
    Scans the DataFrame for duplicate rows and returns a full report.

    Full duplicate detection:
      A full duplicate means every single column in one row matches
      every single column in another row exactly. We use df.duplicated()
      which by default marks every occurrence AFTER the first as a duplicate,
      so the first appearance is never counted as a duplicate itself.

    What the response contains:
      - total_rows:        total number of rows in the dataset
      - duplicate_count:   how many rows are duplicates (not counting the original)
      - duplicate_pct:     what percentage of total rows are duplicates
      - sample_duplicates: up to 5 example duplicate rows so the user can
                           see what is actually being flagged
      - all_strategies:    every available fix with its own xAI explanation
      - subset_report:     a check on commonly important column combinations
                           to catch partial duplicates the user might not notice

    The frontend uses this to render:
      - A summary banner with counts and percentage
      - A sample table showing example duplicate rows
      - A strategy selector with one card per option, each showing
        its explanation before the user makes a choice
    """
    df         = require_df()
    total_rows = len(df)

    # ── Full duplicate detection ──────────────────────────────────────────────

    # df.duplicated() returns a boolean Series where True means that row
    # is an exact copy of a previous row.
    # keep="first" → the FIRST time a row appears it is marked False (not a duplicate).
    #                Every subsequent copy of that same row is marked True.
    # This means the count we get is: "how many rows are redundant copies"
    # and does NOT count the originals.
    duplicate_mask  = df.duplicated(keep="first")
    duplicate_count = int(duplicate_mask.sum())   # count of True values = number of duplicate rows
    duplicate_pct   = round((duplicate_count / total_rows) * 100, 2)

    # Pull up to 5 real duplicate rows as examples for the user to inspect.
    # df[duplicate_mask] filters to only the rows flagged as duplicates.
    # .head(5) caps at 5 rows — we don't want to flood the frontend with data.
    # .to_dict(orient="records") converts to a list of dicts, one per row.
    # safe_json() then converts numpy types to plain Python for JSON serialisation.
    sample_duplicates = safe_json(
        df[duplicate_mask].head(5).to_dict(orient="records")
    )

    # ── Build strategies ──────────────────────────────────────────────────────

    # Each strategy is a dict with a "label" (shown in the dropdown) and an
    # "explanation" (shown beneath the dropdown when that option is selected).
    # Keeping these here means the frontend never has hardcoded strings —
    # it just renders whatever label/explanation the API returns.
    all_strategies = {
        "keep_first": {
            "label":       "Keep first occurrence, remove copies",
            "explanation": explain_duplicates_if_you_keep_first(duplicate_count),
        },
        "keep_last": {
            "label":       "Keep last occurrence, remove earlier copies",
            "explanation": explain_duplicates_if_you_keep_last(duplicate_count),
        },
        "drop_all": {
            "label":       "Remove all copies including the original",
            "explanation": explain_duplicates_if_you_drop_all(duplicate_count),
        },
        "do_nothing": {
            "label":       "Leave duplicates as-is",
            "explanation": explain_duplicates_if_you_do_nothing(),
        },
    }

    # ── Subset duplicate detection ────────────────────────────────────────────

    # Full-row duplicates require every column to match.
    # But often the more meaningful question is: does a column that SHOULD
    # be unique (like email, phone, user_id) contain repeated values?
    #
    # Two rows with the same email are almost certainly the same person,
    # even if other fields like "notes" or "signup_date" differ.
    #
    # We scan column names for identity-related keywords and check each
    # one for duplicates automatically — this is the xAI layer flagging
    # issues the user might not have thought to look for themselves.
    subset_report    = []
    identity_keywords = ("id", "email", "phone", "name", "code", "number")

    # Build a list of columns whose names suggest they should be unique
    identity_cols = [
        col for col in df.columns
        if any(kw in col.lower() for kw in identity_keywords)
    ]

    for col in identity_cols:
        # Check how many rows have a repeated value in this specific column.
        # subset=[col] limits the duplicate check to this one column only.
        # keep="first" means we count copies — not the first occurrence itself.
        subset_dup_count = int(df.duplicated(subset=[col], keep="first").sum())
        if subset_dup_count > 0:
            # Only report if there are actual duplicates in this column
            subset_report.append({
                "column":      col,
                "dup_count":   subset_dup_count,
                "explanation": explain_subset_duplicates([col], subset_dup_count),
            })

    return {
        # Summary numbers for the top banner in the UI
        "total_rows":      total_rows,
        "duplicate_count": duplicate_count,
        "duplicate_pct":   duplicate_pct,

        # Up to 5 real example rows so the user can see what's being flagged
        "sample_duplicates": sample_duplicates,

        # High-level explanation of the overall duplicate situation
        "overview_explanation": explain_duplicates_overview(
            duplicate_count, total_rows, duplicate_pct
        ),

        # All four strategies with labels and explanations for the frontend
        "all_strategies": all_strategies,

        # Partial duplicate warnings for identity-like columns
        "subset_report":  subset_report,

        # Convenience boolean — the frontend can check this instead of dup_count == 0
        "has_duplicates": duplicate_count > 0,
    }


# ── Request body model ────────────────────────────────────────────────────────

class DuplicateFixPayload(BaseModel):
    """
    Defines the shape of the JSON body for POST /task3/fix_duplicates.
    Pydantic validates this automatically before our function runs.

    method:
        One of: keep_first, keep_last, drop_all, do_nothing

    subset:
        Optional list of column names to check for duplicates on.
        When provided, only these columns are compared — two rows are
        considered duplicates if they match on ALL columns in this list,
        even if other columns differ.
        When not provided (or empty), full-row comparison is used.

    Example — full row deduplication:
        {"method": "keep_first"}

    Example — deduplicate based on email column only:
        {"method": "keep_first", "subset": ["email"]}
    """
    method: str                          # required — must be one of the four valid options
    subset: Optional[List[str]] = None   # optional — None means compare all columns


@task3.post("/task3/fix_duplicates")
async def fix_duplicates(payload: DuplicateFixPayload):
    """
    Applies the user's chosen duplicate removal strategy.

    Works on a .copy() as always — dataset_state.df is only overwritten
    at the end if the operation succeeds and the method was not do_nothing.

    keep_first:
        df.drop_duplicates(keep="first") keeps the first occurrence of
        each duplicate group and removes all later copies.
        The index is reset after dropping so rows are numbered from 0.

    keep_last:
        df.drop_duplicates(keep="last") keeps the last occurrence and
        removes all earlier ones. Useful for time-ordered data where
        the most recent entry is the most authoritative.

    drop_all:
        df.drop_duplicates(keep=False) removes EVERY row that has a
        duplicate — including the original first occurrence.
        This is the strictest option and removes the most rows.

    do_nothing:
        Returns immediately without modifying anything. The response
        still includes the current shape so the frontend can update
        its state display consistently.

    subset handling:
        When the user provides a subset list, pandas only looks at those
        columns when deciding if two rows are duplicates. All other columns
        are ignored in the comparison.
    """
    df      = require_df().copy()   # .copy() — never mutate the live DataFrame until we're sure
    applied = []                    # success messages — one per change applied
    errors  = []                    # failure messages — returned to the frontend

    method = payload.method
    # If subset is an empty list, treat it as None (use all columns)
    subset = payload.subset if payload.subset else None

    # ── Validate subset columns ───────────────────────────────────────────────

    # Check all requested subset columns exist BEFORE doing any work.
    # Failing early is better than partially modifying the DataFrame and
    # then discovering a column name was wrong.
    if subset:
        missing_cols = [col for col in subset if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Subset column(s) not found in dataset: {missing_cols}"
            )

    # Record the row count before any changes — needed to calculate rows_removed
    before = len(df)

    try:
        if method == "keep_first":
            # drop_duplicates(keep="first") removes all rows that are duplicates
            # of an earlier row, keeping only the first occurrence of each group.
            # subset=subset passes the column list (or None for all columns).
            df.drop_duplicates(keep="first", subset=subset, inplace=True)
            # reset_index(drop=True) renumbers the index from 0 after row removal.
            # drop=True means the old index is discarded, not added as a new column.
            df.reset_index(drop=True, inplace=True)
            rows_removed = before - len(df)
            # Build a human-readable description of what was compared
            scope = f"columns {subset}" if subset else "all columns"
            applied.append(
                f"Removed {rows_removed} duplicate row(s) based on {scope}. "
                f"First occurrence of each duplicate kept. "
                f"{len(df)} rows remain."
            )

        elif method == "keep_last":
            # keep="last" is the reverse — the LAST occurrence is kept and all
            # earlier duplicates are removed. Useful when data is time-ordered
            # and the most recent entry is the authoritative one.
            df.drop_duplicates(keep="last", subset=subset, inplace=True)
            df.reset_index(drop=True, inplace=True)
            rows_removed = before - len(df)
            scope = f"columns {subset}" if subset else "all columns"
            applied.append(
                f"Removed {rows_removed} duplicate row(s) based on {scope}. "
                f"Last occurrence of each duplicate kept. "
                f"{len(df)} rows remain."
            )

        elif method == "drop_all":
            # keep=False is the strictest option — it removes EVERY row that has
            # any duplicate, including the first occurrence itself.
            # Use this when you want only rows that are completely unique.
            df.drop_duplicates(keep=False, subset=subset, inplace=True)
            df.reset_index(drop=True, inplace=True)
            rows_removed = before - len(df)
            scope = f"columns {subset}" if subset else "all columns"
            applied.append(
                f"Removed {rows_removed} row(s) — all copies including originals dropped "
                f"based on {scope}. "
                f"{len(df)} rows remain."
            )

        elif method == "do_nothing":
            # User chose to leave duplicates in place.
            # We still log it so the applied list reflects the full request.
            applied.append(
                f"No changes made. Dataset still contains {before} rows."
            )

        else:
            # Unknown method name — raise immediately with the valid options listed
            raise HTTPException(
                status_code=400,
                detail=(
                    f'Unknown method "{method}". '
                    "Valid options: keep_first, keep_last, drop_all, do_nothing."
                )
            )

    except HTTPException:
        raise   # re-raise FastAPI HTTP errors as-is — don't swallow them into the errors list
    except Exception as e:
        # Any unexpected runtime error is caught and reported, not re-raised,
        # so a single failure returns a clear message rather than a 500 crash
        errors.append(f"Unexpected error: {str(e)}")

    # ── Save back ─────────────────────────────────────────────────────────────

    # Only save back if:
    #   1. The method was not "do_nothing" (no actual change was made)
    #   2. No errors occurred during the operation
    if method != "do_nothing" and not errors:
        # SNAPSHOT RULE: save BEFORE writing to dataset_state.df
        # require_df() still returns the old unmodified DataFrame at this point
        # because dataset_state.df has not been overwritten yet
        snapshot_store.save(f"Task 3 — duplicates removed ({method})", require_df())
        dataset_state.df = df   # now write the cleaned copy back to shared state

    # ── Return ────────────────────────────────────────────────────────────────

    return {
        "success":      len(errors) == 0,       # True only if zero errors occurred
        "applied":      applied,
        "errors":       errors,
        "rows_removed": before - len(df),        # how many rows were actually deleted
        "new_shape":    list(df.shape),           # list() so it serialises as a JSON array
        # Recount missing cells on the final df for the frontend summary card
        "remaining_missing": int(df.isnull().sum().sum()),
    }
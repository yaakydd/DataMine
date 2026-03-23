from fastapi import HTTPException, APIRouter
from pydantic import BaseModel
from State.dfState import dataset_state
from xAI.duplicates_explainer import (
    explain_duplicates_overview,
    explain_duplicates_if_you_keep_first,
    explain_duplicates_if_you_keep_last,
    explain_duplicates_if_you_drop_all,
    explain_duplicates_if_you_do_nothing,
    explain_subset_duplicates,
)
import pandas as pd
import numpy as np
from typing import Optional, List
from State.snapshotState import snapshot_store

task3 = APIRouter()


# =============================================================================
# HELPER
# =============================================================================

def require_df() -> pd.DataFrame:
    if dataset_state.df is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset loaded. Please upload a file first via POST /api/dataset_info"
        )
    return dataset_state.df


def safe_json(obj):
    """
    Converts numpy/pandas types to plain Python so FastAPI can serialise
    the response to JSON without errors. Called on sample rows before
    returning them so the frontend always receives clean, readable data.
    """
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return safe_json(obj.tolist())
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


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
      - total_rows:       total number of rows in the dataset
      - duplicate_count:  how many rows are duplicates (not counting the original)
      - duplicate_pct:    what percentage of total rows are duplicates
      - sample_duplicates: up to 5 example duplicate rows so the user can
                           see what is actually being flagged
      - all_strategies:   every available fix with its own xAI explanation
      - subset_report:    a check on commonly important column combinations
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

    # df.duplicated() returns a boolean Series — True for every row that is
    # an exact copy of a previous row. keep="first" means the FIRST occurrence
    # is marked False (not a duplicate) and all later copies are marked True.
    duplicate_mask  = df.duplicated(keep="first")
    duplicate_count = int(duplicate_mask.sum())
    duplicate_pct   = round((duplicate_count / total_rows) * 100, 2)

    # Get up to 5 sample duplicate rows so the user can see real examples.
    # safe_json() converts numpy types so the rows serialise cleanly.
    sample_duplicates = safe_json(
        df[duplicate_mask].head(5).to_dict(orient="records")
    )

    # ── Build strategies ──────────────────────────────────────────────────────

    # Every strategy gets its label (shown in the dropdown) and its
    # explanation (shown beneath the dropdown when that option is selected).
    # The frontend renders these without needing any hardcoded strings.

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

    # Full-row duplicates are obvious, but partial duplicates on key columns
    # are often more meaningful. For example, two rows with the same email
    # address probably represent the same person even if other fields differ.
    #
    # We automatically check every column whose name suggests it should be
    # unique: anything containing "id", "email", "phone", "name", "code",
    # or "number". If we find duplicates on those columns we flag them.
    #
    # This is the xAI layer — we don't just report numbers, we point out
    # patterns the user might not have thought to look for.

    subset_report = []
    identity_keywords = ("id", "email", "phone", "name", "code", "number")

    identity_cols = [
        col for col in df.columns
        if any(kw in col.lower() for kw in identity_keywords)
    ]

    for col in identity_cols:
        # Count rows where this single column has a repeated value
        subset_dup_count = int(df.duplicated(subset=[col], keep="first").sum())
        if subset_dup_count > 0:
            subset_report.append({
                "column":      col,
                "dup_count":   subset_dup_count,
                "explanation": explain_subset_duplicates([col], subset_dup_count),
            })

    return {
        # Summary numbers for the top banner
        "total_rows":       total_rows,
        "duplicate_count":  duplicate_count,
        "duplicate_pct":    duplicate_pct,

        # Sample rows — up to 5 examples of actual duplicate rows
        "sample_duplicates": sample_duplicates,

        # The top-level explanation of what duplicates mean in this dataset
        "overview_explanation": explain_duplicates_overview(
            duplicate_count, total_rows, duplicate_pct
        ),

        # All strategies with labels and explanations for the frontend
        "all_strategies": all_strategies,

        # Partial duplicate warnings on identity-like columns
        "subset_report":  subset_report,

        # Convenience flag for the frontend — no need to check dup_count == 0
        "has_duplicates": duplicate_count > 0,
    }


# ── Request body model ────────────────────────────────────────────────────────

class DuplicateFixPayload(BaseModel):
    """
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
    method: str
    subset: Optional[List[str]] = None


@task3.post("/task3/fix_duplicates")
async def fix_duplicates(payload: DuplicateFixPayload):
    """
    Applies the user's chosen duplicate removal strategy.

    Works on a .copy() as always — dataset_state.df is only overwritten
    at the end if the operation succeeds.

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
        This is stricter than keep_first/keep_last.

    do_nothing:
        Returns immediately without modifying anything. The response
        still includes the current shape and missing count so the
        frontend can update its state display.

    subset handling:
        When the user provides a subset list, pandas only looks at those
        columns when deciding if two rows are duplicates. All other columns
        are ignored in the comparison.
    """
    df      = require_df().copy()
    applied = []
    errors  = []

    method = payload.method
    subset = payload.subset if payload.subset else None

    # Validate subset columns exist before doing anything
    if subset:
        missing_cols = [col for col in subset if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Subset column(s) not found in dataset: {missing_cols}"
            )

    before = len(df)

    try:
        if method == "keep_first":
            df.drop_duplicates(keep="first", subset=subset, inplace=True)
            df.reset_index(drop=True, inplace=True)
            rows_removed = before - len(df)
            scope = f"columns {subset}" if subset else "all columns"
            applied.append(
                f"Removed {rows_removed} duplicate row(s) based on {scope}. "
                f"First occurrence of each duplicate kept. "
                f"{len(df)} rows remain."
            )

        elif method == "keep_last":
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
            applied.append(
                f"No changes made. Dataset still contains {before} rows."
            )

        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    f'Unknown method "{method}". '
                    "Valid options: keep_first, keep_last, drop_all, do_nothing."
                )
            )

    except HTTPException:
        raise   # re-raise FastAPI errors as-is
    except Exception as e:
        errors.append(f"Unexpected error: {str(e)}")

    # ── Save back ─────────────────────────────────────────────────────────────

    if method != "do_nothing" and not errors:
        snapshot_store.save(f"Task 3 — duplicates removed ({method})", require_df())
        dataset_state.df = df

    # ── Return ────────────────────────────────────────────────────────────────

    return {
        "success":           len(errors) == 0,
        "applied":           applied,
        "errors":            errors,
        "rows_removed":      before - len(df),
        "new_shape":         list(df.shape),
        "remaining_missing": int(df.isnull().sum().sum()),
    }
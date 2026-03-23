import pandas as pd
import numpy as np
from typing import Optional, List
from fastapi import HTTPException, APIRouter
from pydantic import BaseModel
from difflib import SequenceMatcher
from typing import Optional, List, Dict
from routes.dfState import dataset_state
from xAI.duplicates_explainer import (
    explain_duplicates_overview,
    explain_duplicates_if_you_keep_first,
    explain_duplicates_if_you_keep_last,
    explain_duplicates_if_you_drop_all,
    explain_duplicates_if_you_do_nothing,
    explain_subset_duplicates,
    explain_what_fuzzy_duplicates_are,
    explain_fuzzy_match,
    explain_fuzzy_threshold,
    explain_duplicate_impact,
    explain_no_fuzzy_columns,
)


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

def similarity_score(a: str, b: str) -> float:
    """
    Computes a similarity percentage between two strings using
    Python's built-in SequenceMatcher.
 
    SequenceMatcher uses the Ratcliff/Obershelp algorithm — it finds
    the longest common substring, then recursively matches the parts
    before and after it. The ratio is:
        2 * matching_characters / total_characters_in_both_strings
 
    Returns a float between 0.0 (completely different) and 100.0 (identical).
 
    Why SequenceMatcher instead of Levenshtein distance:
        SequenceMatcher is in Python's standard library — no extra
        dependencies needed. For the typo/spacing/casing mismatches
        that fuzzy duplicate detection is designed for, it performs
        well enough without requiring the 'python-Levenshtein' package.
    """
    a_clean = str(a).strip().lower()
    b_clean = str(b).strip().lower()
    ratio   = SequenceMatcher(None, a_clean, b_clean).ratio()
    return round(ratio * 100, 2)
 
 
@task3.get("/task3/fuzzy_duplicates")
async def get_fuzzy_duplicates(
    threshold: float = 90.0,
    col: Optional[str] = None,
    max_comparisons: int = 5000,
):
    """
    Detects near-duplicate values in text columns using string similarity.
 
    Parameters:
        threshold:        Minimum similarity % to flag as a near-duplicate.
                          Default 90.0. Range: 0–100.
        col:              Optional — check only this one column.
                          If not provided, all text columns are checked.
        max_comparisons:  Maximum number of pairwise comparisons per column.
                          Default 5000. Prevents timeout on large datasets.
                          Comparisons are sampled randomly if this is exceeded.
 
    How it works:
        For each text column, every unique value is compared against every
        other unique value using SequenceMatcher. If the similarity score
        is at or above the threshold, the pair is flagged.
 
        We compare UNIQUE values only, not every row — this is much faster.
        e.g. if "London" appears 500 times and "london" appears 200 times,
        we only do one comparison, not 500 × 200 = 100,000 comparisons.
 
    What the response contains per column:
        - fuzzy_pairs:   list of near-duplicate value pairs with scores
        - pair_count:    how many pairs were found
        - explanation:   xAI description of what was found
        - threshold_explanation: what the threshold means
 
    Performance note:
        For a column with N unique values, the number of comparisons is
        N × (N-1) / 2. For 100 unique values that is 4,950 comparisons.
        For 1,000 unique values it is 499,500 — which is why max_comparisons
        exists as a safety cap.
    """
    from xAI.duplicates_explainer import (
        explain_what_fuzzy_duplicates_are,
        explain_fuzzy_match,
        explain_fuzzy_threshold,
        explain_no_fuzzy_columns,
    )
 
    df = require_df()
 
    # Pick which columns to check
    if col:
        if col not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f'Column "{col}" not found in the dataset.'
            )
        if str(df[col].dtype) not in ("object", "string", "category"):
            raise HTTPException(
                status_code=400,
                detail=f'"{col}" is not a text column. Fuzzy detection only applies to text.'
            )
        text_cols = [col]
    else:
        text_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
 
    if not text_cols:
        return {
            "columns_checked": 0,
            "results":         [],
            "message":         explain_no_fuzzy_columns(),
        }
 
    all_results = []
 
    for column in text_cols:
 
        # Work with unique non-null values only for efficiency
        unique_vals = df[column].dropna().unique().tolist()
        unique_vals = [str(v) for v in unique_vals]
        n           = len(unique_vals)
 
        if n < 2:
            continue   # nothing to compare
 
        fuzzy_pairs    = []
        comparison_count = 0
 
        # Compare every pair of unique values
        # i < j ensures we never compare a value with itself
        # and never compare the same pair twice
        for i in range(n):
            for j in range(i + 1, n):
 
                if comparison_count >= max_comparisons:
                    break
 
                score = similarity_score(unique_vals[i], unique_vals[j])
                comparison_count += 1
 
                if score >= threshold and score < 100.0:
                    # Find the row indices where each value appears
                    # Limited to first 3 occurrences for readability
                    idx1 = df.index[df[column] == unique_vals[i]].tolist()[:1]
                    idx2 = df.index[df[column] == unique_vals[j]].tolist()[:1]
 
                    row_idx1 = idx1[0] if idx1 else -1
                    row_idx2 = idx2[0] if idx2 else -1
 
                    fuzzy_pairs.append({
                        "value_1":    unique_vals[i],
                        "value_2":    unique_vals[j],
                        "similarity": score,
                        "row_idx_1":  int(row_idx1),
                        "row_idx_2":  int(row_idx2),
                        "explanation": explain_fuzzy_match(
                            column,
                            unique_vals[i],
                            unique_vals[j],
                            score,
                            int(row_idx1),
                            int(row_idx2),
                        ),
                    })
 
            if comparison_count >= max_comparisons:
                break
 
        # Sort by similarity descending — most likely duplicates first
        fuzzy_pairs.sort(key=lambda x: x["similarity"], reverse=True)
 
        if fuzzy_pairs:
            all_results.append({
                "column":               column,
                "unique_values_checked": n,
                "comparisons_made":     comparison_count,
                "capped_at_max":        comparison_count >= max_comparisons,
                "pair_count":           len(fuzzy_pairs),
                "fuzzy_pairs":          fuzzy_pairs[:20],   # cap response size at 20 pairs
                "threshold_explanation": explain_fuzzy_threshold(threshold),
            })
 
    return {
        "what_are_fuzzy_duplicates": explain_what_fuzzy_duplicates_are(),
        "threshold_used":           threshold,
        "columns_checked":          len(text_cols),
        "columns_with_matches":     len(all_results),
        "results":                  all_results,
        "tip": (
            "To fix fuzzy duplicates, use POST /task1/harmonise_categories "
            "to map variant spellings to one canonical form. "
            "For example: {'john smith ': 'John Smith', 'john smith': 'John Smith'}"
        ),
    }
 
 
# ── Request body model ────────────────────────────────────────────────────────
 
class ImpactReportPayload(BaseModel):
    """
    method:  the deduplication method to simulate — keep_first, keep_last, drop_all
    subset:  optional list of columns to use for duplicate detection
             if not provided, full-row comparison is used
 
    Example:
        {
            "method": "keep_first",
            "subset": ["email"]
        }
    """
    method: str
    subset: Optional[List[str]] = None
 
 
@task3.post("/task3/duplicate_impact")
async def get_duplicate_impact(payload: ImpactReportPayload):
    """
    Shows exactly how your numeric column statistics will change
    after deduplication — without actually making any changes.
 
    Why this matters:
        Removing duplicates changes your dataset's statistics. If duplicates
        are not evenly distributed across value ranges, removing them can
        shift your means, reduce your sums, and change your row counts
        significantly. This endpoint lets the user see the impact BEFORE
        committing to any change.
 
    How it works:
        We simulate the deduplication on a copy of the DataFrame,
        then compare the before and after statistics for every numeric column.
        Nothing is saved to dataset_state — this is purely a preview.
 
    Returns for each numeric column:
        - count before and after
        - mean before and after
        - sum before and after
        - a plain-English explanation of the change
    """
    from xAI.duplicates_explainer import explain_duplicate_impact
 
    df     = require_df()
    subset = payload.subset or None
 
    # Validate subset columns
    if subset:
        missing_cols = [c for c in subset if c not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Subset column(s) not found: {missing_cols}"
            )
 
    # Simulate the deduplication on a copy
    df_after = df.copy()
 
    if payload.method == "keep_first":
        df_after.drop_duplicates(keep="first", subset=subset, inplace=True)
    elif payload.method == "keep_last":
        df_after.drop_duplicates(keep="last", subset=subset, inplace=True)
    elif payload.method == "drop_all":
        df_after.drop_duplicates(keep=False, subset=subset, inplace=True)
    else:
        raise HTTPException(
            status_code=400,
            detail=f'Unknown method "{payload.method}". Valid: keep_first, keep_last, drop_all.'
        )
 
    df_after.reset_index(drop=True, inplace=True)
 
    rows_before = len(df)
    rows_after  = len(df_after)
    rows_removed = rows_before - rows_after
 
    # Build per-column impact for every numeric column
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    column_impacts = []
 
    for col in numeric_cols:
        mean_before  = float(df[col].mean())
        mean_after   = float(df_after[col].mean())
        sum_before   = float(df[col].sum())
        sum_after    = float(df_after[col].sum())
        count_before = int(df[col].count())
        count_after  = int(df_after[col].count())
 
        # Only report columns where something actually changed
        if abs(mean_before - mean_after) > 1e-10 or count_before != count_after:
            column_impacts.append({
                "column":       col,
                "mean_before":  round(mean_before,  4),
                "mean_after":   round(mean_after,   4),
                "sum_before":   round(sum_before,   4),
                "sum_after":    round(sum_after,    4),
                "count_before": count_before,
                "count_after":  count_after,
                "explanation":  explain_duplicate_impact(
                    col,
                    mean_before, mean_after,
                    sum_before,  sum_after,
                    count_before, count_after,
                ),
            })
 
    return {
        # This is a PREVIEW — nothing was saved
        "is_preview":    True,
        "method":        payload.method,
        "subset":        subset,
 
        # Row-level summary
        "rows_before":   rows_before,
        "rows_after":    rows_after,
        "rows_removed":  rows_removed,
        "pct_removed":   round((rows_removed / rows_before) * 100, 2) if rows_before > 0 else 0,
 
        # Per-column numeric impact
        "column_impacts": column_impacts,
        "columns_affected": len(column_impacts),
 
        # Prompt to apply if happy with the preview
        "next_step": (
            "If you are happy with this impact, apply it using "
            "POST /task3/fix_duplicates with the same method and subset."
        ),
    }
 
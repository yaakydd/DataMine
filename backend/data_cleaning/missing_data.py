# =============================================================================
# missing_data.py
#
# This file handles Task 2 of DataMine: detecting and fixing missing values
# in the uploaded dataset.
#
# It defines 2 endpoints:
#
#   GET  /task2/missing_info  → scan every column and row for missing values,
#                               return counts, percentages, severity labels,
#                               and xAI explanations for every fix strategy.
#                               Never modifies the data.
#
#   POST /task2/fix_missing   → apply the user's chosen fix strategy per column.
#                               Strategies: fill_mean, fill_median, fill_mode,
#                               fill_custom, drop_rows, drop_column, do_nothing.
# =============================================================================


# HTTPException → return a clean error + HTTP status code instead of a raw Python crash
# APIRouter     → groups all Task 2 endpoints under one object registered in main.py
from fastapi import HTTPException, APIRouter

# BaseModel → Pydantic base class that auto-validates incoming JSON request bodies
from pydantic import BaseModel

# Dict, Any, Optional → type hints used in the Pydantic request body model
# Dict[str, Any] means a dict whose keys are strings and values can be anything
from typing import Dict, Any, Optional

# Shared singleton holding the current DataFrame in memory
# The same object is imported by every router in the project
from State.dfState import dataset_state

# Each function returns a plain-English explanation string for the xAI panel.
# They live in a separate file so wording can be updated without touching logic.
from xAI.missingData_explainer import (
    explain_missing_severity,            # describes how bad the missing rate is
    explain_missing_if_you_drop_rows,    # what dropping affected rows means
    explain_missing_if_you_fill_mean,    # what filling with the mean means
    explain_missing_if_you_fill_median,  # what filling with the median means
    explain_missing_if_you_fill_mode,    # what filling with the mode means
    explain_missing_if_you_fill_custom,  # what filling with a custom value means
    explain_missing_if_you_drop_column,  # what dropping the whole column means
    explain_missing_if_you_do_nothing,   # what leaving NaN values in place means
)

import pandas as pd   # main data manipulation library
import numpy as np    # used for numpy type detection (dtype.kind checks)

# Snapshot store saves a copy of the DataFrame before every write operation.
# This powers the undo/rollback system — up to 20 snapshots are kept.
from State.snapshotState import snapshot_store

# All Task 2 endpoints are registered on this router object.
# main.py registers it with: app.include_router(task2, prefix="/api")
task2 = APIRouter()


# =============================================================================
# HELPER
# =============================================================================

def require_df() -> pd.DataFrame:
    """
    Called at the start of every endpoint to guard against missing uploads.

    If dataset_state.df is None (no file uploaded yet), every pandas
    operation would crash with a confusing AttributeError.
    This catches that early and returns a clear HTTP 400 error instead.
    """
    if dataset_state.df is None:
        raise HTTPException(
            status_code=400,   # 400 = Bad Request — the client called this before uploading
            detail="No dataset loaded. Please upload a file first via POST /api/dataset_info"
        )
    return dataset_state.df


# =============================================================================
# TASK 2 — HANDLING MISSING DATA
#
# Two endpoints:
#
#   GET  /task2/missing_info
#        Scans every column and every row for missing values.
#        Returns counts, percentages, severity, and all xAI explanations
#        for every available fix strategy.
#        Never modifies the data.
#
#   POST /task2/fix_missing
#        Applies the user's chosen fix strategy per column.
#        Strategies: mean, median, mode, custom value, drop rows, drop column.
#        Works on a copy — only saves back if changes succeed.
# =============================================================================

@task2.get("/task2/missing_info")
async def get_missing_info():
    """
    Scans the entire DataFrame for missing values and returns a full report.

    Column-level report — for each column that has missing values:
      - How many values are missing and what percentage that is
      - A severity label: low / medium / high
      - A plain-English explanation of what that severity means
      - A full set of available fix strategies, each with its own
        explanation of what will happen if the user picks that option

    Row-level report:
      - How many rows have at least one missing value
      - What percentage of total rows that represents

    Overall summary:
      - Total missing cells across the entire DataFrame
      - Overall missing percentage
    """
    df          = require_df()
    total_rows  = len(df)        # number of rows — used as the denominator for percentages
    total_cells = df.size        # total number of cells = rows × columns

    # ── Column-level analysis ─────────────────────────────────────────────────

    columns_with_missing = []  # we'll append one dict per affected column

    for col in df.columns:
        # .isna() returns a boolean Series — True for each NaN/None/NaT cell
        # .sum() counts the Trues — gives us the total missing count for this column
        missing_count = int(df[col].isna().sum())  # int() ensures JSON serialisable

        if missing_count == 0:
            continue   # column is complete — nothing to report, move on

        # Calculate what percentage of this column's values are missing
        pct = round((missing_count / total_rows) * 100, 2)

        # Assign a severity label based on thresholds.
        # These thresholds are common in data science practice:
        #   < 5%  → low:    usually safe to fill or drop with minimal impact
        #   5–20% → medium: worth thinking carefully about the right strategy
        #   > 20% → high:   the column may be too sparse to be useful at all
        if pct < 5:
            severity = "low"
        elif pct < 20:
            severity = "medium"
        else:
            severity = "high"

        dtype      = str(df[col].dtype)
        strategies = {}  # will hold one entry per available fix strategy for this column

        # ── Numeric strategies (mean and median) ──────────────────────────────
        # Mean and median only make mathematical sense for numeric columns.
        # We check dtype.kind: "i" = integer family, "f" = float family.
        # This is more reliable than checking the dtype string because it
        # catches int8, int16, int32, int64, float32, float64 all at once.
        if dtype in ("int64", "float64") or df[col].dtype.kind in ("i", "f"):
            # float() converts numpy float64 to a plain Python float
            # so FastAPI's JSON serialiser doesn't encounter numpy types
            mean_val   = round(float(df[col].mean()),   4)
            median_val = round(float(df[col].median()), 4)

            strategies["fill_mean"] = {
                "label":       f"Fill with mean ({mean_val})",
                "explanation": explain_missing_if_you_fill_mean(col, mean_val),
            }
            strategies["fill_median"] = {
                "label":       f"Fill with median ({median_val})",
                "explanation": explain_missing_if_you_fill_median(col, median_val),
            }

        # ── Mode strategy (works for any dtype) ──────────────────────────────
        # Mode = most frequently occurring value. Works for text, numbers, booleans.
        # .mode() can return multiple values if there's a tie — we always take [0]
        # (the first one, which pandas sorts alphabetically/numerically for ties).
        mode_val = df[col].mode()
        if not mode_val.empty:   # guard: mode() returns empty if the column is entirely NaN
            mode_display = mode_val[0]
            # .item() converts numpy scalar (e.g. np.int64) to plain Python int/float
            # so it serialises correctly in JSON
            if hasattr(mode_display, "item"):
                mode_display = mode_display.item()
            strategies["fill_mode"] = {
                "label":       f"Fill with mode ({mode_display})",
                "explanation": explain_missing_if_you_fill_mode(col, mode_display),
            }

        # ── Strategies that apply to every column regardless of dtype ─────────

        strategies["fill_custom"] = {
            "label":       "Fill with a custom value",
            "explanation": explain_missing_if_you_fill_custom(col),
        }
        strategies["drop_rows"] = {
            # Show the exact row count in the label so the user knows the cost upfront
            "label":       f"Drop the {missing_count} rows where this is missing",
            "explanation": explain_missing_if_you_drop_rows(col, missing_count, total_rows),
        }
        strategies["drop_column"] = {
            "label":       f"Drop the entire '{col}' column",
            "explanation": explain_missing_if_you_drop_column(col, pct),
        }
        strategies["do_nothing"] = {
            "label":       "Leave as-is (keep NaN)",
            "explanation": explain_missing_if_you_do_nothing(col),
        }

        columns_with_missing.append({
            "column":        col,
            "dtype":         dtype,
            "missing_count": missing_count,
            "missing_pct":   pct,
            "severity":      severity,
            "explanation":   explain_missing_severity(col, pct),   # xAI description of the severity
            "strategies":    strategies,   # all available fix options for this column
        })

    # ── Row-level analysis ────────────────────────────────────────────────────

    # .isnull().any(axis=1) produces a boolean Series — True for each row that
    # has at least one missing value in ANY of its columns.
    # axis=1 means "check across columns" (axis=0 would check across rows).
    rows_with_missing = int(df.isnull().any(axis=1).sum())
    rows_missing_pct  = round((rows_with_missing / total_rows) * 100, 2)

    # ── Overall summary ───────────────────────────────────────────────────────

    # .isnull().sum() gives missing count per column,
    # .sum() again sums those column totals → total missing cells in the whole DataFrame
    total_missing       = int(df.isnull().sum().sum())
    overall_missing_pct = round((total_missing / total_cells) * 100, 2)

    return {
        "total_rows":           total_rows,
        "total_columns":        len(df.columns),
        "total_missing_cells":  total_missing,
        "overall_missing_pct":  overall_missing_pct,
        "rows_with_missing":    rows_with_missing,
        "rows_missing_pct":     rows_missing_pct,
        "columns_with_missing": columns_with_missing,
        # columns without any missing values — useful for the frontend summary card
        "clean_columns":        len(df.columns) - len(columns_with_missing),
    }


# ── Request body model ────────────────────────────────────────────────────────

class MissingFixPayload(BaseModel):
    """
    Defines the shape of the JSON body for POST /task2/fix_missing.
    Pydantic validates this automatically before our function runs.

    strategy is a dict where:
      - the key   is the column name
      - the value is a dict with:
          "method" — one of: fill_mean, fill_median, fill_mode,
                             fill_custom, drop_rows, drop_column, do_nothing
          "value"  — only required when method is fill_custom

    Example:
        {
            "strategy": {
                "age":    {"method": "fill_median"},
                "salary": {"method": "fill_mean"},
                "city":   {"method": "fill_custom", "value": "Unknown"},
                "notes":  {"method": "drop_column"}
            }
        }

    Why Dict[str, Dict[str, Any]]?
        The outer dict maps column names to their config objects.
        The inner dict has a "method" key (always a string) and an optional
        "value" key that could be a string, int, float, or bool depending on
        what the user wants to fill — so Any is the correct type hint here.
    """
    strategy: Dict[str, Dict[str, Any]]


@task2.post("/task2/fix_missing")
async def fix_missing(payload: MissingFixPayload):
    """
    Applies the user's chosen fix strategy to each column they specified.

    Works on a .copy() — dataset_state.df is never touched until all
    operations are done and at least one succeeded.

    Errors are collected and returned rather than raised, so one bad
    column does not block the fixes applied to all the other columns.
    """
    df      = require_df().copy()   # .copy() is critical — never mutate the live DataFrame until we're sure everything worked
    applied = []                    # success messages — one per change that went through
    errors  = []                    # failure messages — returned to the frontend for display

    for col, config in payload.strategy.items():

        if col not in df.columns:
            errors.append(f'"{col}" not found in the dataset.')
            continue   # skip to the next column — don't crash the whole request

        # config is a dict like {"method": "fill_mean"} or {"method": "fill_custom", "value": "N/A"}
        method = config.get("method", "")   # .get() with a default prevents KeyError if "method" is missing

        try:
            if method == "fill_mean":
                # Mean only makes sense for numbers — reject text/category columns early
                # dtype.kind "i" = integers, "f" = floats
                if df[col].dtype.kind not in ("i", "f"):
                    errors.append(
                        f'Cannot fill "{col}" with mean — it is not a numeric column. '
                        f'Use fill_mode or fill_custom instead.'
                    )
                    continue
                mean_val = df[col].mean()   # compute mean ignoring NaN (pandas default)
                df[col].fillna(mean_val, inplace=True)   # replace every NaN with the mean
                applied.append(
                    f'"{col}": filled {df[col].isna().sum()} NaN(s) '
                    f'with mean = {mean_val:.4f}'   # :.4f = 4 decimal places for display
                )

            elif method == "fill_median":
                if df[col].dtype.kind not in ("i", "f"):
                    errors.append(
                        f'Cannot fill "{col}" with median — it is not a numeric column.'
                    )
                    continue
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                applied.append(
                    f'"{col}": filled NaN(s) with median = {median_val:.4f}'
                )

            elif method == "fill_mode":
                mode_series = df[col].mode()
                if mode_series.empty:
                    # An entirely-NaN column has no mode — guard against this
                    errors.append(
                        f'"{col}": cannot compute mode — column may be all NaN.'
                    )
                    continue
                mode_val = mode_series[0]   # take the first mode value (lowest in case of tie)
                df[col].fillna(mode_val, inplace=True)
                applied.append(f'"{col}": filled NaN(s) with mode = {mode_val}')

            elif method == "fill_custom":
                # The user must provide a "value" key alongside "method"
                custom_val = config.get("value")
                if custom_val is None:
                    errors.append(
                        f'"{col}": fill_custom requires a "value" field in the request.'
                    )
                    continue
                df[col].fillna(custom_val, inplace=True)
                applied.append(
                    f'"{col}": filled NaN(s) with custom value = "{custom_val}"'
                )

            elif method == "drop_rows":
                before = len(df)
                # subset=[col] means only drop rows where THIS column is NaN,
                # not rows where any other column is NaN — targeted removal
                df.dropna(subset=[col], inplace=True)
                # reset_index(drop=True) renumbers the index from 0 after rows are removed.
                # Without this the index has gaps (e.g. 0, 2, 5, 7…) which can cause
                # subtle bugs in later operations that assume a clean 0-based index.
                df.reset_index(drop=True, inplace=True)
                rows_dropped = before - len(df)
                applied.append(
                    f'"{col}": dropped {rows_dropped} row(s) where value was missing. '
                    f'{len(df)} rows remain.'
                )

            elif method == "drop_column":
                # Remove the entire column from the DataFrame
                df.drop(columns=[col], inplace=True)
                applied.append(f'"{col}": column dropped entirely.')

            elif method == "do_nothing":
                # User explicitly chose to leave NaN values in place.
                # We still log it so the applied list reflects the full request.
                applied.append(f'"{col}": left as-is (NaN values kept).')

            else:
                # Unrecognised method string — tell the user the valid options
                errors.append(
                    f'"{col}": unknown method "{method}". '
                    f'Valid options: fill_mean, fill_median, fill_mode, '
                    f'fill_custom, drop_rows, drop_column, do_nothing.'
                )

        except Exception as e:
            # Catch any unexpected runtime error and report it rather than crashing
            errors.append(f'"{col}": unexpected error — {str(e)}')

    # ── Save back ─────────────────────────────────────────────────────────────

    if applied:
        # SNAPSHOT RULE: always save the state that existed BEFORE our changes,
        # so the user can undo back to it if needed.
        # require_df() still returns the original unmodified df at this point
        # because we haven't written to dataset_state.df yet.
        snapshot_store.save(f"Task 2 — missing data: {len(applied)} fix(es) applied", require_df())
        dataset_state.df = df   # only write back if at least one change actually succeeded

    # ── Return ────────────────────────────────────────────────────────────────

    # Recount missing cells on the updated df so the frontend can show progress
    remaining_missing = int(df.isnull().sum().sum())

    return {
        "success":           len(errors) == 0,   # True only if zero errors occurred
        "applied":           applied,
        "errors":            errors,
        "new_shape":         list(df.shape),      # list() so it serialises as a JSON array, not a tuple
        "remaining_missing": remaining_missing,   # lets the frontend show "X missing cells left"
    }
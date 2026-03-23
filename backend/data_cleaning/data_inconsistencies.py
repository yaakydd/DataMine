from fastapi import HTTPException, APIRouter
from pydantic import BaseModel
from routes.dfState import dataset_state              # dfState.py sits at the backend root and saves a copy of the dataset object file
from xAI.inconsistencies_explainer import (           # all explanations live in xAI folder
    explain_spaces_in_name,
    explain_special_chars,
    explain_mixed_casing,
    explain_numeric_as_text,
    explain_dtype_change_bool,
    explain_dtype_change_datetime,
    explain_dtype_change_category,
)
import pandas as pd
import numpy as np
import re
from typing import Optional, Dict


# Task1 handles all the data inconsistencies in the dataset. It is more focused on the column names
task1 = APIRouter()


def require_df() -> pd.DataFrame:
    """
    Called at the start of every endpoint.

    Why: if the user hits a cleaning endpoint before uploading a file,
    dataset_state.df is None and Python would crash with a confusing
    AttributeError deep inside pandas. This catches it early and returns
    a clear message instead.
    """
    if dataset_state.df is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset loaded. Please upload a file first via POST /api/dataset_info"
        )
    return dataset_state.df


def auto_clean_name(name: str) -> str:
    """
    Converts any raw column name into a clean, safe, standardised identifier.

    Rules applied in order — order matters because each step feeds the next:

    Rule 1 — strip()
        "  Age  " → "Age"
        Must be first so leading/trailing spaces don't interfere with Rule 2.

    Rule 2 — replace spaces with underscores
        "First Name" → "First_Name"
        Handled separately from Rule 3 so spaces become _ intentionally,
        not just as a side-effect of symbol removal.

    Rule 3 — replace all non-word characters with underscores
        r"[^\\w]" matches anything that is NOT a letter, digit, or underscore.
        "Price ($)" → "Price___"
        The r"" prefix on the string is required — without it, \\w is an
        unsupported escape sequence (one of the Pylance errors we are fixing).

    Rule 4 — collapse multiple underscores into one
        "Price___" → "Price_"

    Rule 5 — strip leading and trailing underscores
        "_price_" → "price"

    Rule 6 — lowercase everything
        "FirstName" → "firstname"
        Always last — earlier rules don't depend on casing.
    """
    cleaned = name.strip()
    cleaned = re.sub(r"\s+", "_", cleaned)      # r"" prefix fixes the escape warning
    cleaned = re.sub(r"[^\w]", "_", cleaned)    # r"" prefix fixes the escape warning
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = cleaned.strip("_")
    cleaned = cleaned.lower()
    return cleaned


def detect_issues(original: str, dtype: str, sample_values: list) -> list:
    """
    Inspects one column and returns a list of plain-English problems.

    The actual explanation strings come from xai/explainer.py.
    This function handles the detection logic only — it decides whether
    an issue exists, then calls the right explainer function to describe it.

    Four checks:
      1. Spaces in the name
      2. Special characters ($, %, ! etc.)
      3. Mixed casing
      4. Numeric data disguised as text (object dtype)
    """
    issues = []

    # Check 1: spaces
    if re.search(r"\s", original):
        issues.append(explain_spaces_in_name(original))

    # Check 2: special characters
    bad_chars = set(re.findall(r"[^\w\s]", original))
    if bad_chars:
        issues.append(explain_special_chars(original, bad_chars))

    # Check 3: mixed casing
    # .isupper() guard avoids flagging ALL-CAPS abbreviations like "ID" or "DOB"
    if original != original.lower() and not original.isupper():
        issues.append(explain_mixed_casing(original))

    # Check 4: numbers stored as text
    # Strip common formatting ($, commas, %) before testing so "$4,500" is
    # correctly identified as a number stored as a string
    if dtype == "object" and sample_values:
        numeric_count = 0
        for v in sample_values:
            try:
                float(
                    str(v)
                    .replace(",", "")
                    .replace("$", "")
                    .replace("%", "")
                    .strip()
                )
                numeric_count += 1
            except ValueError:
                pass
        if numeric_count == len(sample_values):
            issues.append(explain_numeric_as_text(original))

    return issues


# =============================================================================
# TASK 1 — CORRECT DATA INCONSISTENCIES
#
# GET  /task1/columns_info    → scan and report every column, never writes
# POST /task1/update_columns  → apply renames and dtype changes, writes once
# =============================================================================

@task1.get("/task1/columns_info")
async def get_columns_info():
    """
    Read-only scan of every column in the loaded DataFrame.

    Returns for each column:
      - original_name:    the name exactly as it is in the data
      - suggested_name:   the auto-cleaned version
      - needs_fix:        True if suggested differs from original
      - current_dtype:    e.g. "int64", "object", "float64"
      - available_dtypes: the full list to populate the UI dropdown
      - sample_values:    up to 3 real non-null values so the user can
                          see what the data actually looks like
      - issues_detected:  plain-English list from xai/explainer.py

    This endpoint never modifies dataset_state.df. Safe to call repeatedly.
    """
    df = require_df()
    columns_report = []

    for col in df.columns:
        suggested = auto_clean_name(col)

        # .dropna() skips missing values so samples are always real data
        # .item() converts np.int64, np.float32 etc. to plain Python types
        # so that FastAPI's JSON serialiser never encounters unknown types
        sample_values = [
            v.item() if hasattr(v, "item") else v
            for v in df[col].dropna().head(3).tolist()
        ]

        issues = detect_issues(col, str(df[col].dtype), sample_values)

        columns_report.append({
            "original_name":    col,
            "suggested_name":   suggested,
            "needs_fix":        col != suggested,
            "current_dtype":    str(df[col].dtype),
            "available_dtypes": [
                "int64",
                "float64",
                "object",
                "bool",
                "datetime64[ns]",
                "category",
            ],
            "sample_values":    sample_values,
            "issues_detected":  issues,
        })

    return {
        "total_columns": len(df.columns),
        "columns_with_issues": sum(
            1 for c in columns_report
            if c["needs_fix"] or c["issues_detected"]
        ),
        "columns": columns_report,
    }


# ── Request body model ────────────────────────────────────────────────────────

class ColumnUpdatePayload(BaseModel):
    """
    auto_fix_all:
        When True, every column is renamed using auto_clean_name() at once.
        The renames dict is ignored when this is True.

    renames:
        Manual renames. Only used when auto_fix_all is False.
        e.g. {"First Name": "first_name", "Revenue ($)": "revenue_usd"}

    dtype_changes:
        Dtype conversions to apply, independent of renames.
        e.g. {"age": "int64", "signup_date": "datetime64[ns]"}
    """
    auto_fix_all:  Optional[bool]           = False
    renames:       Optional[Dict[str, str]] = {}
    dtype_changes: Optional[Dict[str, str]] = {}


@task1.post("/task1/update_columns")
async def update_columns(payload: ColumnUpdatePayload):
    """
    Applies the user's column changes to the DataFrame.

    Works on a .copy() throughout. dataset_state.df is only overwritten
    at the very end, and only if at least one change succeeded.
    This means a failed conversion never corrupts the stored data.

    Errors are collected and returned rather than raised so that one
    bad conversion does not block all the other valid changes.
    """
    df      = require_df().copy()
    applied = []
    errors  = []

    # ── Renames ───────────────────────────────────────────────────────────────

    if payload.auto_fix_all:
        rename_map = {
            col: auto_clean_name(col)
            for col in df.columns
            if col != auto_clean_name(col)
        }
        if rename_map:
            df.rename(columns=rename_map, inplace=True)
            applied.append(
                f"Auto-fixed {len(rename_map)} column name(s): "
                f"{list(rename_map.keys())} → {list(rename_map.values())}"
            )
        else:
            applied.append("All column names are already clean — nothing to fix.")

    elif payload.renames:
        valid_renames = {}
        for old_name, new_name in payload.renames.items():
            if old_name not in df.columns:
                errors.append(f'Rename failed: "{old_name}" does not exist in the dataset.')
                continue
            if not new_name or not new_name.strip():
                errors.append(f'Rename failed: new name for "{old_name}" is empty.')
                continue
            valid_renames[old_name] = new_name.strip()

        if valid_renames:
            df.rename(columns=valid_renames, inplace=True)
            applied.append(f"Renamed {len(valid_renames)} column(s).")

    # ── Dtype changes ─────────────────────────────────────────────────────────

    for col_name, new_dtype in (payload.dtype_changes or {}).items():

        if col_name not in df.columns:
            errors.append(
                f'Dtype change failed: "{col_name}" not found. '
                f'If you renamed it in this same request, use the new name.'
            )
            continue

        current_dtype = str(df[col_name].dtype)

        if current_dtype == new_dtype:
            continue    # already correct, skip silently

        try:
            if new_dtype == "datetime64[ns]":
                # pd.to_datetime() handles many date string formats automatically.
                # errors="coerce" turns unparseable values into NaT
                # instead of crashing the whole operation.
                df[col_name] = pd.to_datetime(df[col_name], errors="coerce")
                applied.append(
                    f'"{col_name}": {current_dtype} → datetime64[ns]. '
                    + explain_dtype_change_datetime()
                )

            elif new_dtype == "bool":
                # FIX for Pylance error on line 377:
                # Passing the string "bool" to .astype() confuses Pylance's type checker.
                # Passing the actual Python type `bool` resolves it.
                # We also handle text values manually first because .astype(bool)
                # converts any non-empty string to True — so "False" would wrongly
                # become True without this mapping step.
                df[col_name] = df[col_name].map(
                    lambda x:
                        True  if str(x).strip().lower() in ("true",  "1", "yes") else
                        False if str(x).strip().lower() in ("false", "0", "no")  else x
                ).astype(bool)   # <-- bool type, not the string "bool"
                applied.append(
                    f'"{col_name}": {current_dtype} → bool. '
                    + explain_dtype_change_bool()
                )

            elif new_dtype == "category":
                df[col_name] = df[col_name].astype("category")
                applied.append(
                    f'"{col_name}": {current_dtype} → category. '
                    + explain_dtype_change_category()
                )

            elif new_dtype == "int64":
                df[col_name] = df[col_name].astype(np.int64)
                applied.append(f'"{col_name}": {current_dtype} → int64.')

            elif new_dtype == "float64":
                df[col_name] = df[col_name].astype(np.float64)
                applied.append(f'"{col_name}": {current_dtype} → float64.')

            else:
                # object and any other string dtype
                df[col_name] = df[col_name].astype(str)
                applied.append(f'"{col_name}": {current_dtype} → {new_dtype}.')

        except (ValueError, TypeError) as e:
            errors.append(
                f'Could not convert "{col_name}" to {new_dtype}. '
                f'Reason: {str(e)}. '
                f'Tip: check this column for unexpected text or mixed values first.'
            )

    # ── Save back ─────────────────────────────────────────────────────────────

    if applied:
        dataset_state.df = df

    # ── Return ────────────────────────────────────────────────────────────────

    return {
        "success": len(errors) == 0,
        "applied": applied,
        "errors":  errors,
        "updated_columns": [
            {"name": col, "dtype": str(df[col].dtype)}
            for col in df.columns
        ],
    }
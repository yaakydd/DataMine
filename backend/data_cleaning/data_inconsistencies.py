from fastapi import HTTPException, APIRouter
from pydantic import BaseModel
from routes.dfState import dataset_state              # dfState.py sits at the backend root
from xAI.inconsistencies_explainer import (                   # all explanations live in xai/explainer.py
    explain_spaces_in_name,
    explain_special_chars,
    explain_mixed_casing,
    explain_numeric_as_text,
    explain_dtype_change_bool,
    explain_dtype_change_datetime,
    explain_dtype_change_category,
    explain_text_value_cleaning,
    explain_categorical_harmonisation,
    explain_hidden_boolean,
    explain_likely_id_column,
    explain_high_cardinality,
    explain_text_clean_result,
)
import pandas as pd
import numpy as np
import re
from typing import Optional, Dict

task1 = APIRouter()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

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
        '[^\\w]' matches anything that is NOT a letter, digit, or underscore.
        "Price ($)" → "Price___"
        The r prefix on regex strings is required — without it, \\w is an
        unsupported escape sequence that Pylance flags as a warning.

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


# =============================================================================
# TASK 1 UPGRADE A — DATA TYPE INFERENCE
#
# GET /task1/infer_types
#     Scans every column and automatically detects:
#       - Hidden boolean columns  (only 0/1 stored as int/float)
#       - Likely ID columns       (all unique, name suggests identifier)
#       - High cardinality text   (almost every value is unique)
#     Returns suggestions with xAI explanations.
#     Never modifies the data.
# =============================================================================

@task1.get("/task1/infer_types")
async def infer_types():
    """
    Analyses every column for hidden type problems that the dtype alone
    does not reveal.

    Three checks:

    Hidden booleans:
        A column with dtype int64 or float64 that only ever contains
        the values 0 and 1 is almost certainly a boolean flag.
        df[col].nunique() == 2 and set of unique values == {0, 1}.

    Likely ID columns:
        A column where every value is unique (nunique == total rows)
        AND whose name contains a known ID keyword.
        These should never be averaged, summed, or used in models.

    High cardinality text:
        A text (object dtype) column where more than 50% of values
        are unique. Grouping by such a column produces meaningless
        groups of size 1.
    """
    df         = require_df()
    total_rows = len(df)
    findings   = []

    id_keywords = ("id", "key", "code", "uuid", "ref", "reference", "number")

    for col in df.columns:
        dtype      = str(df[col].dtype)
        unique_count = int(df[col].nunique())

        # ── Check 1: hidden boolean ───────────────────────────────────────────
        if df[col].dtype.kind in ("i", "f"):
            non_null = df[col].dropna()
            unique_vals = set(non_null.unique())
            if unique_vals <= {0, 1} and len(unique_vals) == 2:
                findings.append({
                    "column":      col,
                    "finding":     "hidden_boolean",
                    "current_dtype": dtype,
                    "suggested_dtype": "bool",
                    "explanation": explain_hidden_boolean(col),
                })

        # ── Check 2: likely ID column ─────────────────────────────────────────
        # All values unique AND name looks like an identifier
        if unique_count == total_rows and any(
            kw in col.lower() for kw in id_keywords
        ):
            findings.append({
                "column":      col,
                "finding":     "likely_id",
                "current_dtype": dtype,
                "unique_count": unique_count,
                "explanation": explain_likely_id_column(col),
            })

        # ── Check 3: high cardinality text ────────────────────────────────────
        # Object dtype AND more than 50% of values are unique
        if dtype == "object" and total_rows > 0:
            cardinality_pct = unique_count / total_rows
            if cardinality_pct > 0.5 and unique_count > 10:
                findings.append({
                    "column":       col,
                    "finding":      "high_cardinality",
                    "current_dtype": dtype,
                    "unique_count": unique_count,
                    "total_rows":   total_rows,
                    "explanation":  explain_high_cardinality(
                        col, unique_count, total_rows
                    ),
                })

    return {
        "total_columns": len(df.columns),
        "findings_count": len(findings),
        "findings": findings,
        "message": (
            "No type inference issues found."
            if not findings else
            f"{len(findings)} column(s) have hidden type issues worth reviewing."
        ),
    }


# =============================================================================
# TASK 1 UPGRADE B — TEXT VALUE CLEANING
#
# GET  /task1/text_value_info
#      Scans all text (object dtype) columns and reports formatting
#      issues found in the actual cell values — not column names.
#      Reports: whitespace issues, mixed casing, special characters.
#
# POST /task1/clean_text_values
#      Applies the chosen cleaning method to each specified column.
#      Methods: trim, lowercase, uppercase, titlecase, remove_special, all
# =============================================================================

@task1.get("/task1/text_value_info")
async def get_text_value_info():
    """
    Scans every text column and detects formatting problems in the
    actual cell values, not in column names.

    For each text column it checks:

    Whitespace issues:
        Values with leading or trailing spaces — ' London' or 'London '.
        These are invisible in most displays but break exact matching,
        GROUP BY, and merge/join operations completely.
        Detected by: value != value.strip()

    Mixed casing:
        The same word appears in multiple casing variants —
        'london', 'London', 'LONDON' are all in the same column.
        Detected by comparing value.lower() counts vs raw value counts.

    Special characters:
        Values containing symbols like $, %, !, @, # that are unlikely
        to be intentional in a category column.

    Returns a report per column with counts and sample bad values
    so the user can see exactly what is wrong before cleaning.
    """
    df      = require_df()
    reports = []

    # Only check object (text) columns — numeric cleaning is handled elsewhere
    text_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in text_cols:
        series      = df[col].dropna().astype(str)
        total_vals  = len(series)

        if total_vals == 0:
            continue

        # ── Whitespace issues ─────────────────────────────────────────────────
        # A value has a whitespace issue if stripping it changes it
        has_whitespace  = series != series.str.strip()
        whitespace_count = int(has_whitespace.sum())
        whitespace_samples = series[has_whitespace].head(3).tolist()

        # ── Mixed casing ──────────────────────────────────────────────────────
        # Lower all values and compare unique counts.
        # If lowercased unique count < raw unique count, casing variants exist.
        raw_unique    = series.nunique()
        lower_unique  = series.str.lower().nunique()
        casing_issue  = lower_unique < raw_unique
        casing_count  = raw_unique - lower_unique

        # Find actual variant groups — values that are the same when lowercased
        # but differ in the original — limited to top 3 groups for readability
        casing_samples = {}
        if casing_issue:
            grouped = series.groupby(series.str.lower())
            for canonical, group in grouped:
                variants = group.unique().tolist()
                if len(variants) > 1:
                    casing_samples[canonical] = variants
                    if len(casing_samples) >= 3:
                        break

        # ── Special characters ────────────────────────────────────────────────
        import re as _re
        has_special   = series.str.contains(r"[^\w\s]", regex=True, na=False)
        special_count = int(has_special.sum())
        special_samples = series[has_special].head(3).tolist()

        # Only include this column in the report if at least one issue exists
        has_any_issue = (
            whitespace_count > 0 or casing_issue or special_count > 0
        )

        if has_any_issue:
            reports.append({
                "column":      col,
                "total_values": total_vals,

                "whitespace": {
                    "count":   whitespace_count,
                    "pct":     round((whitespace_count / total_vals) * 100, 2),
                    "samples": whitespace_samples,
                },
                "mixed_casing": {
                    "has_issue":      casing_issue,
                    "variant_groups": casing_count,
                    "samples":        casing_samples,
                },
                "special_chars": {
                    "count":   special_count,
                    "pct":     round((special_count / total_vals) * 100, 2),
                    "samples": special_samples,
                },

                # Pre-built explanation for the xAI panel
                "explanation": explain_text_value_cleaning(
                    col,
                    whitespace_count + (casing_count if casing_issue else 0) + special_count
                ),

                # Available cleaning methods for this column's dropdown
                "available_methods": {
                    "trim":           "Remove leading and trailing whitespace from all values",
                    "lowercase":      "Convert all values to lowercase",
                    "uppercase":      "Convert all values to uppercase",
                    "titlecase":      "Capitalise the first letter of each word",
                    "remove_special": "Remove special characters — keep only letters, digits, spaces",
                    "all":            "Apply all of the above in one step (trim + lowercase + remove special)",
                },
            })

    return {
        "text_columns_scanned": len(text_cols),
        "columns_with_issues":  len(reports),
        "reports":              reports,
    }


class TextCleanPayload(BaseModel):
    """
    strategy is a dict mapping column name → cleaning method.

    Valid methods:
      trim           — strip whitespace from both ends of every value
      lowercase      — convert all values to lowercase
      uppercase      — convert all values to uppercase
      titlecase      — capitalise first letter of each word
      remove_special — remove non-alphanumeric characters (keeps spaces)
      all            — apply trim + lowercase + remove_special together

    Example:
        {
            "strategy": {
                "city":    "all",
                "gender":  "lowercase",
                "country": "titlecase"
            }
        }
    """
    strategy: Dict[str, str]


@task1.post("/task1/clean_text_values")
async def clean_text_values(payload: TextCleanPayload):
    """
    Applies the chosen text cleaning method to each specified column.

    All operations use pandas vectorised string methods (.str accessor)
    which apply the operation to every value in the column at once —
    much faster than looping row by row.

    trim:
        series.str.strip() removes spaces, tabs, and newlines from
        both ends of every string value.

    lowercase / uppercase / titlecase:
        series.str.lower() / .str.upper() / .str.title()
        Applied after trimming so casing is consistent.

    remove_special:
        series.str.replace(r'[^\\w\\s]', '', regex=True)
        Keeps letters, digits, underscores, and spaces.
        Removes everything else: $, %, !, @, #, (, ), etc.

    all:
        Applies trim → lowercase → remove_special in that order.
        Order matters: trim first so edge spaces don't survive,
        lowercase before remove_special so casing is uniform.

    NaN values are preserved throughout — we only clean real values,
    not missing ones. That is Task 2's responsibility.
    """
    df      = require_df().copy()
    applied = []
    errors  = []

    import re as _re

    for col, method in payload.strategy.items():

        if col not in df.columns:
            errors.append(f'"{col}" not found in the dataset.')
            continue

        if df[col].dtype.kind not in ("O", "S", "U"):
            # Check for object/string dtype
            if str(df[col].dtype) not in ("object", "string", "category"):
                errors.append(
                    f'"{col}" is not a text column (dtype: {df[col].dtype}). '
                    "Text cleaning only applies to object/string columns."
                )
                continue

        try:
            # Convert to string for .str accessor, preserve NaN
            original_nulls = df[col].isna()

            if method == "trim":
                df[col] = df[col].str.strip()

            elif method == "lowercase":
                df[col] = df[col].str.strip().str.lower()

            elif method == "uppercase":
                df[col] = df[col].str.strip().str.upper()

            elif method == "titlecase":
                df[col] = df[col].str.strip().str.title()

            elif method == "remove_special":
                df[col] = (
                    df[col]
                    .str.strip()
                    .str.replace(r"[^\w\s]", "", regex=True)
                    .str.strip()     # strip again — removing chars can leave edge spaces
                )

            elif method == "all":
                df[col] = (
                    df[col]
                    .str.strip()
                    .str.lower()
                    .str.replace(r"[^\w\s]", "", regex=True)
                    .str.strip()
                )

            else:
                errors.append(
                    f'"{col}": unknown method "{method}". '
                    "Valid: trim, lowercase, uppercase, titlecase, remove_special, all."
                )
                continue

            # Restore NaN positions — .str operations can turn NaN into "nan" string
            df.loc[original_nulls, col] = np.nan

            applied.append(explain_text_clean_result(col, method))

        except Exception as e:
            errors.append(f'"{col}": unexpected error — {str(e)}')

    if applied:
        dataset_state.df = df

    return {
        "success": len(errors) == 0,
        "applied": applied,
        "errors":  errors,
        "new_shape": list(df.shape),
    }


# =============================================================================
# TASK 1 UPGRADE C — CATEGORICAL HARMONISATION
#
# GET  /task1/category_info
#      Detects columns where the same category appears in multiple
#      spelling/casing variants. Shows the variant groups.
#
# POST /task1/harmonise_categories
#      Lets the user define a mapping of variant → canonical form
#      and applies it to the specified column.
# =============================================================================

@task1.get("/task1/category_info")
async def get_category_info():
    """
    Detects inconsistent category values in low-cardinality text columns.

    What we look for:
        Values that are identical when lowercased and stripped but differ
        in their original form. For example:
          "Male", "male", "MALE", "M ", " m" — all the same concept,
          but stored as 5 different strings.

    We only check columns where the number of unique values is low
    (under 50) because high-cardinality columns are handled by the
    text cleaning step, not harmonisation.

    For each group of variants we show:
        - The variants found
        - How many rows contain each variant
        - A suggested canonical form (the most frequent variant)
        - The xAI explanation
    """
    df      = require_df()
    reports = []

    text_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in text_cols:
        series = df[col].dropna().astype(str)

        # Skip high-cardinality columns — harmonisation is for categories
        if series.nunique() > 50:
            continue

        # Group values by their normalised form (lowercase + stripped)
        # If any group has more than one original variant, we have a problem
        variant_groups = {}
        for val in series.unique():
            normalised = val.strip().lower()
            if normalised not in variant_groups:
                variant_groups[normalised] = []
            variant_groups[normalised].append(val)

        # Only report groups that have more than one variant
        problem_groups = {
            k: v for k, v in variant_groups.items()
            if len(v) > 1
        }

        if not problem_groups:
            continue

        # For each problem group, count how many rows have each variant
        # and suggest the most frequent one as the canonical form
        group_details = []
        for normalised, variants in problem_groups.items():
            counts = {
                v: int((series == v).sum())
                for v in variants
            }
            # The canonical suggestion is the most frequently occurring variant
            suggested_canonical = max(counts.keys(), key=lambda k: counts[k])
            group_details.append({
                "variants":          variants,
                "counts":            counts,
                "suggested_canonical": suggested_canonical,
                "normalised_key":    normalised,
            })

        reports.append({
            "column":         col,
            "problem_groups": group_details,
            "total_affected": sum(
                sum(g["counts"].values()) - max(g["counts"].values())
                for g in group_details
            ),
            "explanation": explain_categorical_harmonisation(
                col,
                {g["suggested_canonical"]: g["variants"] for g in group_details}
            ),
        })

    return {
        "columns_checked":     len(text_cols),
        "columns_with_issues": len(reports),
        "reports":             reports,
    }


class HarmonisePayload(BaseModel):
    """
    column:  the column to harmonise
    mapping: a dict of variant → canonical form

    Example — standardise gender values:
        {
            "column": "gender",
            "mapping": {
                "M":    "Male",
                "m":    "Male",
                "male": "Male",
                "MALE": "Male",
                "F":    "Female",
                "f":    "Female",
                "female": "Female"
            }
        }

    Any value in the column that matches a key in the mapping
    will be replaced with the corresponding value.
    Values not in the mapping are left unchanged.
    """
    column:  str
    mapping: Dict[str, str]


@task1.post("/task1/harmonise_categories")
async def harmonise_categories(payload: HarmonisePayload):
    """
    Applies the user's variant → canonical mapping to a column.

    Uses pandas .map() with na_action="ignore" so NaN values are
    preserved and not accidentally mapped to something else.

    For values not in the mapping, we use .fillna() with the original
    value — meaning unmapped values stay exactly as they were.

    This is safer than .replace() because .map() only touches values
    that are explicitly in the mapping, leaving everything else alone.
    """
    df  = require_df().copy()
    col = payload.column

    if col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f'Column "{col}" not found in the dataset.'
        )

    original_uniques = int(df[col].nunique())

    # Map variants to canonical forms.
    # For values not in the mapping, keep the original value.
    # .map() returns NaN for unmapped values, so we fill those back
    # with the original values using .combine_first()
    mapped = df[col].map(payload.mapping)
    df[col] = mapped.combine_first(df[col])

    new_uniques = int(df[col].nunique())
    reduced_by  = original_uniques - new_uniques

    dataset_state.df = df

    return {
        "success":          True,
        "column":           col,
        "variants_before":  original_uniques,
        "variants_after":   new_uniques,
        "reduced_by":       reduced_by,
        "applied": [
            f'"{col}": harmonised {len(payload.mapping)} variant(s) '
            f'into {len(set(payload.mapping.values()))} canonical form(s). '
            f'Unique values reduced from {original_uniques} to {new_uniques}.'
        ],
    }
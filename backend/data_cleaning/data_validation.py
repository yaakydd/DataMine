# =============================================================================
# validation.py
#
# This file handles Task 6 of DataMine: validating column values against
# rules, detecting cross-column logical inconsistencies, and enforcing
# string pattern formats.
#
# It defines 10 endpoints across 4 sections:
#
#   Section 1 — Rule-based validation (original)
#     GET  /task6/validation_info    → scan columns, auto-suggest rules,
#                                      return all available rule types.
#     POST /task6/run_validation     → run a set of rules and report violations.
#     POST /task6/fix_violations     → fix violations in one column at a time.
#
#   Section 2 — Date range validation (Upgrade 1)
#     GET  /task6/date_range_info    → scan datetime and year columns for
#                                      values outside the expected year range.
#     POST /task6/fix_date_range     → set out-of-range values to NaT/NaN.
#
#   Section 3 — Cross-column validation (Upgrade 2)
#     POST /task6/cross_column_check → check logical consistency between
#                                      two columns (e.g. end > start).
#     POST /task6/fix_cross_column   → set the violating column's bad
#                                      values to NaN/NaT.
#
#   Section 4 — String pattern validation (Upgrade 3)
#     GET  /task6/pattern_info       → auto-detect which text columns match
#                                      a known pattern (email, phone, etc.).
#     POST /task6/check_pattern      → validate a column against a named or
#                                      custom regex pattern.
# =============================================================================


# HTTPException → return a clean error + HTTP status code instead of a Python crash
# APIRouter     → groups all Task 6 endpoints under one object registered in main.py
from fastapi import HTTPException, APIRouter

# BaseModel → Pydantic base class that auto-validates incoming JSON request bodies
from pydantic import BaseModel

# Shared singleton holding the current DataFrame in memory across all routers
from State.dfState import dataset_state

# Each function returns a plain-English explanation string for the xAI panel.
from xAI.validation import (
    explain_what_validation_is,             # general intro to data validation
    explain_rule_no_future_year,            # what a future-year violation means
    explain_rule_no_past_year,              # what a past-year violation means
    explain_rule_no_negative,               # what a negative value violation means
    explain_rule_percentage,                # what an out-of-range percentage means
    explain_rule_custom_range,              # what a custom range violation means
    explain_rule_no_nulls,                  # what a required-field violation means
    explain_fix_replace,                    # what the replace fix method does
    explain_fix_set_null,                   # what the set_null fix method does
    explain_fix_drop_rows,                  # what the drop_rows fix method does
    # Upgrade explanations
    explain_what_cross_column_validation_is,  # intro to cross-column checks
    explain_date_range_violation,             # what an out-of-range date means
    explain_end_before_start_violation,       # what end < start means
    explain_age_birth_year_violation,         # what age vs birth year mismatch means
    explain_pattern_violation,                # what a regex pattern violation means
    explain_fix_set_null_cross,               # what nulling a cross-column violation does
)

import pandas as pd                  # main data manipulation library
import numpy as np                   # used for np.nan, dtype detection, and type conversion
import datetime                      # used to get the current year
import re as _re                     # regular expressions for pattern validation
                                     # imported as _re to distinguish from any local 're' variables

# Snapshot store saves a copy of the DataFrame before every write.
# Powers the undo/rollback system — up to 20 snapshots kept.
from State.snapshotState import snapshot_store

# Type hints used in function signatures and Pydantic models
from typing import Dict, Any, Optional, List

# All Task 6 endpoints are attached to this router.
# main.py registers it with: app.include_router(task6, prefix="/api")
task6 = APIRouter()


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
            status_code=400,   # 400 = Bad Request — called before uploading a file
            detail="No dataset loaded. Please upload a file first via POST /api/dataset_info"
        )
    return dataset_state.df


def safe_json(obj: Any) -> Any:
    """
    Recursively converts numpy and pandas types to plain Python equivalents
    so FastAPI's JSON serialiser never encounters a type it doesn't know.

    Why this is needed:
        Pulling values out of a DataFrame gives numpy types like np.int64,
        np.float64, np.bool_. Python's built-in json module does not know
        these types and will raise a TypeError.
        safe_json() walks every level of the object and converts each one.

    Handles:
        dict        → recurse into each value
        list        → recurse into each item
        np.integer  → int()
        np.floating → float(), or None if NaN/Inf (JSON has no representation)
        np.bool_    → bool()
        np.ndarray  → convert to list first, then recurse
        pd.NA/NaT   → None (JSON null for any missing value)
        anything else → return unchanged (already a plain Python type)
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
# PRE-COMPILED REGEX PATTERNS
#
# Why pre-compile?
#   Every time a regex pattern is used, Python must compile it from a string
#   into an internal pattern object before matching.
#   If we compiled inside the endpoint function, the same regex would be
#   recompiled on every API request — wasteful for frequently-called endpoints.
#   Compiling at module load (once, when the server starts) means every
#   request reuses the already-compiled pattern object — much faster.
# =============================================================================

_PATTERNS: Dict[str, Any] = {
    # Standard email format: local@domain.tld
    # [a-zA-Z0-9_.+-]+ → local part (letters, digits, dots, underscores, plus, hyphen)
    # @ → the literal @ separator
    # [a-zA-Z0-9-]+    → domain name
    # \.               → literal dot (escaped because . means "any char" in regex)
    # [a-zA-Z0-9-.]+   → TLD (allows multi-part TLDs like .co.uk)
    "email": _re.compile(
        r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    ),

    # US phone number — accepts many common formats:
    # (555) 123-4567, 555-123-4567, 5551234567, +1 555 123 4567
    # (\+1[\s.-]?)? → optional +1 country code
    # \(?\d{3}\)?    → area code, optionally wrapped in parentheses
    # [\s.-]?        → optional separator (space, dot, or hyphen)
    # \d{3}[\s.-]?\d{4} → 7-digit local number with optional separator
    "phone_us": _re.compile(
        r"^(\+1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}$"
    ),

    # US ZIP code: 5 digits, or 5 digits + hyphen + 4 digits (ZIP+4)
    # e.g. "10001" or "10001-1234"
    "postal_code_us": _re.compile(
        r"^\d{5}(-\d{4})?$"
    ),

    # UK postcode: e.g. "SW1A 2AA", "M1 1AE", "B1 1BB"
    # [A-Z]{1,2}  → 1 or 2 letter area code
    # \d          → district number
    # [A-Z\d]?    → optional sub-district letter or number
    # \s*         → optional space in the middle
    # \d[A-Z]{2}  → sector digit + 2 unit letters
    # re.IGNORECASE → accept lowercase too
    "postal_code_uk": _re.compile(
        r"^[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}$",
        _re.IGNORECASE,
    ),

    # Web URL starting with http:// or https://
    # [^\s/$.?#]  → at least one valid character after the scheme
    # [^\s]*      → rest of the URL (no spaces allowed)
    # re.IGNORECASE → accept HTTP:// and HTTPS://
    "url": _re.compile(
        r"^https?://[^\s/$.?#].[^\s]*$",
        _re.IGNORECASE,
    ),

    # ISO 8601 date format: YYYY-MM-DD (e.g. "2024-01-15")
    # \d{4} → 4-digit year, \d{2} → 2-digit month, \d{2} → 2-digit day
    # Note: this checks FORMAT only — it doesn't verify the date is real
    # (e.g. "2024-13-99" would pass the regex but fail actual date parsing)
    "date_iso": _re.compile(
        r"^\d{4}-\d{2}-\d{2}$"
    ),

    # Whole integer stored as a string, with optional leading minus for negatives
    # e.g. "42", "-7", "1000"
    "integer_string": _re.compile(
        r"^-?\d+$"
    ),

    # Decimal number stored as a string, with optional minus and decimal part
    # e.g. "3.14", "-0.5", "100", "1.0"
    "decimal_string": _re.compile(
        r"^-?\d+(\.\d+)?$"
    ),
}


# =============================================================================
# AUTO-DETECTION LOGIC
# =============================================================================

# Capture the current year once at module load — used as the upper bound
# for year validation rules and compared against year-like column values.
CURRENT_YEAR = datetime.datetime.now().year


def detect_auto_rules(df: pd.DataFrame) -> list:
    """
    Scans all columns and suggests validation rules based on column
    names and dtypes. The user sees these as pre-filled suggestions
    in the validation info endpoint.

    Why keyword-based detection?
        We can't know the intent of a column just from its values —
        a column named "score" with values 0–100 could be a percentage OR
        a raw score on a different scale. Column names give us semantic hints
        that help us make reasonable suggestions. The user can accept or reject them.

    Detection categories:
        year_keywords  → suggest no_future_year + custom_range(1900, CURRENT_YEAR)
        age_keywords   → suggest custom_range(0, 150)
        pct_keywords   → suggest percentage rule (0–100)
        money_keywords → suggest no_negative (prices can't be negative)
        count_keywords → suggest no_negative (counts can't be negative)

    Only suggests rules for NUMERIC columns — a column named "age" that stores
    text like "young/middle/old" doesn't need numeric range validation.

    Note: the original code had an unused 'dtype' variable that was removed here.
    """
    suggestions    = []

    # Keyword lists — all lowercase so we can match against col.lower()
    year_keywords  = ("year", "yr", "born", "dob", "date_of_birth")
    age_keywords   = ("age",)
    pct_keywords   = ("pct", "percent", "rate", "ratio", "score")
    money_keywords = ("price", "cost", "salary", "amount", "revenue",
                      "income", "wage", "fee", "pay", "earning")
    count_keywords = ("count", "qty", "quantity", "num", "number", "total")

    for col in df.columns:
        col_lower  = col.lower()   # lowercase once — reused across all keyword checks below
        # dtype.kind "i" = integer, "f" = float — both are numeric
        is_numeric = df[col].dtype.kind in ("i", "f")

        if any(kw in col_lower for kw in year_keywords) and is_numeric:
            # Year columns get TWO suggestions: no future year AND a full range check
            # The range check catches both future years AND impossibly old years (pre-1900)
            suggestions.append({
                "column":      col,
                "rule":        "no_future_year",
                "description": f'"{col}" looks like a year column — should not exceed {CURRENT_YEAR}.',
                "auto":        True,   # "auto" flags these as system-generated, not user-created
            })
            suggestions.append({
                "column":      col,
                "rule":        "custom_range",
                "min":         1900,
                "max":         CURRENT_YEAR,
                "description": f'"{col}" should be between 1900 and {CURRENT_YEAR}.',
                "auto":        True,
            })

        elif any(kw in col_lower for kw in age_keywords) and is_numeric:
            # 0–150 covers all realistic human ages with a generous upper bound
            suggestions.append({
                "column":      col,
                "rule":        "custom_range",
                "min":         0,
                "max":         150,
                "description": f'"{col}" looks like an age column — should be between 0 and 150.',
                "auto":        True,
            })

        elif any(kw in col_lower for kw in pct_keywords) and is_numeric:
            # Percentages should never fall outside [0, 100]
            suggestions.append({
                "column":      col,
                "rule":        "percentage",
                "description": f'"{col}" looks like a percentage — should be between 0 and 100.',
                "auto":        True,
            })

        elif any(kw in col_lower for kw in money_keywords) and is_numeric:
            # Monetary values are almost never negative in clean business data
            suggestions.append({
                "column":      col,
                "rule":        "no_negative",
                "description": f'"{col}" looks like a monetary value — should not be negative.',
                "auto":        True,
            })

        elif any(kw in col_lower for kw in count_keywords) and is_numeric:
            # Counts (quantities, totals) cannot logically be negative
            suggestions.append({
                "column":      col,
                "rule":        "no_negative",
                "description": f'"{col}" looks like a count — should not be negative.',
                "auto":        True,
            })

    return suggestions


def run_single_rule(df: pd.DataFrame, col: str, rule: str, params: dict) -> dict:
    """
    Runs one validation rule on one column and returns the result.
    READ-ONLY — this function never modifies the DataFrame.

    Why a separate function?
        Both run_validation (which checks multiple rules at once) and
        fix_violations (which needs to rerun the rule to find current violations
        before fixing them) need the same detection logic.
        Extracting it here avoids duplicating the detection code in two places.

    Returns a dict with:
        passed            → True if zero violations found
        violation_count   → how many rows broke the rule
        violation_sample  → up to 5 example bad values (safe_json'd)
        violation_indices → DataFrame index labels of the violating rows
                            (needed by fix_violations to target the right rows)
        explanation       → xAI plain-English description
        error             → only present if the rule name was unrecognised
    """
    series = df[col]   # reference to the column — no copy needed (read-only)

    if rule == "no_future_year":
        # Flag any value greater than the current year.
        # series.notna() guard ensures we only check actual values, not NaN.
        # Without it, NaN > CURRENT_YEAR would raise a comparison warning.
        mask        = series.notna() & (series > CURRENT_YEAR)
        count       = int(mask.sum())
        sample      = safe_json(series[mask].head(5).tolist())
        indices     = series[mask].index.tolist()
        passed      = count == 0
        explanation = explain_rule_no_future_year(col, count, CURRENT_YEAR)

    elif rule == "no_past_year":
        # Flag any value below the minimum acceptable year (default 1900).
        # params.get() with a default avoids KeyError if params is empty.
        min_year    = int(params.get("min_year", 1900))
        mask        = series.notna() & (series < min_year)
        count       = int(mask.sum())
        sample      = safe_json(series[mask].head(5).tolist())
        indices     = series[mask].index.tolist()
        passed      = count == 0
        explanation = explain_rule_no_past_year(col, count, min_year)

    elif rule == "no_negative":
        # Flag any value strictly below zero
        mask        = series.notna() & (series < 0)
        count       = int(mask.sum())
        sample      = safe_json(series[mask].head(5).tolist())
        indices     = series[mask].index.tolist()
        passed      = count == 0
        explanation = explain_rule_no_negative(col, count)

    elif rule == "percentage":
        # Flag any value outside [0, 100]
        # The | operator here is bitwise OR on boolean Series (not Python's 'or')
        mask        = series.notna() & ((series < 0) | (series > 100))
        count       = int(mask.sum())
        sample      = safe_json(series[mask].head(5).tolist())
        indices     = series[mask].index.tolist()
        passed      = count == 0
        explanation = explain_rule_percentage(col, count)

    elif rule == "custom_range":
        # The user specifies a min, max, or both.
        # If neither is provided, the rule is invalid — return an error.
        min_val = params.get("min")
        max_val = params.get("max")

        if min_val is None and max_val is None:
            return {
                "passed":            False,
                "error":             "custom_range requires at least one of: min, max.",
                "violation_count":   0,
                "violation_sample":  [],
                "violation_indices": [],
                "explanation":       "",
            }

        # Build the violation mask based on which bounds were provided.
        # Three cases: both bounds, lower bound only, upper bound only.
        if min_val is not None and max_val is not None:
            mask = series.notna() & ((series < min_val) | (series > max_val))
        elif min_val is not None:
            mask = series.notna() & (series < min_val)   # only a lower bound
        else:
            mask = series.notna() & (series > max_val)   # only an upper bound

        count       = int(mask.sum())
        sample      = safe_json(series[mask].head(5).tolist())
        indices     = series[mask].index.tolist()
        passed      = count == 0
        # float("-inf") / float("inf") are used when one bound is absent —
        # the explanation function uses them to show "no lower/upper bound"
        explanation = explain_rule_custom_range(
            col, count,
            min_val if min_val is not None else float("-inf"),
            max_val if max_val is not None else float("inf"),
        )

    elif rule == "no_nulls":
        # Flag every row where this column is NaN/None/NaT.
        # series.isna() covers all missing value types.
        mask        = series.isna()
        count       = int(mask.sum())
        sample      = []   # NaN values have no meaningful "sample" to show
        indices     = series[mask].index.tolist()
        passed      = count == 0
        explanation = explain_rule_no_nulls(col, count)

    else:
        # Unrecognised rule name — return an error dict instead of raising
        # so the caller can collect it and return it in the errors list
        return {
            "passed":            False,
            "error":             f'Unknown rule "{rule}".',
            "violation_count":   0,
            "violation_sample":  [],
            "violation_indices": [],
            "explanation":       "",
        }

    return {
        "passed":            passed,
        "violation_count":   count,
        "violation_sample":  sample,
        "violation_indices": indices,
        "explanation":       explanation,
    }


# =============================================================================
# TASK 6 — VALIDATION AND CROSS-CHECKING (original endpoints)
# =============================================================================

@task6.get("/task6/validation_info")
async def get_validation_info():
    """
    Returns everything the frontend needs to render the validation panel:

    - Auto-detected rule suggestions for each column
    - A full list of available rule types with their parameters
    - The current min/max of each numeric column (for context)
    - The current year (used to populate year-related rule defaults)

    Never modifies the DataFrame. Safe to call repeatedly.
    """
    df          = require_df()
    suggestions = detect_auto_rules(df)   # auto-suggested rules based on column names

    # Build a summary of every column: name, dtype, and numeric range if applicable
    column_info: List[Dict[str, Any]] = []
    for col in df.columns:
        entry: Dict[str, Any] = {
            "name":  col,
            "dtype": str(df[col].dtype),
        }
        if df[col].dtype.kind in ("i", "f"):
            # Include min/max only for numeric columns — gives context for range rules
            entry["min"] = safe_json(float(df[col].min()))
            entry["max"] = safe_json(float(df[col].max()))
        column_info.append(entry)

    # available_rules describes every rule the user can apply.
    # "params" lists the parameters the user must supply for each rule.
    # An empty params list means no configuration needed.
    available_rules = {
        "no_future_year": {
            "label":       "No future years",
            "description": f"Flags any value greater than {CURRENT_YEAR}.",
            "params":      [],   # no extra config needed — uses CURRENT_YEAR automatically
        },
        "no_past_year": {
            "label":       "No past years before a minimum",
            "description": "Flags any year value below a minimum year you specify.",
            "params":      [{"name": "min_year", "type": "int", "default": 1900}],
        },
        "no_negative": {
            "label":       "No negative values",
            "description": "Flags any value below zero.",
            "params":      [],
        },
        "percentage": {
            "label":       "Valid percentage (0-100)",
            "description": "Flags any value outside the range [0, 100].",
            "params":      [],
        },
        "custom_range": {
            "label":       "Custom range",
            "description": "Flags any value outside a min/max range you define.",
            "params":      [
                {"name": "min", "type": "float", "default": None},
                {"name": "max", "type": "float", "default": None},
            ],
        },
        "no_nulls": {
            "label":       "No missing values (required column)",
            "description": "Flags any row where this column is empty.",
            "params":      [],
        },
    }

    return {
        "what_is_validation": explain_what_validation_is(),
        "total_columns":      len(df.columns),
        "column_info":        column_info,
        "auto_suggestions":   suggestions,
        "available_rules":    available_rules,
        "current_year":       CURRENT_YEAR,
    }


class ValidationRule(BaseModel):
    """
    Defines one rule to run: which column, which rule, and any parameters.
    Used as an item inside the RunValidationPayload list.
    """
    column: str
    rule:   str
    params: Optional[Dict[str, Any]] = {}   # empty dict by default — only needed for some rules


class RunValidationPayload(BaseModel):
    """
    Defines the JSON body for POST /task6/run_validation.
    A list of ValidationRule objects — one per column/rule combination.

    Example:
        {
            "rules": [
                {"column": "age",    "rule": "custom_range", "params": {"min": 0, "max": 150}},
                {"column": "salary", "rule": "no_negative"},
                {"column": "email",  "rule": "no_nulls"}
            ]
        }
    """
    rules: List[ValidationRule]


@task6.post("/task6/run_validation")
async def run_validation(payload: RunValidationPayload):
    """
    Runs every rule in the payload against the DataFrame and returns a report.
    READ-ONLY — never modifies the data.

    Each result includes:
        passed            → True if zero violations
        violation_count   → how many rows failed
        violation_sample  → up to 5 example bad values
        violation_indices → DataFrame index labels of the bad rows
        explanation       → xAI description of what went wrong

    The summary field gives the user a single top-line message they can
    read without parsing the full results array.
    """
    df      = require_df()   # read-only — no .copy() needed
    results = []
    errors  = []

    for rule_def in payload.rules:
        col    = rule_def.column
        rule   = rule_def.rule
        params = rule_def.params or {}   # default to empty dict if not provided

        if col not in df.columns:
            errors.append(f'Column "{col}" not found in the dataset.')
            continue

        # Numeric rules only make sense on numeric columns.
        # Checking before running prevents confusing pandas comparison errors
        # (e.g. comparing a string to an integer with < raises a TypeError).
        if rule in ("no_future_year", "no_past_year", "no_negative",
                    "percentage", "custom_range"):
            if df[col].dtype.kind not in ("i", "f"):
                errors.append(
                    f'Rule "{rule}" on "{col}": only applies to numeric columns. '
                    f'"{col}" has dtype {df[col].dtype}.'
                )
                continue

        result = run_single_rule(df, col, rule, params)

        results.append({
            "column":            col,
            "rule":              rule,
            "params":            params,
            "passed":            result["passed"],
            "violation_count":   result["violation_count"],
            "violation_sample":  result["violation_sample"],
            "violation_indices": result["violation_indices"],
            "explanation":       result["explanation"],
        })

    passed_count = sum(1 for r in results if r["passed"])
    failed_count = sum(1 for r in results if not r["passed"])

    return {
        "total_rules_run": len(results),
        "passed":          passed_count,
        "failed":          failed_count,
        "all_passed":      failed_count == 0,
        "errors":          errors,
        "results":         results,
        # Summary: one friendly message the user can read at a glance
        "summary": (
            f"All {passed_count} rule(s) passed. Your data looks consistent."
            if failed_count == 0 else
            f"{failed_count} rule(s) failed out of {len(results)} checks. "
            "Review the violations below and decide how to fix them."
        ),
    }


class FixViolationPayload(BaseModel):
    """
    Defines the JSON body for POST /task6/fix_violations.

    column:      the column that has violations
    rule:        the rule that was violated (used to re-detect violations)
    params:      the same params used when running the rule
    fix_method:  one of: replace, set_null, drop_rows
    replace_val: only required when fix_method is "replace"

    Example — set all negative salaries to 0:
        {
            "column":      "salary",
            "rule":        "no_negative",
            "fix_method":  "replace",
            "replace_val": 0
        }

    Example — drop rows with future years:
        {
            "column":     "birth_year",
            "rule":       "no_future_year",
            "fix_method": "drop_rows"
        }
    """
    column:      str
    rule:        str
    params:      Optional[Dict[str, Any]] = {}
    fix_method:  str
    replace_val: Optional[Any] = None   # Any because replace_val could be int, float, or string


@task6.post("/task6/fix_violations")
async def fix_violations(payload: FixViolationPayload):
    """
    Fixes violations in one column using the chosen method.

    Why we re-run the rule before fixing:
        The violations were detected at the time of run_validation.
        If the user made other changes between then and now, the violation
        indices could be stale. Re-running the rule gives us fresh indices
        that reflect the current state of the DataFrame.

    Three fix methods:

    replace:
        Sets every violating cell to a specific value the user provides.
        E.g. set all negative salaries to 0, or all future years to CURRENT_YEAR.

    set_null:
        Sets every violating cell to NaN — effectively deleting the bad value
        without removing the entire row.
        Use when you can't choose a good replacement value.

    drop_rows:
        Deletes every row that has a violation.
        Use when the entire row is invalid, not just one cell.
        reset_index(drop=True) renumbers rows from 0 after deletion.
    """
    df      = require_df().copy()   # .copy() — never mutate live DataFrame until the end
    applied = []
    errors  = []
    col     = payload.column
    rule    = payload.rule
    params  = payload.params or {}

    if col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f'Column "{col}" not found in the dataset.'
        )

    # Re-run the rule to get fresh violation indices from the current DataFrame
    result            = run_single_rule(df, col, rule, params)
    violation_count   = result["violation_count"]
    violation_indices = result["violation_indices"]

    # If the rule itself was invalid (e.g. unknown rule name), surface that error
    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])

    # Nothing to fix — return early with a friendly message
    if violation_count == 0:
        return {
            "success":   True,
            "applied":   [f'"{col}": no violations found — nothing to fix.'],
            "errors":    [],
            "new_shape": list(df.shape),
        }

    try:
        if payload.fix_method == "replace":
            # replace_val is required for this method
            if payload.replace_val is None:
                raise HTTPException(
                    status_code=400,
                    detail='fix_method "replace" requires a "replace_val".'
                )
            # .loc[indices, col] targets specific rows by their index labels
            df.loc[violation_indices, col] = payload.replace_val
            applied.append(
                f'"{col}": set {violation_count} violation(s) to {payload.replace_val}. '
                + explain_fix_replace(col, payload.replace_val)
            )

        elif payload.fix_method == "set_null":
            # np.nan is the standard missing value marker for numeric columns.
            # For datetime columns, pd.NaT would be more appropriate — but
            # since this endpoint targets rule violations (which are numeric checks),
            # np.nan is the right choice here.
            df.loc[violation_indices, col] = np.nan
            applied.append(
                f'"{col}": set {violation_count} violation(s) to NaN. '
                + explain_fix_set_null(col)
            )

        elif payload.fix_method == "drop_rows":
            before = len(df)
            # df.drop(index=...) removes rows by their index labels
            df.drop(index=violation_indices, inplace=True)
            # reset_index renumbers rows from 0 to close the gaps left by removal
            df.reset_index(drop=True, inplace=True)
            rows_dropped = before - len(df)
            applied.append(
                f'"{col}": dropped {rows_dropped} row(s) containing violations. '
                + explain_fix_drop_rows(col, rows_dropped)
            )

        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    f'Unknown fix_method "{payload.fix_method}". '
                    "Valid options: replace, set_null, drop_rows."
                )
            )

    except HTTPException:
        raise   # re-raise FastAPI errors as-is — don't swallow them into the errors list
    except Exception as e:
        errors.append(f"Unexpected error: {str(e)}")

    if applied and not errors:
        # SNAPSHOT RULE: save BEFORE writing to dataset_state.df
        snapshot_store.save(f"Task 6 — fixed '{col}' violations ({payload.fix_method})", require_df())
        dataset_state.df = df

    return {
        "success":   len(errors) == 0,
        "applied":   applied,
        "errors":    errors,
        "new_shape": list(df.shape),
    }


# =============================================================================
# TASK 6 UPGRADE 1 — DATE RANGE VALIDATION
#
# This upgrade adds two types of date range checks:
#   1. datetime64[ns] columns → extract the year from each timestamp and
#      check it falls within [min_year, max_year]
#   2. Numeric year columns   → direct numeric comparison against the range
#
# Common problems this catches:
#   - Unix epoch default (1970-01-01) appearing when a null date was stored as 0
#   - Far-future years (e.g. 9999) from date parsing errors
#   - Impossible year values (e.g. year 0 or year 10000)
# =============================================================================

@task6.get("/task6/date_range_info")
async def get_date_range_info(
    min_year: int = 1900,          # oldest acceptable year (default: 1900)
    max_year: int = CURRENT_YEAR,  # newest acceptable year (default: this year)
):
    """
    Scans all datetime and numeric year columns for values outside
    the expected year range.

    Two column types are checked:
        datetime64[ns]: extract the year with .dt.year, then compare
        numeric year columns: columns whose name contains year-like keywords
                              and are numeric — compare directly

    Returns one report entry per column that has at least one violation.
    """
    df = require_df()

    # Select all datetime columns — pandas datetime type is datetime64[ns]
    datetime_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()

    # Find numeric columns whose names suggest they store year values
    year_keywords = ("year", "yr", "born", "dob")
    year_cols = [
        col for col in df.select_dtypes(include=["number"]).columns
        if any(kw in col.lower() for kw in year_keywords)
    ]

    reports = []

    # ── Check datetime columns ────────────────────────────────────────────────
    for col in datetime_cols:
        series       = df[col].dropna()       # skip NaT values
        years        = series.dt.year         # extract just the year part from each timestamp
        out_of_range = (years < min_year) | (years > max_year)
        viol_count   = int(out_of_range.sum())

        if viol_count == 0:
            continue   # column is clean — skip it

        # Convert to string for display — raw timestamps are hard to read in UI
        samples = [str(v) for v in series[out_of_range].head(5).tolist()]
        reports.append({
            "column":            col,
            "dtype":             "datetime64[ns]",
            "violation_count":   viol_count,
            "sample_violations": samples,
            "min_year":          min_year,
            "max_year":          max_year,
            "explanation":       explain_date_range_violation(col, viol_count, min_year, max_year),
        })

    # ── Check numeric year columns ────────────────────────────────────────────
    for col in year_cols:
        series       = df[col].dropna()
        out_of_range = (series < min_year) | (series > max_year)
        viol_count   = int(out_of_range.sum())

        if viol_count == 0:
            continue

        samples = [str(v) for v in series[out_of_range].head(5).tolist()]
        reports.append({
            "column":            col,
            "dtype":             str(df[col].dtype),
            "violation_count":   viol_count,
            "sample_violations": samples,
            "min_year":          min_year,
            "max_year":          max_year,
            "explanation":       explain_date_range_violation(col, viol_count, min_year, max_year),
        })

    return {
        "columns_checked":     len(datetime_cols) + len(year_cols),
        "columns_with_issues": len(reports),
        "min_year_used":       min_year,
        "max_year_used":       max_year,
        "reports":             reports,
        "message": (
            "No date range violations found."
            if not reports else
            f"{len(reports)} column(s) have dates outside [{min_year}, {max_year}]."
        ),
    }


class DateRangeFixPayload(BaseModel):
    """
    Defines the JSON body for POST /task6/fix_date_range.

    column:   the column to fix
    min_year: oldest acceptable year (values below this are cleared)
    max_year: newest acceptable year (values above this are cleared)
    """
    column:   str
    min_year: int = 1900
    max_year: int = CURRENT_YEAR


@task6.post("/task6/fix_date_range")
async def fix_date_range(payload: DateRangeFixPayload):
    """
    Sets out-of-range date values to NaT (for datetime columns) or
    NaN (for numeric year columns).

    All rows are preserved — only the invalid cell values are cleared.
    This is safer than dropping rows when the rest of the row's data is valid.

    Datetime columns use pd.NaT (Not a Time) — pandas' missing value
    for datetime types. Using np.nan on a datetime column would raise a TypeError.
    Numeric columns use np.nan as usual.
    """
    df  = require_df().copy()
    col = payload.column

    if col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f'Column "{col}" not found in the dataset.'
        )

    applied = []

    if str(df[col].dtype) == "datetime64[ns]":
        # Extract the year from each timestamp and build the out-of-range mask
        years        = df[col].dt.year
        out_of_range = (years < payload.min_year) | (years > payload.max_year)
        viol_count   = int(out_of_range.sum())
        # pd.NaT is the correct missing value for datetime columns (not np.nan)
        df.loc[out_of_range, col] = pd.NaT
        applied.append(
            f'"{col}": set {viol_count} out-of-range date(s) to NaT. '
            f"Range enforced: [{payload.min_year}, {payload.max_year}]."
        )

    elif df[col].dtype.kind in ("i", "f"):
        # Numeric year column — direct comparison
        out_of_range = (df[col] < payload.min_year) | (df[col] > payload.max_year)
        viol_count   = int(out_of_range.sum())
        df.loc[out_of_range, col] = np.nan
        applied.append(
            f'"{col}": set {viol_count} out-of-range year(s) to NaN. '
            f"Range enforced: [{payload.min_year}, {payload.max_year}]."
        )

    else:
        # Column is neither datetime nor numeric — can't apply date range logic
        raise HTTPException(
            status_code=400,
            detail=f'"{col}" is not a datetime or numeric column.'
        )

    snapshot_store.save(f"Task 6 — date range fix on '{payload.column}'", require_df())
    dataset_state.df = df
    return {
        "success":   True,
        "applied":   applied,
        "new_shape": list(df.shape),
    }


# =============================================================================
# TASK 6 UPGRADE 2 — CROSS-COLUMN VALIDATION
#
# Standard validation checks one column in isolation.
# Cross-column validation checks logical consistency BETWEEN two columns.
#
# Examples of cross-column inconsistencies:
#   - end_date < start_date   (the end came before the beginning)
#   - age = 30 but birth_year = 1990 (age should be ~35 in 2025)
#   - min_price > max_price   (minimum exceeds maximum)
#
# These errors can't be detected by looking at one column alone.
# =============================================================================

class CrossColumnPayload(BaseModel):
    """
    Defines the JSON body for POST /task6/cross_column_check.

    rule:      the logical relationship to check
    col_a:     the first column (the "left" side of the comparison)
    col_b:     the second column (the "right" side)
    tolerance: only used by age_matches_birth_year — how many years of
               difference to allow before flagging as a violation (default 2)

    Valid rules:
        end_after_start        → col_b (end) must be >= col_a (start)
        age_matches_birth_year → col_a (age) must match CURRENT_YEAR - col_b (birth year)
        a_less_than_b          → col_a must be strictly less than col_b
        a_greater_than_b       → col_a must be strictly greater than col_b
    """
    rule:      str
    col_a:     str
    col_b:     str
    tolerance: Optional[int] = 2   # only used by age_matches_birth_year


@task6.post("/task6/cross_column_check")
async def cross_column_check(payload: CrossColumnPayload):
    """
    Runs a logical cross-column check and returns violations.
    READ-ONLY — never modifies the DataFrame.

    Why we filter to both_present:
        We can only check the relationship between col_a and col_b on rows
        where BOTH columns have a real value. Rows where either is NaN
        would cause comparison errors or misleading results.
        both_present is the mask of rows where neither column is NaN.

    Violations are collected as DataFrame index labels (not positions)
    so they can be passed directly to fix_cross_column.
    """
    df = require_df()

    # Validate both columns exist before doing anything else
    for col in (payload.col_a, payload.col_b):
        if col not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f'Column "{col}" not found in the dataset.'
            )

    col_a = payload.col_a
    col_b = payload.col_b
    rule  = payload.rule

    # Build a mask of rows where BOTH columns have non-null values
    both_present = df[col_a].notna() & df[col_b].notna()
    # Work only on those complete rows — avoids NaN comparison issues
    subset = df[both_present]

    # Default violation_mask to all-False — overwritten by whichever rule matches
    violation_mask = pd.Series([False] * len(subset), index=subset.index)
    explanation    = ""

    if rule == "end_after_start":
        # col_b is the "end" column, col_a is the "start" column.
        # A violation is any row where the end is BEFORE the start.
        violation_mask  = subset[col_b] < subset[col_a]
        violation_count = int(violation_mask.sum())
        explanation     = explain_end_before_start_violation(col_a, col_b, violation_count)

    elif rule == "age_matches_birth_year":
        # Expected age = CURRENT_YEAR - birth_year (col_b).
        # A violation is any row where the actual age (col_a) differs from the
        # expected age by more than the tolerance value.
        # abs() on the diff catches both "too old" and "too young" cases.
        expected_age    = CURRENT_YEAR - subset[col_b]
        diff            = (subset[col_a] - expected_age).abs()
        violation_mask  = diff > (payload.tolerance or 2)
        violation_count = int(violation_mask.sum())
        explanation     = explain_age_birth_year_violation(col_a, col_b, violation_count)

    elif rule == "a_less_than_b":
        # Flag rows where col_a is NOT less than col_b (i.e. col_a >= col_b)
        violation_mask  = subset[col_a] >= subset[col_b]
        violation_count = int(violation_mask.sum())
        explanation     = (
            f"{violation_count} row(s) where '{col_a}' is not less than '{col_b}'."
        )

    elif rule == "a_greater_than_b":
        # Flag rows where col_a is NOT greater than col_b (i.e. col_a <= col_b)
        violation_mask  = subset[col_a] <= subset[col_b]
        violation_count = int(violation_mask.sum())
        explanation     = (
            f"{violation_count} row(s) where '{col_a}' is not greater than '{col_b}'."
        )

    else:
        raise HTTPException(
            status_code=400,
            detail=(
                f'Unknown rule "{rule}". '
                "Valid: end_after_start, age_matches_birth_year, "
                "a_less_than_b, a_greater_than_b."
            )
        )

    # Convert the boolean mask to a list of index labels
    violation_indices = subset.index[violation_mask].tolist()
    violation_count   = len(violation_indices)

    # Build sample rows showing both column values side by side for context
    sample_rows: List[Dict[str, Any]] = []
    for idx in violation_indices[:5]:   # cap at 5 examples
        sample_rows.append({
            "row_index": int(idx),
            col_a:       str(df.loc[idx, col_a]),   # str() ensures JSON safe for all types
            col_b:       str(df.loc[idx, col_b]),
        })

    return {
        "what_is_cross_validation": explain_what_cross_column_validation_is(),
        "rule":              rule,
        "col_a":             col_a,
        "col_b":             col_b,
        # rows_checked = rows where both columns had real values (excludes NaN rows)
        "rows_checked":      int(both_present.sum()),
        "violation_count":   violation_count,
        "violation_indices": violation_indices,   # needed by fix_cross_column
        "sample_violations": sample_rows,
        "explanation":       explanation,
        "passed":            violation_count == 0,
        # Direct guidance to the next endpoint if violations were found
        "next_step": (
            "Use POST /task6/fix_cross_column to set violating values to NaN."
            if violation_count > 0 else
            "No violations found — no action needed."
        ),
    }


class CrossColumnFixPayload(BaseModel):
    """
    Defines the JSON body for POST /task6/fix_cross_column.

    violation_indices: the list of row index labels from /task6/cross_column_check
    col_to_fix:        which column to null out — usually the "less trustworthy" one
                       (e.g. if age and birth_year conflict, fix the age column)
    """
    violation_indices: List[int]   # index labels from the check endpoint
    col_to_fix:        str         # the column whose values will be set to NaN/NaT


@task6.post("/task6/fix_cross_column")
async def fix_cross_column(payload: CrossColumnFixPayload):
    """
    Sets the violating values in col_to_fix to NaN or NaT.
    Only one column is modified — the other column is left completely untouched.

    Why we only fix one column:
        A cross-column violation involves two columns that contradict each other.
        We can't know which one is wrong without domain knowledge.
        The user decides which column to trust and which to null out.
        The frontend surfaces this choice when showing the violation.

    Why we validate indices before changing anything:
        The violation indices came from a previous cross_column_check call.
        If other changes were made since then, those indices might be stale.
        We check all indices upfront and refuse if any are missing — safer
        than silently operating on wrong rows.
    """
    df  = require_df().copy()
    col = payload.col_to_fix

    if col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f'Column "{col}" not found in the dataset.'
        )

    # Validate all provided indices still exist in the current DataFrame
    invalid = [i for i in payload.violation_indices if i not in df.index]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Row indices not found: {invalid[:5]}. "
                "Re-run /task6/cross_column_check to get fresh indices."
            )
        )

    count = len(payload.violation_indices)

    # Use pd.NaT for datetime columns — np.nan would raise TypeError on datetime64
    if str(df[col].dtype) == "datetime64[ns]":
        df.loc[payload.violation_indices, col] = pd.NaT
    else:
        df.loc[payload.violation_indices, col] = np.nan

    snapshot_store.save(f"Task 6 — cross-column fix on '{col}'", require_df())
    dataset_state.df = df

    return {
        "success": True,
        "applied": [
            f'"{col}": set {count} violating value(s) to NaN/NaT. '
            + explain_fix_set_null_cross(col, "the related column")
        ],
        "new_shape": list(df.shape),
    }


# =============================================================================
# TASK 6 UPGRADE 3 — STRING PATTERN VALIDATION
#
# Many text columns are supposed to follow a strict format:
#   - Emails must match user@domain.tld
#   - Phone numbers must match a standard format
#   - Dates stored as strings must match YYYY-MM-DD
#   - ZIP codes must match 5-digit or 5+4 format
#
# This upgrade lets the user check any text column against either a
# built-in named pattern or a custom regex they provide.
# =============================================================================

@task6.get("/task6/pattern_info")
async def get_pattern_info():
    """
    Auto-detects which text columns likely match each built-in pattern
    and returns them as suggestions.

    Detection method:
        For each text column, take up to 20 sample values.
        Test them against every built-in pattern.
        If 80% or more of samples match a pattern, suggest that pattern.
        Pick only the best-matching pattern per column (highest match rate).

    Why 80%?
        80% is strict enough to avoid false positives on free-text columns
        but forgiving enough to catch real format columns with a few bad values.

    For suggested columns it also computes the full violation count across
    all values — not just the sample — so the user knows the real scope.
    """
    df        = require_df()
    text_cols = df.select_dtypes(include=["object"]).columns.tolist()

    suggestions: List[Dict[str, Any]] = []

    for col in text_cols:
        series  = df[col].dropna().astype(str)
        samples = series.head(20).tolist()   # sample 20 values for pattern detection

        if not samples:
            continue

        best_pattern: Optional[str] = None
        best_rate                   = 0.0

        # Test every built-in pattern against the sample values
        for pattern_name, regex in _PATTERNS.items():
            # .strip() normalises whitespace before matching
            matches    = sum(1 for v in samples if regex.match(v.strip()))
            match_rate = matches / len(samples)
            # Keep the pattern with the highest match rate, as long as it exceeds 80%
            if match_rate > best_rate and match_rate >= 0.8:
                best_rate    = match_rate
                best_pattern = pattern_name

        if best_pattern:
            # Count full violations across ALL values in the column (not just the sample)
            full_series = df[col].dropna().astype(str)
            pat_regex   = _PATTERNS[best_pattern]
            violations  = int(
                full_series.apply(
                    # .apply() with a lambda runs the function on every value
                    # not bool(regex.match(v)) → True if the value DOESN'T match (i.e. a violation)
                    lambda v: not bool(pat_regex.match(v.strip()))
                ).sum()
            )
            suggestions.append({
                "column":            col,
                "suggested_pattern": best_pattern,
                "match_rate":        round(best_rate * 100, 1),   # as a percentage e.g. 95.0
                "violation_count":   violations,
                "sample_values":     samples[:3],   # show 3 examples in the UI
            })

    return {
        "available_patterns":   list(_PATTERNS.keys()),   # all patterns the user can choose from
        "text_columns_scanned": len(text_cols),
        "suggestions":          suggestions,
        "message": (
            f"{len(suggestions)} column(s) matched a known pattern."
            if suggestions else
            "No text columns automatically matched a known pattern."
        ),
    }


class PatternCheckPayload(BaseModel):
    """
    Defines the JSON body for POST /task6/check_pattern.

    column:       the text column to validate
    pattern:      a named pattern from _PATTERNS, or "custom"
    custom_regex: required only when pattern is "custom"
                  must be a valid Python regex string

    Example — check an email column:
        {"column": "email_address", "pattern": "email"}

    Example — check a custom internal ID format:
        {"column": "internal_id", "pattern": "custom", "custom_regex": "^EMP-\\d{5}$"}
    """
    column:       str
    pattern:      str
    custom_regex: Optional[str] = None   # only needed when pattern = "custom"


@task6.post("/task6/check_pattern")
async def check_pattern(payload: PatternCheckPayload):
    """
    Validates every non-null value in a text column against a pattern.

    Built-in patterns:
        email, phone_us, postal_code_us, postal_code_uk,
        url, date_iso, integer_string, decimal_string

    Custom patterns:
        The user provides a Python regex string.
        It is compiled and validated before use — an invalid regex raises
        a 400 error with the specific syntax problem rather than crashing.

    pass_rate tells the user what percentage of values conform to the pattern.
    violation_sample shows up to 5 example values that failed — useful for
    diagnosing whether the pattern is wrong or the data is wrong.
    """
    df  = require_df()
    col = payload.column

    if col not in df.columns:
        raise HTTPException(status_code=400, detail=f'Column "{col}" not found.')

    if payload.pattern == "custom":
        # User-provided regex — must be supplied and must be syntactically valid
        if not payload.custom_regex:
            raise HTTPException(
                status_code=400,
                detail='pattern "custom" requires custom_regex to be provided.'
            )
        try:
            regex = _re.compile(payload.custom_regex)
        except _re.error as e:
            # _re.error is raised for invalid regex syntax — surface the specific problem
            raise HTTPException(
                status_code=400,
                detail=f'Invalid regex: {str(e)}'
            )
    elif payload.pattern in _PATTERNS:
        # Named pattern — retrieve the pre-compiled object directly
        regex = _PATTERNS[payload.pattern]
    else:
        raise HTTPException(
            status_code=400,
            detail=(
                f'Unknown pattern "{payload.pattern}". '
                f'Valid: {list(_PATTERNS.keys()) + ["custom"]}'
            )
        )

    # Cast to str and drop NaN — we only check values that actually exist
    series = df[col].dropna().astype(str)
    total  = len(series)

    if total == 0:
        # Empty column after dropping NaN — nothing to check, return clean result
        return {
            "column":            col,
            "pattern":           payload.pattern,
            "values_checked":    0,
            "violation_count":   0,
            "violation_sample":  [],
            "violation_indices": [],
            "pass_rate":         100.0,
            "passed":            True,
            "explanation":       f'"{col}" has no non-null values to check.',
        }

    # .apply() runs the lambda on every value in the Series.
    # not bool(regex.match(...)) → True where the value DOESN'T match the pattern.
    # .strip() removes edge whitespace before matching — "  john@example.com  " should pass.
    bad_mask        = series.apply(lambda v: not bool(regex.match(v.strip())))
    violation_count = int(bad_mask.sum())
    violation_vals  = series[bad_mask].head(5).tolist()   # up to 5 example bad values
    violation_idx   = series[bad_mask].index.tolist()     # their index labels
    # pass_rate = proportion of values that DID match, as a percentage
    pass_rate       = round(((total - violation_count) / total) * 100, 2)

    return {
        "column":            col,
        "pattern":           payload.pattern,
        "values_checked":    total,
        "violation_count":   violation_count,
        "violation_sample":  violation_vals,
        "violation_indices": violation_idx,
        "pass_rate":         pass_rate,
        "passed":            violation_count == 0,
        "explanation":       explain_pattern_violation(col, payload.pattern, violation_count),
    }
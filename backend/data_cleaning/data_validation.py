from fastapi import HTTPException, APIRouter
from pydantic import BaseModel
from routes.dfState import dataset_state
from xAI.validation import (
    explain_what_validation_is,
    explain_rule_no_future_year,
    explain_rule_no_past_year,
    explain_rule_no_negative,
    explain_rule_percentage,
    explain_rule_custom_range,
    explain_rule_no_nulls,
    explain_fix_replace,
    explain_fix_set_null,
    explain_fix_drop_rows,
)
import pandas as pd
import numpy as np
import datetime
from typing import Dict, Any, Optional, List

task6 = APIRouter()


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
    Converts numpy/pandas types to plain Python for JSON serialisation.
    Called on violation sample values before returning them.
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
# AUTO-DETECTION LOGIC
#
# These functions inspect column names and dtypes to automatically suggest
# validation rules the user is likely to care about.
# The user does not have to know what rules to apply — the system figures
# it out from the column name and data characteristics.
# =============================================================================

CURRENT_YEAR = datetime.datetime.now().year


def detect_auto_rules(df: pd.DataFrame) -> list:
    """
    Scans all columns and automatically suggests validation rules
    based on the column name and dtype.

    Rules auto-detected:
      - Year columns (name contains "year", "yr", "born", "dob")
          → no future years, no years before 1900
      - Age columns (name contains "age")
          → no negatives, max reasonable age of 150
      - Percentage columns (name contains "pct", "percent", "rate", "ratio")
          → must be between 0 and 100
      - Price/amount columns (name contains "price", "cost", "salary", "amount",
                              "revenue", "income", "wage", "fee")
          → no negatives
      - Count columns (name contains "count", "qty", "quantity", "num", "number")
          → no negatives

    Returns a list of suggested rule dicts that the GET endpoint includes
    in its response. The user sees these as pre-filled suggestions and
    can choose to run them, modify them, or ignore them.
    """
    suggestions = []

    year_keywords    = ("year", "yr", "born", "dob", "date_of_birth")
    age_keywords     = ("age",)
    pct_keywords     = ("pct", "percent", "rate", "ratio", "score")
    money_keywords   = ("price", "cost", "salary", "amount", "revenue",
                        "income", "wage", "fee", "pay", "earning")
    count_keywords   = ("count", "qty", "quantity", "num", "number", "total")

    for col in df.columns:
        col_lower = col.lower()
        dtype     = str(df[col].dtype)

        # Only suggest numeric rules on numeric columns
        is_numeric = df[col].dtype.kind in ("i", "f")

        if any(kw in col_lower for kw in year_keywords) and is_numeric:
            suggestions.append({
                "column":      col,
                "rule":        "no_future_year",
                "description": f'"{col}" looks like a year column — should not exceed {CURRENT_YEAR}.',
                "auto":        True,
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
            suggestions.append({
                "column":      col,
                "rule":        "custom_range",
                "min":         0,
                "max":         150,
                "description": f'"{col}" looks like an age column — should be between 0 and 150.',
                "auto":        True,
            })

        elif any(kw in col_lower for kw in pct_keywords) and is_numeric:
            suggestions.append({
                "column":      col,
                "rule":        "percentage",
                "description": f'"{col}" looks like a percentage — should be between 0 and 100.',
                "auto":        True,
            })

        elif any(kw in col_lower for kw in money_keywords) and is_numeric:
            suggestions.append({
                "column":      col,
                "rule":        "no_negative",
                "description": f'"{col}" looks like a monetary value — should not be negative.',
                "auto":        True,
            })

        elif any(kw in col_lower for kw in count_keywords) and is_numeric:
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

    Returns a dict containing:
      - passed:            True if no violations found
      - violation_count:   how many values broke the rule
      - violation_sample:  up to 5 actual bad values so the user can see them
      - violation_indices: row indices of violations (for the fix endpoint)
      - explanation:       plain-English description from xai/explainer.py
    """
    series = df[col]

    if rule == "no_future_year":
        # Flag any value greater than the current year
        mask      = series.notna() & (series > CURRENT_YEAR)
        count     = int(mask.sum())
        sample    = safe_json(series[mask].head(5).tolist())
        indices   = series[mask].index.tolist()
        passed    = count == 0
        explanation = explain_rule_no_future_year(col, count, CURRENT_YEAR)

    elif rule == "no_past_year":
        min_year  = int(params.get("min_year", 1900))
        mask      = series.notna() & (series < min_year)
        count     = int(mask.sum())
        sample    = safe_json(series[mask].head(5).tolist())
        indices   = series[mask].index.tolist()
        passed    = count == 0
        explanation = explain_rule_no_past_year(col, count, min_year)

    elif rule == "no_negative":
        mask      = series.notna() & (series < 0)
        count     = int(mask.sum())
        sample    = safe_json(series[mask].head(5).tolist())
        indices   = series[mask].index.tolist()
        passed    = count == 0
        explanation = explain_rule_no_negative(col, count)

    elif rule == "percentage":
        # Percentage must be between 0 and 100 inclusive
        mask      = series.notna() & ((series < 0) | (series > 100))
        count     = int(mask.sum())
        sample    = safe_json(series[mask].head(5).tolist())
        indices   = series[mask].index.tolist()
        passed    = count == 0
        explanation = explain_rule_percentage(col, count)

    elif rule == "custom_range":
        min_val = params.get("min")
        max_val = params.get("max")

        if min_val is None and max_val is None:
            return {
                "passed":          False,
                "error":           "custom_range requires at least one of: min, max.",
                "violation_count": 0,
                "violation_sample": [],
                "violation_indices": [],
                "explanation":     "",
            }

        # Build the mask based on which bounds are provided
        if min_val is not None and max_val is not None:
            mask = series.notna() & ((series < min_val) | (series > max_val))
        elif min_val is not None:
            mask = series.notna() & (series < min_val)
        else:
            mask = series.notna() & (series > max_val)

        count     = int(mask.sum())
        sample    = safe_json(series[mask].head(5).tolist())
        indices   = series[mask].index.tolist()
        passed    = count == 0
        explanation = explain_rule_custom_range(
            col, count,
            min_val if min_val is not None else float("-inf"),
            max_val if max_val is not None else float("inf"),
        )

    elif rule == "no_nulls":
        # Check that a column that should be complete has no missing values
        mask      = series.isna()
        count     = int(mask.sum())
        sample    = []    # null values have nothing to show
        indices   = series[mask].index.tolist()
        passed    = count == 0
        explanation = explain_rule_no_nulls(col, count)

    else:
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
# TASK 6 — VALIDATION AND CROSS-CHECKING
#
# Two endpoints:
#
#   GET  /task6/validation_info
#        Auto-detects sensible validation rules for the dataset.
#        Returns the suggestions with explanations so the user can
#        review them before running any checks.
#        Also returns the full list of available rules with descriptions
#        so the user can add custom rules for any column.
#        Never modifies the data.
#
#   POST /task6/run_validation
#        Runs the user's selected rules and returns a full violation report.
#        Does NOT modify the data — it only reports what it finds.
#
#   POST /task6/fix_violations
#        Applies a fix to the violations found for a specific column:
#          replace   — set violating values to a specific replacement value
#          set_null  — set violating values to NaN
#          drop_rows — remove the rows containing violations
# =============================================================================

@task6.get("/task6/validation_info")
async def get_validation_info():
    """
    Scans the dataset and returns auto-detected rule suggestions.

    The auto-detection looks at column names and dtypes to infer what
    rules make sense. A column called "birth_year" with numeric dtype
    will automatically get a no_future_year suggestion. A column called
    "completion_rate" will get a percentage rule suggestion.

    The user sees these as pre-populated checkboxes in the UI.
    They can run them as-is, modify the parameters, add extra rules,
    or uncheck any they don't want.

    Also returns the full list of available rule types so the user
    can apply any rule to any column manually.
    """
    df          = require_df()
    suggestions = detect_auto_rules(df)

    # Build the column info list — name, dtype, min, max for every column.
    # The frontend uses this to populate the column picker for custom rules.
    column_info = []
    for col in df.columns:
        entry = {
            "name":  col,
            "dtype": str(df[col].dtype),
        }
        # Only include min/max for numeric columns
        if df[col].dtype.kind in ("i", "f"):
            entry["min"] = safe_json(float(df[col].min()))
            entry["max"] = safe_json(float(df[col].max()))
        column_info.append(entry)

    # Available rule types shown in the UI for manual rule creation
    available_rules = {
        "no_future_year": {
            "label":       "No future years",
            "description": f"Flags any value greater than {CURRENT_YEAR}.",
            "params":      [],
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
            "label":       "Valid percentage (0–100)",
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
        "what_is_validation":  explain_what_validation_is(),
        "total_columns":       len(df.columns),
        "column_info":         column_info,
        "auto_suggestions":    suggestions,
        "available_rules":     available_rules,
        "current_year":        CURRENT_YEAR,
    }


# ── Request body models ───────────────────────────────────────────────────────

class ValidationRule(BaseModel):
    """
    One rule to run on one column.

    column: the column name to check
    rule:   one of: no_future_year, no_past_year, no_negative,
                    percentage, custom_range, no_nulls
    params: additional parameters for rules that need them
            e.g. {"min": 0, "max": 120} for custom_range
                 {"min_year": 1900} for no_past_year
    """
    column: str
    rule:   str
    params: Optional[Dict[str, Any]] = {}


class RunValidationPayload(BaseModel):
    """
    List of rules to run. Each rule targets one column.
    Multiple rules can target the same column.

    Example:
        {
            "rules": [
                {"column": "birth_year",  "rule": "no_future_year"},
                {"column": "birth_year",  "rule": "custom_range", "params": {"min": 1900, "max": 2006}},
                {"column": "age",         "rule": "no_negative"},
                {"column": "pass_rate",   "rule": "percentage"},
                {"column": "salary",      "rule": "custom_range", "params": {"min": 0, "max": 10000000}}
            ]
        }
    """
    rules: List[ValidationRule]


@task6.post("/task6/run_validation")
async def run_validation(payload: RunValidationPayload):
    """
    Runs the user's chosen validation rules and returns a full report.

    This endpoint is READ-ONLY — it never modifies the DataFrame.
    It only reports what it finds so the user can review violations
    before deciding how to fix them.

    For each rule:
      - Runs the check on the specified column
      - Returns whether it passed or failed
      - Returns the count and sample of violations
      - Returns the xAI explanation of what the violation means
      - Returns the row indices of violations (used by fix_violations)

    The summary at the top tells the user how many rules passed,
    how many failed, and whether the dataset is ready for use.
    """
    df      = require_df()
    results = []
    errors  = []

    for rule_def in payload.rules:
        col    = rule_def.column
        rule   = rule_def.rule
        params = rule_def.params or {}

        # Validate the column exists before running any check
        if col not in df.columns:
            errors.append(f'Column "{col}" not found in the dataset.')
            continue

        # Validate the rule is applicable to this column's dtype
        if rule in ("no_future_year", "no_past_year", "no_negative",
                    "percentage", "custom_range"):
            if df[col].dtype.kind not in ("i", "f"):
                errors.append(
                    f'Rule "{rule}" on "{col}": this rule only applies to '
                    f'numeric columns. "{col}" has dtype {df[col].dtype}.'
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

    # Summary counts
    passed_count = sum(1 for r in results if r["passed"])
    failed_count = sum(1 for r in results if not r["passed"])

    return {
        "total_rules_run": len(results),
        "passed":          passed_count,
        "failed":          failed_count,
        "all_passed":      failed_count == 0,
        "errors":          errors,
        "results":         results,
        # Convenience message for the banner
        "summary": (
            f"All {passed_count} rule(s) passed. Your data looks consistent."
            if failed_count == 0 else
            f"{failed_count} rule(s) failed out of {len(results)} checks. "
            f"Review the violations below and decide how to fix them."
        ),
    }


class FixViolationPayload(BaseModel):
    """
    Fixes violations found for one column + rule combination.

    column:      the column containing violations
    rule:        the rule that was violated
    params:      the same params used when the rule was run
    fix_method:  one of: replace, set_null, drop_rows
    replace_val: only required when fix_method is "replace"

    Example — cap percentages at 100:
        {
            "column":      "completion_rate",
            "rule":        "percentage",
            "params":      {},
            "fix_method":  "replace",
            "replace_val": 100
        }

    Example — set impossible ages to NaN:
        {
            "column":     "age",
            "rule":       "custom_range",
            "params":     {"min": 0, "max": 150},
            "fix_method": "set_null"
        }

    Example — drop rows with future birth years:
        {
            "column":     "birth_year",
            "rule":       "no_future_year",
            "params":     {},
            "fix_method": "drop_rows"
        }
    """
    column:      str
    rule:        str
    params:      Optional[Dict[str, Any]] = {}
    fix_method:  str
    replace_val: Optional[Any] = None


@task6.post("/task6/fix_violations")
async def fix_violations(payload: FixViolationPayload):
    """
    Applies a fix to the violations found for one column + rule.

    Works on a .copy() — dataset_state.df is only updated at the end
    if the fix succeeds.

    Three fix methods:

      replace:
          Sets every violating value to replace_val.
          e.g. capping completion_rate at 100 by replacing all
          values above 100 with exactly 100.
          The row is kept — only the bad value changes.

      set_null:
          Sets every violating value to NaN.
          Honest about uncertainty — we know the value is wrong
          but we don't know the correct value.
          The row is kept — the bad value becomes missing.
          Can then be handled in Task 2 if needed.

      drop_rows:
          Removes every row that contains a violation.
          Most destructive option — the entire row is lost.
          Use only when the whole row is untrustworthy because
          of this one violation.
    """
    df      = require_df().copy()
    applied = []
    errors  = []

    col    = payload.column
    rule   = payload.rule
    params = payload.params or {}

    if col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f'Column "{col}" not found in the dataset.'
        )

    # Re-run the rule to get the current violation mask
    # We re-run rather than trusting stored indices because the DataFrame
    # may have changed since the user last called run_validation
    result = run_single_rule(df, col, rule, params)

    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])

    violation_count = result["violation_count"]

    if violation_count == 0:
        return {
            "success": True,
            "applied": [f'"{col}": no violations found — nothing to fix.'],
            "errors":  [],
            "new_shape": list(df.shape),
        }

    # Rebuild the mask from the violation indices
    # Using .isin(indices) is safer than re-running the rule logic
    # because indices are exact and don't depend on rule logic edge cases
    violation_indices = result["violation_indices"]

    try:
        if payload.fix_method == "replace":
            if payload.replace_val is None:
                raise HTTPException(
                    status_code=400,
                    detail='fix_method "replace" requires a "replace_val" in the request.'
                )
            df.loc[violation_indices, col] = payload.replace_val
            applied.append(
                f'"{col}": set {violation_count} violation(s) to {payload.replace_val}. '
                + explain_fix_replace(col, payload.replace_val)
            )

        elif payload.fix_method == "set_null":
            df.loc[violation_indices, col] = np.nan
            applied.append(
                f'"{col}": set {violation_count} violation(s) to NaN. '
                + explain_fix_set_null(col)
            )

        elif payload.fix_method == "drop_rows":
            before = len(df)
            df.drop(index=violation_indices, inplace=True)
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
                    'Valid options: replace, set_null, drop_rows.'
                )
            )

    except HTTPException:
        raise
    except Exception as e:
        errors.append(f'Unexpected error: {str(e)}')

    # ── Save back ─────────────────────────────────────────────────────────────

    if applied and not errors:
        dataset_state.df = df

    # ── Return ────────────────────────────────────────────────────────────────

    return {
        "success":   len(errors) == 0,
        "applied":   applied,
        "errors":    errors,
        "new_shape": list(df.shape),
    }
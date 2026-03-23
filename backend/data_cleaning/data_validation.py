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
    # Task 6 upgrades
    explain_what_cross_column_validation_is,
    explain_date_range_violation,
    explain_end_before_start_violation,
    explain_age_birth_year_violation,
    explain_pattern_violation,
    explain_fix_set_null_cross,
)
import pandas as pd
import numpy as np
import datetime
import re as _re
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


def safe_json(obj: Any) -> Any:
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
# Compiled once at module load — no per-request compilation overhead.
# =============================================================================

_PATTERNS: Dict[str, Any] = {
    "email": _re.compile(
        r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    ),
    "phone_us": _re.compile(
        r"^(\+1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}$"
    ),
    "postal_code_us": _re.compile(
        r"^\d{5}(-\d{4})?$"
    ),
    "postal_code_uk": _re.compile(
        r"^[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}$",
        _re.IGNORECASE,
    ),
    "url": _re.compile(
        r"^https?://[^\s/$.?#].[^\s]*$",
        _re.IGNORECASE,
    ),
    "date_iso": _re.compile(
        r"^\d{4}-\d{2}-\d{2}$"
    ),
    "integer_string": _re.compile(
        r"^-?\d+$"
    ),
    "decimal_string": _re.compile(
        r"^-?\d+(\.\d+)?$"
    ),
}


# =============================================================================
# AUTO-DETECTION LOGIC
# =============================================================================

CURRENT_YEAR = datetime.datetime.now().year


def detect_auto_rules(df: pd.DataFrame) -> list:
    """
    Scans all columns and suggests validation rules based on column
    names and dtypes. The user sees these as pre-filled suggestions.

    FIX: removed unused 'dtype' variable that appeared in the original code.
    """
    suggestions    = []
    year_keywords  = ("year", "yr", "born", "dob", "date_of_birth")
    age_keywords   = ("age",)
    pct_keywords   = ("pct", "percent", "rate", "ratio", "score")
    money_keywords = ("price", "cost", "salary", "amount", "revenue",
                      "income", "wage", "fee", "pay", "earning")
    count_keywords = ("count", "qty", "quantity", "num", "number", "total")

    for col in df.columns:
        col_lower  = col.lower()
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
    READ-ONLY — never modifies the DataFrame.
    """
    series = df[col]

    if rule == "no_future_year":
        mask        = series.notna() & (series > CURRENT_YEAR)
        count       = int(mask.sum())
        sample      = safe_json(series[mask].head(5).tolist())
        indices     = series[mask].index.tolist()
        passed      = count == 0
        explanation = explain_rule_no_future_year(col, count, CURRENT_YEAR)

    elif rule == "no_past_year":
        min_year    = int(params.get("min_year", 1900))
        mask        = series.notna() & (series < min_year)
        count       = int(mask.sum())
        sample      = safe_json(series[mask].head(5).tolist())
        indices     = series[mask].index.tolist()
        passed      = count == 0
        explanation = explain_rule_no_past_year(col, count, min_year)

    elif rule == "no_negative":
        mask        = series.notna() & (series < 0)
        count       = int(mask.sum())
        sample      = safe_json(series[mask].head(5).tolist())
        indices     = series[mask].index.tolist()
        passed      = count == 0
        explanation = explain_rule_no_negative(col, count)

    elif rule == "percentage":
        mask        = series.notna() & ((series < 0) | (series > 100))
        count       = int(mask.sum())
        sample      = safe_json(series[mask].head(5).tolist())
        indices     = series[mask].index.tolist()
        passed      = count == 0
        explanation = explain_rule_percentage(col, count)

    elif rule == "custom_range":
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

        if min_val is not None and max_val is not None:
            mask = series.notna() & ((series < min_val) | (series > max_val))
        elif min_val is not None:
            mask = series.notna() & (series < min_val)
        else:
            mask = series.notna() & (series > max_val)

        count       = int(mask.sum())
        sample      = safe_json(series[mask].head(5).tolist())
        indices     = series[mask].index.tolist()
        passed      = count == 0
        explanation = explain_rule_custom_range(
            col, count,
            min_val if min_val is not None else float("-inf"),
            max_val if max_val is not None else float("inf"),
        )

    elif rule == "no_nulls":
        mask        = series.isna()
        count       = int(mask.sum())
        sample      = []
        indices     = series[mask].index.tolist()
        passed      = count == 0
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
# TASK 6 — VALIDATION AND CROSS-CHECKING (original endpoints)
# =============================================================================

@task6.get("/task6/validation_info")
async def get_validation_info():
    df          = require_df()
    suggestions = detect_auto_rules(df)

    column_info: List[Dict[str, Any]] = []
    for col in df.columns:
        entry: Dict[str, Any] = {
            "name":  col,
            "dtype": str(df[col].dtype),
        }
        if df[col].dtype.kind in ("i", "f"):
            entry["min"] = safe_json(float(df[col].min()))
            entry["max"] = safe_json(float(df[col].max()))
        column_info.append(entry)

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
    column: str
    rule:   str
    params: Optional[Dict[str, Any]] = {}


class RunValidationPayload(BaseModel):
    rules: List[ValidationRule]


@task6.post("/task6/run_validation")
async def run_validation(payload: RunValidationPayload):
    df      = require_df()
    results = []
    errors  = []

    for rule_def in payload.rules:
        col    = rule_def.column
        rule   = rule_def.rule
        params = rule_def.params or {}

        if col not in df.columns:
            errors.append(f'Column "{col}" not found in the dataset.')
            continue

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
        "summary": (
            f"All {passed_count} rule(s) passed. Your data looks consistent."
            if failed_count == 0 else
            f"{failed_count} rule(s) failed out of {len(results)} checks. "
            "Review the violations below and decide how to fix them."
        ),
    }


class FixViolationPayload(BaseModel):
    column:      str
    rule:        str
    params:      Optional[Dict[str, Any]] = {}
    fix_method:  str
    replace_val: Optional[Any] = None


@task6.post("/task6/fix_violations")
async def fix_violations(payload: FixViolationPayload):
    df      = require_df().copy()
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

    result            = run_single_rule(df, col, rule, params)
    violation_count   = result["violation_count"]
    violation_indices = result["violation_indices"]

    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])

    if violation_count == 0:
        return {
            "success":   True,
            "applied":   [f'"{col}": no violations found — nothing to fix.'],
            "errors":    [],
            "new_shape": list(df.shape),
        }

    try:
        if payload.fix_method == "replace":
            if payload.replace_val is None:
                raise HTTPException(
                    status_code=400,
                    detail='fix_method "replace" requires a "replace_val".'
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
                    "Valid options: replace, set_null, drop_rows."
                )
            )

    except HTTPException:
        raise
    except Exception as e:
        errors.append(f"Unexpected error: {str(e)}")

    if applied and not errors:
        dataset_state.df = df

    return {
        "success":   len(errors) == 0,
        "applied":   applied,
        "errors":    errors,
        "new_shape": list(df.shape),
    }


# =============================================================================
# TASK 6 UPGRADE 1 — DATE RANGE VALIDATION
# =============================================================================

@task6.get("/task6/date_range_info")
async def get_date_range_info(
    min_year: int = 1900,
    max_year: int = CURRENT_YEAR,
):
    """
    Scans all datetime columns and numeric year columns for values
    outside the expected year range.

    Detects Unix epoch defaults (1970-01-01) and far-future date errors.
    """
    df            = require_df()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
    year_keywords = ("year", "yr", "born", "dob")
    year_cols     = [
        col for col in df.select_dtypes(include=["number"]).columns
        if any(kw in col.lower() for kw in year_keywords)
    ]

    reports = []

    for col in datetime_cols:
        series       = df[col].dropna()
        years        = series.dt.year
        out_of_range = (years < min_year) | (years > max_year)
        viol_count   = int(out_of_range.sum())
        if viol_count == 0:
            continue
        samples = [str(v) for v in series[out_of_range].head(5).tolist()]
        reports.append({
            "column":            col,
            "dtype":             "datetime64[ns]",
            "violation_count":   viol_count,
            "sample_violations": samples,
            "min_year":          min_year,
            "max_year":          max_year,
            "explanation":       explain_date_range_violation(
                col, viol_count, min_year, max_year
            ),
        })

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
            "explanation":       explain_date_range_violation(
                col, viol_count, min_year, max_year
            ),
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
    column:   str
    min_year: int = 1900
    max_year: int = CURRENT_YEAR


@task6.post("/task6/fix_date_range")
async def fix_date_range(payload: DateRangeFixPayload):
    """
    Sets out-of-range date values to NaT (datetime) or NaN (numeric year).
    All rows are preserved — only the invalid values are cleared.
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
        years        = df[col].dt.year
        out_of_range = (years < payload.min_year) | (years > payload.max_year)
        viol_count   = int(out_of_range.sum())
        df.loc[out_of_range, col] = pd.NaT
        applied.append(
            f'"{col}": set {viol_count} out-of-range date(s) to NaT. '
            f"Range enforced: [{payload.min_year}, {payload.max_year}]."
        )

    elif df[col].dtype.kind in ("i", "f"):
        out_of_range = (df[col] < payload.min_year) | (df[col] > payload.max_year)
        viol_count   = int(out_of_range.sum())
        df.loc[out_of_range, col] = np.nan
        applied.append(
            f'"{col}": set {viol_count} out-of-range year(s) to NaN. '
            f"Range enforced: [{payload.min_year}, {payload.max_year}]."
        )

    else:
        raise HTTPException(
            status_code=400,
            detail=f'"{col}" is not a datetime or numeric column.'
        )

    dataset_state.df = df
    return {
        "success":   True,
        "applied":   applied,
        "new_shape": list(df.shape),
    }


# =============================================================================
# TASK 6 UPGRADE 2 — CROSS-COLUMN VALIDATION
# =============================================================================

class CrossColumnPayload(BaseModel):
    rule:      str
    col_a:     str
    col_b:     str
    tolerance: Optional[int] = 2


@task6.post("/task6/cross_column_check")
async def cross_column_check(payload: CrossColumnPayload):
    """
    Runs a logical check across two columns and returns violations.
    READ-ONLY — never modifies the DataFrame.

    Rules:
        end_after_start       — col_b must be >= col_a
        age_matches_birth_year — age must match current_year - birth_year
        a_less_than_b         — col_a must be strictly less than col_b
        a_greater_than_b      — col_a must be strictly greater than col_b
    """
    df = require_df()

    for col in (payload.col_a, payload.col_b):
        if col not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f'Column "{col}" not found in the dataset.'
            )

    col_a        = payload.col_a
    col_b        = payload.col_b
    rule         = payload.rule
    both_present = df[col_a].notna() & df[col_b].notna()
    subset       = df[both_present]
    explanation  = ""

    violation_mask = pd.Series([False] * len(subset), index=subset.index)

    if rule == "end_after_start":
        violation_mask  = subset[col_b] < subset[col_a]
        violation_count = int(violation_mask.sum())
        explanation     = explain_end_before_start_violation(
            col_a, col_b, violation_count
        )

    elif rule == "age_matches_birth_year":
        expected_age    = CURRENT_YEAR - subset[col_b]
        diff            = (subset[col_a] - expected_age).abs()
        violation_mask  = diff > (payload.tolerance or 2)
        violation_count = int(violation_mask.sum())
        explanation     = explain_age_birth_year_violation(
            col_a, col_b, violation_count
        )

    elif rule == "a_less_than_b":
        violation_mask  = subset[col_a] >= subset[col_b]
        violation_count = int(violation_mask.sum())
        explanation     = (
            f"{violation_count} row(s) where '{col_a}' is not less than '{col_b}'."
        )

    elif rule == "a_greater_than_b":
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

    violation_indices = subset.index[violation_mask].tolist()
    violation_count   = len(violation_indices)

    sample_rows: List[Dict[str, Any]] = []
    for idx in violation_indices[:5]:
        sample_rows.append({
            "row_index": int(idx),
            col_a:       str(df.loc[idx, col_a]),
            col_b:       str(df.loc[idx, col_b]),
        })

    return {
        "what_is_cross_validation": explain_what_cross_column_validation_is(),
        "rule":              rule,
        "col_a":             col_a,
        "col_b":             col_b,
        "rows_checked":      int(both_present.sum()),
        "violation_count":   violation_count,
        "violation_indices": violation_indices,
        "sample_violations": sample_rows,
        "explanation":       explanation,
        "passed":            violation_count == 0,
        "next_step": (
            "Use POST /task6/fix_cross_column to set violating values to NaN."
            if violation_count > 0 else
            "No violations found — no action needed."
        ),
    }


class CrossColumnFixPayload(BaseModel):
    violation_indices: List[int]
    col_to_fix:        str


@task6.post("/task6/fix_cross_column")
async def fix_cross_column(payload: CrossColumnFixPayload):
    """
    Sets the violating values in col_to_fix to NaN or NaT.
    Only col_to_fix is modified — the other column is untouched.
    """
    df  = require_df().copy()
    col = payload.col_to_fix

    if col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f'Column "{col}" not found in the dataset.'
        )

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

    if str(df[col].dtype) == "datetime64[ns]":
        df.loc[payload.violation_indices, col] = pd.NaT
    else:
        df.loc[payload.violation_indices, col] = np.nan

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
# =============================================================================

@task6.get("/task6/pattern_info")
async def get_pattern_info():
    """
    Returns available patterns and auto-detects which text columns
    likely match each one (80%+ of sample values must match).
    """
    df        = require_df()
    text_cols = df.select_dtypes(include=["object"]).columns.tolist()

    suggestions: List[Dict[str, Any]] = []

    for col in text_cols:
        series  = df[col].dropna().astype(str)
        samples = series.head(20).tolist()
        if not samples:
            continue

        best_pattern: Optional[str] = None
        best_rate                   = 0.0

        for pattern_name, regex in _PATTERNS.items():
            matches    = sum(1 for v in samples if regex.match(v.strip()))
            match_rate = matches / len(samples)
            if match_rate > best_rate and match_rate >= 0.8:
                best_rate    = match_rate
                best_pattern = pattern_name

        if best_pattern:
            full_series = df[col].dropna().astype(str)
            pat_regex   = _PATTERNS[best_pattern]
            violations  = int(
                full_series.apply(
                    lambda v: not bool(pat_regex.match(v.strip()))
                ).sum()
            )
            suggestions.append({
                "column":            col,
                "suggested_pattern": best_pattern,
                "match_rate":        round(best_rate * 100, 1),
                "violation_count":   violations,
                "sample_values":     samples[:3],
            })

    return {
        "available_patterns":   list(_PATTERNS.keys()),
        "text_columns_scanned": len(text_cols),
        "suggestions":          suggestions,
        "message": (
            f"{len(suggestions)} column(s) matched a known pattern."
            if suggestions else
            "No text columns automatically matched a known pattern."
        ),
    }


class PatternCheckPayload(BaseModel):
    column:       str
    pattern:      str
    custom_regex: Optional[str] = None


@task6.post("/task6/check_pattern")
async def check_pattern(payload: PatternCheckPayload):
    """
    Validates every non-null value in a text column against a pattern.

    Built-in: email, phone_us, postal_code_us, postal_code_uk,
              url, date_iso, integer_string, decimal_string
    Custom:   provide any valid Python regex in custom_regex.
    """
    df  = require_df()
    col = payload.column

    if col not in df.columns:
        raise HTTPException(status_code=400, detail=f'Column "{col}" not found.')

    if payload.pattern == "custom":
        if not payload.custom_regex:
            raise HTTPException(
                status_code=400,
                detail='pattern "custom" requires custom_regex to be provided.'
            )
        try:
            regex = _re.compile(payload.custom_regex)
        except _re.error as e:
            raise HTTPException(
                status_code=400,
                detail=f'Invalid regex: {str(e)}'
            )
    elif payload.pattern in _PATTERNS:
        regex = _PATTERNS[payload.pattern]
    else:
        raise HTTPException(
            status_code=400,
            detail=(
                f'Unknown pattern "{payload.pattern}". '
                f'Valid: {list(_PATTERNS.keys()) + ["custom"]}'
            )
        )

    series = df[col].dropna().astype(str)
    total  = len(series)

    if total == 0:
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

    bad_mask        = series.apply(lambda v: not bool(regex.match(v.strip())))
    violation_count = int(bad_mask.sum())
    violation_vals  = series[bad_mask].head(5).tolist()
    violation_idx   = series[bad_mask].index.tolist()
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
        "explanation":       explain_pattern_violation(
            col, payload.pattern, violation_count
        ),
    }
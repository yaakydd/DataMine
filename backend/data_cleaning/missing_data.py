from fastapi import HTTPException, APIRouter
from pydantic import BaseModel
from typing import Dict, Any, Optional
from routes.dfState import dataset_state
from xAI.missingData_explainer import (
    explain_missing_severity,
    explain_missing_if_you_drop_rows,
    explain_missing_if_you_fill_mean,
    explain_missing_if_you_fill_median,
    explain_missing_if_you_fill_mode,
    explain_missing_if_you_fill_custom,
    explain_missing_if_you_drop_column,
    explain_missing_if_you_do_nothing,
)
import pandas as pd
import numpy as np

task2 = APIRouter()


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
    total_rows  = len(df)
    total_cells = df.size

    # ── Column-level analysis ─────────────────────────────────────────────────

    columns_with_missing = []

    for col in df.columns:
        missing_count = int(df[col].isna().sum())

        if missing_count == 0:
            continue

        pct = round((missing_count / total_rows) * 100, 2)

        if pct < 5:
            severity = "low"
        elif pct < 20:
            severity = "medium"
        else:
            severity = "high"

        dtype      = str(df[col].dtype)
        strategies = {}

        if dtype in ("int64", "float64") or df[col].dtype.kind in ("i", "f"):
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

        mode_val = df[col].mode()
        if not mode_val.empty:
            mode_display = mode_val[0]
            if hasattr(mode_display, "item"):
                mode_display = mode_display.item()
            strategies["fill_mode"] = {
                "label":       f"Fill with mode ({mode_display})",
                "explanation": explain_missing_if_you_fill_mode(col, mode_display),
            }

        strategies["fill_custom"] = {
            "label":       "Fill with a custom value",
            "explanation": explain_missing_if_you_fill_custom(col),
        }
        strategies["drop_rows"] = {
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
            "explanation":   explain_missing_severity(col, pct),
            "strategies":    strategies,
        })

    # ── Row-level analysis ────────────────────────────────────────────────────

    rows_with_missing = int(df.isnull().any(axis=1).sum())
    rows_missing_pct  = round((rows_with_missing / total_rows) * 100, 2)

    # ── Overall summary ───────────────────────────────────────────────────────

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
        "clean_columns":        len(df.columns) - len(columns_with_missing),
    }


# ── Request body model ────────────────────────────────────────────────────────

class MissingFixPayload(BaseModel):
    """
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
    """
    strategy: Dict[str, Dict[str, Any]]


@task2.post("/task2/fix_missing")
async def fix_missing(payload: MissingFixPayload):
    """
    Applies the user's chosen fix strategy to each column they specified.

    Works on a .copy() — dataset_state.df is never touched until all
    operations are done and at least one succeeded.
    """
    df      = require_df().copy()
    applied = []
    errors  = []

    for col, config in payload.strategy.items():

        if col not in df.columns:
            errors.append(f'"{col}" not found in the dataset.')
            continue

        method = config.get("method", "")

        try:
            if method == "fill_mean":
                if df[col].dtype.kind not in ("i", "f"):
                    errors.append(
                        f'Cannot fill "{col}" with mean — it is not a numeric column. '
                        f'Use fill_mode or fill_custom instead.'
                    )
                    continue
                mean_val = df[col].mean()
                df[col].fillna(mean_val, inplace=True)
                applied.append(
                    f'"{col}": filled {df[col].isna().sum()} NaN(s) '
                    f'with mean = {mean_val:.4f}'
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
                    errors.append(
                        f'"{col}": cannot compute mode — column may be all NaN.'
                    )
                    continue
                mode_val = mode_series[0]
                df[col].fillna(mode_val, inplace=True)
                applied.append(f'"{col}": filled NaN(s) with mode = {mode_val}')

            elif method == "fill_custom":
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
                df.dropna(subset=[col], inplace=True)
                df.reset_index(drop=True, inplace=True)
                rows_dropped = before - len(df)
                applied.append(
                    f'"{col}": dropped {rows_dropped} row(s) where value was missing. '
                    f'{len(df)} rows remain.'
                )

            elif method == "drop_column":
                df.drop(columns=[col], inplace=True)
                applied.append(f'"{col}": column dropped entirely.')

            elif method == "do_nothing":
                applied.append(f'"{col}": left as-is (NaN values kept).')

            else:
                errors.append(
                    f'"{col}": unknown method "{method}". '
                    f'Valid options: fill_mean, fill_median, fill_mode, '
                    f'fill_custom, drop_rows, drop_column, do_nothing.'
                )

        except Exception as e:
            errors.append(f'"{col}": unexpected error — {str(e)}')

    # ── Save back ─────────────────────────────────────────────────────────────

    if applied:
        dataset_state.df = df

    # ── Return ────────────────────────────────────────────────────────────────

    remaining_missing = int(df.isnull().sum().sum())

    return {
        "success":           len(errors) == 0,
        "applied":           applied,
        "errors":            errors,
        "new_shape":         list(df.shape),
        "remaining_missing": remaining_missing,
    }
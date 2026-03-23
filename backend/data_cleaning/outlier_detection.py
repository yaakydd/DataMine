from fastapi import HTTPException, APIRouter
from pydantic import BaseModel
from routes.dfState import dataset_state
from xAI.outliers import (
    explain_what_outliers_are,
    explain_zscore_method,
    explain_iqr_method,
    explain_column_outliers,
    explain_outlier_if_you_cap_iqr,
    explain_outlier_if_you_cap_zscore,
    explain_outlier_if_you_remove_rows,
    explain_outlier_if_you_do_nothing,
)
import pandas as pd
import numpy as np
from typing import Dict, Optional

task4 = APIRouter()


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
    the response without errors. Called on any outlier sample values
    before they are returned to the frontend.
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
# TASK 4 — OUTLIER DETECTION
#
# Two endpoints:
#
#   GET  /task4/outliers_info
#        Runs both Z-score and IQR detection on every numeric column.
#        Returns counts, thresholds, actual outlier values (samples),
#        and full xAI explanations for every fix strategy.
#        Never modifies the data.
#
#   POST /task4/fix_outliers
#        Applies the user's chosen fix strategy per column:
#          cap_iqr    — clip values to IQR fences (Winsorisation)
#          cap_zscore — clip values to ±3 standard deviations
#          remove     — drop rows where the value is an outlier
#          do_nothing — leave the column unchanged
# =============================================================================

@task4.get("/task4/outliers_info")
async def get_outliers_info():
    """
    Runs outlier detection on every numeric column using both methods.

    Z-score detection:
        For each value, calculate how many standard deviations it sits
        from the column mean. Flag it if that number exceeds 3 or -3.
        Formula: z = (value - mean) / std
        A value with z > 3 is more than 3 standard deviations above average.

    IQR detection:
        Calculate Q1 (25th percentile) and Q3 (75th percentile).
        IQR = Q3 - Q1
        Lower fence = Q1 - 1.5 * IQR
        Upper fence = Q3 + 1.5 * IQR
        Any value outside these fences is flagged.

    Why run both?
        They disagree on borderline cases. Z-score is sensitive to the mean
        and std — which are themselves affected by outliers. IQR is based on
        percentiles and is therefore more stable on skewed data. Showing
        both gives the user a more complete picture.

    What the response contains per column:
        - zscore_outlier_count:  how many values Z-score flagged
        - iqr_outlier_count:     how many values IQR flagged
        - column statistics:     mean, std, min, max, Q1, Q3, IQR fences
        - sample_outliers:       up to 5 actual outlier values
        - all_strategies:        fix options with explanations
        - column_explanation:    plain-English summary of findings
    """
    df = require_df()

    # Only analyse numeric columns — outlier detection on text makes no sense
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        return {
            "what_are_outliers":  explain_what_outliers_are(),
            "zscore_explanation": explain_zscore_method(),
            "iqr_explanation":    explain_iqr_method(),
            "columns_analysed":   0,
            "column_reports":     [],
            "message": (
                "No numeric columns found in the dataset. "
                "Outlier detection only applies to numeric data. "
                "If you have numeric columns stored as text, "
                "go back to Task 1 and convert their dtype first."
            ),
        }

    column_reports = []

    for col in numeric_cols:

        # Drop NaN before calculating — NaN propagates through all numpy
        # operations and would make every result NaN
        series = df[col].dropna()

        # Need at least 4 values to compute meaningful statistics
        # (Q1 and Q3 need enough points to be meaningful)
        if len(series) < 4:
            column_reports.append({
                "column":  col,
                "skipped": True,
                "reason":  f"Too few non-null values ({len(series)}) to detect outliers reliably.",
            })
            continue

        # ── Column statistics ─────────────────────────────────────────────────

        mean    = float(series.mean())
        std     = float(series.std())
        col_min = float(series.min())
        col_max = float(series.max())
        q1      = float(series.quantile(0.25))
        q3      = float(series.quantile(0.75))
        iqr     = q3 - q1

        # The fences define the "acceptable" range for IQR detection
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr

        # ── Z-score detection ─────────────────────────────────────────────────

        # Guard against std == 0 (all values identical — no outliers possible)
        if std == 0:
            zscore_mask  = pd.Series([False] * len(series), index=series.index)
            zscore_count = 0
        else:
            # Calculate Z-score for every value in the column
            # Then flag any value whose absolute Z-score exceeds 3
            z_scores     = (series - mean) / std
            zscore_mask  = z_scores.abs() > 3
            zscore_count = int(zscore_mask.sum())

        # ── IQR detection ─────────────────────────────────────────────────────

        iqr_mask  = (series < lower_fence) | (series > upper_fence)
        iqr_count = int(iqr_mask.sum())

        # ── Sample outlier values ─────────────────────────────────────────────

        # Show the actual extreme values so the user can judge whether they
        # look like errors or genuine extremes.
        # We use IQR mask for samples since IQR is more reliable on skewed data.
        outlier_values = safe_json(
            series[iqr_mask].head(5).tolist()
        )

        # ── Strategies ───────────────────────────────────────────────────────

        # Precompute the Z-score clip boundaries for the explanation
        zscore_lower = mean - 3 * std
        zscore_upper = mean + 3 * std

        all_strategies = {
            "cap_iqr": {
                "label":       f"Cap to IQR fences [{lower_fence:.2f}, {upper_fence:.2f}]",
                "explanation": explain_outlier_if_you_cap_iqr(
                    col, lower_fence, upper_fence
                ),
            },
            "cap_zscore": {
                "label":       f"Cap to ±3 std [{zscore_lower:.2f}, {zscore_upper:.2f}]",
                "explanation": explain_outlier_if_you_cap_zscore(
                    col, zscore_lower, zscore_upper
                ),
            },
            "remove": {
                "label":       f"Remove {iqr_count} outlier row(s)",
                "explanation": explain_outlier_if_you_remove_rows(col, iqr_count),
            },
            "do_nothing": {
                "label":       "Leave as-is",
                "explanation": explain_outlier_if_you_do_nothing(col),
            },
        }

        column_reports.append({
            "column": col,
            "skipped": False,

            # Detection results
            "zscore_outlier_count": zscore_count,
            "iqr_outlier_count":    iqr_count,
            "has_outliers":         zscore_count > 0 or iqr_count > 0,

            # Column statistics — shown in the detail panel
            "stats": {
                "mean":         round(mean,        4),
                "std":          round(std,         4),
                "min":          round(col_min,     4),
                "max":          round(col_max,     4),
                "q1":           round(q1,          4),
                "q3":           round(q3,          4),
                "iqr":          round(iqr,         4),
                "lower_fence":  round(lower_fence, 4),
                "upper_fence":  round(upper_fence, 4),
            },

            # Actual outlier values so the user can see what was flagged
            "sample_outliers": outlier_values,

            # Plain-English summary for this specific column
            "column_explanation": explain_column_outliers(
                col, zscore_count, iqr_count,
                col_min, col_max, mean, std,
                q1, q3, lower_fence, upper_fence,
            ),

            # Fix strategies with labels and explanations
            "all_strategies": all_strategies,
        })

    # Summary counts for the top banner
    total_cols_with_outliers = sum(
        1 for r in column_reports
        if not r.get("skipped") and r.get("has_outliers")
    )

    return {
        # Shown once at the top of the page before any column results
        "what_are_outliers":  explain_what_outliers_are(),
        "zscore_explanation": explain_zscore_method(),
        "iqr_explanation":    explain_iqr_method(),

        # Summary
        "columns_analysed":          len(numeric_cols),
        "columns_with_outliers":     total_cols_with_outliers,
        "columns_clean":             len(numeric_cols) - total_cols_with_outliers,

        # Per-column reports
        "column_reports": column_reports,
    }


# ── Request body model ────────────────────────────────────────────────────────

class OutlierFixPayload(BaseModel):
    """
    strategy is a dict where:
      - the key   is the column name
      - the value is the method to apply to that column

    Valid methods: cap_iqr, cap_zscore, remove, do_nothing

    Example:
        {
            "strategy": {
                "age":    "cap_iqr",
                "salary": "cap_zscore",
                "score":  "remove",
                "height": "do_nothing"
            }
        }
    """
    strategy: Dict[str, str]


@task4.post("/task4/fix_outliers")
async def fix_outliers(payload: OutlierFixPayload):
    """
    Applies the user's chosen outlier fix strategy to each specified column.

    cap_iqr:
        Uses numpy's .clip() to pull all values into the IQR fence range.
        Values below the lower fence are set to the lower fence value.
        Values above the upper fence are set to the upper fence value.
        No rows are removed — only the extreme values change.
        This is Winsorisation.

    cap_zscore:
        Same as cap_iqr but the clip boundaries are mean ± 3 * std.
        Values outside ±3 standard deviations are clamped to those boundaries.

    remove:
        Drops any row where the column value falls outside the IQR fences.
        The index is reset after dropping so row numbering stays clean.

    do_nothing:
        No change. Logged in applied[] for completeness.
    """
    df      = require_df().copy()
    applied = []
    errors  = []

    for col, method in payload.strategy.items():

        if col not in df.columns:
            errors.append(f'"{col}" not found in the dataset.')
            continue

        if df[col].dtype.kind not in ("i", "f"):
            errors.append(
                f'"{col}" is not numeric (dtype: {df[col].dtype}). '
                "Outlier fixing only applies to numeric columns."
            )
            continue

        series = df[col].dropna()

        if len(series) < 4:
            errors.append(
                f'"{col}" has too few non-null values ({len(series)}) '
                "to compute reliable outlier boundaries."
            )
            continue

        try:
            if method == "cap_iqr":
                q1          = series.quantile(0.25)
                q3          = series.quantile(0.75)
                iqr         = q3 - q1
                lower_fence = q1 - 1.5 * iqr
                upper_fence = q3 + 1.5 * iqr

                # .clip() replaces every value below lower with lower,
                # and every value above upper with upper.
                # lower_inclusive / upper_inclusive ensures the fence
                # values themselves are never flagged as outliers.
                df[col] = df[col].clip(
                    lower=lower_fence,
                    upper=upper_fence
                )
                applied.append(
                    f'"{col}": capped to IQR fences '
                    f'[{lower_fence:.4f}, {upper_fence:.4f}]. '
                    f'No rows removed — extreme values pulled in.'
                )

            elif method == "cap_zscore":
                mean        = series.mean()
                std         = series.std()

                if std == 0:
                    applied.append(
                        f'"{col}": all values are identical — no Z-score capping needed.'
                    )
                    continue

                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std

                df[col] = df[col].clip(
                    lower=lower_bound,
                    upper=upper_bound
                )
                applied.append(
                    f'"{col}": capped to ±3 std '
                    f'[{lower_bound:.4f}, {upper_bound:.4f}]. '
                    f'No rows removed — extreme values pulled in.'
                )

            elif method == "remove":
                q1          = series.quantile(0.25)
                q3          = series.quantile(0.75)
                iqr         = q3 - q1
                lower_fence = q1 - 1.5 * iqr
                upper_fence = q3 + 1.5 * iqr

                before = len(df)

                # Keep only rows where the value is within the fences
                # OR where the value is NaN (we don't touch missing values here —
                # that is Task 2's responsibility)
                within_fences = (df[col] >= lower_fence) & (df[col] <= upper_fence)
                is_null       = df[col].isna()
                df            = df[within_fences | is_null]

                df.reset_index(drop=True, inplace=True)
                rows_removed = before - len(df)

                applied.append(
                    f'"{col}": removed {rows_removed} outlier row(s) '
                    f'outside IQR fences [{lower_fence:.4f}, {upper_fence:.4f}]. '
                    f'{len(df)} rows remain.'
                )

            elif method == "do_nothing":
                applied.append(f'"{col}": left as-is.')

            else:
                errors.append(
                    f'"{col}": unknown method "{method}". '
                    "Valid options: cap_iqr, cap_zscore, remove, do_nothing."
                )

        except Exception as e:
            errors.append(f'"{col}": unexpected error — {str(e)}')

    # ── Save back ─────────────────────────────────────────────────────────────

    if applied and not all("left as-is" in a for a in applied):
        dataset_state.df = df

    # ── Return ────────────────────────────────────────────────────────────────

    return {
        "success":     len(errors) == 0,
        "applied":     applied,
        "errors":      errors,
        "new_shape":   list(df.shape),
    }
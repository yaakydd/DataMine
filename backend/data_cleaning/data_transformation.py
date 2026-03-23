from fastapi import HTTPException, APIRouter
from pydantic import BaseModel
from routes.dfState import dataset_state
from xAI.transformation import (
    explain_what_skewness_is,
    explain_skewness_severity,
    explain_log_transform,
    explain_sqrt_transform,
    explain_boxcox_transform,
    explain_yeojohnson_transform,
    explain_reciprocal_transform,
    explain_transform_result,
    explain_what_scaling_is,
    explain_minmax_scaling,
    explain_zscore_scaling,
    explain_robust_scaling,
    explain_scaling_result,
    explain_what_date_parsing_is,
    explain_date_column_found,
    explain_date_parse_result,
)
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Optional, List, Any


task5 = APIRouter()


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
# TASK 5 — SKEWED DATA AND TRANSFORMATIONS
# =============================================================================

def classify_skewness(skewness: float) -> dict:
    abs_skew  = abs(skewness)
    direction = "right" if skewness > 0 else "left" if skewness < 0 else "none"

    if abs_skew < 0.5:
        level = "symmetric"
    elif abs_skew < 1.0:
        level = "moderate"
    else:
        level = "high"

    return {"level": level, "direction": direction}


def recommend_transform(skewness: float, has_zeros: bool, has_negatives: bool) -> str:
    abs_skew = abs(skewness)

    if abs_skew < 0.5:
        return "do_nothing"
    if has_negatives:
        return "yeojohnson"
    if skewness > 0:
        if abs_skew >= 1.0:
            return "log1p" if has_zeros else "log"
        return "sqrt"
    return "yeojohnson"


@task5.get("/task5/skewness_info")
async def get_skewness_info():
    df           = require_df()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        return {
            "what_is_skewness": explain_what_skewness_is(),
            "columns_analysed": 0,
            "column_reports":   [],
            "message": (
                "No numeric columns found. Skewness analysis only applies to "
                "numeric data. Convert any text columns storing numbers in Task 1 first."
            ),
        }

    column_reports = []

    for col in numeric_cols:
        series = df[col].dropna()

        if len(series) < 3:
            column_reports.append({
                "column":  col,
                "skipped": True,
                "reason":  f"Too few non-null values ({len(series)}) for skewness calculation.",
            })
            continue

        skewness       = round(float(series.skew()), 4)
        classification = classify_skewness(skewness)
        has_zeros      = bool((series == 0).any())
        has_negatives  = bool((series < 0).any())
        col_min        = float(series.min())
        col_max        = float(series.max())
        recommended    = recommend_transform(skewness, has_zeros, has_negatives)

        all_strategies: Dict[str, Any] = {}

        if not has_zeros and not has_negatives:
            all_strategies["log"] = {
                "label":       "Log transform — log(x)",
                "explanation": explain_log_transform(col, has_zeros=False),
                "available":   True,
            }

        if not has_negatives:
            all_strategies["log1p"] = {
                "label":       "Log1p transform — log(1 + x)",
                "explanation": explain_log_transform(col, has_zeros=True),
                "available":   True,
            }

        if not has_negatives:
            all_strategies["sqrt"] = {
                "label":       "Square root transform — sqrt(x)",
                "explanation": explain_sqrt_transform(col),
                "available":   True,
            }

        if not has_zeros and not has_negatives:
            all_strategies["boxcox"] = {
                "label":       "Box-Cox transform (finds optimal lambda automatically)",
                "explanation": explain_boxcox_transform(col),
                "available":   True,
            }

        all_strategies["yeojohnson"] = {
            "label":       "Yeo-Johnson transform (works on any values including negatives)",
            "explanation": explain_yeojohnson_transform(col),
            "available":   True,
        }

        if not has_zeros and not has_negatives:
            all_strategies["reciprocal"] = {
                "label":       "Reciprocal transform — 1/x",
                "explanation": explain_reciprocal_transform(col),
                "available":   True,
            }

        all_strategies["do_nothing"] = {
            "label":       "Leave as-is",
            "explanation": "No transformation applied. The column stays in its current form.",
            "available":   True,
        }

        column_reports.append({
            "column":          col,
            "skipped":         False,
            "skewness":        skewness,
            "level":           classification["level"],
            "direction":       classification["direction"],
            "has_zeros":       has_zeros,
            "has_negatives":   has_negatives,
            "col_min":         round(col_min, 4),
            "col_max":         round(col_max, 4),
            "recommended":     recommended,
            "needs_transform": classification["level"] != "symmetric",
            "explanation":     explain_skewness_severity(col, skewness),
            "all_strategies":  all_strategies,
        })

    needs_transform = sum(
        1 for r in column_reports
        if not r.get("skipped") and r.get("needs_transform")
    )

    return {
        "what_is_skewness":          explain_what_skewness_is(),
        "columns_analysed":          len(numeric_cols),
        "columns_needing_transform": needs_transform,
        "columns_symmetric":         len(numeric_cols) - needs_transform,
        "column_reports":            column_reports,
    }


class SkewnessFixPayload(BaseModel):
    """
    strategy maps column name to transformation method.
    Valid methods: log, log1p, sqrt, boxcox, yeojohnson, reciprocal, do_nothing
    """
    strategy: Dict[str, str]


@task5.post("/task5/fix_skewness")
async def fix_skewness(payload: SkewnessFixPayload):
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
                "Transformations only apply to numeric columns."
            )
            continue

        series = df[col].dropna()

        if len(series) < 3:
            errors.append(
                f'"{col}" has too few non-null values ({len(series)}) '
                "for a meaningful transformation."
            )
            continue

        skew_before = round(float(series.skew()), 4)

        try:
            if method == "log":
                if (series <= 0).any():
                    errors.append(
                        f'"{col}": log requires all values > 0. '
                        "Use log1p or yeojohnson instead."
                    )
                    continue
                df[col] = np.log(df[col])

            elif method == "log1p":
                if (series < 0).any():
                    errors.append(
                        f'"{col}": log1p requires all values >= 0. '
                        "Use yeojohnson instead."
                    )
                    continue
                df[col] = np.log1p(df[col])

            elif method == "sqrt":
                if (series < 0).any():
                    errors.append(
                        f'"{col}": sqrt requires all values >= 0. '
                        "Use yeojohnson instead."
                    )
                    continue
                df[col] = np.sqrt(df[col])

            elif method == "boxcox":
                if (series <= 0).any():
                    errors.append(
                        f'"{col}": Box-Cox requires all values > 0. '
                        "Use yeojohnson instead."
                    )
                    continue
                transformed, lam = stats.boxcox(series)
                result = df[col].copy()
                result[result.notna()] = transformed
                df[col] = result
                applied.append(
                    f'"{col}": Box-Cox applied. Optimal lambda = {lam:.4f}. '
                    + explain_transform_result(
                        col, "Box-Cox", skew_before,
                        round(float(df[col].dropna().skew()), 4)
                    )
                )
                continue

            elif method == "yeojohnson":
                transformed, lam = stats.yeojohnson(series)
                result = df[col].copy()
                result[result.notna()] = transformed
                df[col] = result
                applied.append(
                    f'"{col}": Yeo-Johnson applied. Optimal lambda = {lam:.4f}. '
                    + explain_transform_result(
                        col, "Yeo-Johnson", skew_before,
                        round(float(df[col].dropna().skew()), 4)
                    )
                )
                continue

            elif method == "reciprocal":
                if (series == 0).any():
                    errors.append(
                        f'"{col}": reciprocal undefined for zeros. '
                        "Use log1p or sqrt instead."
                    )
                    continue
                if (series < 0).any():
                    errors.append(
                        f'"{col}": reciprocal works best on positive data. '
                        "Use yeojohnson instead."
                    )
                    continue
                df[col] = 1 / df[col]

            elif method == "do_nothing":
                applied.append(f'"{col}": no transformation applied.')
                continue

            else:
                errors.append(
                    f'"{col}": unknown method "{method}". '
                    "Valid: log, log1p, sqrt, boxcox, yeojohnson, reciprocal, do_nothing."
                )
                continue

            skew_after = round(float(df[col].dropna().skew()), 4)
            applied.append(
                explain_transform_result(col, method, skew_before, skew_after)
            )

        except Exception as e:
            errors.append(f'"{col}": unexpected error during {method} — {str(e)}')

    if [a for a in applied if "no transformation" not in a]:
        dataset_state.df = df

    return {
        "success":   len(errors) == 0,
        "applied":   applied,
        "errors":    errors,
        "new_shape": list(df.shape),
    }


# =============================================================================
# TASK 5 UPGRADE A — FEATURE SCALING
# =============================================================================

@task5.get("/task5/scaling_info")
async def get_scaling_info():
    """
    Scans every numeric column and returns its range, mean, and std.
    Flags which columns have very different magnitudes from each other
    — the signal that scaling is needed.
    Also detects columns already scaled to [0,1] or mean=0/std=1.
    """
    df           = require_df()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        return {
            "what_is_scaling":  explain_what_scaling_is(),
            "columns_analysed": 0,
            "column_reports":   [],
            "message":          "No numeric columns found.",
        }

    column_reports = []

    for col in numeric_cols:
        series = df[col].dropna()

        if len(series) < 2:
            continue

        col_min   = float(series.min())
        col_max   = float(series.max())
        col_mean  = float(series.mean())
        col_std   = float(series.std())
        col_range = col_max - col_min

        already_minmax = col_min >= -0.01 and col_max <= 1.01 and col_range <= 1.01
        already_zscore = abs(col_mean) < 0.1 and abs(col_std - 1.0) < 0.1

        column_reports.append({
            "column":          col,
            "min":             round(col_min,   4),
            "max":             round(col_max,   4),
            "mean":            round(col_mean,  4),
            "std":             round(col_std,   4),
            "range":           round(col_range, 4),
            "already_minmax":  already_minmax,
            "already_zscore":  already_zscore,
            "needs_scaling":   not already_minmax and not already_zscore,
            "available_methods": {
                "minmax": "Rescale to [0, 1] — good for bounded data without outliers",
                "zscore": "Rescale to mean=0, std=1 — good for data with outliers",
                "robust": "Rescale using median and IQR — best when outliers are present",
            },
        })

    column_reports.sort(key=lambda x: x["range"], reverse=True)

    return {
        "what_is_scaling":         explain_what_scaling_is(),
        "columns_analysed":        len(column_reports),
        "columns_needing_scaling": sum(1 for c in column_reports if c["needs_scaling"]),
        "column_reports":          column_reports,
    }


class ScalingPayload(BaseModel):
    """
    strategy maps column name to scaling method.
    Valid methods: minmax, zscore, robust
    """
    strategy: Dict[str, str]


@task5.post("/task5/scale_columns")
async def scale_columns(payload: ScalingPayload):
    """
    Applies the chosen scaling method to each specified column.
    NaN values are preserved throughout all operations.

    minmax  — (x - min) / (max - min) → result in [0, 1]
    zscore  — (x - mean) / std        → result has mean=0, std=1
    robust  — (x - median) / IQR      → centred around 0, outlier-resistant
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
                "Scaling only applies to numeric columns."
            )
            continue

        series = df[col].dropna()

        if len(series) < 2:
            errors.append(f'"{col}" has too few non-null values to scale.')
            continue

        try:
            if method == "minmax":
                col_min   = float(series.min())
                col_max   = float(series.max())
                col_range = col_max - col_min

                if col_range == 0:
                    applied.append(
                        f'"{col}": all values identical — Min-Max scaling not possible.'
                    )
                    continue

                df[col] = (df[col] - col_min) / col_range
                applied.append(
                    explain_minmax_scaling(col, col_min, col_max) + " " +
                    explain_scaling_result(
                        col, "Min-Max",
                        float(df[col].min()),
                        float(df[col].max()),
                        float(df[col].mean()),
                    )
                )

            elif method == "zscore":
                col_mean = float(series.mean())
                col_std  = float(series.std())

                if col_std == 0:
                    applied.append(
                        f'"{col}": standard deviation is 0 — Z-score scaling not possible.'
                    )
                    continue

                df[col] = (df[col] - col_mean) / col_std
                applied.append(
                    explain_zscore_scaling(col, col_mean, col_std) + " " +
                    explain_scaling_result(
                        col, "Z-score",
                        float(df[col].min()),
                        float(df[col].max()),
                        float(df[col].mean()),
                    )
                )

            elif method == "robust":
                col_median = float(series.median())
                col_q1     = float(series.quantile(0.25))
                col_q3     = float(series.quantile(0.75))
                col_iqr    = col_q3 - col_q1

                if col_iqr == 0:
                    df[col] = df[col] - col_median
                    applied.append(
                        f'"{col}": IQR is 0 — applied median centring only '
                        f"(subtracted median = {col_median:.4f})."
                    )
                    continue

                df[col] = (df[col] - col_median) / col_iqr
                applied.append(
                    explain_robust_scaling(col, col_median, col_iqr) + " " +
                    explain_scaling_result(
                        col, "Robust",
                        float(df[col].min()),
                        float(df[col].max()),
                        float(df[col].mean()),
                    )
                )

            else:
                errors.append(
                    f'"{col}": unknown method "{method}". '
                    "Valid options: minmax, zscore, robust."
                )

        except Exception as e:
            errors.append(f'"{col}": unexpected error — {str(e)}')

    if applied:
        dataset_state.df = df

    return {
        "success":   len(errors) == 0,
        "applied":   applied,
        "errors":    errors,
        "new_shape": list(df.shape),
    }


# =============================================================================
# TASK 5 UPGRADE B — DATE AND TIME PARSING
# =============================================================================

@task5.get("/task5/date_columns_info")
async def get_date_columns_info():
    """
    Scans all text (object dtype) columns and detects those that likely
    contain dates stored as strings.

    Uses a 70% parse-success threshold — high enough to avoid false
    positives on free-text columns, low enough to catch real date
    columns that have a few bad values mixed in.
    """
    df        = require_df()
    text_cols = df.select_dtypes(include=["object"]).columns.tolist()

    detected: List[Dict[str, Any]] = []

    for col in text_cols:
        series  = df[col].dropna().astype(str)
        samples = series.head(10).tolist()

        if not samples:
            continue

        parsed       = pd.to_datetime(pd.Series(samples), errors="coerce")
        success      = int(parsed.notna().sum())
        success_rate = success / len(samples)

        if success_rate >= 0.7:
            detected.append({
                "column":        col,
                "sample_values": samples[:5],
                "parse_rate":    round(success_rate * 100, 1),
                "explanation":   explain_date_column_found(col, samples[:3]),
            })

    return {
        "what_is_date_parsing": explain_what_date_parsing_is(),
        "text_columns_scanned": len(text_cols),
        "date_columns_found":   len(detected),
        "detected_columns":     detected,
        "message": (
            "No text columns detected as date columns."
            if not detected else
            f"{len(detected)} column(s) detected as likely date columns."
        ),
    }


class DateParsePayload(BaseModel):
    """
    columns:     list of column names to convert to datetime
    date_format: optional strftime format e.g. "%d/%m/%Y"
                 if omitted, pandas infers the format automatically
    """
    columns:     List[str]
    date_format: Optional[str] = None


@task5.post("/task5/parse_dates")
async def parse_dates(payload: DateParsePayload):
    """
    Converts specified text columns to datetime64[ns].

    If date_format is provided, pandas parses strictly using that template.
    If not, pandas uses automatic format inference — slower but handles
    mixed formats within the same column.

    Values that cannot be parsed become NaT (Not a Time).
    """
    df      = require_df().copy()
    applied = []
    errors  = []

    for col in payload.columns:

        if col not in df.columns:
            errors.append(f'"{col}" not found in the dataset.')
            continue

        if str(df[col].dtype) == "datetime64[ns]":
            applied.append(f'"{col}": already datetime — no change needed.')
            continue

        total = int(df[col].notna().sum())

        try:
            if payload.date_format:
                df[col] = pd.to_datetime(
                    df[col],
                    format=payload.date_format,
                    errors="coerce",
                )
            else:
                df[col] = pd.to_datetime(
                    df[col],
                    infer_datetime_format=True,
                    errors="coerce",
                )

            parsed_count = int(df[col].notna().sum())
            failed_count = int(df[col].isna().sum())
            new_failures = max(0, failed_count - (len(df) - total))

            applied.append(
                explain_date_parse_result(col, parsed_count, new_failures, total)
            )

        except Exception as e:
            errors.append(f'"{col}": unexpected error — {str(e)}')

    if applied:
        dataset_state.df = df

    return {
        "success":   len(errors) == 0,
        "applied":   applied,
        "errors":    errors,
        "new_shape": list(df.shape),
    }
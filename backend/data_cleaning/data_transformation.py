# =============================================================================
# data_transformation.py
#
# This file handles Task 5 of DataMine: transforming numeric columns to fix
# skewness, scaling columns to comparable ranges, and parsing date strings
# into proper datetime columns.
#
# It defines 6 endpoints across 3 sections:
#
#   Section 1 — Skewness and transformations
#     GET  /task5/skewness_info    → scan numeric columns for skewness,
#                                    return severity, direction, and all
#                                    available transforms. Never modifies data.
#     POST /task5/fix_skewness     → apply a chosen transform per column.
#
#   Section 2 — Feature scaling
#     GET  /task5/scaling_info     → scan numeric columns for scale mismatches,
#                                    detect already-scaled columns. Never modifies data.
#     POST /task5/scale_columns    → apply minmax, zscore, or robust scaling.
#
#   Section 3 — Date parsing
#     GET  /task5/date_columns_info  → detect text columns that contain dates.
#     POST /task5/parse_dates        → convert those columns to datetime64[ns].
# =============================================================================


# HTTPException → return a clean error + HTTP status code instead of a Python crash
# APIRouter     → groups all Task 5 endpoints under one object registered in main.py
from fastapi import HTTPException, APIRouter

# BaseModel → Pydantic base class that auto-validates incoming JSON request bodies
from pydantic import BaseModel

# Shared singleton holding the current DataFrame in memory across all routers
from State.dfState import dataset_state

# Each function returns a plain-English explanation string for the xAI panel.
# Keeping explanation text in a separate file means wording can change
# without touching any transformation logic.
from xAI.transformation import (
    explain_what_skewness_is,        # general intro to skewness
    explain_skewness_severity,       # how bad the skewness is for a specific column
    explain_log_transform,           # what log / log1p transform does
    explain_sqrt_transform,          # what square root transform does
    explain_boxcox_transform,        # what Box-Cox transform does
    explain_yeojohnson_transform,    # what Yeo-Johnson transform does
    explain_reciprocal_transform,    # what 1/x transform does
    explain_transform_result,        # before/after skewness comparison
    explain_what_scaling_is,         # general intro to feature scaling
    explain_minmax_scaling,          # what Min-Max scaling does
    explain_zscore_scaling,          # what Z-score scaling does
    explain_robust_scaling,          # what Robust (median/IQR) scaling does
    explain_scaling_result,          # before/after range comparison
    explain_what_date_parsing_is,    # general intro to date parsing
    explain_date_column_found,       # why a column was flagged as a date column
    explain_date_parse_result,       # how many values parsed successfully
)

import pandas as pd   # main data manipulation library
import numpy as np    # numerical operations and array math

# scipy.stats → provides scipy's Box-Cox and Yeo-Johnson implementations.
# These are the standard library functions for power transformations.
from scipy import stats

# Snapshot store saves a copy of the DataFrame before every write.
# Powers the undo/rollback system — up to 20 snapshots kept.
from State.snapshotState import snapshot_store

# Type hints used in function signatures and Pydantic models
from typing import Dict, Optional, List, Any

# All Task 5 endpoints are attached to this router.
# main.py registers it with: app.include_router(task5, prefix="/api")
task5 = APIRouter()


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


def _skew(series: pd.Series) -> float:
    """
    Computes Fisher's corrected skewness as a guaranteed plain Python float.

    WHY WE DON'T USE series.skew() DIRECTLY:
        series.skew() returns a pandas Scalar — a wide union type that can be
        str, bool, datetime, complex, or float. Neither float() nor .item()
        satisfies Pylance on a Scalar because most of those types don't support
        either operation. This causes type-checker errors that can't be silenced
        cleanly.

    THE FIX — compute skewness manually on a plain numpy array:
        series.to_numpy(dtype=float) gives a clean np.ndarray[float64].
        All numpy array operations (.mean(), .std()) return np.float64.
        np.float64 always supports .item() → plain Python float.
        This completely bypasses the pandas Scalar type system.

    THE FORMULA — Fisher's corrected skewness (identical to pandas .skew()):
        1. Standardise every value: z = (x - mean) / std
        2. Cube each standardised value: z^3
        3. Average the cubed values: skew_raw = mean(z^3)
        4. Apply sample correction: skew * sqrt(n * (n-1)) / (n - 2)
           This correction adjusts for the fact that a sample tends to
           underestimate the skewness of the underlying population.

    SKEWNESS INTERPRETATION:
        0         → perfectly symmetric
        positive  → right-skewed (long tail to the right, e.g. income data)
        negative  → left-skewed  (long tail to the left)
        |value| < 0.5  → essentially symmetric
        0.5 to 1.0     → moderate skew
        > 1.0          → high skew — transformation likely needed
    """
    # Convert to a plain float64 numpy array, treating NaN as NaN so we can filter it out
    arr = series.to_numpy(dtype=float, na_value=np.nan)
    # Remove NaN values — skewness is computed on actual data only
    arr = arr[~np.isnan(arr)]

    # Need at least 3 values for the skewness formula to be meaningful
    # (the correction factor divides by n-2, so n must be at least 3)
    if len(arr) < 3:
        return 0.0

    n    = len(arr)
    mean = float(arr.mean())   # arr.mean() → np.float64 → float() → plain Python float
    std  = float(arr.std())    # arr.std()  → np.float64 → float() → plain Python float

    if std == 0.0:
        # All values are identical — no spread, no skewness
        return 0.0

    # Step 1+2+3: standardise, cube, and average in one line
    # arr - mean and / std are element-wise numpy operations → ndarray[float64]
    # ** 3 cubes every element → ndarray[float64]
    # .mean() averages → np.float64
    standardised = ((arr - mean) / std) ** 3
    skew_raw     = float(standardised.mean())

    # Step 4: apply Fisher's sample correction factor
    # sqrt(n * (n-1)) / (n - 2) adjusts for sample bias
    correction = ((n * (n - 1)) ** 0.5) / (n - 2)
    return skew_raw * correction


# =============================================================================
# TASK 5 — SKEWED DATA AND TRANSFORMATIONS
# =============================================================================

def classify_skewness(skewness: float) -> dict:
    """
    Classifies a skewness value into a level and direction.

    Level thresholds (standard data science practice):
        |skew| < 0.5  → "symmetric"  — no transformation needed
        0.5 to 1.0    → "moderate"   — transformation may help
        > 1.0         → "high"       — transformation strongly recommended

    Direction:
        positive skewness → "right" (long tail pulls to the right)
        negative skewness → "left"  (long tail pulls to the left)
        exactly 0         → "none"  (perfectly symmetric)
    """
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
    """
    Suggests the most appropriate transformation based on skewness and data range.

    The recommendation logic (applied in priority order):

    1. If |skew| < 0.5:
          Data is already symmetric → do_nothing.

    2. If the column has negative values:
          Only Yeo-Johnson works on negative values — it handles all real numbers.
          log, sqrt, and Box-Cox all require strictly positive values.

    3. If skewness is positive (right-skewed) AND |skew| >= 1.0:
          Use log1p if zeros exist (log of 0 is undefined).
          Use log if no zeros (log is stronger and simpler).

    4. If skewness is positive but moderate (0.5 to 1.0):
          Use sqrt — a gentler correction for mild positive skew.

    5. Otherwise (left-skewed, no negatives):
          Yeo-Johnson handles left skew well and works on any values.
    """
    abs_skew = abs(skewness)

    if abs_skew < 0.5:
        return "do_nothing"                           # already symmetric
    if has_negatives:
        return "yeojohnson"                           # only option for negative values
    if skewness > 0:
        if abs_skew >= 1.0:
            return "log1p" if has_zeros else "log"   # strong right skew
        return "sqrt"                                  # moderate right skew
    return "yeojohnson"                               # left skew


@task5.get("/task5/skewness_info")
async def get_skewness_info():
    """
    Scans every numeric column for skewness and returns a full report.

    For each column:
      - The computed skewness value
      - A severity level (symmetric / moderate / high)
      - The direction (left / right)
      - Whether zeros or negative values are present (affects which transforms work)
      - The recommended transform
      - All available transforms with their constraints and explanations

    Transform availability depends on the data range:
        log          → requires all values > 0 (no zeros, no negatives)
        log1p        → requires all values >= 0 (allows zeros, no negatives)
        sqrt         → requires all values >= 0 (same as log1p)
        boxcox       → requires all values > 0 (no zeros, no negatives)
        yeojohnson   → works on ALL values including negatives (always available)
        reciprocal   → requires all values > 0 AND no zeros
        do_nothing   → always available
    """
    df           = require_df()
    # select_dtypes(include=[np.number]) selects only integer and float columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Return early with a helpful message if there are no numeric columns at all
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
        series = df[col].dropna()   # drop NaN before computing statistics

        # Need at least 3 values — _skew() requires n >= 3 for its correction formula
        if len(series) < 3:
            column_reports.append({
                "column":  col,
                "skipped": True,
                "reason":  f"Too few non-null values ({len(series)}) for skewness calculation.",
            })
            continue

        skewness       = round(_skew(series), 4)
        classification = classify_skewness(skewness)

        # Check what kinds of values are in this column — determines which
        # transforms are mathematically valid and which must be excluded
        has_zeros     = bool((series == 0).any())       # log and Box-Cox can't handle zeros
        has_negatives = bool((series < 0).any())        # log, sqrt, Box-Cox can't handle negatives
        col_min       = float(series.min())
        col_max       = float(series.max())

        # Get the automatic recommendation based on skewness and data constraints
        recommended = recommend_transform(skewness, has_zeros, has_negatives)

        # ── Build strategy options ────────────────────────────────────────────

        # Only show a strategy if it is valid for this column's data range.
        # Showing an invalid strategy (e.g. log for a column with zeros) would
        # let the user pick it, then hit an error when they apply it.
        # Better to not show it at all and explain why.
        all_strategies: Dict[str, Any] = {}

        if not has_zeros and not has_negatives:
            # log(x) requires all values strictly > 0
            all_strategies["log"] = {
                "label":       "Log transform — log(x)",
                "explanation": explain_log_transform(col, has_zeros=False),
                "available":   True,
            }

        if not has_negatives:
            # log(1 + x) shifts the input by 1 so zeros become log(1) = 0 (safe)
            all_strategies["log1p"] = {
                "label":       "Log1p transform — log(1 + x)",
                "explanation": explain_log_transform(col, has_zeros=True),
                "available":   True,
            }

        if not has_negatives:
            # sqrt(x) requires all values >= 0
            all_strategies["sqrt"] = {
                "label":       "Square root transform — sqrt(x)",
                "explanation": explain_sqrt_transform(col),
                "available":   True,
            }

        if not has_zeros and not has_negatives:
            # Box-Cox requires all values strictly > 0 (same constraint as log)
            # but it automatically finds the optimal power lambda
            all_strategies["boxcox"] = {
                "label":       "Box-Cox transform (finds optimal lambda automatically)",
                "explanation": explain_boxcox_transform(col),
                "available":   True,
            }

        # Yeo-Johnson works on all real numbers — always available, no constraints
        all_strategies["yeojohnson"] = {
            "label":       "Yeo-Johnson transform (works on any values including negatives)",
            "explanation": explain_yeojohnson_transform(col),
            "available":   True,
        }

        if not has_zeros and not has_negatives:
            # 1/x is undefined at 0 and works best on positive data
            all_strategies["reciprocal"] = {
                "label":       "Reciprocal transform — 1/x",
                "explanation": explain_reciprocal_transform(col),
                "available":   True,
            }

        # do_nothing is always available — user may decide no transform is needed
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
            # needs_transform is True for moderate and high skew (not symmetric)
            "needs_transform": classification["level"] != "symmetric",
            "explanation":     explain_skewness_severity(col, skewness),
            "all_strategies":  all_strategies,
        })

    # Count how many columns actually need a transform applied
    needs_transform = sum(
        1 for r in column_reports
        if not r.get("skipped") and r.get("needs_transform")
    )

    return {
        "what_is_skewness":          explain_what_skewness_is(),
        "columns_analysed":          len(numeric_cols),
        "columns_needing_transform": needs_transform,
        # columns_symmetric = those that were fine already (no transform needed)
        "columns_symmetric":         len(numeric_cols) - needs_transform,
        "column_reports":            column_reports,
    }


class SkewnessFixPayload(BaseModel):
    """
    Defines the shape of the JSON body for POST /task5/fix_skewness.
    Pydantic validates this automatically before our function runs.

    strategy maps column name → transformation method.
    Valid methods: log, log1p, sqrt, boxcox, yeojohnson, reciprocal, do_nothing

    Example:
        {
            "strategy": {
                "income":  "log1p",
                "age":     "sqrt",
                "balance": "yeojohnson"
            }
        }
    """
    strategy: Dict[str, str]


@task5.post("/task5/fix_skewness")
async def fix_skewness(payload: SkewnessFixPayload):
    """
    Applies the chosen transformation to each specified numeric column.

    Each method is validated against the column's data range before applying.
    If a constraint is violated (e.g. log on a column with zeros), the error
    is collected and the column is skipped — other columns still proceed.

    skew_before is recorded before the transform and skew_after after, so
    the applied message can report whether skewness actually improved.

    Box-Cox and Yeo-Johnson use scipy and get special handling:
        scipy returns a tuple (transformed_array, optimal_lambda).
        We extract each part by position ([0] and [1]) rather than
        unpacking, to avoid Pylance warnings about ambiguous tuple types.
        The transformed values are written back to the column while
        preserving NaN positions.
    """
    df      = require_df().copy()   # .copy() — never mutate the live DataFrame until the end
    applied = []
    errors  = []

    for col, method in payload.strategy.items():

        if col not in df.columns:
            errors.append(f'"{col}" not found in the dataset.')
            continue

        # All transforms are mathematical — only numeric columns qualify
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

        # Record skewness BEFORE the transform so we can compare in the result message
        skew_before = round(_skew(series), 4)

        try:
            if method == "log":
                # np.log() is undefined for 0 and negative values —
                # it returns -inf for 0 and NaN for negatives, corrupting the column
                if (series <= 0).any():
                    errors.append(
                        f'"{col}": log requires all values > 0. '
                        "Use log1p or yeojohnson instead."
                    )
                    continue
                # Apply natural log (base e) to every value in the column.
                # NaN values are left as NaN — np.log(NaN) = NaN naturally.
                df[col] = np.log(df[col])

            elif method == "log1p":
                # np.log1p(x) = log(1 + x) — the +1 shift ensures log(0) = log(1) = 0
                # instead of -infinity, making it safe for zero values
                if (series < 0).any():
                    errors.append(
                        f'"{col}": log1p requires all values >= 0. '
                        "Use yeojohnson instead."
                    )
                    continue
                df[col] = np.log1p(df[col])

            elif method == "sqrt":
                # Square root is a gentler correction than log — good for moderate right skew.
                # Undefined for negative values (returns NaN in numpy).
                if (series < 0).any():
                    errors.append(
                        f'"{col}": sqrt requires all values >= 0. '
                        "Use yeojohnson instead."
                    )
                    continue
                df[col] = np.sqrt(df[col])

            elif method == "boxcox":
                # Box-Cox finds the optimal power lambda automatically using maximum likelihood.
                # It is the most powerful transform available but has the strictest constraint:
                # all values must be strictly positive (> 0).
                if (series <= 0).any():
                    errors.append(
                        f'"{col}": Box-Cox requires all values > 0. '
                        "Use yeojohnson instead."
                    )
                    continue

                # scipy's stats.boxcox() returns a tuple: (transformed_values, optimal_lambda)
                # We index by position ([0] and [1]) rather than unpacking to a, b = ...
                # because Pylance has trouble inferring the types when unpacking scipy tuples.
                bc_result   = stats.boxcox(series)
                transformed = np.asarray(bc_result[0])   # the transformed values as a numpy array
                # np.float64(...).item() extracts the lambda as a guaranteed plain Python float,
                # regardless of what type Pylance thinks bc_result[1] is
                lam: float  = np.float64(bc_result[1]).item()

                # Write transformed values back while preserving NaN positions.
                # series only contains non-null values (from .dropna() above),
                # so transformed has one value per non-null row.
                # We use result.notna() as the mask to write only to non-null positions.
                result = df[col].copy()
                result[result.notna()] = transformed
                df[col] = result

                skew_after = round(_skew(df[col].dropna()), 4)
                applied.append(
                    f'"{col}": Box-Cox applied. Optimal lambda = {lam:.4f}. '
                    + explain_transform_result(col, "Box-Cox", skew_before, skew_after)
                )
                continue   # skip the generic skew_after + applied.append below

            elif method == "yeojohnson":
                # Yeo-Johnson is a generalisation of Box-Cox that also handles
                # zero and negative values — it is the most flexible transform.
                # Like Box-Cox, it finds the optimal lambda automatically.
                yj_result   = stats.yeojohnson(series)
                transformed = np.asarray(yj_result[0])
                lam_raw     = yj_result[1]
                # np.asarray(...).flat[0] safely extracts a single scalar from whatever
                # type scipy returns for the lambda, then float() converts it
                lam = float(np.asarray(lam_raw).flat[0])

                # Same NaN-preserving write-back pattern as Box-Cox above
                result = df[col].copy()
                result[result.notna()] = transformed
                df[col] = result

                skew_after = round(_skew(df[col].dropna()), 4)
                applied.append(
                    f'"{col}": Yeo-Johnson applied. Optimal lambda = {lam:.4f}. '
                    + explain_transform_result(col, "Yeo-Johnson", skew_before, skew_after)
                )
                continue   # skip the generic skew_after + applied.append below

            elif method == "reciprocal":
                # 1/x reverses the order of values and compresses large ones.
                # Undefined at 0 (division by zero → inf).
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
                df[col] = 1 / df[col]   # element-wise reciprocal; NaN stays NaN

            elif method == "do_nothing":
                applied.append(f'"{col}": no transformation applied.')
                continue   # skip to the next column — no change to make

            else:
                errors.append(
                    f'"{col}": unknown method "{method}". '
                    "Valid: log, log1p, sqrt, boxcox, yeojohnson, reciprocal, do_nothing."
                )
                continue

            # For all methods except boxcox, yeojohnson, and do_nothing —
            # compute skewness after the transform and build the result message
            skew_after = round(_skew(df[col].dropna()), 4)
            applied.append(
                explain_transform_result(col, method, skew_before, skew_after)
            )

        except Exception as e:
            errors.append(f'"{col}": unexpected error during {method} — {str(e)}')

    # ── Save back ─────────────────────────────────────────────────────────────

    # Only save if at least one REAL transformation was applied.
    # "no transformation applied" entries from do_nothing don't count.
    # The list comprehension filters those out before checking if anything remains.
    if [a for a in applied if "no transformation" not in a]:
        snapshot_store.save(f"Task 5 — transformations ({len(applied)} column(s))", require_df())
        dataset_state.df = df

    return {
        "success":   len(errors) == 0,
        "applied":   applied,
        "errors":    errors,
        "new_shape": list(df.shape),
    }


# =============================================================================
# TASK 5 UPGRADE A — FEATURE SCALING
#
# Why scaling matters:
#   Many machine learning algorithms (neural networks, SVM, k-NN, PCA) compute
#   distances or gradients between values. If one column is in the range 0–1
#   and another is in the range 0–1,000,000, the large-range column will
#   completely dominate the result — not because it's more important, but simply
#   because its numbers are bigger. Scaling puts all columns on the same footing.
#
# Three methods:
#   Min-Max  → rescales to [0, 1]. Simple and interpretable. Sensitive to outliers.
#   Z-score  → rescales to mean=0, std=1. Good for normally distributed data.
#   Robust   → rescales using median and IQR. Outlier-resistant version of Z-score.
# =============================================================================

@task5.get("/task5/scaling_info")
async def get_scaling_info():
    """
    Scans every numeric column and returns its range, mean, and std.

    Flags which columns have very different magnitudes from each other —
    the key signal that scaling is needed before running ML models.

    Also detects columns that are ALREADY scaled:
        already_minmax → min ≈ 0 and max ≈ 1 (within small tolerance)
        already_zscore → mean ≈ 0 and std ≈ 1 (within small tolerance)

    Columns are sorted by range (largest first) so the user can immediately
    see which column has the most extreme scale difference.
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

        # Need at least 2 values to compute a meaningful range or std
        if len(series) < 2:
            continue

        col_min   = float(series.min())
        col_max   = float(series.max())
        col_mean  = float(series.mean())
        col_std   = float(series.std())
        col_range = col_max - col_min

        # ── Already-scaled detection ──────────────────────────────────────────

        # already_minmax: values roughly span [0, 1]
        # Tolerance of ±0.01 handles floating-point imprecision from previous scaling
        already_minmax = col_min >= -0.01 and col_max <= 1.01 and col_range <= 1.01

        # already_zscore: mean is close to 0 AND std is close to 1
        # Tolerance of 0.1 handles slight drift from rounding
        already_zscore = abs(col_mean) < 0.1 and abs(col_std - 1.0) < 0.1

        column_reports.append({
            "column":         col,
            "min":            round(col_min,   4),
            "max":            round(col_max,   4),
            "mean":           round(col_mean,  4),
            "std":            round(col_std,   4),
            "range":          round(col_range, 4),
            "already_minmax": already_minmax,
            "already_zscore": already_zscore,
            # needs_scaling is True if the column is NOT already in either scaled form
            "needs_scaling":  not already_minmax and not already_zscore,
            "available_methods": {
                "minmax": "Rescale to [0, 1] — good for bounded data without outliers",
                "zscore": "Rescale to mean=0, std=1 — good for data with outliers",
                "robust": "Rescale using median and IQR — best when outliers are present",
            },
        })

    # Sort by range descending so the most extreme columns appear first —
    # these are the ones most in need of scaling
    column_reports.sort(key=lambda x: x["range"], reverse=True)

    return {
        "what_is_scaling":         explain_what_scaling_is(),
        "columns_analysed":        len(column_reports),
        "columns_needing_scaling": sum(1 for c in column_reports if c["needs_scaling"]),
        "column_reports":          column_reports,
    }


class ScalingPayload(BaseModel):
    """
    Defines the shape of the JSON body for POST /task5/scale_columns.
    Pydantic validates this automatically before our function runs.

    strategy maps column name → scaling method.
    Valid methods: minmax, zscore, robust

    Example:
        {
            "strategy": {
                "age":    "minmax",
                "salary": "robust",
                "score":  "zscore"
            }
        }
    """
    strategy: Dict[str, str]


@task5.post("/task5/scale_columns")
async def scale_columns(payload: ScalingPayload):
    """
    Applies the chosen scaling method to each specified column.
    NaN values are preserved throughout — scaling only touches real values.

    minmax  → (x - min) / (max - min)
              Compresses every value into the range [0, 1].
              Problem: sensitive to outliers — one extreme value can compress
              everything else to a tiny range near 0 or 1.

    zscore  → (x - mean) / std
              Centres the data at 0 with a spread of 1.
              Mean and std are affected by outliers but less severely than min/max.

    robust  → (x - median) / IQR
              Like Z-score but uses median and IQR instead of mean and std.
              Median and IQR are not affected by outliers at all —
              the best choice when the column has extreme values.

    Edge cases handled:
        - range == 0 for minmax: all values identical — scaling impossible
        - std == 0 for zscore: same issue — fall back to a clear message
        - IQR == 0 for robust: applies median centring only (subtracts median)
          even though dividing by IQR isn't possible
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
                    # All values are identical — dividing by 0 would give NaN everywhere
                    applied.append(
                        f'"{col}": all values identical — Min-Max scaling not possible.'
                    )
                    continue

                # Subtract the minimum and divide by the range → every value lands in [0, 1]
                # pandas broadcasts scalar operations across the whole column automatically
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
                    # All values identical — std is 0, division would give NaN
                    applied.append(
                        f'"{col}": standard deviation is 0 — Z-score scaling not possible.'
                    )
                    continue

                # Subtract mean (centres at 0) then divide by std (normalises spread to 1)
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
                    # IQR is 0 when over 50% of values are identical —
                    # we can still centre at the median, just can't normalise the spread
                    df[col] = df[col] - col_median
                    applied.append(
                        f'"{col}": IQR is 0 — applied median centring only '
                        f"(subtracted median = {col_median:.4f})."
                    )
                    continue

                # Subtract median (centres at 0) then divide by IQR (normalises spread)
                # Result: 0 at the median, ±0.5 at Q1/Q3, and outliers pull further out
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
        snapshot_store.save(f"Task 5 — scaling ({len(applied)} column(s))", require_df())
        dataset_state.df = df

    return {
        "success":   len(errors) == 0,
        "applied":   applied,
        "errors":    errors,
        "new_shape": list(df.shape),
    }


# =============================================================================
# TASK 5 UPGRADE B — DATE AND TIME PARSING
#
# Why this matters:
#   When a dataset is loaded, date columns are often read as plain text —
#   "2024-01-15" stored as the string "2024-01-15". pandas can't do date
#   arithmetic (calculate age, find the most recent row, extract the year)
#   on a string. Converting to datetime64[ns] unlocks all of pandas'
#   time-series functionality.
# =============================================================================

@task5.get("/task5/date_columns_info")
async def get_date_columns_info():
    """
    Scans all text (object dtype) columns and detects those that likely
    contain dates stored as strings.

    Detection method:
        Take the first 10 non-null values from each text column.
        Try to parse all of them with pd.to_datetime(errors="coerce").
        If 70% or more parse successfully, flag the column as a likely date column.

    Why 70% and not 100%?
        A real date column might have a few badly formatted values mixed in
        (e.g. "N/A", "TBD", or an empty string). 70% is high enough to avoid
        false positives on free-text columns while still catching real date
        columns that aren't perfectly clean.
    """
    df        = require_df()
    # Only check text (object dtype) columns — datetime64 columns are already parsed,
    # and numeric columns can't be dates
    text_cols = df.select_dtypes(include=["object"]).columns.tolist()

    detected: List[Dict[str, Any]] = []

    for col in text_cols:
        # Drop NaN and cast to string — .astype(str) ensures no mixed types
        series  = df[col].dropna().astype(str)
        # Take up to 10 sample values for the parse test — enough to be representative
        # without being slow on large datasets
        samples = series.head(10).tolist()

        if not samples:
            continue   # empty column after dropping NaN — skip

        # Try to parse the samples as dates.
        # errors="coerce" turns unparseable values into NaT instead of crashing.
        parsed       = pd.to_datetime(pd.Series(samples), errors="coerce")
        success      = int(parsed.notna().sum())   # count of values that parsed successfully
        success_rate = success / len(samples)      # proportion 0.0–1.0

        if success_rate >= 0.7:
            detected.append({
                "column":        col,
                "sample_values": samples[:5],              # show 5 examples in the UI
                "parse_rate":    round(success_rate * 100, 1),   # e.g. 80.0%
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
    Defines the shape of the JSON body for POST /task5/parse_dates.
    Pydantic validates this automatically before our function runs.

    columns:
        List of column names to convert to datetime64[ns].

    date_format:
        Optional strftime format string, e.g. "%d/%m/%Y" for day/month/year.
        If omitted, pandas infers the format automatically.
        Use this when automatic inference gets the format wrong.

    Example — automatic format inference:
        {"columns": ["signup_date", "last_login"]}

    Example — explicit format:
        {"columns": ["signup_date"], "date_format": "%d/%m/%Y"}
    """
    columns:     List[str]
    date_format: Optional[str] = None   # None = let pandas infer the format


@task5.post("/task5/parse_dates")
async def parse_dates(payload: DateParsePayload):
    """
    Converts specified text columns to datetime64[ns].

    How it works:
        pd.to_datetime() tries to parse every value in the column as a date.
        errors="coerce" turns any value that can't be parsed into NaT
        (Not a Time — pandas' equivalent of NaN for datetime columns).
        This means the operation never crashes, even on imperfect data.

    Format handling:
        If date_format is provided: pandas parses strictly using that template.
        If not: pandas uses automatic format inference (standard since pandas 2.0).
        Note: infer_datetime_format was removed in pandas 2.2 — we don't use it.

    new_failures tracking:
        Some values may already be NaN before parsing (genuinely missing).
        We count how many were NaN before and after, then subtract to find how
        many NEW NaT values were introduced by failed parsing — these are the
        values that had content but couldn't be parsed as a date.
    """
    df      = require_df().copy()
    applied = []
    errors  = []

    for col in payload.columns:

        if col not in df.columns:
            errors.append(f'"{col}" not found in the dataset.')
            continue

        # Skip if already the correct type — no need to re-parse
        if str(df[col].dtype) == "datetime64[ns]":
            applied.append(f'"{col}": already datetime — no change needed.')
            continue

        # Count non-null values BEFORE parsing — used to compute new_failures below
        total = int(df[col].notna().sum())

        try:
            if payload.date_format:
                # Strict format parsing — only values matching this exact format will parse
                df[col] = pd.to_datetime(
                    df[col],
                    format=payload.date_format,
                    errors="coerce",   # unparseable values → NaT instead of crashing
                )
            else:
                # Automatic format inference — pandas 2.0+ does this by default
                df[col] = pd.to_datetime(
                    df[col],
                    errors="coerce",
                )

            # Count successfully parsed values (not NaT/NaN after parsing)
            parsed_count = int(df[col].notna().sum())
            # Total NaN/NaT in the column after parsing
            failed_count = int(df[col].isna().sum())
            # New failures = values that existed before but couldn't be parsed.
            # max(0, ...) guards against any edge case where this could go negative.
            new_failures = max(0, failed_count - (len(df) - total))

            applied.append(
                explain_date_parse_result(col, parsed_count, new_failures, total)
            )

        except Exception as e:
            errors.append(f'"{col}": unexpected error — {str(e)}')

    if applied:
        snapshot_store.save(f"Task 5 — date parsing ({len(payload.columns)} column(s))", require_df())
        dataset_state.df = df

    return {
        "success":   len(errors) == 0,
        "applied":   applied,
        "errors":    errors,
        "new_shape": list(df.shape),
    }
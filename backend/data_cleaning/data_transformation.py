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
)
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict

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
#
# Two endpoints:
#
#   GET  /task5/skewness_info
#        Measures skewness of every numeric column.
#        Returns the score, direction, severity, recommended transformation,
#        and xAI explanations for every available method.
#        Never modifies the data.
#
#   POST /task5/fix_skewness
#        Applies the user's chosen transformation per column:
#          log         — log(x) for strictly positive columns
#          log1p       — log(1+x) for columns with zeros
#          sqrt        — square root, handles zeros
#          boxcox      — optimal power transform, strictly positive only
#          yeojohnson  — optimal power transform, handles any values
#          reciprocal  — 1/x, for extreme right skew, no zeros allowed
#          do_nothing  — skip this column
#        Records skewness before and after so the user can see the improvement.
# =============================================================================

def classify_skewness(skewness: float) -> dict:
    """
    Converts a raw skewness score into a human-readable label and direction.

    Skewness scale:
      |skew| < 0.5   → approximately symmetric, no action needed
      |skew| < 1.0   → moderate skew, transformation recommended
      |skew| >= 1.0  → high skew, transformation strongly recommended

    Returns a dict with level and direction so the frontend can
    colour-code the card (green/yellow/red) and show an arrow.
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
    Suggests the most appropriate transformation based on the skewness
    score and the column's value range.

    Decision logic:
      - Symmetric data → do_nothing
      - Right skew, no zeros or negatives → log (most powerful for positive data)
      - Right skew, has zeros but no negatives → log1p (handles zeros safely)
      - Right skew, has negatives → yeojohnson (only option that handles negatives)
      - Left skew → yeojohnson (handles left skew better than log/sqrt)
      - Moderate right skew → sqrt (gentler than log, safer starting point)
    """
    abs_skew = abs(skewness)

    if abs_skew < 0.5:
        return "do_nothing"

    if has_negatives:
        # Log and sqrt cannot handle negatives — Yeo-Johnson is the only safe choice
        return "yeojohnson"

    if skewness > 0:
        # Right skew
        if abs_skew >= 1.0:
            if has_zeros:
                return "log1p"
            return "log"
        else:
            # Moderate right skew — sqrt is gentler and usually enough
            return "sqrt"

    # Left skew — Yeo-Johnson handles this better than log or sqrt
    return "yeojohnson"


@task5.get("/task5/skewness_info")
async def get_skewness_info():
    """
    Measures skewness on every numeric column and returns a full report.

    For each column we calculate:
      - The skewness score using pandas .skew()
        (This uses Fisher's definition, the same as scipy and most textbooks)
      - The direction: right (positive), left (negative), or symmetric
      - The severity: symmetric / moderate / high
      - Whether the column has zeros or negative values
        (this determines which transformations are available)
      - The recommended transformation based on the data characteristics
      - All available strategies with explanations so the user can
        understand each option before choosing

    Columns with fewer than 3 non-null values are skipped — you need
    at least 3 data points for a skewness calculation to be meaningful.
    """
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

        # ── Core skewness measurement ─────────────────────────────────────────

        skewness = round(float(series.skew()), 4)
        classification = classify_skewness(skewness)

        # ── Column value characteristics ──────────────────────────────────────

        # These flags determine which transformations are available.
        # Log and sqrt break on zeros/negatives, so we check upfront.
        has_zeros     = bool((series == 0).any())
        has_negatives = bool((series < 0).any())
        col_min       = float(series.min())
        col_max       = float(series.max())

        recommended = recommend_transform(skewness, has_zeros, has_negatives)

        # ── Build strategies ──────────────────────────────────────────────────

        # Only show transformations that are valid for this column's value range.
        # Offering log to a column with negatives would just cause a runtime error.

        all_strategies = {}

        # Log — strictly positive only
        if not has_zeros and not has_negatives:
            all_strategies["log"] = {
                "label":       "Log transform — log(x)",
                "explanation": explain_log_transform(col, has_zeros=False),
                "available":   True,
            }

        # Log1p — handles zeros, not negatives
        if not has_negatives:
            all_strategies["log1p"] = {
                "label":       "Log1p transform — log(1 + x)",
                "explanation": explain_log_transform(col, has_zeros=True),
                "available":   True,
            }

        # Sqrt — handles zeros, not negatives
        if not has_negatives:
            all_strategies["sqrt"] = {
                "label":       "Square root transform — sqrt(x)",
                "explanation": explain_sqrt_transform(col),
                "available":   True,
            }

        # Box-Cox — strictly positive only
        if not has_zeros and not has_negatives:
            all_strategies["boxcox"] = {
                "label":       "Box-Cox transform (finds optimal lambda automatically)",
                "explanation": explain_boxcox_transform(col),
                "available":   True,
            }

        # Yeo-Johnson — works on any values including zeros and negatives
        all_strategies["yeojohnson"] = {
            "label":       "Yeo-Johnson transform (works on any values including negatives)",
            "explanation": explain_yeojohnson_transform(col),
            "available":   True,
        }

        # Reciprocal — strictly positive only
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

    # Summary counts for the top banner
    needs_transform = sum(
        1 for r in column_reports
        if not r.get("skipped") and r.get("needs_transform")
    )

    return {
        "what_is_skewness":      explain_what_skewness_is(),
        "columns_analysed":      len(numeric_cols),
        "columns_needing_transform": needs_transform,
        "columns_symmetric":     len(numeric_cols) - needs_transform,
        "column_reports":        column_reports,
    }


# ── Request body model ────────────────────────────────────────────────────────

class SkewnessFixPayload(BaseModel):
    """
    strategy is a dict where:
      - the key   is the column name
      - the value is the transformation method to apply

    Valid methods:
      log, log1p, sqrt, boxcox, yeojohnson, reciprocal, do_nothing

    Example:
        {
            "strategy": {
                "income":      "log",
                "age":         "sqrt",
                "score":       "yeojohnson",
                "temperature": "do_nothing"
            }
        }
    """
    strategy: Dict[str, str]


@task5.post("/task5/fix_skewness")
async def fix_skewness(payload: SkewnessFixPayload):
    """
    Applies the user's chosen transformation to each specified column.

    For every column we:
      1. Record the skewness BEFORE the transform
      2. Apply the transformation
      3. Record the skewness AFTER the transform
      4. Return both values so the user can see the improvement

    This before/after comparison is the most important piece of feedback
    we can give the user — it shows whether the transformation actually worked.

    Transformation implementations:

      log:
          np.log(x) — natural logarithm.
          Requires all values > 0. Compresses large values dramatically.
          A value of 1 becomes 0. A value of 100 becomes 4.6.

      log1p:
          np.log1p(x) — equivalent to log(1 + x).
          Handles zeros safely. log1p(0) = 0.
          A value of 0 stays 0. A value of 99 becomes 4.6.

      sqrt:
          np.sqrt(x) — square root.
          Gentler compression than log. Works on zeros.
          A value of 100 becomes 10. A value of 0 stays 0.

      boxcox:
          scipy.stats.boxcox(x) — finds the lambda that minimises skewness.
          Returns the transformed values and the optimal lambda.
          Requires all values strictly > 0.

      yeojohnson:
          scipy.stats.yeojohnson(x) — like Box-Cox but for any real values.
          Works on zeros and negatives. The lambda is found automatically.

      reciprocal:
          1 / x — flips the scale completely.
          Very aggressive. Use only for extreme right skew.
          Cannot handle zeros (1/0 is undefined).

      do_nothing:
          No change. Logged for completeness.
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

        # Record skewness before for the before/after comparison
        skew_before = round(float(series.skew()), 4)

        try:
            if method == "log":
                if (series <= 0).any():
                    errors.append(
                        f'"{col}": log transform requires all values to be strictly '
                        f'positive. This column has values <= 0. Use log1p or yeojohnson instead.'
                    )
                    continue
                df[col] = np.log(df[col])

            elif method == "log1p":
                if (series < 0).any():
                    errors.append(
                        f'"{col}": log1p requires all values >= 0. '
                        f'This column has negative values. Use yeojohnson instead.'
                    )
                    continue
                df[col] = np.log1p(df[col])

            elif method == "sqrt":
                if (series < 0).any():
                    errors.append(
                        f'"{col}": sqrt requires all values >= 0. '
                        f'This column has negative values. Use yeojohnson instead.'
                    )
                    continue
                df[col] = np.sqrt(df[col])

            elif method == "boxcox":
                if (series <= 0).any():
                    errors.append(
                        f'"{col}": Box-Cox requires all values to be strictly positive. '
                        f'This column has values <= 0. Use yeojohnson instead.'
                    )
                    continue
                # scipy returns (transformed_array, optimal_lambda)
                # We only need the transformed array — lambda is informational
                transformed, lam = stats.boxcox(series)

                # We need to assign back to the full column including NaN positions.
                # Create a new series with the original index, fill transformed values.
                result = df[col].copy()
                result[result.notna()] = transformed
                df[col] = result

                applied.append(
                    f'"{col}": Box-Cox transform applied. '
                    f'Optimal lambda = {lam:.4f}. '
                    + explain_transform_result(
                        col, "Box-Cox", skew_before,
                        round(float(df[col].dropna().skew()), 4)
                    )
                )
                continue   # skip the generic applied.append below

            elif method == "yeojohnson":
                # scipy returns (transformed_array, optimal_lambda)
                transformed, lam = stats.yeojohnson(series)

                result = df[col].copy()
                result[result.notna()] = transformed
                df[col] = result

                applied.append(
                    f'"{col}": Yeo-Johnson transform applied. '
                    f'Optimal lambda = {lam:.4f}. '
                    + explain_transform_result(
                        col, "Yeo-Johnson", skew_before,
                        round(float(df[col].dropna().skew()), 4)
                    )
                )
                continue

            elif method == "reciprocal":
                if (series == 0).any():
                    errors.append(
                        f'"{col}": reciprocal (1/x) is undefined for zero values. '
                        f'This column contains zeros. Use log1p or sqrt instead.'
                    )
                    continue
                if (series < 0).any():
                    errors.append(
                        f'"{col}": reciprocal works best on positive data. '
                        f'This column has negative values — use yeojohnson instead.'
                    )
                    continue
                df[col] = 1 / df[col]

            elif method == "do_nothing":
                applied.append(f'"{col}": no transformation applied.')
                continue

            else:
                errors.append(
                    f'"{col}": unknown method "{method}". '
                    "Valid options: log, log1p, sqrt, boxcox, yeojohnson, reciprocal, do_nothing."
                )
                continue

            # ── Record skewness after and append to applied ───────────────────

            skew_after = round(float(df[col].dropna().skew()), 4)
            applied.append(
                explain_transform_result(col, method, skew_before, skew_after)
            )

        except Exception as e:
            errors.append(f'"{col}": unexpected error during {method} — {str(e)}')

    # ── Save back ─────────────────────────────────────────────────────────────

    actual_changes = [a for a in applied if "no transformation" not in a]
    if actual_changes and not errors:
        dataset_state.df = df
    elif actual_changes:
        # Some succeeded, some errored — save the successful ones
        dataset_state.df = df

    # ── Return ────────────────────────────────────────────────────────────────

    return {
        "success":   len(errors) == 0,
        "applied":   applied,
        "errors":    errors,
        "new_shape": list(df.shape),
    }
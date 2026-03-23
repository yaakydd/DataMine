# =============================================================================
# outlier_detection.py
#
# This file handles Task 4 of DataMine: detecting and fixing outliers in
# numeric columns, plus correlation analysis between columns.
#
# It defines 5 endpoints across 2 sections:
#
#   Section 1 — Univariate outlier detection (one column at a time)
#     GET  /task4/outliers_info     → scan every numeric column using Z-score
#                                     and IQR methods, return counts and strategies.
#                                     Never modifies the data.
#     POST /task4/fix_outliers      → cap or remove outliers per column.
#
#   Section 2 — Multivariate outlier detection (across multiple columns at once)
#     GET  /task4/multivariate_outliers        → detect outliers using Mahalanobis
#                                               distance across all numeric columns.
#     POST /task4/fix_multivariate_outliers    → drop or flag the detected rows.
#
#   Section 3 — Correlation analysis
#     GET  /task4/correlation_analysis         → find pairs of columns that are
#                                               highly correlated and therefore
#                                               redundant.
# =============================================================================


# HTTPException → return a clean error + HTTP status code instead of a Python crash
# APIRouter     → groups all Task 4 endpoints under one object registered in main.py
from fastapi import HTTPException, APIRouter

# BaseModel → Pydantic base class that auto-validates incoming JSON request bodies
from pydantic import BaseModel

# scipy.stats → used to compute the chi-squared threshold for Mahalanobis distance.
# scipy is a scientific computing library built on top of numpy.
from scipy import stats

# Type hints for function signatures and Pydantic models
from typing import Dict, Optional, List, Any

# Shared singleton holding the current DataFrame in memory across all routers
from State.dfState import dataset_state

# Each function returns a plain-English explanation for the xAI panel.
# Keeping explanation text here means wording can be updated without
# touching any cleaning logic.
from xAI.outliers import (
    explain_what_outliers_are,            # general intro to what outliers are
    explain_zscore_method,                # how Z-score detection works
    explain_iqr_method,                   # how IQR detection works
    explain_column_outliers,              # findings for one specific column
    explain_outlier_if_you_cap_iqr,       # what IQR capping does to this column
    explain_outlier_if_you_cap_zscore,    # what Z-score capping does to this column
    explain_outlier_if_you_remove_rows,   # what removing outlier rows does
    explain_outlier_if_you_do_nothing,    # what leaving outliers in place means
    explain_what_multivariate_outliers_are,   # general intro to multivariate outliers
    explain_mahalanobis_method,               # how Mahalanobis distance works
    explain_mahalanobis_result,               # findings when outliers were found
    explain_mahalanobis_no_result,            # findings when no outliers were found
    explain_mahalanobis_singular_matrix,      # error when columns are perfectly correlated
    explain_correlation_overview,             # general intro to correlation analysis
    explain_high_correlation_pair,            # what a specific high-correlation pair means
    explain_no_high_correlations,             # when no high-correlation pairs were found
)

import pandas as pd   # main data manipulation library
import numpy as np    # numerical operations, array math, and type detection

# Snapshot store saves a copy of the DataFrame before every write.
# Powers the undo/rollback system — up to 20 snapshots kept.
from State.snapshotState import snapshot_store

# All Task 4 endpoints are attached to this router.
# main.py registers it with: app.include_router(task4, prefix="/api")
task4 = APIRouter()


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
        np.float64, np.bool_. Python's built-in json module (used by FastAPI)
        does not know these types and will raise a TypeError.
        safe_json() walks every level of the object and converts each one.

    Handles:
        dict        → recurse into each value
        list        → recurse into each item
        np.integer  → int()
        np.floating → float(), or None if NaN/Inf (JSON has no representation for those)
        np.bool_    → bool()
        np.ndarray  → convert to list first, then recurse
        pd.NA/NaT/NaN → None  (JSON null for any missing value)
        anything else → return unchanged (already a plain Python type)
    """
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        # JSON cannot represent NaN or Infinity — convert to null instead
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return safe_json(obj.tolist())   # convert to list first, then recurse
    try:
        # pd.isna() catches pd.NA, pd.NaT, float('nan'), and None all at once
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        # pd.isna() raises on some types like lists — ignore and fall through
        pass
    return obj


# =============================================================================
# TASK 4 — OUTLIER DETECTION
# =============================================================================

@task4.get("/task4/outliers_info")
async def get_outliers_info():
    """
    Scans every numeric column for outliers using two methods simultaneously:
    Z-score and IQR. Returns statistics, outlier counts, and all fix strategies.

    Why two methods?
        Z-score assumes the data follows a normal (bell-curve) distribution.
        IQR makes no assumption about distribution — it just looks at the
        spread of the middle 50% of values.
        Showing both gives the user a more complete picture — a value might
        be flagged by one method but not the other, which itself is informative.

    Skips columns with fewer than 4 non-null values because you need a
    reasonable sample size to compute meaningful statistics.
    """
    df           = require_df()
    # select_dtypes(include=[np.number]) selects only integer and float columns.
    # Text, boolean, and datetime columns are excluded — outlier detection
    # is a numeric concept.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # If there are no numeric columns at all, return early with a helpful message
    # rather than returning an empty report with no explanation.
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

    column_reports = []   # one entry per numeric column

    for col in numeric_cols:
        # .dropna() removes missing values before computing statistics —
        # NaN would make mean(), std(), quantile() return NaN themselves
        series = df[col].dropna()

        # Guard: need at least 4 values to compute meaningful statistics.
        # With fewer than 4 values, std() is unreliable and quantiles are
        # essentially meaningless — the result would just confuse the user.
        if len(series) < 4:
            column_reports.append({
                "column":  col,
                "skipped": True,
                "reason":  f"Too few non-null values ({len(series)}) to detect outliers reliably.",
            })
            continue

        # ── Compute column statistics ─────────────────────────────────────────

        mean    = float(series.mean())     # arithmetic average
        std     = float(series.std())      # standard deviation — spread of the data
        col_min = float(series.min())      # smallest value in the column
        col_max = float(series.max())      # largest value in the column

        # IQR (Interquartile Range) statistics:
        # Q1 = 25th percentile (bottom quarter boundary)
        # Q3 = 75th percentile (top quarter boundary)
        # IQR = distance between Q1 and Q3 (the "middle 50%" spread)
        q1  = float(series.quantile(0.25))
        q3  = float(series.quantile(0.75))
        iqr = q3 - q1

        # IQR fences — the standard Tukey method:
        # Anything below Q1 - 1.5×IQR or above Q3 + 1.5×IQR is an outlier.
        # 1.5 is the standard multiplier used in most boxplot implementations.
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr

        # ── Z-score outlier detection ─────────────────────────────────────────

        if std == 0:
            # If std is 0, all values are identical — the Z-score formula
            # would divide by zero. There are no outliers in a constant column.
            zscore_mask  = pd.Series([False] * len(series), index=series.index)
            zscore_count = 0
        else:
            # Z-score = (value - mean) / std
            # A Z-score tells you how many standard deviations a value is
            # from the mean. Values beyond ±3 are considered outliers by convention
            # (they sit in the extreme 0.3% tails of a normal distribution).
            z_scores     = (series - mean) / std
            zscore_mask  = z_scores.abs() > 3   # True where |z| > 3
            zscore_count = int(zscore_mask.sum())

        # ── IQR outlier detection ─────────────────────────────────────────────

        # A value is an IQR outlier if it falls outside the computed fences.
        # The | operator here is a logical OR on two boolean Series —
        # True where either condition is met (below lower OR above upper fence).
        iqr_mask  = (series < lower_fence) | (series > upper_fence)
        iqr_count = int(iqr_mask.sum())

        # Get up to 5 sample outlier values (using IQR definition) so the user
        # can see real examples of what is being flagged
        outlier_values = safe_json(series[iqr_mask].head(5).tolist())

        # Pre-compute the Z-score boundary values for the strategy label
        zscore_lower = mean - 3 * std   # lower Z-score boundary as an actual value
        zscore_upper = mean + 3 * std   # upper Z-score boundary as an actual value

        # ── Build fix strategies ──────────────────────────────────────────────

        # Every strategy gets a label (shown in the dropdown) and an explanation
        # (shown when that strategy is selected). The frontend renders these
        # without any hardcoded strings — everything comes from the API.
        all_strategies = {
            "cap_iqr": {
                # Show the actual fence values in the label so the user knows
                # exactly what range their data will be clipped to
                "label":       f"Cap to IQR fences [{lower_fence:.2f}, {upper_fence:.2f}]",
                "explanation": explain_outlier_if_you_cap_iqr(col, lower_fence, upper_fence),
            },
            "cap_zscore": {
                "label":       f"Cap to ±3 std [{zscore_lower:.2f}, {zscore_upper:.2f}]",
                "explanation": explain_outlier_if_you_cap_zscore(col, zscore_lower, zscore_upper),
            },
            "remove": {
                # Tell the user upfront exactly how many rows will be deleted
                "label":       f"Remove {iqr_count} outlier row(s)",
                "explanation": explain_outlier_if_you_remove_rows(col, iqr_count),
            },
            "do_nothing": {
                "label":       "Leave as-is",
                "explanation": explain_outlier_if_you_do_nothing(col),
            },
        }

        column_reports.append({
            "column":               col,
            "skipped":              False,
            "zscore_outlier_count": zscore_count,
            "iqr_outlier_count":    iqr_count,
            # True if either method found at least one outlier
            "has_outliers":         zscore_count > 0 or iqr_count > 0,
            # Full descriptive statistics for the frontend's summary table
            "stats": {
                "mean":        round(mean,        4),
                "std":         round(std,         4),
                "min":         round(col_min,     4),
                "max":         round(col_max,     4),
                "q1":          round(q1,          4),
                "q3":          round(q3,          4),
                "iqr":         round(iqr,         4),
                "lower_fence": round(lower_fence, 4),
                "upper_fence": round(upper_fence, 4),
            },
            "sample_outliers":    outlier_values,
            # Column-level xAI explanation built from all the stats computed above
            "column_explanation": explain_column_outliers(
                col, zscore_count, iqr_count,
                col_min, col_max, mean, std,
                q1, q3, lower_fence, upper_fence,
            ),
            "all_strategies": all_strategies,
        })

    # Count how many columns have at least one outlier (by either method)
    total_cols_with_outliers = sum(
        1 for r in column_reports
        if not r.get("skipped") and r.get("has_outliers")   # skip columns that were skipped
    )

    return {
        # Top-level xAI explanations for the page header
        "what_are_outliers":     explain_what_outliers_are(),
        "zscore_explanation":    explain_zscore_method(),
        "iqr_explanation":       explain_iqr_method(),
        "columns_analysed":      len(numeric_cols),
        "columns_with_outliers": total_cols_with_outliers,
        # columns_clean = numeric columns that had no outliers at all
        "columns_clean":         len(numeric_cols) - total_cols_with_outliers,
        "column_reports":        column_reports,
    }


class OutlierFixPayload(BaseModel):
    """
    Defines the shape of the JSON body for POST /task4/fix_outliers.
    Pydantic validates this automatically before our function runs.

    strategy maps column name → fix method.
    Valid methods: cap_iqr, cap_zscore, remove, do_nothing

    Example:
        {
            "strategy": {
                "age":    "cap_iqr",
                "salary": "remove",
                "score":  "do_nothing"
            }
        }
    """
    strategy: Dict[str, str]


@task4.post("/task4/fix_outliers")
async def fix_outliers(payload: OutlierFixPayload):
    """
    Applies the chosen fix strategy to each specified numeric column.

    cap_iqr:
        .clip(lower=lower_fence, upper=upper_fence) replaces any value below
        the lower IQR fence with the lower fence value, and any value above
        the upper fence with the upper fence value.
        No rows are removed — extreme values are pulled in to the boundary.

    cap_zscore:
        Same clip approach but uses mean ± 3×std as the boundaries instead
        of IQR fences. Better for normally distributed data.

    remove:
        Drops all rows where the column's value falls outside the IQR fences.
        This removes entire rows, not just the outlier cell — chosen when
        the row as a whole is considered bad data.
        reset_index(drop=True) renumbers rows from 0 after removal.

    do_nothing:
        Logs the column as processed without changing it.

    Works on a .copy() — dataset_state.df is never touched until the end.
    """
    df      = require_df().copy()   # .copy() — never mutate the live DataFrame until we're sure
    applied = []
    errors  = []

    for col, method in payload.strategy.items():

        if col not in df.columns:
            errors.append(f'"{col}" not found in the dataset.')
            continue

        # Reject non-numeric columns early — outlier operations are math-based
        # dtype.kind: "i" = integer family, "f" = float family
        if df[col].dtype.kind not in ("i", "f"):
            errors.append(
                f'"{col}" is not numeric (dtype: {df[col].dtype}). '
                "Outlier fixing only applies to numeric columns."
            )
            continue

        series = df[col].dropna()

        # Need at least 4 values to compute reliable statistics
        if len(series) < 4:
            errors.append(
                f'"{col}" has too few non-null values ({len(series)}) '
                "to compute reliable outlier boundaries."
            )
            continue

        try:
            if method == "cap_iqr":
                # Recompute fences from the current DataFrame state
                q1          = float(series.quantile(0.25))
                q3          = float(series.quantile(0.75))
                iqr         = q3 - q1
                lower_fence = q1 - 1.5 * iqr
                upper_fence = q3 + 1.5 * iqr
                # .clip() replaces values outside the range with the boundary value.
                # lower= replaces anything below lower_fence with lower_fence.
                # upper= replaces anything above upper_fence with upper_fence.
                # NaN values are left as NaN — clip() ignores them by default.
                df[col] = df[col].clip(lower=lower_fence, upper=upper_fence)
                applied.append(
                    f'"{col}": capped to IQR fences '
                    f'[{lower_fence:.4f}, {upper_fence:.4f}]. '
                    f'No rows removed — extreme values pulled in.'
                )

            elif method == "cap_zscore":
                mean = float(series.mean())
                std  = float(series.std())
                if std == 0:
                    # All values are identical — there are no outliers to cap
                    applied.append(
                        f'"{col}": all values are identical — no Z-score capping needed.'
                    )
                    continue
                # mean ± 3×std gives the Z-score boundaries as actual data values
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
                df[col]     = df[col].clip(lower=lower_bound, upper=upper_bound)
                applied.append(
                    f'"{col}": capped to ±3 std '
                    f'[{lower_bound:.4f}, {upper_bound:.4f}]. '
                    f'No rows removed — extreme values pulled in.'
                )

            elif method == "remove":
                q1          = float(series.quantile(0.25))
                q3          = float(series.quantile(0.75))
                iqr         = q3 - q1
                lower_fence = q1 - 1.5 * iqr
                upper_fence = q3 + 1.5 * iqr
                before      = len(df)

                # Build a mask of rows that are WITHIN the fences (rows we keep)
                within_fences = (df[col] >= lower_fence) & (df[col] <= upper_fence)
                # Also keep rows where this column is NaN — we only remove rows
                # with an actual outlier value, not rows with a missing value.
                # Missing values are Task 2's responsibility.
                is_null = df[col].isna()
                df      = df[within_fences | is_null]  # keep rows that are either in-fence OR null

                # reset_index renumbers rows from 0 to avoid index gaps after removal
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
            # Catch any unexpected runtime error and report it cleanly
            errors.append(f'"{col}": unexpected error — {str(e)}')

    # ── Save back ─────────────────────────────────────────────────────────────

    # Only save if:
    #   1. Something was applied
    #   2. Not ALL applied entries are "do_nothing" (which means no actual data change)
    # The "left as-is" check prevents an unnecessary snapshot when the user
    # selected do_nothing for every column.
    if applied and not all("left as-is" in a for a in applied):
        snapshot_store.save(f"Task 4 — outlier fixes ({len(applied)} column(s))", require_df())
        dataset_state.df = df

    return {
        "success":   len(errors) == 0,
        "applied":   applied,
        "errors":    errors,
        "new_shape": list(df.shape),
    }


# =============================================================================
# TASK 4 UPGRADE — MULTIVARIATE OUTLIER DETECTION (Mahalanobis Distance)
#
# Univariate outlier detection checks one column at a time.
# But some rows are only unusual when you look at multiple columns together.
# For example: someone 7 feet tall is unusual on its own, but a 7-foot person
# who weighs 100 lbs is a multivariate outlier — the combination is the problem.
#
# Mahalanobis distance measures how far each row is from the centre of the
# data cloud in multi-dimensional space, while accounting for the correlation
# between columns. A row with a large Mahalanobis distance is a multivariate outlier.
# =============================================================================

@task4.get("/task4/multivariate_outliers")
async def get_multivariate_outliers(
    columns:    Optional[str] = None,   # comma-separated column names, or None = use all numeric cols
    confidence: float = 0.975,          # chi-squared percentile for the outlier threshold (default 97.5%)
):
    """
    Detects multivariate outliers using Mahalanobis distance.

    Algorithm (step by step):
      1. Select the numeric columns to analyse.
      2. Drop rows with any NaN in those columns (Mahalanobis needs complete rows).
      3. Compute the mean vector (centroid of the data cloud).
      4. Compute the covariance matrix (how columns vary together).
      5. Invert the covariance matrix — needed for the distance formula.
      6. For each row, compute: distance = sqrt((row - mean) × inv_cov × (row - mean)ᵀ)
      7. Compare each distance to a threshold from the chi-squared distribution.
         Rows above the threshold are multivariate outliers.

    Why chi-squared?
        Under the assumption that the data is multivariate normal, Mahalanobis
        distance squared follows a chi-squared distribution with degrees of
        freedom equal to the number of columns. The threshold is the point
        beyond which only (1 - confidence)% of normal data would fall.

    Why singular matrix error?
        The covariance matrix cannot be inverted if two or more columns are
        perfectly correlated (one is a linear combination of another).
        If this happens, run /task4/correlation_analysis to find and remove
        the redundant column.
    """
    df = require_df()

    # ── Column selection and validation ───────────────────────────────────────

    if columns:
        # User specified particular columns — validate them before proceeding
        col_list     = [c.strip() for c in columns.split(",")]
        missing_cols = [c for c in col_list if c not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Column(s) not found: {missing_cols}")
        # Mahalanobis only works on numeric data — reject non-numeric columns early
        non_numeric = [c for c in col_list if df[c].dtype.kind not in ("i", "f")]
        if non_numeric:
            raise HTTPException(
                status_code=400,
                detail=f"Column(s) {non_numeric} are not numeric."
            )
    else:
        # No columns specified — use all numeric columns automatically
        col_list = df.select_dtypes(include=[np.number]).columns.tolist()

    # Mahalanobis distance requires at least 2 dimensions (columns) to be meaningful
    if len(col_list) < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "Mahalanobis distance requires at least 2 numeric columns. "
                f"Only {len(col_list)} found."
            )
        )

    # ── Prepare data ──────────────────────────────────────────────────────────

    # Drop rows where ANY of the selected columns has a NaN value.
    # Mahalanobis distance requires complete rows — a partial row cannot
    # be placed in multi-dimensional space.
    data         = df[col_list].dropna()
    n_rows       = len(data)       # number of complete rows available for analysis
    n_cols       = len(col_list)   # number of dimensions
    dropped_rows = len(df) - n_rows  # how many rows were excluded due to NaN

    # Need more rows than columns to compute a non-singular covariance matrix.
    # If n_rows ≤ n_cols, the matrix will be singular (non-invertible).
    if n_rows < n_cols + 1:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Not enough complete rows ({n_rows}) to compute Mahalanobis distance "
                f"for {n_cols} columns. Need at least {n_cols + 1} rows with no missing values."
            )
        )

    # ── Compute Mahalanobis distances ─────────────────────────────────────────

    # Convert to a plain float64 numpy array.
    # .astype(float) ensures we have a clean, fully typed array — no mixed types,
    # no pandas index complications. Every subsequent operation is pure numpy.
    data_array = data.values.astype(float)

    # mean_vec: the centroid of the data cloud — one mean value per column
    # axis=0 means "compute the mean down each column" (across all rows)
    mean_vec = np.mean(data_array, axis=0)

    # cov_matrix: (n_cols × n_cols) matrix describing how columns co-vary.
    # rowvar=False means each COLUMN is a variable (standard data science convention).
    # The diagonal contains each column's variance; off-diagonal entries contain
    # pairwise covariances between columns.
    cov_matrix = np.cov(data_array, rowvar=False)

    # Invert the covariance matrix — required by the Mahalanobis formula.
    # np.linalg.inv() raises LinAlgError if the matrix is singular
    # (i.e. two columns are perfectly correlated — one is redundant).
    try:
        inv_cov = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        # Return a helpful error instead of crashing — guide the user to fix the cause
        return {
            "success":     False,
            "explanation": explain_mahalanobis_singular_matrix(),
            "tip": (
                "Run GET /task4/correlation_analysis to identify which columns "
                "are perfectly correlated and should be removed before retrying."
            ),
        }

    # Compute the Mahalanobis distance for each row.
    # Formula: d = sqrt((row - mean) @ inv_cov @ (row - mean)ᵀ)
    # @ is the matrix multiplication operator in Python.
    #
    # TYPE SAFETY NOTE:
    # distances is declared as List[float] — plain Python floats only.
    # Every .append() calls .item() on the np.float64 result of np.sqrt(),
    # which always extracts a plain Python float. This avoids numpy scalar
    # type ambiguity in the list and ensures safe_json() never receives
    # anything unexpected.
    distances: List[float] = []
    for row in data_array:
        diff    = row - mean_vec              # vector from this row to the centroid
        dist_sq = float(diff @ inv_cov @ diff.T)   # Mahalanobis distance SQUARED
        sq_root = np.sqrt(max(dist_sq, 0.0))       # sqrt to get the actual distance
                                                    # max(..., 0.0) guards against tiny
                                                    # negative values from floating-point errors
        distances.append(sq_root.item())            # .item() → plain Python float

    dist_array = np.array(distances)   # convert back to numpy array for easy comparison

    # ── Compute the outlier threshold ─────────────────────────────────────────

    # The threshold is derived from the chi-squared distribution.
    # stats.chi2.ppf(confidence, df=n_cols) gives the value below which
    # `confidence`% of the chi-squared distribution falls.
    # We take the sqrt because our distances are already sqrt'd above.
    #
    # .item() is called on the final np.float64 to get a plain Python float.
    threshold_sq: float = float(stats.chi2.ppf(confidence, df=n_cols))
    threshold_raw       = np.sqrt(threshold_sq)   # np.float64
    threshold: float    = threshold_raw.item()    # plain Python float — safe for JSON

    # ── Identify outliers ─────────────────────────────────────────────────────

    outlier_mask  = dist_array > threshold   # True where this row's distance exceeds the threshold
    outlier_count = int(outlier_mask.sum())

    # np.where(outlier_mask)[0] returns an array of integer positions where outlier_mask is True.
    # .tolist() converts from np.ndarray to list, then int() converts each to plain Python int.
    outlier_positions: List[int] = [
        int(p) for p in np.where(outlier_mask)[0].tolist()
    ]

    # Map integer positions back to DataFrame index labels.
    # The `data` DataFrame might have a non-contiguous index (due to NaN row drops),
    # so we can't assume position == index label. We look up the actual label.
    data_index_list: List[Any] = data.index.tolist()
    outlier_indices: List[Any] = [data_index_list[p] for p in outlier_positions]

    # Build sample rows — up to 5 for display in the frontend
    sample_positions = outlier_positions[:5]
    sample_indices   = outlier_indices[:5]

    # Retrieve the actual row data for those sample rows from the original DataFrame
    # safe_json() converts all numpy types for clean JSON serialisation
    sample_rows: List[Dict[str, Any]] = safe_json(
        df.loc[sample_indices, col_list].to_dict(orient="records")
    )

    # Annotate each sample row with its Mahalanobis distance so the user can
    # see how far each flagged row is from the data centre
    for i, pos in enumerate(sample_positions):
        dist_val: float = distances[pos]   # plain Python float from the List[float]
        sample_rows[i]["__mahalanobis_distance__"] = round(dist_val, 4)

    return {
        "what_are_multivariate_outliers": explain_what_multivariate_outliers_are(),
        "method_explanation":             explain_mahalanobis_method(),
        "columns_used":                   col_list,
        "rows_analysed":                  n_rows,
        "rows_dropped_nan":               dropped_rows,   # rows excluded due to missing values
        "threshold":                      round(threshold, 4),
        "confidence_pct":                 round(confidence * 100, 1),
        "outlier_count":                  outlier_count,
        "outlier_indices":                outlier_indices,  # needed by the fix endpoint
        "sample_outliers":                sample_rows,
        # Show the relevant explanation depending on whether outliers were found
        "result_explanation": (
            explain_mahalanobis_result(outlier_count, n_rows, col_list, threshold)
            if outlier_count > 0
            else explain_mahalanobis_no_result(col_list)
        ),
        # Direct the user to the next step — fix endpoint if outliers exist
        "next_step": (
            "To remove these rows, use POST /task4/fix_multivariate_outliers "
            "with the outlier_indices from this response."
            if outlier_count > 0
            else "No action needed."
        ),
    }


class MultivariateOutlierFixPayload(BaseModel):
    """
    Defines the JSON body for POST /task4/fix_multivariate_outliers.

    outlier_indices:
        The list of row index labels returned by GET /task4/multivariate_outliers.
        Must be passed back directly from that response — don't guess or
        construct these manually, as the index labels may not be sequential.

    method:
        "drop" → permanently remove the outlier rows from the DataFrame.
        "flag" → add a boolean column '__multivariate_outlier__' that marks
                 each outlier row as True (and all other rows as False).
                 Useful when you want to keep the rows but mark them for
                 downstream filtering.

    Example:
        {
            "outlier_indices": [4, 17, 83],
            "method": "drop"
        }
    """
    outlier_indices: List[int]   # row index labels from the detection endpoint
    method:          str         # "drop" or "flag"


@task4.post("/task4/fix_multivariate_outliers")
async def fix_multivariate_outliers(payload: MultivariateOutlierFixPayload):
    """
    Drops or flags the multivariate outlier rows identified by the detection endpoint.

    Why we validate indices before doing anything:
        The outlier indices were computed at the time of the GET request.
        If the user has made other changes to the DataFrame since then
        (e.g. removed rows in Task 3), those index labels may no longer exist.
        We check all indices upfront and refuse to proceed if any are missing,
        rather than silently operating on the wrong rows.
    """
    df      = require_df().copy()
    applied = []
    errors  = []

    # Validate that every provided index still exists in the current DataFrame.
    # Show only the first 5 invalid indices to keep the error message readable.
    invalid = [i for i in payload.outlier_indices if i not in df.index]
    if invalid:
        errors.append(
            f"Row index/indices not found in dataset: {invalid[:5]}. "
            "Re-run GET /task4/multivariate_outliers to get fresh indices."
        )
        # Return early — don't make any changes if the indices are stale
        return {"success": False, "applied": [], "errors": errors}

    if payload.method == "drop":
        before = len(df)
        # df.drop(index=...) removes rows by their index labels
        df.drop(index=payload.outlier_indices, inplace=True)
        # reset_index renumbers rows from 0 to close the gaps left by removal
        df.reset_index(drop=True, inplace=True)
        removed = before - len(df)
        applied.append(
            f"Dropped {removed} multivariate outlier row(s). {len(df)} rows remain."
        )

    elif payload.method == "flag":
        # Create a new boolean column initialised to False for every row.
        # pd.array with dtype="boolean" uses pandas' nullable boolean type —
        # this is better than plain bool because it supports NA values,
        # which may exist in some rows due to earlier cleaning steps.
        df["__multivariate_outlier__"] = pd.array(
            [False] * len(df), dtype="boolean"
        )
        # Set True only for the outlier rows by their index labels
        df.loc[payload.outlier_indices, "__multivariate_outlier__"] = True
        # Count how many rows were actually flagged
        flagged = int(df["__multivariate_outlier__"].sum())
        applied.append(
            f"Added '__multivariate_outlier__' column. "
            f"{flagged} row(s) flagged as True, "
            f"{len(df) - flagged} row(s) flagged as False."
        )

    else:
        raise HTTPException(
            status_code=400,
            detail=f'Unknown method "{payload.method}". Valid options: drop, flag.'
        )

    # SNAPSHOT RULE: save BEFORE writing to dataset_state.df
    snapshot_store.save(f"Task 4 — multivariate outliers ({payload.method})", require_df())
    dataset_state.df = df

    return {
        "success":   len(errors) == 0,
        "applied":   applied,
        "errors":    errors,
        "new_shape": list(df.shape),
    }


# =============================================================================
# TASK 4 UPGRADE — CORRELATION AND REDUNDANCY ANALYSIS
#
# Correlation measures how strongly two columns move together.
# A correlation of +1.0 means they increase perfectly in lockstep.
# A correlation of -1.0 means one increases as the other decreases perfectly.
# A correlation near 0 means they are independent of each other.
#
# When two columns are highly correlated (e.g. |r| > 0.95), keeping both
# adds almost no new information — they are redundant. This matters for
# machine learning models, where redundant columns can cause instability.
#
# This endpoint also identifies the problematic pairs that would cause
# the singular matrix error in /task4/multivariate_outliers.
# =============================================================================

@task4.get("/task4/correlation_analysis")
async def get_correlation_analysis(threshold: float = 0.95):
    """
    Computes pairwise Pearson correlations between all numeric columns
    and identifies pairs that are highly correlated (above the threshold).

    For each high-correlation pair it suggests which column to drop —
    the one with more missing values, since it carries less information.

    The full correlation matrix is also returned so the frontend can
    render a heatmap visualisation.

    Why the numpy array approach (not .loc)?
        pandas' .corr() returns a DataFrame where every cell value is
        technically a Scalar type — a pandas abstraction that can hold
        many different things. Calling .item() on a Scalar is not always
        valid and can confuse Pylance's type checker.
        Converting to a plain numpy float64 array first (.to_numpy(dtype=float))
        gives us a clean (n × n) matrix where every element is np.float64,
        and .item() on np.float64 always returns a plain Python float.
        This avoids all type ambiguity and Pylance warnings.
    """
    df           = require_df()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Need at least 2 numeric columns to compute any correlations
    if len(numeric_cols) < 2:
        return {
            "numeric_columns":        numeric_cols,
            "message":                "Need at least 2 numeric columns to compute correlations.",
            "high_correlation_pairs": [],
            "correlation_matrix":     {},
        }

    # ── Build the correlation matrix ──────────────────────────────────────────

    # .corr() computes the Pearson correlation between every pair of columns.
    # Result is a (n_cols × n_cols) DataFrame — diagonal is always 1.0 (each
    # column correlates perfectly with itself).
    corr_df = df[numeric_cols].corr()

    # Convert immediately to a plain float64 numpy array.
    # dtype=float ensures no mixed types. Shape: (n, n).
    # Every subsequent index operation [i, j] returns np.float64 — fully typed.
    corr_array = corr_df.to_numpy(dtype=float)

    n = len(numeric_cols)

    # Build the correlation dict for the frontend heatmap.
    # Using integer indexing [i, j] on corr_array (not .loc on corr_df)
    # avoids all Scalar type ambiguity.
    corr_dict: Dict[str, Dict[str, float]] = {}
    for i, col in enumerate(numeric_cols):
        corr_dict[col] = {}
        for j, other in enumerate(numeric_cols):
            # .item() converts np.float64 → plain Python float for JSON serialisation
            corr_dict[col][other] = round(corr_array[i, j].item(), 4)

    # ── Find high-correlation pairs ───────────────────────────────────────────

    high_pairs: List[Dict[str, Any]] = []

    # Only iterate the upper triangle of the matrix (i < j) to avoid
    # reporting each pair twice (e.g. A-B and B-A are the same pair).
    for i in range(n):
        for j in range(i + 1, n):   # j starts at i+1 so we never check a column against itself
            col1     = numeric_cols[i]
            col2     = numeric_cols[j]
            # corr_array[i, j] is np.float64 — .item() gives plain Python float
            corr_val: float = corr_array[i, j].item()

            if abs(corr_val) >= threshold:   # check absolute value — negative correlation is equally redundant
                # Suggest dropping the column with MORE missing values —
                # it carries less information than the one with fewer gaps
                missing1       = int(df[col1].isna().sum())
                missing2       = int(df[col2].isna().sum())
                suggested_drop = col2 if missing2 >= missing1 else col1

                high_pairs.append({
                    "column_1":        col1,
                    "column_2":        col2,
                    "correlation":     round(corr_val, 4),
                    "abs_correlation": round(abs(corr_val), 4),   # for sorting by strength
                    "suggested_drop":  suggested_drop,
                    "explanation":     explain_high_correlation_pair(col1, col2, corr_val),
                })

    # Sort by absolute correlation descending — strongest relationships first
    high_pairs.sort(key=lambda x: x["abs_correlation"], reverse=True)

    return {
        "overview":               explain_correlation_overview(),
        "numeric_columns":        numeric_cols,
        "threshold_used":         threshold,
        "high_correlation_pairs": high_pairs,
        "pairs_found":            len(high_pairs),
        # Show a different explanation depending on whether any pairs were found
        "result_explanation": (
            explain_no_high_correlations()
            if not high_pairs
            else f"{len(high_pairs)} highly correlated pair(s) found above |r| = {threshold}."
        ),
        # Full matrix for the frontend heatmap — all values pre-converted to plain floats
        "correlation_matrix": corr_dict,
    }
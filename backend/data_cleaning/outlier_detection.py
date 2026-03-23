from fastapi import HTTPException, APIRouter
from pydantic import BaseModel
from scipy import stats
from typing import Dict, Optional, List, Any
from State.dfState import dataset_state
from xAI.outliers import (
    explain_what_outliers_are,
    explain_zscore_method,
    explain_iqr_method,
    explain_column_outliers,
    explain_outlier_if_you_cap_iqr,
    explain_outlier_if_you_cap_zscore,
    explain_outlier_if_you_remove_rows,
    explain_outlier_if_you_do_nothing,
    explain_what_multivariate_outliers_are,
    explain_mahalanobis_method,
    explain_mahalanobis_result,
    explain_mahalanobis_no_result,
    explain_mahalanobis_singular_matrix,
    explain_correlation_overview,
    explain_high_correlation_pair,
    explain_no_high_correlations,
)
import pandas as pd
import numpy as np
from State.snapshotState import snapshot_store

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


def safe_json(obj: Any) -> Any:
    """
    Converts numpy/pandas types to plain Python so FastAPI can serialise
    the response without errors.
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
# =============================================================================

@task4.get("/task4/outliers_info")
async def get_outliers_info():
    df           = require_df()
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
        series = df[col].dropna()

        if len(series) < 4:
            column_reports.append({
                "column":  col,
                "skipped": True,
                "reason":  f"Too few non-null values ({len(series)}) to detect outliers reliably.",
            })
            continue

        mean        = float(series.mean())
        std         = float(series.std())
        col_min     = float(series.min())
        col_max     = float(series.max())
        q1          = float(series.quantile(0.25))
        q3          = float(series.quantile(0.75))
        iqr         = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr

        if std == 0:
            zscore_mask  = pd.Series([False] * len(series), index=series.index)
            zscore_count = 0
        else:
            z_scores     = (series - mean) / std
            zscore_mask  = z_scores.abs() > 3
            zscore_count = int(zscore_mask.sum())

        iqr_mask  = (series < lower_fence) | (series > upper_fence)
        iqr_count = int(iqr_mask.sum())

        outlier_values = safe_json(series[iqr_mask].head(5).tolist())

        zscore_lower = mean - 3 * std
        zscore_upper = mean + 3 * std

        all_strategies = {
            "cap_iqr": {
                "label":       f"Cap to IQR fences [{lower_fence:.2f}, {upper_fence:.2f}]",
                "explanation": explain_outlier_if_you_cap_iqr(col, lower_fence, upper_fence),
            },
            "cap_zscore": {
                "label":       f"Cap to ±3 std [{zscore_lower:.2f}, {zscore_upper:.2f}]",
                "explanation": explain_outlier_if_you_cap_zscore(col, zscore_lower, zscore_upper),
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
            "column":               col,
            "skipped":              False,
            "zscore_outlier_count": zscore_count,
            "iqr_outlier_count":    iqr_count,
            "has_outliers":         zscore_count > 0 or iqr_count > 0,
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
            "column_explanation": explain_column_outliers(
                col, zscore_count, iqr_count,
                col_min, col_max, mean, std,
                q1, q3, lower_fence, upper_fence,
            ),
            "all_strategies": all_strategies,
        })

    total_cols_with_outliers = sum(
        1 for r in column_reports
        if not r.get("skipped") and r.get("has_outliers")
    )

    return {
        "what_are_outliers":     explain_what_outliers_are(),
        "zscore_explanation":    explain_zscore_method(),
        "iqr_explanation":       explain_iqr_method(),
        "columns_analysed":      len(numeric_cols),
        "columns_with_outliers": total_cols_with_outliers,
        "columns_clean":         len(numeric_cols) - total_cols_with_outliers,
        "column_reports":        column_reports,
    }


class OutlierFixPayload(BaseModel):
    """
    strategy maps column name to fix method.
    Valid methods: cap_iqr, cap_zscore, remove, do_nothing
    """
    strategy: Dict[str, str]


@task4.post("/task4/fix_outliers")
async def fix_outliers(payload: OutlierFixPayload):
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
                q1          = float(series.quantile(0.25))
                q3          = float(series.quantile(0.75))
                iqr         = q3 - q1
                lower_fence = q1 - 1.5 * iqr
                upper_fence = q3 + 1.5 * iqr
                df[col]     = df[col].clip(lower=lower_fence, upper=upper_fence)
                applied.append(
                    f'"{col}": capped to IQR fences '
                    f'[{lower_fence:.4f}, {upper_fence:.4f}]. '
                    f'No rows removed — extreme values pulled in.'
                )

            elif method == "cap_zscore":
                mean = float(series.mean())
                std  = float(series.std())
                if std == 0:
                    applied.append(
                        f'"{col}": all values are identical — no Z-score capping needed.'
                    )
                    continue
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
                df[col]     = df[col].clip(lower=lower_bound, upper=upper_bound)
                applied.append(
                    f'"{col}": capped to ±3 std '
                    f'[{lower_bound:.4f}, {upper_bound:.4f}]. '
                    f'No rows removed — extreme values pulled in.'
                )

            elif method == "remove":
                q1            = float(series.quantile(0.25))
                q3            = float(series.quantile(0.75))
                iqr           = q3 - q1
                lower_fence   = q1 - 1.5 * iqr
                upper_fence   = q3 + 1.5 * iqr
                before        = len(df)
                within_fences = (df[col] >= lower_fence) & (df[col] <= upper_fence)
                is_null        = df[col].isna()
                df             = df[within_fences | is_null]
                df.reset_index(drop=True, inplace=True)
                rows_removed   = before - len(df)
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
# =============================================================================

@task4.get("/task4/multivariate_outliers")
async def get_multivariate_outliers(
    columns: Optional[str] = None,
    confidence: float = 0.975,
):
    df = require_df()

    if columns:
        col_list     = [c.strip() for c in columns.split(",")]
        missing_cols = [c for c in col_list if c not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Column(s) not found: {missing_cols}")
        non_numeric = [c for c in col_list if df[c].dtype.kind not in ("i", "f")]
        if non_numeric:
            raise HTTPException(
                status_code=400,
                detail=f"Column(s) {non_numeric} are not numeric."
            )
    else:
        col_list = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(col_list) < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "Mahalanobis distance requires at least 2 numeric columns. "
                f"Only {len(col_list)} found."
            )
        )

    data         = df[col_list].dropna()
    n_rows       = len(data)
    n_cols       = len(col_list)
    dropped_rows = len(df) - n_rows

    if n_rows < n_cols + 1:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Not enough complete rows ({n_rows}) to compute Mahalanobis distance "
                f"for {n_cols} columns. Need at least {n_cols + 1} rows with no missing values."
            )
        )

    # .astype(float) gives us a plain float64 numpy array — fully typed
    data_array = data.values.astype(float)
    mean_vec   = np.mean(data_array, axis=0)
    cov_matrix = np.cov(data_array, rowvar=False)

    try:
        inv_cov = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        return {
            "success":     False,
            "explanation": explain_mahalanobis_singular_matrix(),
            "tip": (
                "Run GET /task4/correlation_analysis to identify which columns "
                "are perfectly correlated and should be removed before retrying."
            ),
        }

    # distances is a plain Python List[float].
    # Every append uses .item() on np.float64 which always returns a plain float.
    # This is the correct way to extract a scalar from numpy without Scalar ambiguity.
    distances: List[float] = []
    for row in data_array:
        diff    = row - mean_vec
        dist_sq = float(diff @ inv_cov @ diff.T)
        sq_root = np.sqrt(max(dist_sq, 0.0))       # np.float64
        distances.append(sq_root.item())            # .item() → plain Python float

    dist_array = np.array(distances)

    # threshold uses the same .item() pattern — np.sqrt returns np.float64
    threshold_sq: float = float(stats.chi2.ppf(confidence, df=n_cols))
    threshold_raw       = np.sqrt(threshold_sq)     # np.float64
    threshold: float    = threshold_raw.item()      # plain Python float

    outlier_mask  = dist_array > threshold
    outlier_count = int(outlier_mask.sum())

    # np.where()[0] → ndarray of integer positions → plain List[int]
    outlier_positions: List[int] = [
        int(p) for p in np.where(outlier_mask)[0].tolist()
    ]

    data_index_list: List[Any] = data.index.tolist()
    outlier_indices: List[Any] = [data_index_list[p] for p in outlier_positions]

    sample_positions = outlier_positions[:5]
    sample_indices   = outlier_indices[:5]

    sample_rows: List[Dict[str, Any]] = safe_json(
        df.loc[sample_indices, col_list].to_dict(orient="records")
    )

    # distances[pos] is a plain Python float — List[float] indexed by int
    for i, pos in enumerate(sample_positions):
        dist_val: float = distances[pos]
        sample_rows[i]["__mahalanobis_distance__"] = round(dist_val, 4)

    return {
        "what_are_multivariate_outliers": explain_what_multivariate_outliers_are(),
        "method_explanation":             explain_mahalanobis_method(),
        "columns_used":                   col_list,
        "rows_analysed":                  n_rows,
        "rows_dropped_nan":               dropped_rows,
        "threshold":                      round(threshold, 4),
        "confidence_pct":                 round(confidence * 100, 1),
        "outlier_count":                  outlier_count,
        "outlier_indices":                outlier_indices,
        "sample_outliers":                sample_rows,
        "result_explanation": (
            explain_mahalanobis_result(outlier_count, n_rows, col_list, threshold)
            if outlier_count > 0
            else explain_mahalanobis_no_result(col_list)
        ),
        "next_step": (
            "To remove these rows, use POST /task4/fix_multivariate_outliers "
            "with the outlier_indices from this response."
            if outlier_count > 0
            else "No action needed."
        ),
    }


class MultivariateOutlierFixPayload(BaseModel):
    """
    outlier_indices: row index labels from /task4/multivariate_outliers
    method:          "drop" removes rows, "flag" adds a boolean marker column
    """
    outlier_indices: List[int]
    method:          str


@task4.post("/task4/fix_multivariate_outliers")
async def fix_multivariate_outliers(payload: MultivariateOutlierFixPayload):
    df      = require_df().copy()
    applied = []
    errors  = []

    invalid = [i for i in payload.outlier_indices if i not in df.index]
    if invalid:
        errors.append(
            f"Row index/indices not found in dataset: {invalid[:5]}. "
            "Re-run GET /task4/multivariate_outliers to get fresh indices."
        )
        return {"success": False, "applied": [], "errors": errors}

    if payload.method == "drop":
        before = len(df)
        df.drop(index=payload.outlier_indices, inplace=True)
        df.reset_index(drop=True, inplace=True)
        removed = before - len(df)
        applied.append(
            f"Dropped {removed} multivariate outlier row(s). {len(df)} rows remain."
        )

    elif payload.method == "flag":
        df["__multivariate_outlier__"] = pd.array(
            [False] * len(df), dtype="boolean"
        )
        df.loc[payload.outlier_indices, "__multivariate_outlier__"] = True
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
# =============================================================================

@task4.get("/task4/correlation_analysis")
async def get_correlation_analysis(threshold: float = 0.95):
    df           = require_df()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        return {
            "numeric_columns":        numeric_cols,
            "message":                "Need at least 2 numeric columns to compute correlations.",
            "high_correlation_pairs": [],
            "correlation_matrix":     {},
        }

    # KEY FIX: convert the entire correlation DataFrame to a plain float64
    # numpy array immediately. Every subsequent index operation on corr_array
    # returns np.float64, on which .item() is always valid and always returns
    # a plain Python float. This avoids all Scalar / .loc type ambiguity.
    corr_df    = df[numeric_cols].corr()
    corr_array = corr_df.to_numpy(dtype=float)   # shape (n, n), dtype float64

    n = len(numeric_cols)

    # Build the full matrix dict using integer array indexing — no .loc, no Scalar
    corr_dict: Dict[str, Dict[str, float]] = {}
    for i, col in enumerate(numeric_cols):
        corr_dict[col] = {}
        for j, other in enumerate(numeric_cols):
            corr_dict[col][other] = round(corr_array[i, j].item(), 4)

    # Find high correlation pairs — again using integer array indexing
    high_pairs: List[Dict[str, Any]] = []

    for i in range(n):
        for j in range(i + 1, n):
            col1     = numeric_cols[i]
            col2     = numeric_cols[j]
            # corr_array[i, j] is np.float64 — .item() gives plain Python float
            corr_val: float = corr_array[i, j].item()

            if abs(corr_val) >= threshold:
                missing1       = int(df[col1].isna().sum())
                missing2       = int(df[col2].isna().sum())
                suggested_drop = col2 if missing2 >= missing1 else col1

                high_pairs.append({
                    "column_1":        col1,
                    "column_2":        col2,
                    "correlation":     round(corr_val, 4),
                    "abs_correlation": round(abs(corr_val), 4),
                    "suggested_drop":  suggested_drop,
                    "explanation":     explain_high_correlation_pair(
                        col1, col2, corr_val
                    ),
                })

    high_pairs.sort(key=lambda x: x["abs_correlation"], reverse=True)

    return {
        "overview":               explain_correlation_overview(),
        "numeric_columns":        numeric_cols,
        "threshold_used":         threshold,
        "high_correlation_pairs": high_pairs,
        "pairs_found":            len(high_pairs),
        "result_explanation": (
            explain_no_high_correlations()
            if not high_pairs
            else f"{len(high_pairs)} highly correlated pair(s) found above |r| = {threshold}."
        ),
        "correlation_matrix": corr_dict,
    }
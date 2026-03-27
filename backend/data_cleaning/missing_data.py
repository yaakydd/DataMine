# =============================================================================
# missing_data.py
#
# Task 2 of DataMine — detect and fix missing values in the uploaded dataset.
#
# Endpoints
# ---------
#   GET  /task2/missing_info
#       Read-only scan. Returns column-level and row-level missing reports,
#       severity labels, xAI explanations, before-fix distribution snapshots,
#       disguised sentinel counts, and correlated-missing pair warnings.
#
#   POST /task2/fix_missing
#       Applies user-chosen fix strategies per column.
#       Strategies: fill_mean, fill_median, fill_mode, fill_custom,
#                   drop_rows, drop_column, do_nothing.
#       Works on a copy — only saves back if at least one fix succeeds.
# =============================================================================


# --- Standard library --------------------------------------------------------

# builtins is imported explicitly so we always have a guaranteed reference to
# the Python built-in sum() function.
#
# Why this is necessary:
#   numpy's wildcard presence in a module's namespace can cause some type
#   checkers (notably Pylance) to resolve the bare name "sum" to numpy's
#   overloaded sum rather than the Python built-in, producing a
#   reportCallIssue: "Object of type np_2darray is not callable".
#   By importing the builtins module and calling builtins.sum(...) we give
#   both the runtime and the type checker an unambiguous reference.
import builtins

# --- FastAPI imports ----------------------------------------------------------
# HTTPException  lets us return a structured HTTP error (status code + message)
#                instead of letting Python raise an unhandled 500 crash.
# APIRouter      groups all Task-2 routes under one object; main.py mounts it
#                with app.include_router(task2, prefix="/api").
from fastapi import HTTPException, APIRouter

# --- Pydantic ----------------------------------------------------------------
# BaseModel is the base class for request-body schemas.
# Pydantic automatically validates and coerces incoming JSON before our
# function body runs — a bad payload gets a 422 response, never a raw crash.
from pydantic import BaseModel

# --- Standard-library type hints ---------------------------------------------
# Dict   — a mapping with typed key and value, e.g. Dict[str, int]
# Any    — a value of any type (used when the inner type is genuinely dynamic)
# List   — a list with a typed element, e.g. List[str]
# Tuple  — used for function return-type annotations instead of the built-in
#          tuple[...] subscript syntax.
#
#          FIX (reportIndexIssue):
#          The built-in tuple[X, Y] subscript syntax was introduced in
#          Python 3.9.  Pylance raises reportIndexIssue when the project
#          targets an earlier Python version and sees `tuple[...]` in a
#          type annotation.  Importing Tuple from typing and writing
#          Tuple[X, Y] instead is compatible with Python 3.7+ and silences
#          the diagnostic entirely.
from typing import Dict, Any, List, Tuple

# --- Shared application state ------------------------------------------------
# dataset_state is a singleton that holds the current DataFrame in memory.
# Every router imports the same object so they all operate on the same data.
from State.dfState import dataset_state

# --- xAI explanation functions -----------------------------------------------
# Each function returns one plain-English string shown in the UI's xAI panel.
# They live in a separate module so wording can be updated without touching
# data logic.
#
# FIX (reportAttributeAccessIssue x4):
# The four functions below — explain_type_mismatch_warning,
# explain_disguised_missing, explain_distribution_change,
# explain_correlated_missing — are NEW additions that must exist in
# xAI/missingData_explainer.py.  Pylance flagged them as unknown import
# symbols because the old explainer file did not define them.  The updated
# explainer file (delivered alongside this file) now defines all four.
from xAI.missingData_explainer import (
    explain_missing_severity,             # describes severity band (low/medium/high)
    explain_missing_if_you_drop_rows,     # explains what dropping affected rows means
    explain_missing_if_you_fill_mean,     # explains what filling with the mean means
    explain_missing_if_you_fill_median,   # explains what filling with the median means
    explain_missing_if_you_fill_mode,     # explains what filling with the mode means
    explain_missing_if_you_fill_custom,   # explains what filling with a custom value means
    explain_missing_if_you_drop_column,   # explains what dropping the whole column means
    explain_missing_if_you_do_nothing,    # explains what leaving NaN values in place means
    explain_type_mismatch_warning,        # warns when custom fill value type clashes with column dtype
    explain_disguised_missing,            # explains why string sentinels were replaced with NaN
    explain_distribution_change,          # interprets the before/after distribution snapshot
    explain_correlated_missing,           # warns when two columns share missing rows
)

# --- Third-party data libraries ----------------------------------------------
import pandas as pd   # DataFrame manipulation — the core library used throughout
import numpy as np    # np.nan is the canonical missing-value sentinel;
                      # dtype.kind is used to classify columns:
                      #   "i" = integer family, "f" = float family, "O" = object/string

# --- Snapshot store ----------------------------------------------------------
# Saves a copy of the DataFrame before every write operation, powering the
# undo/rollback system (up to 20 snapshots are kept).
from State.snapshotState import snapshot_store


# =============================================================================
# MODULE-LEVEL CONSTANT — DISGUISED MISSING SENTINELS
#
# Many real-world datasets represent "no value" with a string like "N/A",
# "-", "?", or an empty string.  pandas .isna() does NOT flag these — they
# look like ordinary string data.  We normalise them to np.nan before any
# analysis so all downstream logic only has to handle one kind of missing.
#
# Defined at module level so it is easy to extend without touching any
# function internals.
# =============================================================================
DISGUISED_MISSING_STRINGS: List[str] = [
    "n/a", "na", "n.a.", "n.a",          # common abbreviations for "not available"
    "none", "null",                        # programmatic nulls written as strings
    "missing", "unknown",                  # explicit human labels
    "-", "--", "---",                      # dash placeholders
    "?", "??",                             # question-mark placeholders
    "",                                    # empty string (after stripping whitespace)
    "not available", "not applicable",     # long-form variants
]

# Router object — all Task-2 endpoints attach to this.
# main.py registers it with: app.include_router(task2, prefix="/api")
task2 = APIRouter()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def require_df() -> pd.DataFrame:
    """
    Guard called at the start of every endpoint.

    If no file has been uploaded yet, dataset_state.df is None and every
    pandas operation would raise a confusing AttributeError.  This catches
    that early and returns a structured HTTP 400 instead.
    """
    if dataset_state.df is None:
        # 400 = Bad Request — the client called this before uploading a file
        raise HTTPException(
            status_code=400,
            detail="No dataset loaded. Please upload a file first via POST /api/dataset_info",
        )
    return dataset_state.df   # return the live DataFrame reference


def _normalise_disguised_missing(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    # FIX (reportIndexIssue):
    # Return type is written as Tuple[...] (from typing) not tuple[...] (built-in).
    # The built-in subscript syntax requires Python 3.9+; typing.Tuple works on
    # Python 3.7+ and silences Pylance's reportIndexIssue diagnostic.
    """
    Scans every object/string column and replaces disguised missing sentinels
    with np.nan so that downstream .isna() calls detect them correctly.

    Works on a copy — never modifies the caller's DataFrame in place.

    Parameters
    ----------
    df : the DataFrame to scan

    Returns
    -------
    df_clean : pd.DataFrame
        Copy of df with string sentinels replaced by np.nan.
    disguised_counts : Dict[str, int]
        Maps column name → number of cells replaced.
        Only columns where at least one replacement occurred are included.
    """
    df_clean: pd.DataFrame           = df.copy()   # always work on a copy
    disguised_counts: Dict[str, int] = {}           # accumulates replacement counts

    for col in df_clean.columns:

        # Only string/object columns can contain text sentinels.
        # Numeric and datetime columns are skipped entirely.
        if df_clean[col].dtype == object:

            # Normalise every cell to a stripped, lower-cased string for
            # comparison.  .astype(str) converts actual NaN cells (internally
            # stored as float) to the string "nan" — but "nan" is not in our
            # sentinel list, so genuine NaN cells are not double-counted.
            normalised: pd.Series = (
                df_clean[col]
                .astype(str)     # convert every cell value to a Python string
                .str.strip()     # remove leading/trailing whitespace
                .str.lower()     # lower-case so "N/A" == "n/a"
            )

            # Boolean mask — True for every cell whose normalised value
            # appears in the sentinel list.
            mask: pd.Series = normalised.isin(DISGUISED_MISSING_STRINGS)

            # int() converts the numpy scalar returned by .sum() to a plain
            # Python int, which is JSON-serialisable and type-checker-friendly.
            count: int = int(mask.sum())

            if count > 0:
                # Replace flagged cells with np.nan in the working copy.
                # .loc[mask, col] selects only the sentinel rows for this column.
                df_clean.loc[mask, col] = np.nan
                disguised_counts[col]   = count

    return df_clean, disguised_counts


def _distribution_summary(series: pd.Series) -> Dict[str, Any]:
    """
    Produces a lightweight snapshot of a column's value distribution.

    Used to generate before/after comparisons when a fix strategy is applied.
    All numpy scalar types are cast to plain Python types so FastAPI's JSON
    serialiser never encounters np.int64 or np.float64 objects.

    Parameters
    ----------
    series : column to summarise (NaN values are dropped before computing stats)

    Returns
    -------
    dict with a "type" key ("numeric" or "categorical") plus relevant stats.
    """
    # Drop NaN before computing stats so we describe the observed values only.
    # This is a local reassignment — the original Series is not mutated.
    series = series.dropna()

    if series.dtype.kind in ("i", "f"):
        # Numeric column — return descriptive statistics
        return {
            "type":  "numeric",
            "count": int(series.count()),              # non-null value count
            "mean":  round(float(series.mean()),  4),  # arithmetic mean, 4 decimal places
            "std":   round(float(series.std()),   4),  # standard deviation, 4 decimal places
            "min":   round(float(series.min()),   4),  # smallest observed value
            "max":   round(float(series.max()),   4),  # largest observed value
        }
    else:
        # Categorical / text column — return frequency of top 5 values
        top = series.value_counts().head(5)   # Series: value → count, sorted descending
        return {
            "type":       "categorical",
            "count":      int(series.count()),                         # non-null value count
            "top_values": {str(k): int(v) for k, v in top.items()},   # top-5 as plain dict
        }


def _find_correlated_missing(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Identifies pairs of columns that are BOTH missing on many of the same rows.

    Why this matters: if two columns share the same missing rows, applying
    drop_rows to one will silently remove those rows from the other too.
    Surfacing this lets the user make a coordinated decision.

    A pair is only reported when the row overlap covers at least 50 % of the
    missing rows in EITHER column — weaker overlaps are likely coincidental.

    Parameters
    ----------
    df : the DataFrame to inspect (not modified)

    Returns
    -------
    List of dicts, each describing one correlated pair:
        {"col_a": str, "col_b": str, "shared_rows": int,
         "pct_of_a": float, "pct_of_b": float}
    """
    # Only compare columns that actually have at least one missing value —
    # checking all column pairs would be wasteful for large, mostly-clean datasets.
    missing_cols: List[str]         = [c for c in df.columns if df[c].isna().any()]
    pairs:        List[Dict[str, Any]] = []

    # i < j ensures each pair is reported only once (not once each direction).
    for i in range(len(missing_cols)):
        for j in range(i + 1, len(missing_cols)):
            col_a: str = missing_cols[i]
            col_b: str = missing_cols[j]

            # Boolean Series — True for every row where the column is NaN
            mask_a: pd.Series = df[col_a].isna()
            mask_b: pd.Series = df[col_b].isna()

            # Count rows where BOTH columns are NaN simultaneously.
            # Bitwise & performs element-wise AND on two boolean Series.
            # int() converts the resulting numpy scalar to a plain Python int.
            shared: int = int((mask_a & mask_b).sum())

            if shared == 0:
                continue   # no row overlap — skip this pair

            # FIX (reportGeneralTypeIssues):
            # mask_a.sum() / mask_b.sum() return numpy scalars that Pylance
            # types as "Series[bool] | bool".  Using that directly as a
            # divisor makes Pylance complain "Invalid conditional operand"
            # because it cannot rule out the Series[bool] branch, whose
            # __bool__ raises a ValueError at runtime.
            #
            # Wrapping each .sum() call in int() narrows the type to a plain
            # Python int, which is unambiguously safe to divide by and
            # silences the reportGeneralTypeIssues diagnostic.
            count_a: int = int(mask_a.sum())   # total NaN count for col_a
            count_b: int = int(mask_b.sum())   # total NaN count for col_b

            pct_of_a: float = round(shared / count_a * 100, 1)
            pct_of_b: float = round(shared / count_b * 100, 1)

            # Only surface pairs where the overlap is substantial for either column.
            if pct_of_a >= 50 or pct_of_b >= 50:
                pairs.append({
                    "col_a":       col_a,
                    "col_b":       col_b,
                    "shared_rows": shared,
                    "pct_of_a":    pct_of_a,
                    "pct_of_b":    pct_of_b,
                })

    return pairs


# =============================================================================
# ENDPOINT 1 — GET /task2/missing_info
# =============================================================================

@task2.get("/task2/missing_info")
async def get_missing_info():
    """
    Read-only scan of the entire DataFrame for missing values.

    Steps
    -----
    1.  Run disguised-sentinel detection on a working copy (not saved back).
    2.  For every column with missing values, build a full report entry.
    3.  Compute row-level and whole-DataFrame summaries.
    4.  Find correlated missing pairs and attach xAI explanations.

    Returns a JSON object — see inline comments on the return dict for the
    meaning of each key.
    """
    raw_df = require_df()   # raises HTTP 400 if no file uploaded yet

    # ── Step 1: detect and replace disguised sentinels ────────────────────────
    # Must be read-only — we work on a copy and never write back to
    # dataset_state.df from a GET endpoint.
    # disguised_counts tells us which columns had sentinels and how many,
    # purely for advisory display in the UI.
    df, disguised_counts = _normalise_disguised_missing(raw_df)

    total_rows:  int = len(df)    # used as denominator for per-column percentages
    total_cells: int = df.size    # total cells = rows × columns, for overall pct

    # ── Step 2: column-level analysis ─────────────────────────────────────────
    columns_with_missing: List[Dict[str, Any]] = []

    for col in df.columns:

        # .isna().sum() counts True values in the boolean NaN mask.
        # int() converts the numpy scalar to a plain Python int.
        missing_count: int = int(df[col].isna().sum())

        if missing_count == 0:
            continue   # column is complete — skip it entirely

        # Percentage of this column's values that are missing
        pct: float = round((missing_count / total_rows) * 100, 2)

        # Severity bands (standard data-science thresholds):
        #   < 5%  → low:    filling or dropping rows has minimal impact
        #   5–20% → medium: requires careful consideration
        #   > 20% → high:   column may be too sparse to be reliable
        if pct < 5:
            severity = "low"
        elif pct < 20:
            severity = "medium"
        else:
            severity = "high"

        dtype: str       = str(df[col].dtype)   # e.g. "int64", "float64", "object"
        strategies: Dict = {}                    # holds one entry per available fix

        # ── Numeric-only strategies (mean & median) ───────────────────────────
        # dtype.kind "i" = all integer widths (int8…int64)
        # dtype.kind "f" = all float widths (float16…float64)
        # Checking .kind is more reliable than comparing the dtype string
        # because it handles all width variants in one check.
        if df[col].dtype.kind in ("i", "f"):
            mean_val:   float = round(float(df[col].mean()),   4)
            median_val: float = round(float(df[col].median()), 4)

            strategies["fill_mean"] = {
                "label":       f"Fill with mean ({mean_val})",
                "explanation": explain_missing_if_you_fill_mean(col, mean_val),
            }
            strategies["fill_median"] = {
                "label":       f"Fill with median ({median_val})",
                "explanation": explain_missing_if_you_fill_median(col, median_val),
            }

        # ── Mode strategy (works for any dtype) ───────────────────────────────
        # .mode() returns a Series; there can be multiple values on a tie.
        # We always take index [0] — the first value (alphabetical/numerical).
        mode_series: pd.Series = df[col].mode()
        if not mode_series.empty:   # guard: entirely-NaN column has no mode
            mode_display = mode_series[0]
            # numpy scalars (np.int64, np.float64, etc.) are not JSON-safe.
            # .item() converts them to the equivalent plain Python type.
            if hasattr(mode_display, "item"):
                mode_display = mode_display.item()
            strategies["fill_mode"] = {
                "label":       f"Fill with mode ({mode_display})",
                "explanation": explain_missing_if_you_fill_mode(col, mode_display),
            }

        # ── Universal strategies — available for every dtype ──────────────────
        strategies["fill_custom"] = {
            "label":       "Fill with a custom value",
            "explanation": explain_missing_if_you_fill_custom(col),
        }
        strategies["drop_rows"] = {
            # Show the exact row count in the label so the user knows the cost upfront
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

        # ── Sample of affected row indices (up to 5) ──────────────────────────
        # df.index[df[col].isna()] gives the index labels of the NaN rows.
        # .tolist()[:5] caps at 5 so the response stays compact.
        sample_indices: List = df.index[df[col].isna()].tolist()[:5]

        # ── Distribution snapshot BEFORE any fix ──────────────────────────────
        # Stored here so the frontend can show a before/after diff after POST,
        # without needing a second GET call.
        dist_before: Dict = _distribution_summary(df[col])

        # ── Disguised sentinel count for this specific column ─────────────────
        # .get(col, 0) returns 0 rather than raising KeyError when no sentinels
        # were found in this column.
        # The result is a plain int, so the `> 0` comparison below is safe.
        disguised_in_col: int = disguised_counts.get(col, 0)

        columns_with_missing.append({
            "column":              col,
            "dtype":               dtype,
            "missing_count":       missing_count,
            "missing_pct":         pct,
            "severity":            severity,
            "explanation":         explain_missing_severity(col, pct),   # xAI severity text
            "strategies":          strategies,
            "sample_missing_rows": sample_indices,
            "distribution_before": dist_before,
            "disguised_replaced":  disguised_in_col,
            # None signals to the frontend that no advisory banner is needed.
            # disguised_in_col is a plain int — comparing it with > 0 is safe.
            "disguised_explanation": (
                explain_disguised_missing(col, disguised_in_col)
                if disguised_in_col > 0
                else None
            ),
        })

    # ── Step 3: row-level analysis ─────────────────────────────────────────────
    # .isnull().any(axis=1) — axis=1 means "scan across columns for each row",
    # producing a boolean Series that is True for rows with at least one NaN.
    rows_with_missing: int   = int(df.isnull().any(axis=1).sum())
    rows_missing_pct:  float = round((rows_with_missing / total_rows) * 100, 2)

    # ── Step 4: whole-DataFrame summary ───────────────────────────────────────
    # .isnull().sum()  → per-column missing counts (a Series)
    # .sum() again     → sums those column totals → total missing cells
    total_missing:       int   = int(df.isnull().sum().sum())
    overall_missing_pct: float = round((total_missing / total_cells) * 100, 2)

    # ── Step 5: correlated missing pairs ──────────────────────────────────────
    correlated_pairs: List = _find_correlated_missing(df)

    # Attach an xAI explanation to each pair dict.
    # {**pair, "explanation": ...} spreads all existing keys and adds the new one.
    correlated_with_explanations: List = [
        {
            **pair,
            "explanation": explain_correlated_missing(
                pair["col_a"],
                pair["col_b"],
                pair["shared_rows"],
                pair["pct_of_a"],
                pair["pct_of_b"],
            ),
        }
        for pair in correlated_pairs
    ]

    # ── Step 6: safe total-disguised count ────────────────────────────────────
    # FIX (reportCallIssue):
    # The previous version wrote `sum(disguised_counts.values())` directly.
    # Pylance resolved "sum" to numpy's overloaded sum in this module's scope
    # (because numpy is imported), producing:
    #   "Object of type np_2darray is not callable"
    # Using builtins.sum() instead gives an explicit, unambiguous reference to
    # the Python built-in and eliminates the diagnostic entirely.
    total_disguised: int = builtins.sum(disguised_counts.values())

    return {
        "total_rows":               total_rows,
        "total_columns":            len(df.columns),
        "total_missing_cells":      total_missing,
        "overall_missing_pct":      overall_missing_pct,
        "rows_with_missing":        rows_with_missing,
        "rows_missing_pct":         rows_missing_pct,
        "columns_with_missing":     columns_with_missing,
        "clean_columns":            len(df.columns) - len(columns_with_missing),
        "total_disguised_replaced": total_disguised,          # total sentinel cells replaced
        "disguised_by_column":      disguised_counts,         # per-column sentinel counts
        "correlated_missing_pairs": correlated_with_explanations,
    }


# =============================================================================
# REQUEST BODY MODEL
# =============================================================================

class MissingFixPayload(BaseModel):
    """
    Pydantic schema for POST /task2/fix_missing.

    Fields
    ------
    strategy : Dict[str, Dict[str, Any]]
        Maps each column name to a fix-config dict.
        Every config dict must have a "method" key.
        The fill_custom method additionally requires a "value" key.

    replace_disguised : bool  (default True)
        When True, disguised sentinels ("N/A", "-", etc.) in object columns
        are replaced with NaN before fix strategies run, so fills operate on
        the complete set of missing values.
        Set to False only when those strings are intentional data, not placeholders.

    Example payload
    ---------------
    {
        "strategy": {
            "age":    {"method": "fill_median"},
            "salary": {"method": "fill_mean"},
            "city":   {"method": "fill_custom", "value": "Unknown"},
            "notes":  {"method": "drop_column"}
        },
        "replace_disguised": true
    }
    """
    strategy:          Dict[str, Dict[str, Any]]   # col_name → {method, optional value}
    replace_disguised: bool = True                 # default: normalise sentinels first


# =============================================================================
# ENDPOINT 2 — POST /task2/fix_missing
# =============================================================================

@task2.post("/task2/fix_missing")
async def fix_missing(payload: MissingFixPayload):
    """
    Applies the user's chosen fix strategy to each column in the payload.

    Design decisions
    ----------------
    • Works on a copy — dataset_state.df is only overwritten after all ops
      complete and at least one succeeded.
    • Errors are collected per column, not raised immediately, so one bad
      column does not block fixes on all others.
    • All fill and drop_column ops run BEFORE drop_rows ops.  drop_rows
      removes entire rows, which changes row counts mid-loop and would
      invalidate the NaN counts captured for fill operations.
    • nan_count_before is captured BEFORE any modification so the success
      message correctly reports how many cells were filled.  The original
      code read .isna().sum() after the fill — it always reported 0.
    """
    raw_df = require_df()   # raises HTTP 400 if no file uploaded yet

    # ── Step 1: optional sentinel normalisation ────────────────────────────────
    if payload.replace_disguised:
        # Replace "N/A", "-", etc. with np.nan on a fresh copy.
        df, disguised_counts = _normalise_disguised_missing(raw_df)
    else:
        df               = raw_df.copy()   # always copy — never mutate raw_df directly
        disguised_counts = {}              # nothing replaced — empty dict for the response

    applied:    List[str]      = []   # one success message per column successfully changed
    errors:     List[str]      = []   # one failure message per column that failed
    warnings:   List[str]      = []   # non-fatal advisory messages (e.g. type mismatch)
    dist_after: Dict[str, Any] = {}   # before/after distribution snapshot per changed column

    # ── Step 2: sort operations — fills first, row-drops last ─────────────────
    # Separating into two dicts and merging guarantees execution order regardless
    # of how the user ordered keys in the payload.
    fill_ops: Dict = {
        col: cfg
        for col, cfg in payload.strategy.items()
        if cfg.get("method") != "drop_rows"   # everything that is NOT a row drop
    }
    drop_ops: Dict = {
        col: cfg
        for col, cfg in payload.strategy.items()
        if cfg.get("method") == "drop_rows"   # only row-drop operations
    }

    # In Python 3.7+ dict preserves insertion order, so ** merge keeps the
    # two groups in the correct sequence: fills first, then row drops.
    ordered_strategy: Dict = {**fill_ops, **drop_ops}

    # ── Step 3: apply each fix strategy ───────────────────────────────────────
    for col, config in ordered_strategy.items():

        # Skip columns that are no longer in the DataFrame.
        # A column might have been dropped by drop_column earlier in this loop.
        if col not in df.columns:
            errors.append(f'"{col}" not found in the dataset.')
            continue

        # .get() with a default avoids KeyError if the "method" key is absent.
        method: str = config.get("method", "")

        # Capture NaN count BEFORE any modification — this value appears in
        # the success message ("filled N NaN(s)").
        nan_count_before: int = int(df[col].isna().sum())

        # Snapshot the distribution before this fix for the before/after UI.
        dist_before_col: Dict = _distribution_summary(df[col])

        try:

            # ── fill_mean ─────────────────────────────────────────────────────
            if method == "fill_mean":
                # Mean only makes mathematical sense for numeric columns.
                if df[col].dtype.kind not in ("i", "f"):
                    errors.append(
                        f'Cannot fill "{col}" with mean — it is not a numeric column. '
                        f'Use fill_mode or fill_custom instead.'
                    )
                    continue

                mean_val = df[col].mean()            # ignores NaN by default
                df[col]  = df[col].fillna(mean_val)  # replace every NaN with the mean
                applied.append(
                    f'"{col}": filled {nan_count_before} NaN(s) with mean = {mean_val:.4f}'
                )

            # ── fill_median ───────────────────────────────────────────────────
            elif method == "fill_median":
                if df[col].dtype.kind not in ("i", "f"):
                    errors.append(
                        f'Cannot fill "{col}" with median — it is not a numeric column.'
                    )
                    continue

                median_val = df[col].median()
                df[col]    = df[col].fillna(median_val)
                applied.append(
                    f'"{col}": filled {nan_count_before} NaN(s) with median = {median_val:.4f}'
                )

            # ── fill_mode ─────────────────────────────────────────────────────
            elif method == "fill_mode":
                mode_series = df[col].mode()
                if mode_series.empty:
                    # An entirely-NaN column has no mode
                    errors.append(
                        f'"{col}": cannot compute mode — column may be all NaN.'
                    )
                    continue

                mode_val = mode_series[0]    # first value, lowest on a tie
                df[col]  = df[col].fillna(mode_val)
                applied.append(
                    f'"{col}": filled {nan_count_before} NaN(s) with mode = {mode_val}'
                )

            # ── fill_custom ───────────────────────────────────────────────────
            elif method == "fill_custom":
                custom_val = config.get("value")   # None if the key is absent

                if custom_val is None:
                    errors.append(
                        f'"{col}": fill_custom requires a "value" field in the request.'
                    )
                    continue

                # Type-compatibility check:
                # We warn (not block) because pandas will still perform the fill
                # by upcasting the column to object dtype.  The user should know
                # their column type is changing.
                col_kind:   str  = df[col].dtype.kind
                val_is_num: bool = isinstance(custom_val, (int, float))
                val_is_str: bool = isinstance(custom_val, str)

                if col_kind in ("i", "f") and val_is_str:
                    # Numeric column + string fill → column becomes object dtype
                    warnings.append(
                        explain_type_mismatch_warning(
                            col, "numeric", custom_val, "string",
                            "the column will be converted to a mixed text/number type",
                        )
                    )
                elif col_kind == "O" and val_is_num:
                    # Text column + numeric fill → harmless but possibly unintended
                    warnings.append(
                        explain_type_mismatch_warning(
                            col, "text", custom_val, "number",
                            "the number will be stored as text",
                        )
                    )

                df[col] = df[col].fillna(custom_val)
                applied.append(
                    f'"{col}": filled {nan_count_before} NaN(s) '
                    f'with custom value = "{custom_val}"'
                )

            # ── drop_rows ─────────────────────────────────────────────────────
            elif method == "drop_rows":
                before: int = len(df)

                # subset=[col] drops rows where only THIS column is NaN.
                # Without subset, pandas would drop rows missing in ANY column.
                df = df.dropna(subset=[col])

                # reset_index(drop=True) renumbers the index from 0 after removal.
                # Without it the index has gaps (0, 2, 5, …) which can cause
                # subtle bugs in later index-based operations.
                df = df.reset_index(drop=True)

                rows_dropped: int = before - len(df)
                applied.append(
                    f'"{col}": dropped {rows_dropped} row(s) where value was missing. '
                    f'{len(df)} rows remain.'
                )

            # ── drop_column ───────────────────────────────────────────────────
            elif method == "drop_column":
                df = df.drop(columns=[col])   # remove the entire column
                applied.append(f'"{col}": column dropped entirely.')
                # There is no "after" distribution snapshot for a dropped column.
                continue

            # ── do_nothing ────────────────────────────────────────────────────
            elif method == "do_nothing":
                # User explicitly chose to leave NaN values in place.
                # Log it so the applied list reflects the full request.
                applied.append(f'"{col}": left as-is (NaN values kept).')
                continue   # nothing changed — no distribution snapshot needed

            # ── unknown method ────────────────────────────────────────────────
            else:
                errors.append(
                    f'"{col}": unknown method "{method}". '
                    f'Valid options: fill_mean, fill_median, fill_mode, '
                    f'fill_custom, drop_rows, drop_column, do_nothing.'
                )
                continue

            # ── Distribution snapshot AFTER the fix ───────────────────────────
            # Only reached for fill_* and drop_rows (drop_column and do_nothing
            # use `continue` above and never reach this point).
            # We compute the after-snapshot once and reuse it for both the dict
            # value and the explanation call — avoids calling the function twice.
            if col in df.columns:
                dist_after_col: Dict = _distribution_summary(df[col])
                dist_after[col] = {
                    "before":      dist_before_col,
                    "after":       dist_after_col,
                    "explanation": explain_distribution_change(
                        col, dist_before_col, dist_after_col
                    ),
                }

        except Exception as e:
            # Catch unexpected runtime errors per column and surface them to the
            # caller rather than crashing the entire request.
            errors.append(f'"{col}": unexpected error — {str(e)}')

    # ── Step 4: commit changes only if at least one fix succeeded ─────────────
    if applied:
        # Snapshot the state that existed BEFORE our changes so the user can
        # undo back to it.  require_df() still returns the original unmodified
        # DataFrame here because we have not yet written to dataset_state.df.
        snapshot_store.save(
            f"Task 2 — missing data: {len(applied)} fix(es) applied",
            require_df(),
        )
        dataset_state.df = df   # commit the modified copy as the new live DataFrame

    # Count remaining missing cells on the (possibly updated) DataFrame so the
    # frontend can show "N missing cells left" without a second GET call.
    remaining_missing: int = int(df.isnull().sum().sum())

    return {
        "success":              len(errors) == 0,   # True only when zero errors occurred
        "applied":              applied,             # success messages
        "errors":               errors,              # failure messages
        "warnings":             warnings,            # non-fatal advisory messages
        "new_shape":            list(df.shape),      # [rows, cols] as a JSON array
        "remaining_missing":    remaining_missing,
        "distribution_changes": dist_after,          # before/after snapshots per column
        "disguised_normalised": disguised_counts,    # per-column sentinel replacement counts
    }
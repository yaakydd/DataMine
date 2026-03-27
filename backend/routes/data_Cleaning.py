# =============================================================================
# data_cleaning.py
#
# This file is the ORCHESTRATION LAYER of DataMine.
# It sits between data_info.py (the upload step) and the individual task
# routers (task1 through task6) and does two things:
#
#   1. SCAN — runs every task's read-only info endpoint in sequence and
#             returns a combined report the frontend uses to show the user
#             what problems exist in their dataset, task by task.
#
#   2. APPLY — proxies fix requests to the correct task router so the
#              frontend always calls one consistent URL pattern.
#
# =============================================================================
#
# HOW THE PIPELINE WORKS (end-to-end flow):
#
#   Step 1 — User uploads a file
#     POST /api/dataset_info (data_info.py)
#     → DataFrame is stored in dataset_state.df
#     → Frontend receives the dataset overview card (filename, shape, etc.)
#
#   Step 2 — Frontend immediately calls the pipeline (no button press needed)
#     POST /api/pipeline/run
#     → This endpoint calls every task's scan function in order
#     → Returns a combined JSON object with one result block per task
#     → Frontend renders each task panel with its findings and suggested fixes
#
#   Step 3 — User reviews each task panel and selects a fix option
#     The frontend calls the individual task fix endpoints directly
#     (e.g. POST /api/task1/update_columns, POST /api/task2/fix_missing, etc.)
#     → The fix is applied to dataset_state.df
#     → The frontend re-renders only that task's panel
#
#   Step 4 — User moves to the next task panel
#     The cycle repeats: review findings → choose fix → apply → next task
#
# =============================================================================
#
# WHY A SEPARATE PIPELINE FILE?
#
#   Each task file (task1–task6) is responsible for ONE cleaning concern.
#   None of them know about the others. data_cleaning.py is the only place
#   that knows about all tasks and can run them in the correct order.
#
#   This separation means:
#     - Task files stay focused and independent
#     - The pipeline order can be changed in one place without touching any task
#     - The frontend calls one endpoint to get a complete picture of all issues
#
# =============================================================================


# HTTPException → return a clean error + HTTP status code instead of a Python crash
# APIRouter     → groups all pipeline endpoints under one object registered in main.py
from fastapi import HTTPException, APIRouter

# Shared singleton holding the current DataFrame in memory.
# The pipeline reads from this — it never writes to it directly.
# All writes are done by the individual task fix endpoints.
from State.dfState import dataset_state

# ── Import every task's read-only scan function ───────────────────────────────
#
# We import the underlying async functions, NOT the routers.
# This lets us call them directly as Python functions (passing the DataFrame)
# rather than making HTTP requests to ourselves, which would be slow and fragile.
#
# Each function is the GET info endpoint from its respective task file:
#   get_columns_info()    → Task 1: column name and dtype problems
#   get_missing_info()    → Task 2: missing value analysis
#   get_duplicates_info() → Task 3: duplicate row detection
#   get_outliers_info()   → Task 4: outlier detection (Z-score + IQR)
#   get_skewness_info()   → Task 5: skewness analysis
#   get_validation_info() → Task 6: auto-suggested validation rules

from data_cleaning.data_inconsistencies import get_columns_info
from data_cleaning.missing_data         import get_missing_info
from data_cleaning.duplicates           import get_duplicates_info
from data_cleaning.outlier_detection    import get_outliers_info
from data_cleaning.data_transformation  import get_skewness_info
from data_cleaning.data_validation           import get_validation_info

# All pipeline endpoints are attached to this router.
# main.py registers it with: app.include_router(pipeline, prefix="/api")
# so every route here becomes /api/pipeline/...
pipeline = APIRouter()


# =============================================================================
# HELPER
# =============================================================================

def require_df():
    """
    Called at the start of every endpoint to guard against missing uploads.

    If no file has been uploaded yet, dataset_state.df is None.
    Calling any pandas operation on None would crash with an unhelpful
    AttributeError. This catches it early and returns a clear HTTP 400.
    """
    if dataset_state.df is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "No dataset loaded. "
                "Please upload a file first via POST /api/dataset_info"
            )
        )
    return dataset_state.df


# =============================================================================
# POST /pipeline/run
#
# THE MAIN PIPELINE ENDPOINT.
#
# Called automatically by the frontend immediately after a successful upload.
# The user never clicks a button — this fires as soon as dataset_state.df
# is populated by data_info.py.
#
# It calls every task's scan function in the correct cleaning order and
# returns one combined JSON response.
#
# The frontend uses this response to:
#   1. Render each task panel with its findings
#   2. Show the recommended fixes for each task
#   3. Let the user choose which fix to apply before moving on
#
# Each task result is wrapped in a try/except so that a failure in one
# task does not prevent the other tasks from running and reporting their
# findings. The user sees results for every task that succeeded, plus an
# error message for any that failed.
# =============================================================================

@pipeline.post("/pipeline/run")
async def run_pipeline():
    """
    Runs all 6 task scans in sequence and returns a combined report.

    The scan order matches the recommended data cleaning workflow:
        1. Fix column names and dtypes first — every subsequent task
           depends on columns being correctly named and typed.
        2. Handle missing values — outlier and skewness calculations
           are distorted by NaN values.
        3. Remove duplicates — duplicates inflate counts and skew statistics.
        4. Detect outliers — after duplicates are removed, outlier detection
           is more accurate.
        5. Check skewness and scaling — only meaningful after the data is
           clean of the above issues.
        6. Validate rules — the final check to ensure business logic is met.

    Each result key matches the task number so the frontend can route
    each result block to the correct task panel without any mapping logic.
    """
    require_df()   # guard: raises 400 if no file uploaded

    # Each task result is stored here — one key per task.
    # If a task fails, its key contains an error dict instead of scan results,
    # so the frontend can show an error card for that task and continue
    # rendering the others normally.
    results = {}

    # ── Task 1: Column name and dtype issues ──────────────────────────────────
    # Must run first. Every other task uses column names and dtypes —
    # if names are broken or dtypes are wrong, downstream scans will
    # produce misleading results.
    try:
        results["task1_result"] = await get_columns_info()
    except Exception as e:
        results["task1_result"] = {
            "error": f"Task 1 scan failed: {str(e)}",
            "task":  "data_inconsistencies"
        }

    # ── Task 2: Missing value analysis ────────────────────────────────────────
    # Run second. Missing values affect mean, median, std, and correlation
    # calculations used by Tasks 4 and 5. The user should decide how to
    # handle them before those analyses run.
    try:
        results["task2_result"] = await get_missing_info()
    except Exception as e:
        results["task2_result"] = {
            "error": f"Task 2 scan failed: {str(e)}",
            "task":  "missing_data"
        }

    # ── Task 3: Duplicate row detection ───────────────────────────────────────
    # Run third. Duplicate rows inflate row counts and can skew every
    # statistic computed in Tasks 4 and 5. Remove them before analysing
    # value distributions.
    try:
        results["task3_result"] = await get_duplicates_info()
    except Exception as e:
        results["task3_result"] = {
            "error": f"Task 3 scan failed: {str(e)}",
            "task":  "duplicates"
        }

    # ── Task 4: Outlier detection ─────────────────────────────────────────────
    # Run fourth. With missing values handled and duplicates removed,
    # outlier detection is now operating on a cleaner dataset and the
    # IQR/Z-score statistics are more representative.
    try:
        results["task4_result"] = await get_outliers_info()
    except Exception as e:
        results["task4_result"] = {
            "error": f"Task 4 scan failed: {str(e)}",
            "task":  "outlier_detection"
        }

    # ── Task 5: Skewness and transformation analysis ──────────────────────────
    # Run fifth. Skewness analysis is most meaningful after outliers have
    # been addressed — extreme outliers artificially inflate skewness scores
    # and can trigger false transform recommendations.
    #try:
       # results["task5_result"] = await get_skewness_info
    #except Exception as e:
        results["task5_result"] = {
            "error": f"Task 5 scan failed: {str(e)}",
            "task":  "data_transformation"
        }

    # ── Task 6: Rule-based validation ─────────────────────────────────────────
    # Run last. Validation checks business logic rules (e.g. age must be
    # between 0 and 150) — these checks are most useful after the structural
    # data quality issues above have been resolved.
    try:
        results["task6_result"] = await get_validation_info()
    except Exception as e:
        results["task6_result"] = {
            "error": f"Task 6 scan failed: {str(e)}",
            "task":  "validation"
        }

    # ── Build the pipeline summary ────────────────────────────────────────────

    # Count how many tasks found at least one issue — used for the
    # top-level summary banner in the frontend ("5 of 6 tasks found issues").
    tasks_with_issues = _count_tasks_with_issues(results)

    return {
        # Top-level summary for the dashboard banner
        "pipeline_summary": {
            "tasks_run":         6,
            "tasks_with_issues": tasks_with_issues,
            "tasks_clean":       6 - tasks_with_issues,
            "message": (
                "Your dataset looks clean across all 6 checks!"
                if tasks_with_issues == 0 else
                f"{tasks_with_issues} of 6 task(s) found issues. "
                "Review each panel below and apply the recommended fixes."
            ),
        },

        # One result block per task — the frontend routes each to its panel.
        # Keys are named task1_result through task6_result so the frontend
        # can loop through them or access them by name without any mapping.
        **results,   # unpack the results dict directly into the response
    }


def _count_tasks_with_issues(results: dict) -> int:
    """
    Counts how many tasks found at least one issue in their scan.

    Each task returns a different structure, so we check for the most
    common "has issues" signals across all tasks:

        task1: columns_with_issues > 0
        task2: total_missing_cells > 0
        task3: duplicate_count > 0
        task4: columns_with_outliers > 0
        task5: columns_needing_transform > 0
        task6: auto_suggestions has entries

    If a task result contains an "error" key, it means the scan itself
    failed — we count that as having an issue (the user needs to know).

    This function is intentionally defensive: if a key doesn't exist in
    the result dict, .get() returns 0 or [] rather than raising a KeyError.
    """
    count = 0

    task1 = results.get("task1_result", {})
    if task1.get("error") or task1.get("columns_with_issues", 0) > 0:
        count += 1

    task2 = results.get("task2_result", {})
    if task2.get("error") or task2.get("total_missing_cells", 0) > 0:
        count += 1

    task3 = results.get("task3_result", {})
    if task3.get("error") or task3.get("duplicate_count", 0) > 0:
        count += 1

    task4 = results.get("task4_result", {})
    if task4.get("error") or task4.get("columns_with_outliers", 0) > 0:
        count += 1

    task5 = results.get("task5_result", {})
    if task5.get("error") or task5.get("columns_needing_transform", 0) > 0:
        count += 1

    task6 = results.get("task6_result", {})
    if task6.get("error") or len(task6.get("auto_suggestions", [])) > 0:
        count += 1

    return count


# =============================================================================
# GET /pipeline/status
#
# A lightweight endpoint the frontend can call at any time to check
# whether a dataset is currently loaded and get a quick health summary
# without re-running the full pipeline scan.
#
# Useful for:
#   - Checking on page load if a dataset is already in memory
#   - Refreshing the top-level summary after a fix is applied
#   - Showing a "dataset ready" indicator in the UI header
# =============================================================================

@pipeline.get("/pipeline/status")
async def get_pipeline_status():
    """
    Returns a lightweight summary of the current DataFrame state.

    Does NOT re-run any task scans — just reads what's already in memory.
    Safe to call frequently (e.g. on every page navigation) without
    triggering expensive computations.

    Returns:
        dataset_loaded: bool — whether a DataFrame is in memory
        shape:          [rows, cols] — current dimensions after any applied fixes
        columns:        list of current column names
        missing_cells:  total NaN count across the whole DataFrame
        message:        a one-line status summary
    """
    df = dataset_state.df

    if df is None:
        # No file has been uploaded yet — return a clean "not ready" state
        # rather than raising a 400, because this endpoint is used for polling
        return {
            "dataset_loaded": False,
            "shape":          None,
            "columns":        [],
            "missing_cells":  0,
            "message":        "No dataset loaded. Upload a file to begin.",
        }

    # DataFrame is in memory — return its current state.
    # int() converts numpy int64 to plain Python int for JSON serialisation.
    total_missing = int(df.isnull().sum().sum())

    return {
        "dataset_loaded": True,
        "shape":          list(df.shape),      # list() so it serialises as [rows, cols]
        "columns":        df.columns.tolist(),
        "missing_cells":  total_missing,
        "message": (
            f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns. "
            f"{total_missing} missing cell(s) remaining."
        ),
    }
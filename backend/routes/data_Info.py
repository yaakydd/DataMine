# =============================================================================
# data_info.py
#
# This file is the ENTRY POINT of DataMine.
# It handles one job: accept an uploaded file, read it into a pandas DataFrame,
# store it in shared memory, and return a summary of the dataset.
#
# After this endpoint responds, the frontend immediately calls
# POST /api/pipeline/run (in data_cleaning.py) which scans the DataFrame
# and returns a full report across all 6 cleaning tasks.
#
# Supported file formats:
#   CSV, TSV, JSON, XLSX/XLS, Parquet, ORC
#
# Endpoint:
#   POST /api/dataset_info
# =============================================================================


# File       → FastAPI's type that represents an incoming uploaded file object
# UploadFile → the actual file object with .filename, .content_type, and .read()
# APIRouter  → groups this endpoint under one object registered in main.py
# HTTPException → return a clean error + HTTP status code instead of a Python crash
from fastapi import File, UploadFile, APIRouter, HTTPException

# Shared singleton holding the current DataFrame in memory.
# Every other router in the project imports and uses the same object.
from State.dfState import dataset_state

import pandas as pd   # main data manipulation library
import io             # used to wrap raw bytes into file-like objects pandas can read
import json           # used to safely serialise DataFrame stats to JSON
import pyarrow.parquet as pq   # explicit pyarrow import for Parquet support

# Snapshot store saves copies of the DataFrame for undo/rollback.
# We clear all snapshots on a new upload so the undo history doesn't
# carry over from a previous session's data.
from State.snapshotState import snapshot_store


# All endpoints in this file are attached to this router.
# main.py registers it with: app.include_router(data_info, prefix="/api")
# so the full path becomes POST /api/dataset_info
data_info = APIRouter()


# =============================================================================
# POST /dataset_info
#
# This is the ONLY endpoint in this file.
# It accepts a file upload, reads it into a DataFrame, stores it in
# dataset_state, and returns a metadata summary the frontend can use
# to populate the dataset overview panel.
#
# It does NOT display anything to the user directly — the cleaning UI
# (data_cleaning.py) handles all interactive display.
# The response here is used by the frontend purely for the overview card
# (filename, shape, column list, first 5 rows, basic statistics).
# =============================================================================

@data_info.post("/dataset_info")
async def upload_file(file: UploadFile = File(...)):
    """
    Accepts a file upload, reads it into a DataFrame, and returns metadata.

    File(...) means the file parameter is REQUIRED — FastAPI will reject
    the request with a 422 Unprocessable Entity if no file is attached.

    The function is async because await file.read() is a non-blocking I/O
    operation — while the file is being read from the network, the server
    can handle other requests instead of blocking and waiting.
    """

    # Guard: reject requests where the file object has no name.
    # This shouldn't happen in normal use but can occur if the client
    # sends a malformed multipart request.
    if not file.filename:
        raise HTTPException(status_code=400, detail="File has no name")

    try:
        # Lowercase the filename so all extension checks below work consistently.
        # Without this, "Data.CSV" and "data.csv" would be treated differently.
        filename = file.filename.lower()

        # await file.read() reads the entire file content into memory as raw bytes.
        # await is used because .read() is an async operation — it waits for the
        # full file to arrive before continuing, without blocking other requests.
        file_content = await file.read()

        # io.BytesIO wraps raw bytes in a file-like object that pandas can read.
        # pandas' read functions (read_csv, read_excel, etc.) expect a file-like
        # object, not raw bytes — BytesIO is the bridge between the two.
        file_memory_buffer = io.BytesIO(file_content)

        # ── File format detection and loading ─────────────────────────────────

        # We check BOTH the filename extension AND the MIME content_type so that
        # the correct reader is used even if one of the two is missing or wrong.
        # The extension check is first because it's more reliable than MIME types,
        # which can vary between browsers and OS configurations.

        if filename.endswith(".csv") or file.content_type == "text/csv":
            # sep=','      → explicitly set comma as delimiter (CSV standard)
            # engine='python' → more flexible than the default 'c' engine;
            #                   handles irregular formatting that the c engine rejects
            # on_bad_lines='skip' → skip malformed rows instead of crashing;
            #                       a real-world CSV may have a few broken lines
            df = pd.read_csv(file_memory_buffer, sep=',', engine='python', on_bad_lines='skip')

        elif filename.endswith(".tsv") or file.content_type == "text/tab-separated-values":
            # TSV is identical to CSV but uses tab as the separator instead of comma
            df = pd.read_csv(file_memory_buffer, sep='\t', engine='python')

        elif filename.endswith(".json") or file.content_type == "application/json":
            # pd.read_json() handles both array-of-objects and object-of-arrays formats.
            # It infers the structure automatically.
            df = pd.read_json(file_memory_buffer)

        elif filename.endswith((".xlsx", ".xls")):
            # openpyxl is the required engine for .xlsx files.
            # Without specifying engine='openpyxl', newer pandas versions may
            # raise an error if the default engine is not available.
            # sheet_name defaults to 0 — always reads the first sheet.
            df = pd.read_excel(file_memory_buffer, engine='openpyxl')

        elif filename.endswith(".parquet") or file.content_type == "application/vnd.apache.parquet":
            # Parquet is a columnar binary format — common in data engineering pipelines.
            # engine='pyarrow' is specified explicitly to avoid ambiguity between
            # pyarrow and fastparquet (both are valid engines but behave differently).
            df = pd.read_parquet(file_memory_buffer, engine='pyarrow')

        elif filename.endswith(".orc") or file.content_type == "application/vnd.apache.orc":
            # ORC (Optimized Row Columnar) is similar to Parquet —
            # used in Hive and Spark pipelines.
            # pd.read_orc() uses pyarrow internally.
            df = pd.read_orc(file_memory_buffer)

        else:
            # File type is not supported — return a clear 400 error
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}"
            )

        # ── Store in shared state ─────────────────────────────────────────────

        # Write the loaded DataFrame to the shared singleton.
        # Every cleaning endpoint (task1 through task6) reads from this same object.
        # This is the moment the dataset becomes available to the rest of the app.
        dataset_state.set_df(user_id, df)   # replaces dataset_state.df = df
     

        # Clear all undo snapshots from any previous session.
        # Without this, the user could undo back to a state from a completely
        # different dataset — which would be confusing and dangerous.
        snapshot_store.clear(user_id)  

        # ── Build the metadata response ───────────────────────────────────────

        # df.info() writes a summary of dtypes, non-null counts, and memory usage
        # to a buffer. We capture it as a string and return it in the response.
        # buf=buffer redirects the output from stdout to our StringIO object.
        buffer    = io.StringIO()
        df.info(buf=buffer)
        info_data = buffer.getvalue()

        # df.memory_usage(deep=True) returns the RAM size of each column in bytes.
        # deep=True is critical for object (text) columns — without it, pandas
        # returns only the size of the pointer, not the actual string content.
        # .sum() adds up all columns to get the total DataFrame size in bytes.
        file_size = df.memory_usage(deep=True).sum()

        # Convert bytes to megabytes for human-readable display.
        # 1 MB = 1024 * 1024 bytes.
        filesize_in_mb = f"{round(file_size / (1024 * 1024), 2)} MB"

        # ── Separate columns by type for statistics ───────────────────────────

        # select_dtypes splits the DataFrame into numeric-only and text/category-only
        # subsets. describe() on each gives different statistics:
        #   numeric:     count, mean, std, min, 25%, 50%, 75%, max
        #   categorical: count, unique, top (most common value), freq
        numerical_df   = df.select_dtypes(include=['number'])
        categorical_df = df.select_dtypes(include=['object', 'category'])

        # ── JSON-safe serialisation ───────────────────────────────────────────

        # json.loads(df.to_json()) is used instead of df.to_dict() because:
        #   1. pandas' .to_json() handles NaN, Inf, and special types (dates,
        #      decimals from Parquet/ORC) much better than the standard json library
        #   2. json.loads() converts the result back to a plain Python dict/list
        #      that FastAPI can serialise cleanly without extra type handling
        # orient='records' → list of dicts, one dict per row
        first_five = json.loads(df.head().to_json(orient='records'))

        # fillna(0) replaces NaN in numeric stats — NaN can't be serialised to JSON
        numeric_stats = json.loads(numerical_df.describe().fillna(0).to_json())

        # fillna("NaN") replaces NaN in categorical stats with the string "NaN"
        # so the frontend can display "NaN" rather than a missing value
        category_stats = json.loads(categorical_df.describe().fillna("NaN").to_json())

        # ── Return metadata ───────────────────────────────────────────────────

        # This response is used by the frontend for the dataset overview card.
        # It does NOT trigger the cleaning pipeline — that is done separately
        # by POST /api/pipeline/run in data_cleaning.py, which the frontend
        # calls immediately after this response arrives.
        return {
            "filename":               filename,
            "file_size":              filesize_in_mb,
            "shape":                  df.shape,        # (rows, columns) tuple
            "columns":                df.columns.tolist(),
            "first_five_rows":        first_five,      # list of dicts, one per row
            "numeric_statistics":     numeric_stats,   # describe() output for numeric cols
            "categorical_statistics": category_stats,  # describe() output for text cols
            "df_info":                info_data,        # raw df.info() string output
        }

    except Exception as e:
        # Catch any unexpected error (corrupt file, unsupported encoding, etc.)
        # and return a 400 with the specific error message rather than a 500 crash
        raise HTTPException(
            status_code=400,
            detail=f"Error processing file: {str(e)}"
        )
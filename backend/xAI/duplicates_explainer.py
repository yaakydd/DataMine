"""
xai/explainer.py

All plain-English explanations used across every cleaning task live here.
Keeping them separate from the endpoint logic means:
  - You can update wording without touching any route file
  - You can later swap these out for Claude API calls per-column
  - The route files stay clean and focused on logic only

Each function takes the relevant data as arguments and returns either
a string explanation or a dict of multiple explanations.
"""


# =============================================================================
# TASK 1 — COLUMN INCONSISTENCIES
# =============================================================================

def explain_spaces_in_name(col_name: str) -> str:
    return (
        f'"{col_name}" contains spaces. '
        "This breaks SQL queries and pandas dot-notation. "
        "For example: df.First Name is a syntax error, "
        "but df.first_name works perfectly fine."
    )


def explain_special_chars(col_name: str, bad_chars: set) -> str:
    return (
        f'"{col_name}" contains special characters {bad_chars}. '
        "These are treated as mathematical operators in SQL and most query tools. "
        'For example, a column called "Revenue ($)" in SQL would be misread as '
        '"Revenue" multiplied by something, causing silent errors.'
    )


def explain_mixed_casing(col_name: str) -> str:
    return (
        f'"{col_name}" uses mixed casing. '
        "Python, SQL, and R are all case-sensitive in different ways. "
        'A column called "Revenue" and one called "revenue" are treated as '
        "two completely different columns, which causes key-not-found errors."
    )


def explain_numeric_as_text(col_name: str) -> str:
    return (
        f'"{col_name}" is stored as text (object) but all sample values look like numbers. '
        "This means you cannot calculate averages, sums, or run comparisons on it. "
        "Pandas will raise a TypeError the moment you try df['column'].mean(). "
        "Convert it to float64 or int64 to unlock all numeric operations."
    )


def explain_dtype_change_bool() -> str:
    return (
        "Converting to bool. Values like 'true', '1', 'yes' will become True. "
        "Values like 'false', '0', 'no' will become False. "
        "Any other value will cause an error — check your data first."
    )


def explain_dtype_change_datetime() -> str:
    return (
        "Converting to datetime. Pandas will try to parse common formats like "
        "'2024-01-15', 'Jan 15 2024', '15/01/2024'. "
        "Values it cannot parse will silently become NaT (Not a Time). "
        "Check for NaT values after conversion."
    )


def explain_dtype_change_category() -> str:
    return (
        "Converting to category dtype. This is ideal for columns with a small "
        "number of repeated values like gender, country, or status. "
        "Pandas stores them as integer codes internally, which can cut memory "
        "usage by up to 90% compared to storing raw text."
    )


# =============================================================================
# TASK 2 — MISSING DATA
# =============================================================================

def explain_missing_severity(col_name: str, pct: float) -> str:
    """
    Explains what the missing percentage means for a specific column.
    Three severity bands: low (under 5%), medium (5-20%), high (over 20%).
    """
    if pct < 5:
        return (
            f'"{col_name}" is missing {pct:.1f}% of its values. '
            "This is a small amount. You can safely fill these gaps with the "
            "mean, median, or mode without meaningfully distorting your data. "
            "Dropping the affected rows is also low-risk since so few are involved."
        )
    elif pct < 20:
        return (
            f'"{col_name}" is missing {pct:.1f}% of its values. '
            "This is a moderate amount. Filling with mean or median introduces "
            "some bias — the filled values pull results toward the average "
            "and reduce the natural spread of your data. "
            "Consider whether this column is critical to your analysis before deciding."
        )
    else:
        return (
            f'"{col_name}" is missing {pct:.1f}% of its values. '
            "This is a high amount — more than 1 in 5 values are unknown. "
            "Imputing this many values can seriously distort your results because "
            "you are essentially inventing a large portion of the data. "
            "Dropping the column is often the safest choice unless it is essential "
            "to your analysis — in which case, document the imputation clearly."
        )


def explain_missing_if_you_drop_rows(col_name: str, missing_count: int, total_rows: int) -> str:
    remaining = total_rows - missing_count
    pct_lost  = (missing_count / total_rows) * 100
    return (
        f"Dropping rows: you will lose {missing_count} rows and keep {remaining}. "
        f"That is {pct_lost:.1f}% of your dataset removed. "
        "Only do this if the rows with missing values are not representative "
        "of your population — otherwise you introduce selection bias."
    )


def explain_missing_if_you_fill_mean(col_name: str, mean_val: float) -> str:
    return (
        f"Filling with mean ({mean_val:.2f}): every missing cell gets the column average. "
        "This preserves your row count and keeps the mean unchanged, "
        "but reduces variance — your data looks more uniform than it really is. "
        "Best for columns with a roughly symmetric distribution and no extreme outliers."
    )


def explain_missing_if_you_fill_median(col_name: str, median_val: float) -> str:
    return (
        f"Filling with median ({median_val:.2f}): every missing cell gets the middle value. "
        "The median is more robust than the mean when your data has outliers "
        "because extreme values do not pull it in either direction. "
        "Best for skewed columns or columns with outliers."
    )


def explain_missing_if_you_fill_mode(col_name: str, mode_val) -> str:
    return (
        f"Filling with mode ({mode_val}): every missing cell gets the most frequent value. "
        "This is the standard approach for text (categorical) columns "
        "where mean and median do not apply. "
        "Be cautious if one value dominates — you may over-represent it."
    )


def explain_missing_if_you_fill_custom(col_name: str) -> str:
    return (
        f"Filling with a custom value: every missing cell gets exactly what you specify. "
        "Use this when missing has a real-world meaning — for example, "
        "filling 'discount' with 0 because missing means no discount was applied, "
        "or filling 'notes' with 'None' because missing means no note was written."
    )


def explain_missing_if_you_drop_column(col_name: str, pct: float) -> str:
    return (
        f"Dropping the column: '{col_name}' ({pct:.1f}% missing) will be removed entirely. "
        "Do this only if the column is not important to your goal, or if the missing "
        "rate is so high the column cannot be trusted. "
        "Once dropped, you cannot recover it without re-uploading the original file."
    )


def explain_missing_if_you_do_nothing(col_name: str) -> str:
    return (
        f"Leaving as-is: missing values stay as NaN in '{col_name}'. "
        "Many pandas operations skip NaN automatically (e.g. .mean(), .sum()), "
        "but machine learning models will fail or produce errors when they encounter NaN. "
        "Only leave it if your next step explicitly handles missing values."
    )


# =============================================================================
# TASK 3 — DUPLICATES
# =============================================================================

def explain_duplicates_overview(dup_count: int, total_rows: int, pct: float) -> str:
    """
    Top-level explanation shown in the summary banner.
    Explains what duplicates are and what the count means for this dataset.
    """
    if dup_count == 0:
        return (
            "No duplicate rows were found. Every row in your dataset is unique. "
            "You can safely move on to the next cleaning step."
        )
    return (
        f"{dup_count} duplicate row(s) found out of {total_rows} total rows ({pct:.1f}%). "
        "A duplicate row is an exact copy of another row — every single column "
        "has the same value. These can happen from data entry errors, "
        "system glitches that log the same event twice, or merging datasets "
        "that overlap. Duplicates inflate your counts, skew your averages, "
        "and can cause machine learning models to overfit on repeated examples."
    )


def explain_duplicates_if_you_keep_first(dup_count: int) -> str:
    return (
        f"Keep first occurrence: the first time a row appears is kept, "
        f"and all {dup_count} subsequent copies are removed. "
        "This is the most common choice. It assumes the first entry is the "
        "original and the rest are accidental repeats."
    )


def explain_duplicates_if_you_keep_last(dup_count: int) -> str:
    return (
        f"Keep last occurrence: the most recent copy of each duplicate is kept, "
        f"and all earlier ones are removed. "
        "Use this when your data is ordered by time and the latest entry "
        "is the most up-to-date — for example, a customer record that was "
        "updated and re-submitted."
    )


def explain_duplicates_if_you_drop_all(dup_count: int) -> str:
    return (
        f"Drop all copies: every row that appears more than once is removed entirely — "
        f"including the original. "
        "Use this when any repeated row is considered invalid, such as in "
        "experiment logs where each event should only ever appear once. "
        "This is the most aggressive option and removes more rows than keep_first or keep_last."
    )


def explain_duplicates_if_you_do_nothing() -> str:
    return (
        "Leave as-is: all duplicate rows are kept. "
        "Your row count, averages, and totals will include the repeated entries. "
        "Only choose this if duplicates are valid in your context — "
        "for example, a sales log where the same product can legitimately "
        "appear multiple times in one day."
    )


def explain_subset_duplicates(cols: list, dup_count: int) -> str:
    col_list = ", ".join(f'"{c}"' for c in cols)
    return (
        f"{dup_count} row(s) share identical values in the columns: {col_list}. "
        "These are not full row duplicates — other columns may differ — but "
        "these specific fields repeat. This often signals a data entry error "
        "or a join that produced unintended matches on those columns."
    )

# =============================================================================
# TASK 3 UPGRADES — paste at the bottom of your duplicates_explainer.py
# =============================================================================

def explain_what_fuzzy_duplicates_are() -> str:
    return (
        "Exact duplicate detection only catches rows where every character matches perfectly. "
        "Fuzzy duplicates are rows that represent the same real-world entity but differ "
        "slightly due to typos, spacing, or casing — for example: "
        "'John Smith' and 'john smith ', or 'New York' and 'new york'. "
        "These pass through exact deduplication undetected but still corrupt your "
        "aggregations, inflate row counts, and produce misleading group statistics. "
        "Fuzzy detection uses string similarity scoring to catch these near-matches."
    )


def explain_fuzzy_match(
    col_name: str,
    val1: str,
    val2: str,
    similarity: float,
    row_idx1: int,
    row_idx2: int,
) -> str:
    return (
        f'Near-duplicate found in "{col_name}": '
        f'row {row_idx1} ("{val1}") and row {row_idx2} ("{val2}") '
        f"are {similarity:.1f}% similar. "
        "These may represent the same entity with a data entry inconsistency. "
        "Review both rows and decide whether to merge, standardise, or keep them separate."
    )


def explain_fuzzy_threshold(threshold: float) -> str:
    return (
        f"Similarity threshold set to {threshold:.0f}%. "
        f"Any two values with a similarity score at or above {threshold:.0f}% "
        "will be flagged as potential duplicates. "
        "Higher threshold (e.g. 95%) = fewer but more certain matches. "
        "Lower threshold (e.g. 80%) = more matches but more false positives. "
        "Start at 90% and lower it if you are missing obvious duplicates."
    )


def explain_duplicate_impact(
    col_name: str,
    mean_before: float,
    mean_after: float,
    sum_before: float,
    sum_after: float,
    count_before: int,
    count_after: int,
) -> str:
    mean_change = mean_after - mean_before
    sum_change  = sum_after  - sum_before
    direction   = "increase" if mean_change > 0 else "decrease"

    return (
        f'Impact on "{col_name}" after removing duplicates: '
        f"row count {count_before} -> {count_after} ({count_before - count_after} removed). "
        f"Mean: {mean_before:.4f} -> {mean_after:.4f} "
        f"({direction} of {abs(mean_change):.4f}). "
        f"Sum: {sum_before:.4f} -> {sum_after:.4f} "
        f"(change of {sum_change:.4f}). "
        "If the mean changed significantly, your duplicates were not evenly distributed "
        "across the value range — removing them will affect your statistical summaries."
    )


def explain_no_fuzzy_columns() -> str:
    return (
        "No text columns were found to perform fuzzy duplicate detection on. "
        "Fuzzy detection only applies to text (object dtype) columns. "
        "Your dataset may be entirely numeric, in which case exact duplicate "
        "detection in the main task is sufficient."
    )
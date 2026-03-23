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
# TASK 4 — OUTLIER DETECTION
# =============================================================================

def explain_what_outliers_are() -> str:
    """
    General explanation of what outliers are.
    Shown once at the top of the outlier detection page
    before any column-specific results are displayed.
    """
    return (
        "An outlier is a data point that sits far away from the rest of your data. "
        "They are not always errors — a billionaire's salary in a household income "
        "dataset is a genuine extreme value, while an age of 999 is clearly a "
        "data entry mistake. "
        "Outliers matter because they can pull your averages, distort your "
        "statistical summaries, and cause machine learning models to learn the "
        "wrong patterns. "
        "We use two detection methods: Z-score (how many standard deviations "
        "a value is from the mean) and IQR (how far a value sits beyond the "
        "middle 50% of the data). IQR is more reliable when your data is already "
        "skewed, because the mean and standard deviation used in Z-score are "
        "themselves pulled by extreme values."
    )


def explain_zscore_method() -> str:
    return (
        "Z-score measures how many standard deviations a value is from the mean. "
        "A Z-score of 0 means the value equals the mean exactly. "
        "A Z-score of 2 means the value is 2 standard deviations above the mean. "
        "We flag values with a Z-score beyond +3 or below -3 as outliers. "
        "In a normal distribution, only 0.3% of values fall beyond ±3 — "
        "so anything flagged here is genuinely unusual."
    )


def explain_iqr_method() -> str:
    return (
        "IQR (Interquartile Range) measures the spread of the middle 50% of your data. "
        "Q1 is the 25th percentile, Q3 is the 75th percentile, and IQR = Q3 - Q1. "
        "The lower fence is Q1 - 1.5 x IQR. The upper fence is Q3 + 1.5 x IQR. "
        "Any value outside these fences is flagged as an outlier. "
        "IQR is more robust than Z-score for skewed data because it is based "
        "on the middle of the distribution and is not pulled by extreme values."
    )


def explain_column_outliers(
    col_name: str,
    zscore_count: int,
    iqr_count: int,
    col_min: float,
    col_max: float,
    mean: float,
    std: float,
    q1: float,
    q3: float,
    lower_fence: float,
    upper_fence: float,
) -> str:
    """
    Column-specific explanation combining both detection results.
    Tells the user what was found and what it means for this specific column.
    """
    if zscore_count == 0 and iqr_count == 0:
        return (
            f'"{col_name}" has no outliers by either method. '
            f"All values sit within the expected range "
            f"[{lower_fence:.2f} – {upper_fence:.2f}] (IQR fences) "
            f"and within ±3 standard deviations of the mean ({mean:.2f}). "
            "This column looks clean."
        )

    higher = max(zscore_count, iqr_count)
    return (
        f'"{col_name}" has up to {higher} outlier(s) detected. '
        f"The column ranges from {col_min:.2f} to {col_max:.2f}, "
        f"with a mean of {mean:.2f} and standard deviation of {std:.2f}. "
        f"Z-score flagged {zscore_count} value(s) beyond ±3 standard deviations. "
        f"IQR flagged {iqr_count} value(s) outside the fences "
        f"[{lower_fence:.2f}, {upper_fence:.2f}]. "
        "If this column represents something like age, salary, or test scores, "
        "check whether the extreme values are real data points or entry errors."
    )


def explain_outlier_if_you_cap_iqr(col_name: str, lower: float, upper: float) -> str:
    return (
        f"Capping to IQR fences: any value in '{col_name}' below {lower:.2f} "
        f"will be set to {lower:.2f}, and any value above {upper:.2f} "
        f"will be set to {upper:.2f}. "
        "The outlier rows are kept — only the extreme values are pulled in. "
        "This is called Winsorisation. It reduces the influence of extremes "
        "without losing any rows, making it the least destructive option."
    )


def explain_outlier_if_you_cap_zscore(col_name: str, lower: float, upper: float) -> str:
    return (
        f"Capping to ±3 standard deviations: values in '{col_name}' below "
        f"{lower:.2f} will be set to {lower:.2f}, and values above {upper:.2f} "
        f"will be set to {upper:.2f}. "
        "Similar to IQR capping but uses the mean and standard deviation "
        "to define the boundaries instead of quartiles. "
        "Better suited for symmetric, bell-shaped distributions."
    )


def explain_outlier_if_you_remove_rows(col_name: str, iqr_count: int) -> str:
    return (
        f"Removing outlier rows: the {iqr_count} row(s) in '{col_name}' "
        "that fall outside the IQR fences will be deleted entirely. "
        "This is the most aggressive option — you lose those data points completely. "
        "Use this when outliers are confirmed data entry errors and keeping "
        "them (even capped) would mislead your analysis."
    )


def explain_outlier_if_you_do_nothing(col_name: str) -> str:
    return (
        f"Leaving '{col_name}' as-is: all outlier values are kept unchanged. "
        "Choose this when the extreme values are real and meaningful — "
        "for example, a top-earning employee's salary in a payroll dataset "
        "is a genuine data point that should not be removed or altered. "
        "Be aware that these values will continue to influence your means "
        "and standard deviations."
    )
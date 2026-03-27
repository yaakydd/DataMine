"""
xAI/missingData_explainer.py

All plain-English explanations for Task 2 — Missing Data.

Functions added in this revision
---------------------------------
explain_type_mismatch_warning   — warns when a custom fill value's type
                                   does not match the column's dtype.
explain_disguised_missing       — explains what a disguised sentinel is and
                                   why it was replaced with NaN.
explain_distribution_change     — interprets the before/after distribution
                                   snapshot produced after a fix is applied.
explain_correlated_missing      — warns when two columns share missing rows,
                                   so the user understands the knock-on effect
                                   of fixing one column on the other.
"""

from typing import Dict, Any


# =============================================================================
# ORIGINAL FUNCTIONS (unchanged wording, kept intact)
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
        "Filling with a custom value: every missing cell gets exactly what you specify. "
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
        "but some operations and tools will fail when they encounter NaN. "
        "Only leave it if your next step explicitly handles missing values."
    )


# =============================================================================
# NEW FUNCTIONS
# =============================================================================

def explain_type_mismatch_warning(
    col_name:     str,
    col_type:     str,
    custom_val:   Any,
    val_type:     str,
    consequence:  str,
) -> str:
    """
    Warns the user when the type of their custom fill value does not match
    the column's dtype.

    Parameters
    ----------
    col_name    : column being filled
    col_type    : human-readable column type ("numeric", "text", etc.)
    custom_val  : the value the user provided
    val_type    : human-readable type of that value ("string", "number", etc.)
    consequence : plain-English description of what pandas will do
    """
    return (
        f'Type mismatch warning for "{col_name}": '
        f"the column is {col_type} but you are filling it with the {val_type} "
        f'value "{custom_val}". '
        f"The fill will still be applied, but {consequence}. "
        "If this is not what you intended, cancel and choose a value that "
        "matches the column's type, or convert the column type first."
    )


def explain_disguised_missing(col_name: str, count: int) -> str:
    """
    Explains why cells containing strings like "N/A", "-", "none", or "?"
    were treated as missing even though they are not technically NaN.

    Parameters
    ----------
    col_name : the column where sentinels were found
    count    : how many cells were replaced
    """
    return (
        f'"{col_name}" contained {count} cell(s) with placeholder text such as '
        '"N/A", "-", "none", "?", or similar — values that represent '
        '"no data" in plain language but are stored as text, not as true blanks. '
        "These have been converted to proper missing values (NaN) so that "
        "your chosen fix strategy applies to all of them, not just the ones "
        "that were already blank. "
        "If any of these strings were intentional data rather than placeholders, "
        "use the custom fill option to restore a specific value after cleaning."
    )


def explain_distribution_change(
    col_name:    str,
    before:      Dict[str, Any],
    after:       Dict[str, Any],
) -> str:
    """
    Interprets the before/after distribution snapshots produced when a fill
    strategy is applied, and explains what changed in plain English.

    Parameters
    ----------
    col_name : column that was modified
    before   : _distribution_summary() result captured before the fix
    after    : _distribution_summary() result captured after the fix
    """
    if before["type"] == "numeric":
        mean_before = before["mean"]
        mean_after  = after["mean"]
        std_before  = before["std"]
        std_after   = after["std"]
        count_added = after["count"] - before["count"]

        mean_delta  = round(mean_after  - mean_before, 4)
        std_delta   = round(std_after   - std_before,  4)

        mean_note = (
            "The mean did not change meaningfully."
            if abs(mean_delta) < 0.01
            else (
                f"The mean shifted from {mean_before} to {mean_after} "
                f"({'up' if mean_delta > 0 else 'down'} by {abs(mean_delta)})."
            )
        )
        std_note = (
            "The standard deviation fell slightly, which is expected — "
            "imputed values reduce spread because they are all the same number."
            if std_delta < -0.01
            else (
                "The standard deviation is roughly unchanged."
                if abs(std_delta) < 0.01
                else f"The standard deviation changed from {std_before} to {std_after}."
            )
        )

        return (
            f'"{col_name}": {count_added} missing value(s) were filled. '
            f"{mean_note} {std_note} "
            f"Range: {after['min']} – {after['max']} (unchanged by filling)."
        )

    else:
        # Categorical column
        count_added = after["count"] - before["count"]
        top_before  = list(before.get("top_values", {}).keys())
        top_after   = list(after.get("top_values",  {}).keys())
        top_changed = top_before != top_after

        top_note = (
            "The most common values did not change."
            if not top_changed
            else (
                f"The ranking of most common values shifted — "
                f"previously {top_before[:3]}, now {top_after[:3]}. "
                "This may mean the fill value became one of the dominant categories."
            )
        )

        return (
            f'"{col_name}": {count_added} missing value(s) were filled. {top_note}'
        )


def explain_correlated_missing(
    col_a:       str,
    col_b:       str,
    shared_rows: int,
    pct_of_a:    float,
    pct_of_b:    float,
) -> str:
    """
    Warns the user when two columns are missing values on many of the same rows.

    Parameters
    ----------
    col_a, col_b  : the two columns that share missing rows
    shared_rows   : number of rows where BOTH columns are NaN
    pct_of_a      : what percentage of col_a's missing rows overlap with col_b
    pct_of_b      : what percentage of col_b's missing rows overlap with col_a
    """
    return (
        f'"{col_a}" and "{col_b}" are both missing on {shared_rows} of the same row(s) '
        f"({pct_of_a}% of {col_a}'s gaps, {pct_of_b}% of {col_b}'s gaps overlap). "
        "This suggests the two columns may be related — for example, they could both "
        "come from an optional section of a form, or from a join that produced NaN "
        "in multiple columns at once. "
        "What this means practically: "
        f"if you choose 'drop rows' for '{col_a}', those same {shared_rows} rows will "
        f"also disappear from '{col_b}', potentially resolving part of its missing "
        "problem without a separate action. "
        "Conversely, if you fill both columns independently, you are making two "
        "separate assumptions about the same missing event, which may be inconsistent. "
        "Consider deciding on a single strategy that covers both columns together."
    )
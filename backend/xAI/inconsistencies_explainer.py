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
# TASK 1 UPGRADES — TEXT CLEANING, CATEGORICAL HARMONISATION, TYPE INFERENCE
# =============================================================================

def explain_text_value_cleaning(col_name: str, issues_found: int) -> str:
    return (
        f'"{col_name}" has {issues_found} value(s) with text formatting issues. '
        "Cell values like 'london', 'London', 'LONDON ', ' london' all mean "
        "the same thing but Python and SQL treat them as completely different categories. "
        "This silently inflates your category counts, breaks GROUP BY queries, "
        "and causes value_counts() to show duplicates that are not real duplicates. "
        "Cleaning trims whitespace, standardises casing, and removes invisible characters."
    )


def explain_categorical_harmonisation(col_name: str, variants: dict) -> str:
    groups = "; ".join(
        f"{canonical} <- {v}"
        for canonical, v in list(variants.items())[:3]
    )
    return (
        f'"{col_name}" has inconsistent category values that mean the same thing. '
        f"Example groups detected: {groups}. "
        "These variants will be counted separately in every analysis — "
        "a pivot table will show 'Male', 'male', 'M', and 'MALE' as four different "
        "groups instead of one. Harmonising maps all variants to one canonical form."
    )


def explain_hidden_boolean(col_name: str) -> str:
    return (
        f'"{col_name}" only contains 0 and 1 but is stored as int64 or float64. '
        "This is almost certainly a boolean column (True/False, Yes/No). "
        "Converting it to bool dtype makes the column meaning explicit, "
        "reduces memory usage, and prevents accidental arithmetic on flag columns "
        "like summing a column that means 'is_active'."
    )


def explain_likely_id_column(col_name: str) -> str:
    return (
        f'"{col_name}" appears to be an ID column — every value is unique '
        "and the name contains 'id', 'key', 'code', or 'uuid'. "
        "ID columns should never be used in statistical calculations. "
        "Taking the mean of a customer ID is meaningless. "
        "Consider excluding this column from any numeric analysis."
    )


def explain_high_cardinality(col_name: str, unique_count: int, total: int) -> str:
    pct = round((unique_count / total) * 100, 1)
    return (
        f'"{col_name}" has {unique_count} unique values out of {total} rows ({pct}%). '
        "High cardinality text columns are rarely useful as categories — "
        "if almost every row has a different value, grouping by this column "
        "produces groups of size 1, which is statistically meaningless. "
        "This column may be a free-text field, a name, or an ID stored as text. "
        "Consider dropping it or extracting structured information from it."
    )


def explain_text_clean_result(col_name: str, method: str) -> str:
    methods = {
        "trim":           f'Trimmed whitespace from all values in "{col_name}". Leading and trailing spaces removed.',
        "lowercase":      f'Lowercased all values in "{col_name}". Every value is now lowercase.',
        "uppercase":      f'Uppercased all values in "{col_name}". Every value is now uppercase.',
        "titlecase":      f'Title-cased all values in "{col_name}". First letter of each word is capitalised.',
        "remove_special": f'Removed special characters from values in "{col_name}". Only letters, digits, and spaces kept.',
        "all":            f'Applied full text cleaning to "{col_name}": trimmed, lowercased, and removed special characters.',
    }
    return methods.get(method, f'Applied {method} cleaning to "{col_name}".')
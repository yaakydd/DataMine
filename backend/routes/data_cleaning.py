from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
import pandas as pd
import sys
import os

# Add parent directory to path to import from logic_biz
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logic_biz.data_validator import DataValidator
from logic_biz.data_quality_checker import DataCleaner

# Import the centralized storage
from storage.datasets import DatasetStorage

# Create router
router = APIRouter()


# ============================================
# 1. VALIDATE DATASET (Find all problems)
# ============================================

@router.post("/validate/{dataset_id}")
async def validate_dataset(dataset_id: str) -> Dict[str, Any]:
    """
    Run validation checks on uploaded dataset.
    
    Plain English: Inspect the data and create a report of all issues found.
    Like a doctor examining a patient and listing all health problems.
    
    Args:
        dataset_id: The unique ID of uploaded dataset
    
    Returns:
        Comprehensive validation report with all issues
    """
    
    # Check if dataset exists
    # Get the dataset
    dataset = DatasetStorage.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Dataset not found",
                "explanation": (
                    "Can't find this dataset. Maybe it was deleted or the ID is wrong? "
                    "It's like trying to clean a car that's not in the garage!"
                )
            }
        )
    
    df = dataset["dataframe"]
    
    # Run validation
    validator = DataValidator(df)
    report = validator.validate_all()
    
    return {
        "success": True,
        "dataset_id": dataset_id,
        "validation_report": report,
        "explanation": (
            f"✅ Validation complete! "
            f"Found {report['summary']['total_issues']} issues and "
            f"{report['summary']['total_warnings']} warnings. "
            f"\n\nThink of this like a health checkup - we've identified all the problems "
            f"that need fixing before your data is analysis-ready."
        )
    }


# ============================================
# 2. REMOVE DUPLICATES
# ============================================

@router.post("/clean/remove-duplicates/{dataset_id}")
async def remove_duplicates(
    dataset_id: str,
    subset: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Remove duplicate rows from dataset.
    
    Plain English: Delete exact copies of rows, keeping only one.
    Like removing duplicate photos from your camera roll.
    
    Args:
        dataset_id: The unique ID of uploaded dataset
        subset: Optional list of columns to check for duplicates.
                If None, checks entire row.
    
    Example:
        POST /clean/remove-duplicates/abc-123
        Body: {"subset": ["email", "phone"]}
        
        This removes rows with duplicate email+phone combinations
    """
    
    dataset = DatasetStorage.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = dataset["dataframe"]
    
    # Create cleaner and remove duplicates
    cleaner = DataCleaner(df)
    result = cleaner.remove_duplicates(subset=subset)
    
    # Update stored dataset with cleaned version
    dataset["dataframe"] = cleaner.get_cleaned_data()
    dataset["rows"] = len(cleaner.get_cleaned_data())
    DatasetStorage.save_dataset(dataset_id, dataset)
    
    return {
        "success": True,
        "dataset_id": dataset_id,
        "cleaning_result": result,
        "new_row_count": len(cleaner.get_cleaned_data())
    }


# ============================================
# 3. FILL MISSING VALUES
# ============================================

@router.post("/clean/fill-missing/{dataset_id}")
async def fill_missing_values(
    dataset_id: str,
    column: str,
    method: str,
    custom_value: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Fill missing values in a column.
    
    Plain English: Fill in the blanks in your data.
    Like completing a form with missing answers.
    
    Args:
        dataset_id: The unique ID
        column: Which column to fill
        method: How to fill - 'mean', 'median', 'mode', 'forward_fill', 
                'backward_fill', or 'custom'
        custom_value: Value to use if method='custom'
    
    Example:
        POST /clean/fill-missing/abc-123
        Body: {
            "column": "age",
            "method": "mean"
        }
        
        This fills missing ages with the average age
    """
    
    dataset = DatasetStorage.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = dataset["dataframe"]
    
    # Validate column exists
    if column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Column not found",
                "explanation": f"The column '{column}' doesn't exist in your dataset.",
                "available_columns": list(df.columns)
            }
        )
    
    # Validate method
    valid_methods = ['mean', 'median', 'mode', 'forward_fill', 'backward_fill', 'custom']
    if method not in valid_methods:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid method",
                "explanation": f"Method must be one of: {', '.join(valid_methods)}",
                "provided": method
            }
        )
    
    # Fill missing values
    cleaner = DataCleaner(df)
    result = cleaner.fill_missing_values(column, method, custom_value)
    
    # Update stored dataset
    dataset["dataframe"] = cleaner.get_cleaned_data()
    DatasetStorage.save_dataset(dataset_id, dataset)
    
    return {
        "success": True,
        "dataset_id": dataset_id,
        "cleaning_result": result
    }


# ============================================
# 4. REMOVE OUTLIERS
# ============================================

@router.post("/clean/remove-outliers/{dataset_id}")
async def remove_outliers(
    dataset_id: str,
    column: str,
    method: str = "iqr",
    threshold: float = 1.5
) -> Dict[str, Any]:
    """
    Remove outlier values from a numeric column.
    
    Plain English: Remove extremely unusual values that don't fit the pattern.
    Like removing a "height: 500 feet" entry from human height data.
    
    Args:
        dataset_id: The unique ID
        column: Which column to clean
        method: 'iqr' (default) or 'zscore'
        threshold: How strict to be (default 1.5 for IQR, 3 for z-score)
    
    Example:
        POST /clean/remove-outliers/abc-123
        Body: {
            "column": "salary",
            "method": "iqr",
            "threshold": 1.5
        }
        
        This removes salary values that are way outside normal range
    """
    
    dataset = DatasetStorage.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = dataset["dataframe"]
    
    # Validate column exists
    if column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{column}' not found"
        )
    
    # Validate column is numeric
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Column must be numeric",
                "explanation": (
                    f"The column '{column}' contains {df[column].dtype} data, not numbers. "
                    f"Outlier removal only works on numeric columns like age, price, weight, etc."
                )
            }
        )
    
    # Remove outliers
    cleaner = DataCleaner(df)
    result = cleaner.remove_outliers(column, method, threshold)
    
    # Update stored dataset
    dataset["dataframe"] = cleaner.get_cleaned_data()
    dataset["rows"] = len(cleaner.get_cleaned_data())
    DatasetStorage.save_dataset(dataset_id, dataset)
    
    return {
        "success": True,
        "dataset_id": dataset_id,
        "cleaning_result": result,
        "new_row_count": len(cleaner.get_cleaned_data())
    }


# ============================================
# 5. STANDARDIZE TEXT
# ============================================

@router.post("/clean/standardize-text/{dataset_id}")
async def standardize_text(
    dataset_id: str,
    column: str,
    operations: List[str]
) -> Dict[str, Any]:
    """
    Clean and standardize text in a column.
    
    Plain English: Make text consistent - lowercase, remove spaces, etc.
    Like auto-correcting and formatting text in a document.
    
    Args:
        dataset_id: The unique ID
        column: Which column to clean
        operations: List of operations - 'lowercase', 'strip', 'remove_special'
    
    Example:
        POST /clean/standardize-text/abc-123
        Body: {
            "column": "country",
            "operations": ["lowercase", "strip"]
        }
        
        This converts "  USA  " to "usa" (lowercase and trim spaces)
    """
    
    dataset = DatasetStorage.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = dataset["dataframe"]
    
    if column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{column}' not found")
    
    # Validate operations
    valid_operations = ['lowercase', 'strip', 'remove_special']
    invalid_ops = [op for op in operations if op not in valid_operations]
    
    if invalid_ops:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid operations",
                "explanation": f"Valid operations are: {', '.join(valid_operations)}",
                "invalid": invalid_ops
            }
        )
    
    # Standardize text
    cleaner = DataCleaner(df)
    result = cleaner.standardize_text(column, operations)
    
    # Update stored dataset
    dataset["dataframe"] = cleaner.get_cleaned_data()
    DatasetStorage.save_dataset(dataset_id, dataset)
    
    return {
        "success": True,
        "dataset_id": dataset_id,
        "cleaning_result": result
    }


# ============================================
# 6. CONVERT DATA TYPE
# ============================================

@router.post("/clean/convert-type/{dataset_id}")
async def convert_data_type(
    dataset_id: str,
    column: str,
    target_type: str
) -> Dict[str, Any]:
    """
    Convert a column to a different data type.
    
    Plain English: Change how data is stored (text to numbers, etc.)
    Like converting a Word file to PDF - same content, different format.
    
    Args:
        dataset_id: The unique ID
        column: Which column to convert
        target_type: 'int', 'float', 'string', 'datetime', 'category'
    
    Example:
        POST /clean/convert-type/abc-123
        Body: {
            "column": "age",
            "target_type": "int"
        }
        
        This converts age from text "25" to number 25
    """
    
    dataset = DatasetStorage.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = dataset["dataframe"]
    
    if column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{column}' not found")
    
    # Validate target type
    valid_types = ['int', 'float', 'string', 'datetime', 'category']
    if target_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid type",
                "explanation": f"Type must be one of: {', '.join(valid_types)}",
                "provided": target_type
            }
        )
    
    # Convert type
    cleaner = DataCleaner(df)
    result = cleaner.convert_data_type(column, target_type)
    
    # Update stored dataset
    dataset["dataframe"] = cleaner.get_cleaned_data()
    DatasetStorage.save_dataset(dataset_id, dataset)
    
    return {
        "success": True,
        "dataset_id": dataset_id,
        "cleaning_result": result
    }


# ============================================
# 7. REMOVE LOW VARIANCE COLUMNS
# ============================================

@router.post("/clean/remove-low-variance/{dataset_id}")
async def remove_low_variance_columns(
    dataset_id: str,
    threshold: float = 0.01
) -> Dict[str, Any]:
    """
    Remove columns where values barely change.
    
    Plain English: Delete columns that are almost always the same.
    Like removing a "Country: USA" column if all rows are USA.
    
    Args:
        dataset_id: The unique ID
        threshold: Minimum variance to keep column (default 0.01)
    
    Example:
        POST /clean/remove-low-variance/abc-123
        Body: {"threshold": 0.01}
        
        This removes columns where values barely vary
    """
    
    dataset = DatasetStorage.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = dataset["dataframe"]
    
    # Remove low variance columns
    cleaner = DataCleaner(df)
    result = cleaner.remove_low_variance_columns(threshold)
    
    # Update stored dataset
    # Update stored dataset
    dataset["dataframe"] = cleaner.get_cleaned_data()
    dataset["columns"] = len(cleaner.get_cleaned_data().columns)
    dataset["column_names"] = list(cleaner.get_cleaned_data().columns)
    DatasetStorage.save_dataset(dataset_id, dataset)
    
    return {
        "success": True,
        "dataset_id": dataset_id,
        "cleaning_result": result,
        "new_column_count": len(cleaner.get_cleaned_data().columns)
    }


# ============================================
# 8. GET CLEANING LOG
# ============================================

@router.get("/clean/log/{dataset_id}")
async def get_cleaning_log(dataset_id: str) -> Dict[str, Any]:
    """
    Get the history of all cleaning operations performed.
    
    Plain English: See a log of everything you've cleaned.
    Like a receipt showing all services performed at a car wash.
    
    Returns:
        List of all cleaning operations with explanations
    """
    
    dataset = DatasetStorage.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = dataset["dataframe"]
    
    # Create cleaner to access log (empty log if no operations yet)
    cleaner = DataCleaner(df)
    
    return {
        "success": True,
        "dataset_id": dataset_id,
        "log": cleaner.get_cleaning_log(),
        "explanation": (
            "This shows all cleaning operations performed on your dataset. "
            "Each entry explains what was done and why."
        )
    }


# ============================================
# 9. GET CLEANING SUMMARY
# ============================================

@router.get("/clean/summary/{dataset_id}")
async def get_cleaning_summary(dataset_id: str) -> Dict[str, Any]:
    """
    Get a summary of all cleaning performed on the dataset.
    
    Plain English: Get a report card of data cleaning.
    Shows how many operations were done and final dataset size.
    """
    
    dataset = DatasetStorage.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = dataset["dataframe"]
    
    cleaner = DataCleaner(df)
    summary = cleaner.export_summary()
    
    return {
        "success": True,
        "dataset_id": dataset_id,
        "summary": summary
    }


# ============================================
# 10. DOWNLOAD CLEANED DATA
# ============================================

@router.get("/clean/download/{dataset_id}")
async def download_cleaned_data(
    dataset_id: str,
    format: str = "csv"
) -> Dict[str, Any]:
    """
    Download the cleaned dataset.
    
    Plain English: Export your cleaned data as a file.
    Like saving a cleaned-up document after editing.
    
    Args:
        dataset_id: The unique ID
        format: 'csv' or 'excel' (default 'csv')
    
    Returns:
        Download link or file data
    """
    
    dataset = DatasetStorage.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = dataset["dataframe"]
    
    # For now, return preview and info
    # In production, this would return actual file download
    
    return {
        "success": True,
        "dataset_id": dataset_id,
        "format": format,
        "rows": len(df),
        "columns": len(df.columns),
        "preview": df.head(10).to_dict('records'),
        "explanation": (
            "In production, this endpoint would trigger a file download. "
            "For now, here's a preview of your cleaned data."
        )
    }
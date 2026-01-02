import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from scipy import stats

class DataCleaner:
    """
    Cleans data based on user decisions.
    
    Plain English: This is your data janitor - it fixes problems
    after you tell it what to do.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()  # Work on a copy, don't modify original
        self.cleaning_log = []  # Track what we do
        
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Remove duplicate rows.
        
        Args:
            subset: Specific columns to check for duplicates. 
                   If None, checks entire row.
        
        Plain English: Remove photocopies, keep only originals.
        """
        
        initial_rows = len(self.df)
        
        # Remove duplicates
        self.df = self.df.drop_duplicates(subset=subset, keep='first')
        
        removed_count = initial_rows - len(self.df)
        
        log_entry = {
            "action": "remove_duplicates",
            "rows_before": initial_rows,
            "rows_after": len(self.df),
            "rows_removed": removed_count,
            "explanation": (
                f"Removed {removed_count} duplicate rows. "
                f"Think of it like removing duplicate photos from an album - "
                f"we kept the first occurrence and deleted the copies."
            )
        }
        
        self.cleaning_log.append(log_entry)
        
        return log_entry
    
    def fill_missing_values(self, column: str, method: str, 
                           custom_value: Any = None) -> Dict[str, Any]:
        """
        Fill missing values in a column.
        
        Args:
            column: Column name to fill
            method: 'mean', 'median', 'mode', 'forward_fill', 'backward_fill', 'custom'
            custom_value: Value to use if method='custom'
        
        Plain English: Fill in blank answers on a form.
        """
        
        missing_before = self.df[column].isnull().sum()
        
        if missing_before == 0:
            return {
                "action": "fill_missing",
                "column": column,
                "filled_count": 0,
                "explanation": f"No missing values found in '{column}' - nothing to fill!"
            }
        
        # Apply filling method
        if method == 'mean':
            fill_value = self.df[column].mean()
            self.df[column].fillna(fill_value, inplace=True)
            explanation = (
                f"Filled {missing_before} missing values with the average ({fill_value:.2f}). "
                f"Like asking 'What's a typical value?' and using that for blanks."
            )
            
        elif method == 'median':
            fill_value = self.df[column].median()
            self.df[column].fillna(fill_value, inplace=True)
            explanation = (
                f"Filled {missing_before} missing values with the median ({fill_value:.2f}). "
                f"The median is the 'middle value' when sorted - less affected by extremes than average."
            )
            
        elif method == 'mode':
            fill_value = self.df[column].mode()[0]
            self.df[column].fillna(fill_value, inplace=True)
            explanation = (
                f"Filled {missing_before} missing values with the most common value ('{fill_value}'). "
                f"Like a survey where you use the most popular answer for blank responses."
            )
            
        elif method == 'forward_fill':
            self.df[column] = self.df[column].ffill()
            explanation = (
            f"Filled {missing_before} missing values by copying the previous value. "
            f"Think of it like 'ditto marks' - when something is blank, assume it's the same as above."
        )

        elif method == 'backward_fill':
            self.df[column] = self.df[column].bfill()
            explanation = (
            f"Filled {missing_before} missing values by copying the next value. "
            f"Like looking ahead to see what comes next and using that."
        )

            
        elif method == 'custom' and custom_value is not None:
            self.df[column].fillna(custom_value, inplace=True)
            explanation = (
                f"Filled {missing_before} missing values with your custom value ('{custom_value}'). "
                f"You decided what to put in the blanks!"
            )
            
        else:
            raise ValueError(f"Invalid method: {method}")
        
        missing_after = self.df[column].isnull().sum()
        
        log_entry = {
            "action": "fill_missing",
            "column": column,
            "method": method,
            "filled_count": missing_before - missing_after,
            "missing_before": int(missing_before),
            "missing_after": int(missing_after),
            "explanation": explanation
        }
        
        self.cleaning_log.append(log_entry)
        
        return log_entry
    
    def remove_outliers(self, column: str, method: str = 'iqr', 
                       threshold: float = 1.5) -> Dict[str, Any]:
        """
        Remove outlier values from a numeric column.
        
        Args:
            column: Column name
            method: 'iqr' (Interquartile Range) or 'zscore' (Standard Deviation)
            threshold: How strict to be (default 1.5 for IQR, 3 for z-score)
        
        Plain English: Remove extreme values that don't fit the pattern.
        """
        
        initial_rows = len(self.df)
        
        if method == 'iqr':
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Keep only values within bounds
            self.df = self.df[
                (self.df[column] >= lower_bound) & 
                (self.df[column] <= upper_bound)
            ]
            
            explanation = (
                f"Removed {initial_rows - len(self.df)} rows with outliers in '{column}'. "
                f"Values outside the range {lower_bound:.2f} to {upper_bound:.2f} were removed. "
                f"Think of it like removing test scores that are impossibly high or low."
            )
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(self.df[column].dropna()))
            self.df = self.df[z_scores < threshold]
            
            explanation = (
                f"Removed {initial_rows - len(self.df)} rows with outliers in '{column}'. "
                f"Removed values more than {threshold} standard deviations from the mean. "
                f"Like removing extremely unusual measurements."
            )
        
        log_entry = {
            "action": "remove_outliers",
            "column": column,
            "method": method,
            "rows_before": initial_rows,
            "rows_after": len(self.df),
            "rows_removed": initial_rows - len(self.df),
            "explanation": explanation
        }
        
        self.cleaning_log.append(log_entry)
        
        return log_entry
    
    def standardize_text(self, column: str, 
                        operations: List[str]) -> Dict[str, Any]:
        """
        Clean text in a column.
        
        Args:
            column: Column name
            operations: List of operations - 'lowercase', 'strip', 'remove_special'
        
        Plain English: Make text consistent and clean.
        """
        
        changes_made = []
        
        if 'lowercase' in operations:
            self.df[column] = self.df[column].str.lower()
            changes_made.append("converted to lowercase")
        
        if 'strip' in operations:
            self.df[column] = self.df[column].str.strip()
            changes_made.append("removed extra spaces")
        
        if 'remove_special' in operations:
            self.df[column] = self.df[column].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
            changes_made.append("removed special characters")
        
        log_entry = {
            "action": "standardize_text",
            "column": column,
            "operations": operations,
            "explanation": (
                f"Cleaned the '{column}' column by: {', '.join(changes_made)}. "
                f"Think of it like spell-checking and formatting a document."
            )
        }
        
        self.cleaning_log.append(log_entry)
        
        return log_entry
    
    def convert_data_type(self, column: str, 
                         target_type: str) -> Dict[str, Any]:
        """
        Convert column to different data type.
        
        Args:
            column: Column name
            target_type: 'int', 'float', 'string', 'datetime', 'category'
        
        Plain English: Change how data is stored (like converting text to numbers).
        """
        
        original_type = str(self.df[column].dtype)
        
        try:
            if target_type == 'int':
                self.df[column] = pd.to_numeric(self.df[column], errors='coerce').astype('Int64')
            elif target_type == 'float':
                self.df[column] = pd.to_numeric(self.df[column], errors='coerce')
            elif target_type == 'string':
                self.df[column] = self.df[column].astype(str)
            elif target_type == 'datetime':
                self.df[column] = pd.to_datetime(self.df[column], errors='coerce')
            elif target_type == 'category':
                self.df[column] = self.df[column].astype('category')
            else:
                raise ValueError(f"Unknown type: {target_type}")
            
            log_entry = {
                "action": "convert_type",
                "column": column,
                "from_type": original_type,
                "to_type": target_type,
                "explanation": (
                    f"Converted '{column}' from {original_type} to {target_type}. "
                    f"Like changing a Word document to PDF - same content, different format."
                )
            }
            
            self.cleaning_log.append(log_entry)
            return log_entry
            
        except Exception as e:
            return {
                "action": "convert_type",
                "column": column,
                "success": False,
                "error": str(e),
                "explanation": (
                    f"Couldn't convert '{column}' to {target_type}. "
                    f"Some values might not be compatible. "
                    f"It's like trying to fit a square peg in a round hole."
                )
            }
    
    def remove_low_variance_columns(self, threshold: float = 0.01) -> Dict[str, Any]:
        """
        Remove columns where values barely change.
        
        Args:
            threshold: Minimum variance to keep column
        
        Plain English: Remove columns that are almost always the same value.
        """
        
        removed_columns = []
        
        for col in self.df.select_dtypes(include=[np.number]).columns:
            if self.df[col].var() < threshold:
                removed_columns.append(col)
                self.df = self.df.drop(columns=[col])
        
        log_entry = {
            "action": "remove_low_variance",
            "removed_columns": removed_columns,
            "explanation": (
                f"Removed {len(removed_columns)} columns with low variance: {', '.join(removed_columns)}. "
                f"These columns barely changed across rows, like asking 'Is water wet?' every time - "
                f"the answer is always yes, so it doesn't help analysis."
            ) if removed_columns else "No low variance columns found."
        }
        
        self.cleaning_log.append(log_entry)
        
        return log_entry
    
    def get_cleaned_data(self) -> pd.DataFrame:
        """Return the cleaned dataframe"""
        return self.df
    
    def get_cleaning_log(self) -> List[Dict[str, Any]]:
        """Return log of all cleaning operations"""
        return self.cleaning_log
    
    def export_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all cleaning performed.
        
        Plain English: A report card of what was cleaned.
        """
        
        return {
            "total_operations": len(self.cleaning_log),
            "operations": self.cleaning_log,
            "final_shape": {
                "rows": len(self.df),
                "columns": len(self.df.columns)
            },
            "summary": (
                f"Performed {len(self.cleaning_log)} cleaning operations. "
                f"Your dataset now has {len(self.df)} rows and {len(self.df.columns)} columns. "
                f"Think of it like cleaning a messy room - everything is now organized!"
            )
        }
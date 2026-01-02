import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from typing import Dict, List, Any
from datetime import datetime

class DataValidator:
    """
    Validates uploaded datasets and checks for common issues.
    Think of this as a 'health inspector' for your data.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.issues = []
        self.warnings = []
        self.info = []
        
    def validate_all(self) -> Dict[str, Any]:
        """
        Run all validation checks.
        Returns a comprehensive report of issues found.
        """
        report = {
            "basic_info": self._get_basic_info(),
            "missing_values": self._check_missing_values(),
            "duplicates": self._check_duplicates(),
            "data_types": self._check_data_types(),
            "outliers": self._check_outliers(),
            "formatting": self.check_formatting(),
            "date_issues": self._check_dates(),
            "cardinality": self._check_cardinality(),
            "variance": self._check_variance(),
            "summary": {
                "total_issues": len(self.issues),
                "total_warnings": len(self.warnings),
                "issues": self.issues,
                "warnings": self.warnings,
                "info": self.info
            }
        }
        
        return report
    
    def _get_basic_info(self) -> Dict[str, Any]:
        """Get basic dataset information"""
        return {
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "column_names": list(self.df.columns),
            "memory_usage": f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
            "dtypes": self.df.dtypes.astype(str).to_dict()
        }
    
    def _check_missing_values(self) -> Dict[str, Any]:
        """
        Check for missing values (NaN, None, empty strings).
        
        Plain English: Like checking if survey forms are incomplete.
        """
        missing = self.df.isnull().sum()
        missing_percent = (missing / len(self.df)) * 100
        
        problematic_cols = []
        
        for col in self.df.columns:
            missing_count = missing[col]
            missing_pct = missing_percent[col]
            
            if missing_count > 0:
                explanation = self._explain_missing_values(
                    col, missing_count, missing_pct
                )
                
                issue = {
                    "column": col,
                    "missing_count": int(missing_count),
                    "missing_percentage": round(missing_pct, 2),
                    "explanation": explanation,
                    "severity": self._assess_missing_severity(missing_pct),
                    "recommendations": self._recommend_missing_fix(col, missing_pct)
                }
                
                problematic_cols.append(issue)
                
                if missing_pct > 50:
                    self.issues.append(
                        f"Column '{col}' is missing {missing_pct:.1f}% of values - consider removing it"
                    )
                elif missing_pct > 5:
                    self.warnings.append(
                        f"Column '{col}' has {missing_pct:.1f}% missing values"
                    )
        
        return {
            "has_missing": len(problematic_cols) > 0,
            "affected_columns": problematic_cols,
            "total_missing_cells": int(missing.sum())
        }
    
    def _explain_missing_values(self, col: str, count: int, percent: float) -> str:
        """Generate plain English explanation for missing values"""
        if percent < 5:
            return (
                f"The '{col}' column has {count} empty cells ({percent:.1f}% of all rows). "
                f"This is a small amount and easily fixable. Think of it like a few blank "
                f"answers on a survey - we can fill them in without affecting the overall results much."
            )
        elif percent < 20:
            return (
                f"The '{col}' column is missing {count} values ({percent:.1f}% of data). "
                f"This is noticeable but manageable. It's like having incomplete medical records "
                f"for 1 in 5 patients - we need to decide how to handle these gaps carefully."
            )
        elif percent < 50:
            return (
                f"The '{col}' column has {count} missing values ({percent:.1f}%). "
                f"This is significant - almost half the data is missing! It's like trying to "
                f"watch a movie where every other scene is cut out. We might need to remove this "
                f"column or collect more data."
            )
        else:
            return (
                f"The '{col}' column is missing {count} values ({percent:.1f}% - more than half!). "
                f"This column is mostly empty. It's like buying a book where most pages are blank. "
                f"I strongly recommend removing this column as it won't help with predictions."
            )
    
    def _assess_missing_severity(self, percent: float) -> str:
        """Categorize severity of missing values"""
        if percent < 5:
            return "low"
        elif percent < 20:
            return "medium"
        elif percent < 50:
            return "high"
        else:
            return "critical"
    
    def _recommend_missing_fix(self, col: str, percent: float) -> List[str]:
        """Provide recommendations for handling missing values"""
        recommendations = []
        
        if percent < 5:
            recommendations.append("Fill with mean/median (for numbers)")
            recommendations.append("Fill with mode (most common value)")
            recommendations.append("Use 'forward fill' (copy previous value)")
        elif percent < 20:
            recommendations.append("Analyze if missing values have a pattern")
            recommendations.append("Consider predictive imputation (ML-based filling)")
            recommendations.append("Create a 'missing' category if meaningful")
        elif percent < 50:
            recommendations.append("Consider removing this column")
            recommendations.append("Check if missing data is 'informative' (means something)")
            recommendations.append("Collect more data if possible")
        else:
            recommendations.append("Remove this column - too much missing data")
            recommendations.append("Or collect new data for this field")
        
        return recommendations
    
    def _check_duplicates(self) -> Dict[str, Any]:
        """
        Check for duplicate rows and column-level duplicates.
        
        Plain English: Finding if the same information appears twice.
        """
        # Full row duplicates
        duplicated_rows = self.df.duplicated().sum()
        duplicate_indices = self.df[self.df.duplicated()].index.tolist()
        
        # Column-level duplicates
        column_duplicates = []
        
        for col in self.df.columns:
            unique_count = self.df[col].nunique()
            total_count = len(self.df)
            duplicate_count = total_count - unique_count
            
            if duplicate_count > 0:
                dup_percentage = (duplicate_count / total_count) * 100
                
                explanation = self._explain_column_duplicates(
                    col, unique_count, total_count, dup_percentage
                )
                
                column_duplicates.append({
                    "column": col,
                    "unique_values": int(unique_count),
                    "total_values": int(total_count),
                    "duplicate_count": int(duplicate_count),
                    "duplicate_percentage": round(dup_percentage, 2),
                    "explanation": explanation,
                    "is_problematic": self._is_duplicate_problematic(col, dup_percentage)
                })
        
        if duplicated_rows > 0:
            self.issues.append(
                f"Found {duplicated_rows} duplicate rows - these should be removed"
            )
        
        return {
            "has_duplicates": duplicated_rows > 0,
            "duplicate_row_count": int(duplicated_rows),
            "duplicate_indices": duplicate_indices[:100],  # First 100 for display
            "column_level_duplicates": column_duplicates,
            "explanation": self._explain_row_duplicates(duplicated_rows)
        }
    
    def _explain_column_duplicates(self, col: str, unique: int, 
                                   total: int, dup_pct: float) -> str:
        """Explain what column duplicates mean in context"""
        
        # Check if column name suggests it should be unique
        unique_identifiers = ['id', 'email', 'phone', 'ssn', 'passport', 
                             'username', 'account']
        
        is_identifier = any(identifier in col.lower() for identifier in unique_identifiers)
        
        if is_identifier and dup_pct > 0:
            return (
                f"⚠️ The '{col}' column appears to be an identifier (like a passport number), "
                f"but it has {total - unique} duplicate values. Each {col} should be unique! "
                f"This could mean: (1) Data entry errors, (2) Multiple records for same entity, "
                f"or (3) The column name is misleading. You should investigate these duplicates."
            )
        elif dup_pct < 10:
            return (
                f"The '{col}' column has some repeated values ({dup_pct:.1f}% duplicates). "
                f"This is normal for categorical data like 'Gender' or 'City'. "
                f"Example: If you survey 100 people, many will say 'Male' or 'Female' repeatedly."
            )
        elif dup_pct < 50:
            return (
                f"The '{col}' column has moderate repetition ({dup_pct:.1f}% duplicates). "
                f"Out of {total} values, only {unique} are unique. This could be fine "
                f"(like repeated product categories) or problematic (like repeated customer IDs). "
                f"Context matters!"
            )
        else:
            return (
                f"The '{col}' column has HIGH repetition ({dup_pct:.1f}% duplicates). "
                f"Only {unique} unique values out of {total} total! This means many rows "
                f"have the same value. If this column should vary (like 'Transaction Amount'), "
                f"this is suspicious. If it's meant to be repeated (like 'Product Type'), it's okay."
            )
    
    def _is_duplicate_problematic(self, col: str, dup_pct: float) -> bool:
        """Determine if duplicates in a column are problematic"""
        unique_identifiers = ['id', 'email', 'phone', 'ssn', 'passport', 
                             'username', 'account']
        
        is_identifier = any(identifier in col.lower() for identifier in unique_identifiers)
        
        return is_identifier and dup_pct > 0
    
    def _explain_row_duplicates(self, count: int) -> str:
        """Explain row duplicates in plain English"""
        if count == 0:
            return "✅ Great news! No duplicate rows found. Each row in your dataset is unique."
        elif count < 10:
            return (
                f"Found {count} duplicate rows. These are exact copies of other rows. "
                f"Think of it like accidentally photocopying the same page twice - "
                f"we only need one copy. These should be removed to avoid double-counting."
            )
        else:
            return (
                f"Found {count} duplicate rows! These are complete copies that appear multiple times. "
                f"Imagine counting the same person twice in a census - it inflates your numbers "
                f"and skews results. We should definitely remove these duplicates before analysis."
            )
    
    def _check_data_types(self) -> Dict[str, Any]:
        """Check if columns have correct data types"""
        issues = []
        
        for col in self.df.columns:
            dtype = self.df[col].dtype
            sample_values = self.df[col].dropna().head(5).tolist()
            
            # Check if numeric column stored as object
            if dtype == 'object':
                # Try converting to numeric
                try:
                    pd.to_numeric(self.df[col], errors='coerce')
                    non_numeric = self.df[col][
                        pd.to_numeric(self.df[col], errors='coerce').isna() & 
                        self.df[col].notna()
                    ]
                    
                    if len(non_numeric) > 0:
                        issues.append({
                            "column": col,
                            "current_type": str(dtype),
                            "suggested_type": "numeric",
                            "problematic_values": non_numeric.head(5).tolist(),
                            "explanation": (
                                f"The '{col}' column should contain numbers, but has text values "
                                f"like {non_numeric.iloc[0]}. It's like trying to add 'apple' + 5 - "
                                f"doesn't work! You need to clean these values first."
                            )
                        })
                        
                        self.warnings.append(
                            f"Column '{col}' has mixed numeric and text values"
                        )
                except:
                    pass
        
        return {
            "has_type_issues": len(issues) > 0,
            "problematic_columns": issues
        }
    
    def _check_outliers(self) -> Dict[str, Any]:
        """
        Detect outliers using IQR method.
        
        Plain English: Finding values that are extremely different from others.
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outlier_report = []
        
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[
                (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            ][col]
            
            if len(outliers) > 0:
                outlier_report.append({
                    "column": col,
                    "outlier_count": len(outliers),
                    "outlier_percentage": round((len(outliers) / len(self.df)) * 100, 2),
                    "outlier_values": outliers.head(10).tolist(),
                    "normal_range": f"{lower_bound:.2f} to {upper_bound:.2f}",
                    "explanation": self._explain_outliers(col, outliers, lower_bound, upper_bound)
                })
                
                if len(outliers) / len(self.df) > 0.05:
                    self.warnings.append(
                        f"Column '{col}' has {len(outliers)} outlier values"
                    )
        
        return {
            "has_outliers": len(outlier_report) > 0,
            "affected_columns": outlier_report
        }
    
    def _explain_outliers(self, col: str, outliers, lower: float, upper: float) -> str:
        """Explain outliers in plain English"""
        min_outlier = outliers.min()
        max_outlier = outliers.max()
        
        return (
            f"The '{col}' column has {len(outliers)} unusual values. "
            f"Most values fall between {lower:.2f} and {upper:.2f}, "
            f"but some are as low as {min_outlier:.2f} or as high as {max_outlier:.2f}. "
            f"\n\nThink of it like height measurements: if most people are 5-6 feet tall, "
            f"but someone records 15 feet, that's clearly wrong (unless it's a giraffe!). "
            f"\n\nOutliers can be: "
            f"(1) Data entry errors, (2) Genuine extreme cases, or (3) Different units mixed up. "
            f"You should review these values before deciding what to do."
        )
    
    def check_formatting(self) -> Dict[str, Any]:
        """Check for formatting inconsistencies"""
        issues = []
        
        for col in self.df.select_dtypes(include=['object']).columns:
            # Check for leading/trailing spaces
            has_spaces = self.df[col].astype(str).str.strip() != self.df[col].astype(str)
            space_count = has_spaces.sum()
            
            if space_count > 0:
                issues.append({
                    "column": col,
                    "issue_type": "whitespace",
                    "affected_rows": int(space_count),
                    "explanation": (
                        f"The '{col}' column has {space_count} values with extra spaces. "
                        f"For example, 'Apple ' and 'Apple' look the same to us but computers "
                        f"see them as different. It's like having two files named 'Report.pdf' "
                        f"and 'Report .pdf' - confusing!"
                    )
                })
            
            # Check for case inconsistencies
            if self.df[col].nunique() > 1:
                lower_unique = self.df[col].astype(str).str.lower().nunique()
                if lower_unique < self.df[col].nunique():
                    issues.append({
                        "column": col,
                        "issue_type": "case_inconsistency",
                        "explanation": (
                            f"The '{col}' column has values that differ only in capitalization. "
                            f"For example: 'USA', 'usa', 'Usa'. Humans know these mean the same thing, "
                            f"but computers treat them as completely different values!"
                        )
                    })
        
        return {
            "has_formatting_issues": len(issues) > 0,
            "issues": issues
        }
    
    def _check_dates(self) -> Dict[str, Any]:
        """Check for date/datetime columns and validate them"""
        date_issues = []
        
        # Find potential date columns
        for col in self.df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    converted = pd.to_datetime(self.df[col], errors='coerce')
                    invalid_dates = converted.isna() & self.df[col].notna()
                    invalid_count = invalid_dates.sum()
                    
                    if invalid_count > 0:
                        date_issues.append({
                            "column": col,
                            "invalid_count": int(invalid_count),
                            "sample_invalid": self.df[col][invalid_dates].head(5).tolist(),
                            "explanation": (
                                f"The '{col}' column should contain dates, but {invalid_count} "
                                f"values can't be recognized as dates. Examples: {self.df[col][invalid_dates].iloc[0]}. "
                                f"It's like writing 'February 30th' - that date doesn't exist!"
                            )
                        })
                except:
                    pass
        
        return {
            "has_date_issues": len(date_issues) > 0,
            "issues": date_issues
        }
    
    def _check_cardinality(self) -> Dict[str, Any]:
        """Check for high cardinality (too many unique values)"""
        issues = []
        
        for col in self.df.columns:
            unique_count = self.df[col].nunique()
            unique_ratio = unique_count / len(self.df)
            
            # High cardinality if more than 50% unique and not numeric
            if unique_ratio > 0.5 and self.df[col].dtype == 'object':
                issues.append({
                    "column": col,
                    "unique_count": int(unique_count),
                    "total_rows": len(self.df),
                    "uniqueness_ratio": round(unique_ratio * 100, 2),
                    "explanation": (
                        f"The '{col}' column has {unique_count} unique values out of {len(self.df)} rows "
                        f"({unique_ratio*100:.1f}% unique). This is very high! "
                        f"\n\nImagine trying to find patterns in a library where every book has "
                        f"its own unique category - too specific! "
                        f"\n\nHigh cardinality can: "
                        f"(1) Slow down analysis, (2) Make ML models confused, (3) Hide useful patterns. "
                        f"\n\nConsider grouping similar values or creating broader categories."
                    )
                })
                
                self.warnings.append(
                    f"Column '{col}' has very high cardinality ({unique_count} unique values)"
                )
        
        return {
            "has_high_cardinality": len(issues) > 0,
            "affected_columns": issues
        }
    
    def _check_variance(self) -> Dict[str, Any]:
        """Check for low variance columns (mostly same value)"""
        
        low_variance_cols = []
        
        for col in self.df.columns:
            column_data = self.df[col]
            
            # ===============================
            # CATEGORICAL / OBJECT COLUMNS
            # ===============================
            if column_data.dtype in ["object", "category"]:
                value_counts = column_data.value_counts(dropna=True)
                
                if not value_counts.empty:
                    most_common_pct = (value_counts.iloc[0] / len(self.df)) * 100
                    
                    if most_common_pct > 95:
                        low_variance_cols.append({
                            "column": col,
                            "most_common_value": str(value_counts.index[0]),
                            "most_common_percentage": round(most_common_pct, 2),
                            "explanation": (
                                f"The '{col}' column has the same value "
                                f"('{value_counts.index[0]}') in {most_common_pct:.1f}% of rows. "
                                f"This column barely changes and adds little analytical value. "
                                f"Recommendation: consider removing it."
                            )
                        })
                        
                        self.info.append(
                            f"Column '{col}' has very low variance (categorical)"
                        )
            
            # ===============================
            # NUMERIC COLUMNS
            # ===============================
            elif is_numeric_dtype(column_data):
                try:
                    # Calculate variance
                    raw_variance = column_data.var()
                    
                    # Skip if variance is None or NaN
                    if pd.isna(raw_variance):
                        continue
                    
                    # Safely convert to float
                    variance = float(pd.to_numeric(raw_variance))
                    
                    if variance < 0.01:
                        low_variance_cols.append({
                            "column": col,
                            "variance": round(variance, 6),
                            "explanation": (
                                f"The '{col}' column has very low variance ({variance:.4f}). "
                                f"Most values are nearly identical, so this column may not be useful."
                            )
                        })
                        
                        self.info.append(
                            f"Column '{col}' has very low variance (numeric)"
                        )
                
                except (TypeError, ValueError):
                    # Skip columns that can't be processed
                    continue
        
        return {
            "has_low_variance": len(low_variance_cols) > 0,
            "affected_columns": low_variance_cols
        }
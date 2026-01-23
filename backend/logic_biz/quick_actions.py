from storage.datasets import DatasetStorage
import pandas as pd
from typing import Dict, Any

class QuickActions:
    @staticmethod
    def auto_clean(dataset_id: str) -> Dict[str, Any]:
        """
        Automatically cleans a dataset by:
        1. Dropping duplicates
        2. Dropping rows with missing values
        """
        dataset = DatasetStorage.get_dataset(dataset_id)
        if not dataset:
            raise ValueError("Dataset not found")

        df = dataset["dataframe"]
        original_rows = len(df)
        
        # 1. Drop Duplicates
        df_cleaned = df.drop_duplicates()
        duplicates_removed = original_rows - len(df_cleaned)
        
        # 2. Drop Missing
        rows_before_na = len(df_cleaned)
        df_cleaned = df_cleaned.dropna()
        missing_removed = rows_before_na - len(df_cleaned)
        
        # Update Storage
        dataset["dataframe"] = df_cleaned
        dataset["rows"] = len(df_cleaned)
        DatasetStorage.save_dataset(dataset_id, dataset)

        # Log to History
        from storage.history import HistoryStorage
        HistoryStorage.log_action(dataset_id, "Auto-Clean", {
            "duplicates_removed": duplicates_removed,
            "missing_rows_removed": missing_removed,
            "original_rows": original_rows,
            "final_rows": len(df_cleaned)
        })
        
        return {
            "action": "auto_clean",
            "original_rows": original_rows,
            "final_rows": len(df_cleaned),
            "duplicates_removed": duplicates_removed,
            "missing_rows_removed": missing_removed,
            "message": f"Cleaned data! Removed {duplicates_removed} duplicates and {missing_removed} rows with missing values."
        }

    @staticmethod
    def generate_summary(dataset_id: str) -> Dict[str, Any]:
        """
        Generates quick statistical summary
        """
        dataset = DatasetStorage.get_dataset(dataset_id)
        if not dataset:
            raise ValueError("Dataset not found")

        df = dataset["dataframe"]
        
        # Select numeric columns for stats
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty:
            return {"message": "No numeric columns to summarize."}
            
        description = numeric_df.describe().to_dict()
        return description

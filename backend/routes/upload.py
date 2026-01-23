from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict, Any
import pandas as pd
import io
import uuid
from datetime import datetime

router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    allowed_types = [
        "text/csv",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/json",
        "application/octet-stream"
    ]
    
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Invalid file type")

    try:
        contents = await file.read()
        filename = (file.filename or "").lower()

        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith(".tsv"):
            df = pd.read_csv(io.BytesIO(contents), sep="\t")
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))
        elif filename.endswith(".json"):
            df = pd.read_json(io.BytesIO(contents))
        elif filename.endswith(".parquet"):
            df = pd.read_parquet(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        if df.empty:
            raise HTTPException(status_code=400, detail="Empty file")

        dataset_id = str(uuid.uuid4())

        from storage.datasets import DatasetStorage
        
        dataset_data = {
            "dataframe": df,
            "filename": file.filename,
            "uploaded_at": datetime.now().isoformat(),
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "size_mb": len(contents) / (1024 * 1024),
            "dataset_id": dataset_id
        }
        
        DatasetStorage.save_dataset(dataset_id, dataset_data)
        
        # Log to History
        from storage.history import HistoryStorage
        HistoryStorage.log_action(dataset_id, "Dataset Uploaded", {
            "filename": file.filename,
            "size_mb": float(f"{len(contents) / (1024 * 1024):.2f}")
        })

        return {
            "success": True,
            "message": "File uploaded successfully!",
            "dataset_id": dataset_id,
            "info": {
                "filename": file.filename,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
                "size": f"{len(contents) / (1024 * 1024):.2f} MB",
                "preview": df.head(5).to_dict('records')
            },
            "explanation": f"Received '{file.filename}' with {len(df)} rows and {len(df.columns)} columns."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
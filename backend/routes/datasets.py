from fastapi import APIRouter
from storage.datasets import DatasetStorage
from typing import List, Dict, Any

router = APIRouter(prefix="/datasets", tags=["Datasets"])

@router.get("/")
async def list_datasets() -> List[Dict[str, Any]]:
    # Return list of datasets (metadata only, no dataframe to save bandwidth)
    datasets = DatasetStorage.list_datasets()
    # Filter out dataframe object for JSON response
    return [
        {k: v for k, v in d.items() if k != 'dataframe'}
        for d in datasets
    ]

@router.get("/{dataset_id}")
async def get_dataset(dataset_id: str):
    d = DatasetStorage.get_dataset(dataset_id)
    if d:
        return {k: v for k, v in d.items() if k != 'dataframe'}
    return None

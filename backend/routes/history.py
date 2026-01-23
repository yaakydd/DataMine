from fastapi import APIRouter
from storage.history import HistoryStorage

router = APIRouter(prefix="/history", tags=["History"])

@router.get("/{dataset_id}")
async def get_history(dataset_id: str):
    return HistoryStorage.get_history(dataset_id)

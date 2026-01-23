from fastapi import APIRouter, HTTPException
from logic_biz.quick_actions import QuickActions

router = APIRouter(prefix="/quick-actions", tags=["Quick Actions"])

@router.post("/auto-clean/{dataset_id}")
async def auto_clean(dataset_id: str):
    try:
        return QuickActions.auto_clean(dataset_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/summary/{dataset_id}")
async def get_summary(dataset_id: str):
    try:
        return QuickActions.generate_summary(dataset_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

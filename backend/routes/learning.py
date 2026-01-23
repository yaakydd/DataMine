from fastapi import APIRouter
from logic_biz.learning_content import LearningContentManager

router = APIRouter(prefix="/learning", tags=["Learning"])

@router.get("/tutorial/{concept}")
async def get_tutorial(concept: str):
    return LearningContentManager.get_tutorial(concept)

@router.get("/recommendations/{dataset_id}")
async def get_recommendations(dataset_id: str):
    return LearningContentManager.recommend_topics(dataset_id)

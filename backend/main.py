from fastapi import FastAPI
from routes.upload import router as upload_router
from routes.analysis import router as analysis_router
from routes.learning import router as learning_router
from routes.workspace import router as workspace_router
from routes.history import router as history_router
from routes.quick_actions import router as quick_actions_router
from routes.datasets import router as datasets_router

app = FastAPI(title="FairData")

app.include_router(upload_router, prefix="/api")
app.include_router(analysis_router, prefix="/api")
app.include_router(learning_router, prefix="/api")
app.include_router(workspace_router, prefix="/api")
app.include_router(history_router, prefix="/api")
app.include_router(quick_actions_router, prefix="/api")
app.include_router(datasets_router, prefix="/api")

@app.get("/")
def home():
    return {"message": "FairData backend running"}


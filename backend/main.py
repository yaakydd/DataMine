from fastapi import FastAPI
from routes.upload import router as upload_router
from routes.analysis import router as analysis_router

app = FastAPI(title="FairData")

app.include_router(upload_router, prefix="/api")
app.include_router(analysis_router, prefix="/api")

@app.get("/")
def home():
    return {"message": "FairData backend running"}


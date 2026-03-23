from fastapi import FastAPI
from routes.data_Info import data_info
from routes.data_Cleaning import pipeline

app = FastAPI(title="DataMine API")

app.include_router(data_info, prefix="/api", tags=["dataset_info"])
app.include_router(pipeline, prefix="/cleaningAPI", tags=["data_pipeline"])

@app.get("/")
async def root():
    return{"message": "DataMine API is running"}
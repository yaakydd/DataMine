from fastapi import FastAPI
from backend.routes.data_Info import data_info

app = FastAPI()

app.include_router(data_info, prefix="/api", tags=["dataset_info"])

@app.get("/")
async def root():
    return{"message": "DataMine API is running"}
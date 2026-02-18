from fastapi import FastAPI, HTTPException, APIRouter
import pandas as pd
from routes.dfState import dataset_state


clean = APIRouter()

@clean.post("/dataset_cleaning")
async def cleaning():
    return







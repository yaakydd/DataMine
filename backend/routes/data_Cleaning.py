from fastapi import FastAPI, HTTPException, APIRouter
import pandas as pd
from .dfState import dataset_state


clean = APIRouter()


@clean.post("/dataset_cleaning")
#Function to handle wrong datatypes
async def wrong_datatypes_cleaning():
    if dataset_state.df is None:
        raise HTTPException(status_code=400, detail="Dataset isn't loaded and stored")

    try:   
        clean_df = dataset_state.df
        #Identifying the column datatypes
        column_datatypes = clean_df.dtypes
        return {
            'Column_Datatpes' : column_datatypes
        }
    except Exception as e:
        raise HTTPException(status_code=400)
        







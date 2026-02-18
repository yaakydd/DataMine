from fastapi import FastAPI, HTTPException, APIRouter
from .dfState import dataset_state
import pandas as pd
import re


clean = APIRouter()


@clean.post("/dataset_cleaning")
#Function to handle wrong datatypes
async def wrong_datatypes_cleaning():
    if dataset_state.df is None:
        raise HTTPException(status_code=400, detail="Dataset isn't loaded and stored")

    try:   
        clean_df = dataset_state.df
      # Convert the dtypes to strings, then to a dictionary
      # This is how it works: Goes through the list one-by-one(.apply(...)), A placeholder that says: "For each item in the list, call it 'x' for a second." lambda x: and The action: "Look at item 'x' and grab only its 'name' property." (x.name) only refers to the values{int64...} not the labels{column_name}.
        column_datatypes = clean_df.dtypes.apply(lambda x: x.name).to_dict()
        return {
            'Column_Datatpes' : column_datatypes
        }
    except Exception as e:
        raise HTTPException(status_code=400)
        







from fastapi import File, UploadFile, APIRouter
import pandas as pd

"""
APIRouter acts as the blueprint used to group the related endpoints in this file.
It used to organize the Dataset information endpoints 
into modular and reusable components. By using APIRouter, we can easily manage and maintain the endpoints related to dataset information,
and also allows us to include this router in the main FastAPI application without cluttering the main app with all the endpoint definitions.
"""

data_info = APIRouter()

"""
The path or route created will be /api/dataset_info. The endpoint is /dataset_info 
and requests the user to upload a file(csv file), reads and stores it
in a variable called df and returns some basic information 
about the dataset such as the file name
"""


@data_info.post("/dataset_info")
def upload_file(file : UploadFile):
    df = pd.read_csv(file.file)
    return{
        "filename" : file.filename,
        "first_five_rows" : df.head().to_dict(),
        "shape" : df.shape,
        "columns" : df.columns.tolist(),
        "numerical_columns_statistics": df.describe().to_dict(),
        "column_datatypes": df.dtypes.astype(str).to_dict(),
    }


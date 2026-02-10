from fastapi import File, UploadFile, APIRouter, HTTPException
import pandas as pd
import pandas as pd
import io
import pyarrow.parquet as pq

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

# Validate file extension and content type
@data_info.post("/dataset_info")
async def upload_file(file: UploadFile = File(...)):
    # Checking for file name
    if not file.filename:
        raise HTTPException(status_code=400, detail="File has no name")
    # Converting filenames into lowercase for consistency
    file.filename = file.filename.lower()
    file_content = await file.read() #Awaits for the file to be read before storesd in file_content
    file_memory_buffer = io.BytesIO(file_content) #Reads the stores the memory as a buffer(temporarily)
    
    try:
        # Identifying the various file formats and loading it into a Dataframe and stored in the variable called df
        if filename.endswith(".csv") or file.content_type == "text/csv":
            df = pd.read_csv(file_memory_buffer, sep=',')
        elif filename.endswith(".tsv") or file.content_type == "text/tab-separated-values":
            df = pd.read_csv(file_memory_buffer, sep='\t')
        elif filename.endswith(".json") or file.content_type == "application/json": 
            df = pd.read_json(file_memory_buffer)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_memory_buffer, engine='openpyxl')
        elif filename.endswith(".parquet") or "parquet" in file.content_type:
            # Explicitly using pyarrow to avoid engine errors
            df = pd.read_parquet(file_memory_buffer, engine='pyarrow')
       elif filename.endswith(".orc") or file.content_type == "application/vnd.apache.orc":
           # Using pyarrow as the default engine for ORC files
       df = pd.read_orc(file_memory_buffer)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

        # Processing the metadata to be stored
        buffer = io.StringIO()
        df.info(buf=buffer)
        data_info = buffer.getvalue()

        # To know how much RAM python is using to hold your data.
        file_size = df.memory_usage(deep=True).sum() #deep=True tells pandas to get the actual sizes of all the strings in the column in the dataframe. 
        
        # Converting the RAM size of your data into Megabytes
        filesize_in_mb = f"{round(file_size / (1024 * 1024), 2)} MB"

        # Displays the basic information about the datset.
        return {
            "filename": file.filename,
            "file_size": filesize_in_mb,
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "first_five_rows": df.head().to_dict(),
            "numerical_columns_statistics": df.describe().to_dict(),
            "df_info": data_info
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

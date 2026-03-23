from typing import Optional
import pandas as pd

#Creating the dataset State class 
class dfState:
    # Using the class constructor to setup the dataframe
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None  # This will hold the DataFrame globally but for now its empty

# Created one instance to be imported everywhere across all the endpoints
dataset_state = dfState()
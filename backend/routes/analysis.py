from fastapi import APIRouter, HTTPException
from storage.datasets import DatasetStorage
from routes.EDA import DataAnalyzer, pd

router = APIRouter()

@router.get("/analyze/{dataset_id}")
async def analyze_dataset(dataset_id: str):
    """
    Perform full EDA on a specific dataset.
    """
    dataset_info = DatasetStorage.get_dataset(dataset_id)
    if not dataset_info:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = dataset_info["dataframe"]
    
    # Initialize analyzer
    analyzer = DataAnalyzer(df)
    
    try:
        # Generate full report
        report = analyzer.generate_full_report()
        
        # Prepare visualization data separately or include it
        viz_data = analyzer.prepare_visualization_data()
        
        return {
            "success": True,
            "report": report,
            "visualizations": viz_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

import requests
import json
import os

BASE_URL = "http://localhost:8000/api"
FILE_PATH = "test_data.csv"

def test_workflow():
    print("1. Testing Upload...")
    if not os.path.exists(FILE_PATH):
        print(f"Error: {FILE_PATH} not found")
        return

    files = {'file': open(FILE_PATH, 'rb')}
    try:
        response = requests.post(f"{BASE_URL}/upload", files=files)
        if response.status_code != 200:
            print(f"Upload failed: {response.text}")
            return
        
        data = response.json()
        print(f"Upload success! Dataset ID: {data['dataset_id']}")
        dataset_id = data['dataset_id']
        
        print("\n2. Testing Analysis...")
        response = requests.get(f"{BASE_URL}/analyze/{dataset_id}")
        if response.status_code != 200:
            print(f"Analysis failed: {response.text}")
            return
            
        analysis_data = response.json()
        print("Analysis success!")
        print(f"Overall Quality Score: {analysis_data['report']['data_quality_score']['overall_score']}")
        print(f"Insights found: {len(analysis_data['report']['insights'])}")
        
    except Exception as e:
        print(f"Test failed with exception: {str(e)}")

if __name__ == "__main__":
    test_workflow()

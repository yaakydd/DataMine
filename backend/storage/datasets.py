from typing import Dict, List, Optional
import uuid
from datetime import datetime

# In-memory storage for now (replace with DB later)
_datasets: Dict[str, Dict] = {}

class DatasetStorage:
    @staticmethod
    def save_dataset(dataset_id: str, metadata: Dict) -> Dict:
        metadata["saved_at"] = datetime.now().isoformat()
        _datasets[dataset_id] = metadata
        return metadata

    @staticmethod
    def get_dataset(dataset_id: str) -> Optional[Dict]:
        return _datasets.get(dataset_id)

    @staticmethod
    def list_datasets() -> List[Dict]:
        return list(_datasets.values())

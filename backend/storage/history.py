from typing import Dict, List
from datetime import datetime
import uuid

# In-memory storage for history
_history: Dict[str, List[Dict]] = {} # Keyed by dataset_id

class HistoryStorage:
    @staticmethod
    def log_action(dataset_id: str, action: str, details: Dict = None):
        if dataset_id not in _history:
            _history[dataset_id] = []
        
        entry = {
            "id": str(uuid.uuid4()),
            "action": action,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }
        _history[dataset_id].append(entry)
        return entry

    @staticmethod
    def get_history(dataset_id: str) -> List[Dict]:
        return _history.get(dataset_id, [])

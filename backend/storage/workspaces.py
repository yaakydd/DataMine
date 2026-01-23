from typing import Dict, List, Optional
import uuid
from datetime import datetime

# In-memory storage for workspaces
_workspaces: Dict[str, Dict] = {}

class WorkspaceStorage:
    @staticmethod
    def create_workspace(name: str) -> Dict:
        workspace_id = str(uuid.uuid4())
        workspace = {
            "id": workspace_id,
            "name": name,
            "datasets": [],
            "created_at": datetime.now().isoformat()
        }
        _workspaces[workspace_id] = workspace
        return workspace

    @staticmethod
    def get_workspace(workspace_id: str) -> Optional[Dict]:
        return _workspaces.get(workspace_id)

    @staticmethod
    def add_dataset_to_workspace(workspace_id: str, dataset_id: str):
        if workspace_id in _workspaces:
            if dataset_id not in _workspaces[workspace_id]["datasets"]:
                _workspaces[workspace_id]["datasets"].append(dataset_id)

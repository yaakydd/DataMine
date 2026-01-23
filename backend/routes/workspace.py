from fastapi import APIRouter, HTTPException
from storage.workspaces import WorkspaceStorage

router = APIRouter(prefix="/workspaces", tags=["Workspaces"])

@router.post("/")
async def create_workspace(name: str):
    return WorkspaceStorage.create_workspace(name)

@router.get("/{workspace_id}")
async def get_workspace(workspace_id: str):
    ws = WorkspaceStorage.get_workspace(workspace_id)
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return ws

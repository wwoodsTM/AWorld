import logging
import os
from typing import List
from fastapi import APIRouter, HTTPException, status, Query

from aworld.output import WorkSpace, ArtifactType
from config import WORKSPACE_PATH, WORKSPACE_TYPE

router = APIRouter()

@router.get("/{workspace_id}/tree")
async def get_workspace_tree(workspace_id: str):
    logging.info(f"get_workspace_tree: {workspace_id}")
    workspace = await load_workspace(workspace_id)
    return workspace.generate_tree_data()


@router.post("/{workspace_id}/artifacts")
async def get_workspace_artifacts(workspace_id: str, artifact_types: List[str] = Query(None)):
    """
    Get artifacts by workspace id and filter by a list of artifact types.
    Args:
        workspace_id: Workspace ID
        artifact_types: List of artifact type names (optional)
    Returns:
        Dict with filtered artifacts
    """
    if artifact_types:
        # Validate all types
        invalid_types = [t for t in artifact_types if t not in ArtifactType.__members__]
        if invalid_types:
            logging.error(f"Invalid artifact_types: {invalid_types}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid artifact types: {invalid_types}")
        logging.info(f"Fetching artifacts of types: {artifact_types}")
    else:
        logging.info(f"Fetching all artifacts (no type filter)")

    workspace = await load_workspace(workspace_id)
    all_artifacts = workspace.list_artifacts()
    if artifact_types:
        filtered_artifacts = [a for a in all_artifacts if a.artifact_type.name in artifact_types]
    else:
        filtered_artifacts = all_artifacts
    return {
        "data": filtered_artifacts
    }


@router.get("/{workspace_id}/file/{artifact_id}/content")
async def get_workspace_file_content(workspace_id: str, artifact_id: str):
    logging.info(f"get_workspace_file_content: {workspace_id}, {artifact_id}")
    workspace = await load_workspace(workspace_id)
    return {
        "data": workspace.get_file_content_by_artifact_id(artifact_id)
    }

    
async def load_workspace(workspace_id: str):
    
    """
    This function is used to get the workspace by its id.
    It first checks the workspace type and then creates the workspace accordingly.
    If the workspace type is not valid, it raises an HTTPException.
    """
    if workspace_id is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Workspace ID is required")
    
    if WORKSPACE_TYPE == "local":
        workspace = WorkSpace.from_local_storages(workspace_id, storage_path=os.path.join(WORKSPACE_PATH, workspace_id))
    elif WORKSPACE_TYPE == "oss":
        workspace = WorkSpace.from_oss_storages(workspace_id, storage_path=os.path.join(WORKSPACE_PATH, workspace_id))
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid workspace type")
    return workspace
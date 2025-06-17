import asyncio
import os

from aworld.core.exceptions import AworldException
from aworld.output import WorkSpace

from config import WORKSPACE_TYPE, WORKSPACE_PATH


async def load_workspace(user_id: str = None, session_id: str = None):
    if WORKSPACE_TYPE == "local":
        workspace = WorkSpace.from_local_storages(session_id, storage_path=await build_workspace_path(session_id,user_id
                                                                                                      ))
    elif WORKSPACE_TYPE == "oss":
        workspace = WorkSpace.from_oss_storages(session_id, storage_path=await build_workspace_path(session_id,user_id
                                                                                                      ))
    else:
        raise AworldException(message=f"Invalid workspace type#{WORKSPACE_TYPE}")
    return workspace


async def build_workspace_path(workspace_id: str, user_id: str = None) -> str:
    user_id = user_id if user_id is not None else ""
    return os.path.join(WORKSPACE_PATH, user_id, workspace_id)

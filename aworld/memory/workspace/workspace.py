import os
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Callable

from aworld.memory.workspace.artifact import ArtifactType, Artifact
from aworld.memory.workspace.artifact_repository import LocalArtifactRepository


class ArtifactWorkSpace:
    """
    Artifact workspace, managing a group of related artifacts
    
    Provides collaborative editing features, supporting version management, update notifications, etc. for multiple Artifacts
    """

    def __init__(
            self,
            workspace_id: Optional[str] = None,
            name: Optional[str] = None,
            storage_path: Optional[str] = None
    ):
        self.workspace_id = workspace_id or str(uuid.uuid4())
        self.name = name or f"Workspace-{self.workspace_id[:8]}"
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self.artifacts: Dict[str, Artifact] = {}
        self.metadata: Dict[str, Any] = {}
        self.observers: List[Callable[[str, Artifact], None]] = []

        # Initialize repository
        storage_dir = storage_path or os.path.join("data", "workspaces", self.workspace_id)
        self.repository = LocalArtifactRepository(storage_dir)

    def create_artifact(
            self,
            artifact_type: Union[ArtifactType, str],
            artifact_id: Optional[str] = None,
            content: Optional[Any] = None,
            metadata: Optional[Dict[str, Any]] = None,
            render_type: Optional[str] = None
    ) -> Artifact:
        """
        Create a new artifact in the workspace
        
        Args:
            artifact_type: Artifact type
            content: Artifact content
            metadata: Artifact metadata
            render_type: Rendering type
            
        Returns:
            The created artifact object
        """
        # If a string is passed, convert to enum type
        if isinstance(artifact_type, str):
            artifact_type = ArtifactType(artifact_type)

        # Create new artifact
        artifact = Artifact(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            content=content,
            metadata=metadata,
            render_type=render_type
        )

        # Add to workspace
        self.artifacts[artifact.artifact_id] = artifact

        # Store in repository
        self._store_artifact(artifact)

        # Update workspace time
        self.updated_at = datetime.now().isoformat()

        # Notify observers
        self._notify_observers("create", artifact)

        return artifact

    def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        """Get artifact with the specified ID"""
        return self.artifacts.get(artifact_id)

    def update_artifact(
            self,
            artifact_id: str,
            content: Any,
            description: str = "Content update"
    ) -> Optional[Artifact]:
        """
        Update artifact content
        
        Args:
            artifact_id: Artifact ID
            content: New content
            description: Update description
            
        Returns:
            Updated artifact, or None if it doesn't exist
        """
        artifact = self.get_artifact(artifact_id)
        if artifact:
            artifact.update_content(content, description)

            # Update storage
            self._store_artifact(artifact)

            # Update workspace time
            self.updated_at = datetime.now().isoformat()

            # Notify observers
            self._notify_observers("update", artifact)

            return artifact
        return None

    def delete_artifact(self, artifact_id: str) -> bool:
        """
        Delete an artifact from the workspace
        
        Args:
            artifact_id: Artifact ID
            
        Returns:
            Whether deletion was successful
        """
        artifact = self.artifacts.pop(artifact_id, None)
        if artifact:
            # Mark as archived
            artifact.archive()

            # Update storage
            self._store_artifact(artifact)

            # Update workspace time
            self.updated_at = datetime.now().isoformat()

            # Notify observers
            self._notify_observers("delete", artifact)

            return True
        return False

    def list_artifacts(self, filter_type: Optional[ArtifactType] = None) -> List[Artifact]:
        """
        List all artifacts in the workspace
        
        Args:
            filter_type: Optional filter type
            
        Returns:
            List of artifacts
        """
        if filter_type:
            return [a for a in self.artifacts.values() if a.artifact_type == filter_type]
        return list(self.artifacts.values())

    def add_observer(self, callback: Callable[[str, Artifact], None]) -> None:
        """
        Add an artifact change observer
        
        Args:
            callback: Callback function, receives operation type and artifact object
        """
        self.observers.append(callback)

    def remove_observer(self, callback: Callable[[str, Artifact], None]) -> None:
        """Remove an observer"""
        if callback in self.observers:
            self.observers.remove(callback)

    def _notify_observers(self, operation: str, artifact: Artifact) -> None:
        """Notify all observers of artifact changes"""
        for observer in self.observers:
            try:
                observer(operation, artifact)
            except Exception as e:
                print(f"Observer notification failed: {e}")

    def _store_artifact(self, artifact: Artifact) -> None:
        """Store artifact in repository"""
        artifact_data = artifact.to_dict()

        # Include complete version history
        artifact_data["version_history"] = artifact.version_history

        # Store in repository
        self.repository.store(
            artifact_id=artifact.artifact_id,
            data=artifact_data,
            metadata={"workspace_id": self.workspace_id}
        )

    def save(self) -> str:
        """
        Save workspace state
        
        Returns:
            Workspace storage ID
        """
        workspace_data = {
            "workspace_id": self.workspace_id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            "artifact_ids": list(self.artifacts.keys())
        }

        # Store workspace information
        return self.repository.store(
            artifact_id=f"workspace_{self.workspace_id}",
            data=workspace_data
        )

    @classmethod
    def load(cls, workspace_id: str, storage_path: Optional[str] = None) -> Optional["ArtifactWorkSpace"]:
        """
        Load workspace
        
        Args:
            workspace_id: Workspace ID
            storage_path: Optional storage path
            
        Returns:
            Loaded workspace, or None if it doesn't exist
        """
        # Initialize storage path
        storage_dir = storage_path or os.path.join("data", "workspaces", workspace_id)
        repository = LocalArtifactRepository(storage_dir)

        # Get workspace versions
        workspace_versions = repository.get_versions(f"workspace_{workspace_id}")
        if not workspace_versions:
            return None

        # Get latest version
        latest_version_id = workspace_versions[-1]["id"]
        workspace_data = repository.retrieve(latest_version_id)

        if not workspace_data:
            return None

        # Create workspace instance
        workspace = cls(
            workspace_id=workspace_data["workspace_id"],
            name=workspace_data["name"],
            storage_path=storage_dir
        )
        workspace.created_at = workspace_data["created_at"]
        workspace.updated_at = workspace_data["updated_at"]
        workspace.metadata = workspace_data["metadata"]

        # Load artifacts
        for artifact_id in workspace_data["artifact_ids"]:
            artifact_versions = repository.get_versions(artifact_id)
            if artifact_versions:
                latest_version_id = artifact_versions[-1]["id"]
                artifact_data = repository.retrieve(latest_version_id)
                if artifact_data:
                    workspace.artifacts[artifact_id] = Artifact.from_dict(artifact_data)

        return workspace

from enum import Enum
from typing import Dict, Any, Optional, List
import hashlib
import json
import os
import time
from pathlib import Path


class ArtifactRepository:
    def __init__(self):
        """
        Initialize the artifact repository
        """
        pass

    def _load_index(self) -> Dict[str, Any]:
        """Load or create index file"""
        pass

    def _save_index(self, index: Dict[str, Any]) -> None:
        """Save index to file"""
        pass

    def _compute_content_hash(self, data: Any) -> str:
        """
        Calculate the hash value of the content as a storage identifier

        Args:
            data: Data to be stored

        Returns:
            SHA-256 hash value of the content
        """
        pass

    def store(self,
              artifact_id: str,
              data: Dict[str, Any],
              metadata: Optional[Dict[str, Any]] = None
              ) -> str:
        """
        Store artifact and return its version identifier

        Args:
            artifact_id: Unique identifier of the artifact
            data: Data to be stored
            metadata: Optional metadata

        Returns:
            Version identifier
        """
        pass

    def retrieve(self, version_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve artifact based on version ID

        Args:
            version_id: Version identifier

        Returns:
            Stored data, or None if it doesn't exist
        """
        pass

    def get_versions(self, artifact_id: str) -> List[Dict[str, Any]]:
        """
        Get information about all versions of an artifact

        Args:
            artifact_id: Artifact identifier

        Returns:
            List of version information
        """
        pass



class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return {"__enum__": True, "__enum_type__": obj.__class__.__name__, "__enum_value__": obj.name}
        return json.JSONEncoder.default(self, obj)

class EnumDecoder(json.JSONDecoder):
    def decode(self, s, **kwargs):
        parsed_json = super().decode(s, **kwargs)
        for key, value in parsed_json.items():
            if isinstance(value, dict) and value.get("__enum__"):
                enum_type = globals()[value["__enum_type__"]]
                enum_value = enum_type[value["__enum_value__"]]
                parsed_json[key] = enum_value
        return parsed_json


class LocalArtifactRepository(ArtifactRepository):
    """Artifact storage layer: manages versioned artifacts through content-addressable storage"""

    def __init__(self, storage_path: str):
        """
        Initialize the artifact repository
        
        Args:
            storage_path: Directory path for storing data
        """
        super().__init__()
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.index_path = self.storage_path / "index.json"
        self.index = self._load_index()

    def _load_index(self) -> Dict[str, Any]:
        """Load or create index file"""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {"artifacts": {}, "versions": {}}
        else:
            index = {"artifacts": {}, "versions": {}}
            self._save_index(index)
            return index

    def _save_index(self, index: Dict[str, Any]) -> None:
        """Save index to file"""
        with open(self.index_path, 'w') as f:
            json.dump(index, f, indent=2)

    def _compute_content_hash(self, data: Any) -> str:
        """
        Calculate the hash value of the content as a storage identifier
        
        Args:
            data: Data to be stored
            
        Returns:
            SHA-256 hash value of the content
        """
        content = json.dumps(data, sort_keys=True, cls=EnumEncoder).encode('utf-8')
        return hashlib.sha256(content).hexdigest()

    def store(self,
              artifact_id: str,
              data: Dict[str, Any],
              metadata: Optional[Dict[str, Any]] = None
              ) -> str:
        """
        Store artifact and return its version identifier
        
        Args:
            artifact_id: Unique identifier of the artifact
            data: Data to be stored
            metadata: Optional metadata
            
        Returns:
            Version identifier
        """
        # Calculate content hash
        content_hash = self._compute_content_hash(data)

        # Create version record
        version = {
            "hash": content_hash,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }

        # Store content
        content_path = self.storage_path / f"{content_hash}.json"
        if not content_path.exists():
            with open(content_path, 'w') as f:
                json.dump(data, f, indent=2, cls=EnumEncoder)

        # Update index
        if artifact_id not in self.index["artifacts"]:
            self.index["artifacts"][artifact_id] = []

        version_id = f"{artifact_id}_{len(self.index['artifacts'][artifact_id])}"
        self.index["artifacts"][artifact_id].append(version_id)
        self.index["versions"][version_id] = version

        self._save_index(self.index)
        return version_id

    def retrieve(self, version_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve artifact based on version ID
        
        Args:
            version_id: Version identifier
            
        Returns:
            Stored data, or None if it doesn't exist
        """
        if version_id not in self.index["versions"]:
            return None

        version = self.index["versions"][version_id]
        content_hash = version["hash"]
        content_path = self.storage_path / f"{content_hash}.json"

        if not content_path.exists():
            return None

        with open(content_path, 'r') as f:
            return json.load(f)

    def get_versions(self, artifact_id: str) -> List[Dict[str, Any]]:
        """
        Get information about all versions of an artifact
        
        Args:
            artifact_id: Artifact identifier
            
        Returns:
            List of version information
        """
        if artifact_id not in self.index["artifacts"]:
            return []

        versions = []
        for version_id in self.index["artifacts"][artifact_id]:
            version_info = self.index["versions"][version_id].copy()
            version_info["id"] = version_id
            versions.append(version_info)

        return versions

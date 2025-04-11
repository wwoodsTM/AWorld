import abc
import json
from typing import Any, Dict, Generator, AsyncGenerator, Union, Optional, TypedDict

from pydantic import Field, BaseModel, model_validator


class OutputPart(BaseModel):
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="metadata")

    @model_validator(mode='after')
    def setup_metadata(self):
        # Ensure metadata is initialized
        if self.metadata is None:
            self.metadata = {}
        return self
    

class Output(BaseModel):
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="metadata")
    parts: Any = Field(default_factory=list, exclude=True, description="parts of Output")

    @model_validator(mode='after')
    def setup_defaults(self):
        # Ensure metadata and parts are initialized
        if self.metadata is None:
            self.metadata = {}
        if self.parts is None:
            self.parts = []
        return self

class CommonOutput(Output):
    data: Any = Field(default=None, exclude=True, description="Output Data")

class MessageOutput(Output):
    reasoning: Any = Field(default=None, exclude=True, description="reasoning")
    response: Any = Field(default=None, exclude=True, description="response")

class Event(Output):
    pass

class ToolOutput(Output):
    pass

class SearchItem(BaseModel):
    title: str = Field(default="", description="search result title")
    url: str= Field(default="", description="search result url")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="metadata")


class SearchOutput(ToolOutput):
    query: str = Field(..., description="Search query string")
    results: list[SearchItem] = Field(default_factory=list, description="List of search results")

    @classmethod
    def from_dict(cls, data: dict) -> "SearchOutput":
        if not isinstance(data, dict):
            data = {}

        query = data.get("query")
        if query is None:
            raise ValueError("query is required")

        results_data = data.get("results", [])
        
        search_items = []
        for result in results_data:
            if isinstance(result, SearchItem):
                search_items.append(result)
            elif isinstance(result, dict):
                search_items.append(SearchItem(**result))
            else:
                raise ValueError(f"Invalid result type: {type(result)}")

        return cls(
            query=query,
            results=search_items
        )
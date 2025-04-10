import abc
import json
from typing import Any, Dict, Generator, AsyncGenerator, Union, Optional, TypedDict

from pydantic import Field, BaseModel


# The agent's output type
class Output:
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="metadata")



class Event(Output):
    pass

class ToolOutput(Output):
    result: Any | None = Field(description="tool result")

class SearchItem(BaseModel):
    title: str
    url: str
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="metadata")

class SearchOutput(Output, BaseModel):
    query: str
    results: list[SearchItem]


    @classmethod
    def from_dict(cls, data: dict) -> "SearchOutput":
        return cls(query=data.get("query"), results=data.get("results"))


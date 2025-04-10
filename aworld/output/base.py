import abc
from typing import Any, Dict, Generator, AsyncGenerator, Union, Optional

from pydantic import Field


# The agent's output type
class Output:
    pass

class Event(Output):
    pass


class MessageOutput(Output):
    reasoning: Union[str, Generator[str, None, None], AsyncGenerator[str, None]] = Field(description="llm think")
    response: Union[str, Generator[str, None, None], AsyncGenerator[str, None]] = Field(description="llm response")
    metadata: Optional[Dict[str, Any]] = Field(description="metadata, such as usage")



class ToolOutput(Output):
    result: Any | None = Field(description="tool result")


class SearchOutput(Output):
    result: Any | None = Field(description="tool result")

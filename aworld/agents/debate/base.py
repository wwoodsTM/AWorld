from pydantic import BaseModel, Field

from aworld.output import Output


class DebateSpeech(Output, BaseModel):
    name: str = Field(default="", description="name of the speaker")
    type: str = Field(default="", description="speech type")
    stance: str = Field(default="", description="stance of the speech")
    content: str = Field(default="", description="content of the speech")
    round: int = Field(default=0, description="round of the speech")
    metadata: dict = Field(default_factory=dict, description="metadata of the speech")



    @classmethod
    def from_dict(cls, data: dict) -> "DebateSpeech":
        return cls(
            name=data.get("name", ""),
            type=data.get("type", ""),
            stance=data.get("stance", ""),
            content=data.get("content", ""),
            round=data.get("round", 0),
            metadata=data.get("metadata", {})
        )

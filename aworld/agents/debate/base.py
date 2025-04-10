from pydantic import BaseModel

from aworld.output import Output


class DebateSpeech(Output, BaseModel):
    name: str
    type: str
    content: str
    round: int

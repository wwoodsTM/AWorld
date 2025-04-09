from pydantic import BaseModel


class DebateSpeech(BaseModel):
    name: str
    type: str
    content: str
    round: int

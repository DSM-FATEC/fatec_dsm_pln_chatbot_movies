from pydantic import BaseModel


class MessageModel(BaseModel):
    message: str


class MessageModelResponse(BaseModel):
    answer: str
    sentiment: str

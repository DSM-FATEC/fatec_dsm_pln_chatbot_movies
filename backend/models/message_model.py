from pydantic import BaseModel


class MessageModel(BaseModel):
    message: str


class MessageModelResponse(BaseModel):
    answer: str
    question_sentiment: str
    answer_sentiment: str

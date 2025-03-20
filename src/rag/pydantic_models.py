from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class ModelName(str, Enum):
    GPT4_O_MINI = "gpt-4o-mini"
    GEMINI_2_0_FLASH_THINKING_EXP_01_21 = "gemini-2.0-flash-thinking-exp-01-21"


class QueryInput(BaseModel):
    question: str
    session_id: str = Field(
        default=None,
        title="Session ID",
        description="Unique session ID for the conversation",
    )
    model: ModelName = Field(
        default=ModelName.GEMINI_2_0_FLASH_THINKING_EXP_01_21,
        title="Model Name",
        description="Name of the model to use for the response",
    )


class QueryAnswer(BaseModel):
    answer: str
    session_id: str
    model: ModelName

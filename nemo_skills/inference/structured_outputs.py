from typing import Literal
from pydantic import BaseModel


class HLEJudgeAAResponseFormat(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int
    strict: Literal[True] # 100% reliability


STRUCTURED_OUTPUTS = {
    "HLE_JUDGE_AA": HLEJudgeAAResponseFormat,
}

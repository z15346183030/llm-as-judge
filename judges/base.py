"""评委基类 — 定义评估接口。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from core.llm import LLMClient


@dataclass
class JudgeScore:
    """评委评分。"""
    judge_name: str
    correctness: float       # 0-100
    readability: float       # 0-100
    efficiency: float        # 0-100
    robustness: float        # 0-100
    style: float             # 0-100
    overall: float = 0.0     # 加权总分
    feedback: str = ""       # 文字反馈

    def __post_init__(self):
        if self.overall == 0.0:
            self.overall = (
                self.correctness * 0.30 +
                self.readability * 0.20 +
                self.efficiency * 0.20 +
                self.robustness * 0.15 +
                self.style * 0.15
            )


class BaseJudge(ABC):
    """评委基类。"""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    @abstractmethod
    def evaluate(self, code: str, task: str) -> JudgeScore:
        """评估代码，返回评分。"""
        ...

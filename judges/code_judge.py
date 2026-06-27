"""代码评估评委 — 用 LLM 评估代码质量。"""

import json
import re

from judges.base import BaseJudge, JudgeScore

JUDGE_SYSTEM = """你是一个资深代码审查专家，负责给代码打分。

评估维度（每项 0-100 分）:
1. 正确性 (correctness): 逻辑是否正确，是否满足需求
2. 可读性 (readability): 命名、结构、注释是否清晰
3. 效率 (efficiency): 时间/空间复杂度是否合理
4. 健壮性 (robustness): 异常处理、边界情况
5. 代码风格 (style): 是否遵循最佳实践

请严格按以下 JSON 格式返回（不要包含其他文字）:
{
  "correctness": 分数,
  "readability": 分数,
  "efficiency": 分数,
  "robustness": 分数,
  "style": 分数,
  "feedback": "简短的文字评价（50字以内）"
}"""

JUDGE_USER = """请评估以下代码:

任务描述: {task}

```python
{code}
```

请按 JSON 格式返回评分:"""


class CodeJudge(BaseJudge):
    """代码评估评委。"""

    def __init__(self, llm, name: str = None):
        super().__init__(llm)
        self.name = name or llm.model

    def evaluate(self, code: str, task: str) -> JudgeScore:
        """评估代码质量。"""
        prompt = JUDGE_USER.format(task=task, code=code)

        response = self.llm.chat(
            system=JUDGE_SYSTEM,
            user=prompt,
            temperature=0.2,
        )

        # 提取 JSON
        content = response.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # 尝试用正则提取数字
            data = self._extract_scores(content)

        return JudgeScore(
            judge_name=self.name,
            correctness=float(data.get("correctness", 50)),
            readability=float(data.get("readability", 50)),
            efficiency=float(data.get("efficiency", 50)),
            robustness=float(data.get("robustness", 50)),
            style=float(data.get("style", 50)),
            feedback=data.get("feedback", ""),
        )

    def _extract_scores(self, text: str) -> dict:
        """从文本中提取分数。"""
        scores = {}
        for dim in ["correctness", "readability", "efficiency", "robustness", "style"]:
            match = re.search(rf'{dim}["\s:]+(\d+)', text)
            if match:
                scores[dim] = int(match.group(1))
            else:
                scores[dim] = 50
        return scores

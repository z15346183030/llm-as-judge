"""评分校准 — 用参考答案校准评委的评分标准。"""

import json
from judges.base import BaseJudge, JudgeScore


class Calibrator:
    """
    评分校准器。

    通过让评委评估参考答案，建立评分基线，
    用于校准后续评估的分数。
    """

    def __init__(self):
        self.baseline_scores: dict[str, JudgeScore] = {}

    def calibrate(self, judge: BaseJudge, reference_code: str, task: str, expected_score: float = 90.0):
        """
        校准评委。

        Args:
            judge: 评委实例
            reference_code: 参考答案代码
            task: 任务描述
            expected_score: 参考答案的预期分数
        """
        score = judge.evaluate(reference_code, task)
        self.baseline_scores[judge.name] = {
            "score": score,
            "expected": expected_score,
            "offset": expected_score - score.overall,
        }

    def calibrate_score(self, judge_name: str, raw_score: JudgeScore) -> JudgeScore:
        """
        校准分数。

        Args:
            judge_name: 评委名称
            raw_score: 原始评分

        Returns:
            校准后的评分
        """
        if judge_name not in self.baseline_scores:
            return raw_score

        offset = self.baseline_scores[judge_name]["offset"]

        return JudgeScore(
            judge_name=raw_score.judge_name,
            correctness=min(100, max(0, raw_score.correctness + offset)),
            readability=min(100, max(0, raw_score.readability + offset * 0.5)),
            efficiency=min(100, max(0, raw_score.efficiency + offset * 0.5)),
            robustness=min(100, max(0, raw_score.robustness + offset * 0.5)),
            style=min(100, max(0, raw_score.style + offset * 0.5)),
            feedback=raw_score.feedback,
        )

    def get_calibration_report(self) -> str:
        """生成校准报告。"""
        lines = ["## 校准报告\n"]
        for name, data in self.baseline_scores.items():
            lines.append(f"- {name}: 预期={data['expected']}, 实际={data['score'].overall:.1f}, 偏移={data['offset']:.1f}")
        return "\n".join(lines)

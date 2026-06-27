#!/usr/bin/env python3
"""LLM-as-a-Judge — 主程序入口。"""

import argparse
import json
import os
import sys

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from core.llm import LLMClient
from judges.code_judge import CodeJudge
from judges.base import JudgeScore

console = Console()


def calculate_icc(scores: list[list[float]]) -> float:
    """
    计算组内相关系数 (ICC) — 衡量评委间一致性。

    简化版 ICC(2,1): Two-way random, single measures, absolute agreement.
    """
    if len(scores) < 2:
        return 1.0

    data = np.array(scores)
    n_items = data.shape[1]
    n_judges = data.shape[0]

    # 总均值
    grand_mean = data.mean()

    # 项目均值
    item_means = data.mean(axis=0)

    # 评委均值
    judge_means = data.mean(axis=1)

    # 各种平方和
    ss_total = ((data - grand_mean) ** 2).sum()
    ss_between_items = n_judges * ((item_means - grand_mean) ** 2).sum()
    ss_between_judges = n_items * ((judge_means - grand_mean) ** 2).sum()
    ss_error = ss_total - ss_between_items - ss_between_judges

    # 均方
    ms_error = ss_error / ((n_items - 1) * (n_judges - 1)) if (n_items - 1) * (n_judges - 1) > 0 else 0

    # ICC
    ms_items = ss_between_items / (n_items - 1) if n_items > 1 else 0
    icc = (ms_items - ms_error) / (ms_items + (n_judges - 1) * ms_error) if (ms_items + (n_judges - 1) * ms_error) > 0 else 0

    return max(0, min(1, icc))


def evaluate_code(code: str, task: str, judge_models: list[str]) -> list[JudgeScore]:
    """用多个评委评估代码。"""
    scores = []
    for model in judge_models:
        console.print(f"[dim]  评委 {model} 评分中...[/]")
        llm = LLMClient(model=model)
        judge = CodeJudge(llm, name=model)
        score = judge.evaluate(code, task)
        scores.append(score)
    return scores


def print_results(scores: list[JudgeScore]):
    """打印评估结果。"""
    table = Table(title="⚖️ 评委评分", show_lines=True)
    table.add_column("评委", style="cyan")
    table.add_column("正确性", justify="right")
    table.add_column("可读性", justify="right")
    table.add_column("效率", justify="right")
    table.add_column("健壮性", justify="right")
    table.add_column("风格", justify="right")
    table.add_column("总分", justify="right", style="bold")

    for s in scores:
        table.add_row(
            s.judge_name,
            f"{s.correctness:.0f}",
            f"{s.readability:.0f}",
            f"{s.efficiency:.0f}",
            f"{s.robustness:.0f}",
            f"{s.style:.0f}",
            f"{s.overall:.1f}",
        )

    console.print()
    console.print(table)

    # 统计分析
    if len(scores) >= 2:
        all_scores = [[s.correctness, s.readability, s.efficiency, s.robustness, s.style] for s in scores]
        icc = calculate_icc(all_scores)

        avg = sum(s.overall for s in scores) / len(scores)
        std = np.std([s.overall for s in scores])

        console.print(f"\n[bold]📊 统计分析[/]")
        console.print(f"  加权平均分: {avg:.1f}")
        console.print(f"  标准差: {std:.1f}")
        console.print(f"  评委一致性 (ICC): {icc:.2f}", end="")

        if icc > 0.75:
            console.print(" [green](高一致性)[/]")
        elif icc > 0.5:
            console.print(" [yellow](中等一致性)[/]")
        else:
            console.print(" [red](低一致性 — 建议增加评委数量)[/]")

    # 文字反馈
    console.print(f"\n[bold]💬 评委反馈[/]")
    for s in scores:
        if s.feedback:
            console.print(f"  {s.judge_name}: {s.feedback}")


def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge — AI 代码质量评估")
    parser.add_argument("--code", help="要评估的代码（字符串）")
    parser.add_argument("--file", help="要评估的代码文件")
    parser.add_argument("--task", required=True, help="任务描述")
    parser.add_argument("--judges", default="gpt-4o", help="评委模型，逗号分隔")
    parser.add_argument("--output", help="输出报告到文件")
    args = parser.parse_args()

    console.print(Panel("[bold cyan]⚖️ LLM-as-a-Judge[/]\nAI 代码质量评估框架", style="cyan"))

    # 获取代码
    if args.code:
        code = args.code
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            code = f.read()
    else:
        console.print("[yellow]请输入代码（输入空行结束）:[/]")
        lines = []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)
        code = "\n".join(lines)

    if not code.strip():
        console.print("[red]错误: 未提供代码[/]")
        sys.exit(1)

    # 解析评委列表
    judge_models = [m.strip() for m in args.judges.split(",")]

    # 评估
    console.print(f"\n[cyan]📋 任务: {args.task}[/]")
    console.print(f"[cyan]⚖️ 评委: {', '.join(judge_models)}[/]\n")

    scores = evaluate_code(code, args.task, judge_models)

    # 输出结果
    print_results(scores)

    # 保存报告
    if args.output:
        report = {
            "task": args.task,
            "code": code,
            "scores": [
                {
                    "judge": s.judge_name,
                    "correctness": s.correctness,
                    "readability": s.readability,
                    "efficiency": s.efficiency,
                    "robustness": s.robustness,
                    "style": s.style,
                    "overall": s.overall,
                    "feedback": s.feedback,
                }
                for s in scores
            ],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        console.print(f"\n[dim]报告已保存: {args.output}[/]")


if __name__ == "__main__":
    main()

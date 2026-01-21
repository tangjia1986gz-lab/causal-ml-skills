#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OrchestratorAgent - 多智能体校准系统协调器

职责:
1. 管理整体工作流
2. 协调各 Agent 之间的通信
3. 质量门控和进度追踪
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Windows 编码修复
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
SKILLS_DIR = PROJECT_ROOT / "skills"
CALIBRATION_NOTES_DIR = PROJECT_ROOT / "calibration_notes"


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class CalibrationTask:
    """校准任务定义"""
    skill_name: str
    queries: List[str]
    status: TaskStatus = TaskStatus.PENDING
    papers_found: int = 0
    notes_generated: int = 0
    gaps_identified: int = 0
    updates_applied: int = 0
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class CalibrationResult:
    """校准结果"""
    skill_name: str
    papers_processed: List[Dict[str, Any]]
    gaps: List[Dict[str, Any]]
    recommendations: List[str]
    files_updated: List[str]
    success: bool
    error: Optional[str] = None


class OrchestratorAgent:
    """
    协调器 Agent

    管理整个校准流程：
    1. 接收技能列表和校准配置
    2. 调度 LiteratureAgent 搜索论文
    3. 调度 ExtractorAgent 提取内容
    4. 调度 CalibrationAgent 识别差距
    5. 调度 UpdateAgent 生成更新
    6. 汇总结果并生成报告
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.tasks: Dict[str, CalibrationTask] = {}
        self.results: Dict[str, CalibrationResult] = {}
        self._agents = {}

    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            "max_papers_per_skill": 5,
            "min_citations": 100,
            "target_journals": [
                "American Economic Review",
                "Econometrica",
                "Journal of Econometrics",
                "Review of Economic Studies",
                "Quarterly Journal of Economics",
                "Journal of Political Economy",
                "Review of Economics and Statistics",
                "Journal of the American Statistical Association"
            ],
            "parallel_tasks": 3,
            "output_format": "markdown",
            "auto_update": False,  # 是否自动应用更新
        }

    def register_skill(self, skill_name: str, queries: List[str]) -> None:
        """注册技能进行校准"""
        self.tasks[skill_name] = CalibrationTask(
            skill_name=skill_name,
            queries=queries
        )
        print(f"[Orchestrator] 已注册技能: {skill_name}")

    def register_skills_from_config(self, skills_config: Dict[str, List[str]]) -> None:
        """从配置批量注册技能"""
        for skill_name, queries in skills_config.items():
            self.register_skill(skill_name, queries)

    async def run_calibration(self, skill_name: str) -> CalibrationResult:
        """
        对单个技能运行完整校准流程
        """
        from .literature_agent import LiteratureAgent
        from .extractor_agent import ExtractorAgent
        from .calibration_agent import CalibrationAgent

        task = self.tasks.get(skill_name)
        if not task:
            raise ValueError(f"技能未注册: {skill_name}")

        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()

        print(f"\n{'='*60}")
        print(f"[Orchestrator] 开始校准: {skill_name}")
        print(f"{'='*60}")

        try:
            # Step 1: 搜索文献
            print(f"\n[Step 1] 搜索高引文献...")
            literature_agent = LiteratureAgent(
                min_citations=self.config["min_citations"],
                target_journals=self.config["target_journals"]
            )

            papers = []
            for query in task.queries:
                found = await literature_agent.search(
                    query,
                    limit=self.config["max_papers_per_skill"]
                )
                papers.extend(found)

            # 去重并按引用数排序
            papers = self._deduplicate_papers(papers)
            papers = sorted(papers, key=lambda x: x.get('citations', 0), reverse=True)
            papers = papers[:self.config["max_papers_per_skill"]]

            task.papers_found = len(papers)
            print(f"    找到 {len(papers)} 篇相关论文")

            # Step 2: 提取内容
            print(f"\n[Step 2] 提取论文内容...")
            extractor_agent = ExtractorAgent()

            extracted_content = []
            for paper in papers:
                content = await extractor_agent.extract(paper)
                if content:
                    extracted_content.append(content)

            task.notes_generated = len(extracted_content)
            print(f"    成功提取 {len(extracted_content)} 篇论文内容")

            # Step 3: 对比分析
            print(f"\n[Step 3] 分析差距...")
            calibration_agent = CalibrationAgent(skill_name)

            gaps = await calibration_agent.analyze_gaps(extracted_content)
            task.gaps_identified = len(gaps)
            print(f"    识别到 {len(gaps)} 个差距")

            # Step 4: 生成建议
            print(f"\n[Step 4] 生成更新建议...")
            recommendations = calibration_agent.generate_recommendations(gaps)

            # Step 5: 应用更新 (如果启用自动更新)
            files_updated = []
            if self.config["auto_update"]:
                print(f"\n[Step 5] 应用更新...")
                from .update_agent import UpdateAgent
                update_agent = UpdateAgent(skill_name)
                files_updated = await update_agent.apply_updates(recommendations)
                task.updates_applied = len(files_updated)

            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()

            result = CalibrationResult(
                skill_name=skill_name,
                papers_processed=[{
                    'title': p.get('title'),
                    'year': p.get('year'),
                    'citations': p.get('citations')
                } for p in papers],
                gaps=gaps,
                recommendations=recommendations,
                files_updated=files_updated,
                success=True
            )

            self.results[skill_name] = result

            print(f"\n[Orchestrator] 校准完成: {skill_name}")
            return result

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.now()

            result = CalibrationResult(
                skill_name=skill_name,
                papers_processed=[],
                gaps=[],
                recommendations=[],
                files_updated=[],
                success=False,
                error=str(e)
            )
            self.results[skill_name] = result

            print(f"\n[Orchestrator] 校准失败: {skill_name} - {e}")
            return result

    async def run_all_calibrations(self) -> Dict[str, CalibrationResult]:
        """运行所有技能的校准"""
        import asyncio

        print(f"\n{'#'*60}")
        print(f"# 开始批量校准 ({len(self.tasks)} 个技能)")
        print(f"{'#'*60}")

        # 分批并行处理
        skill_names = list(self.tasks.keys())
        batch_size = self.config["parallel_tasks"]

        for i in range(0, len(skill_names), batch_size):
            batch = skill_names[i:i + batch_size]
            print(f"\n--- 批次 {i // batch_size + 1}: {batch} ---")

            tasks = [self.run_calibration(skill) for skill in batch]
            await asyncio.gather(*tasks)

        # 生成汇总报告
        self._generate_summary_report()

        return self.results

    def _deduplicate_papers(self, papers: List[Dict]) -> List[Dict]:
        """论文去重"""
        seen = set()
        unique = []
        for paper in papers:
            paper_id = paper.get('paper_id') or paper.get('title', '')
            if paper_id not in seen:
                seen.add(paper_id)
                unique.append(paper)
        return unique

    def _generate_summary_report(self) -> None:
        """生成汇总报告"""
        report_path = CALIBRATION_NOTES_DIR / "calibration_summary.md"

        total_papers = sum(r.papers_processed.__len__() for r in self.results.values())
        total_gaps = sum(len(r.gaps) for r in self.results.values())
        successful = sum(1 for r in self.results.values() if r.success)

        report = f"""# 校准汇总报告

> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 概览

| 指标 | 数值 |
|------|------|
| 技能总数 | {len(self.tasks)} |
| 成功校准 | {successful} |
| 处理论文 | {total_papers} |
| 识别差距 | {total_gaps} |

## 各技能详情

"""

        for skill_name, result in self.results.items():
            status = "✅" if result.success else "❌"
            report += f"""### {status} {skill_name}

- **论文数**: {len(result.papers_processed)}
- **差距数**: {len(result.gaps)}
- **建议数**: {len(result.recommendations)}

"""
            if result.gaps:
                report += "**主要差距**:\n"
                for gap in result.gaps[:5]:
                    report += f"- {gap.get('description', 'N/A')}\n"
                report += "\n"

        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n[Orchestrator] 汇总报告已保存: {report_path}")

    def get_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            "total_tasks": len(self.tasks),
            "completed": sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED),
            "failed": sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED),
            "in_progress": sum(1 for t in self.tasks.values() if t.status == TaskStatus.IN_PROGRESS),
            "pending": sum(1 for t in self.tasks.values() if t.status == TaskStatus.PENDING),
            "tasks": {
                name: {
                    "status": task.status.value,
                    "papers": task.papers_found,
                    "gaps": task.gaps_identified
                }
                for name, task in self.tasks.items()
            }
        }


# 预定义技能查询配置
SKILL_CALIBRATION_CONFIG = {
    "estimator-did": [
        "Callaway Sant'Anna difference-in-differences multiple time periods",
        "Goodman-Bacon decomposition difference-in-differences",
        "Sun Abraham event study heterogeneous treatment",
        "de Chaisemartin D'Haultfoeuille two-way fixed effects",
    ],
    "estimator-rd": [
        "Cattaneo Idrobo Titiunik regression discontinuity practical",
        "Imbens Kalyanaraman optimal bandwidth regression discontinuity",
        "McCrary density test manipulation",
        "Calonico Cattaneo Titiunik robust inference RD",
    ],
    "estimator-iv": [
        "Andrews Stock Sun weak instruments many",
        "Staiger Stock instrumental variables weak",
        "Lee McCrary Moreira Vytlacil robust weak IV",
        "Angrist Krueger instrumental variables",
    ],
    "estimator-psm": [
        "Imbens Rubin propensity score matching causal",
        "King Nielsen propensity score matching",
        "Rosenbaum bounds sensitivity analysis matching",
        "Abadie Imbens matching estimators",
    ],
    "causal-ddml": [
        "Chernozhukov double debiased machine learning",
        "Chernozhukov Victor causal inference machine learning",
        "DoubleML Neyman orthogonal score",
    ],
    "causal-forest": [
        "Athey Wager causal forest heterogeneous",
        "Athey Imbens recursive partitioning causal effects",
        "Wager Athey estimation inference heterogeneous",
    ],
    "structural-equation-modeling": [
        "Bollen structural equation modeling",
        "Rosseel lavaan structural equation modeling R",
        "Kline principles practice structural equation modeling",
        "Muthén Muthén Mplus latent variable",
    ],
}


if __name__ == "__main__":
    import asyncio

    async def main():
        orchestrator = OrchestratorAgent()

        # 注册所有技能
        orchestrator.register_skills_from_config(SKILL_CALIBRATION_CONFIG)

        # 运行单个技能校准
        # result = await orchestrator.run_calibration("estimator-did")

        # 或运行所有技能
        # results = await orchestrator.run_all_calibrations()

        print(json.dumps(orchestrator.get_status(), indent=2, ensure_ascii=False))

    asyncio.run(main())

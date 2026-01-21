#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Full Calibration - 完整校准流程主脚本

实现六阶段校准流程:
1. 文献库构建 (LiteratureAgent)
2. 内容深度提取 (ExtractorAgent)
3. 组件级差距分析 (GapAnalyzer)
4. 交叉验证 (FormulaAgent + CitationAgent)
5. 质量门控 (ValidatorAgent)
6. 自动更新 (UpdaterAgent)

使用方法:
    # 校准所有技能
    python run_full_calibration.py --all --parallel 5

    # 校准单个技能
    python run_full_calibration.py --skill estimator-did

    # 按优先级校准
    python run_full_calibration.py --priority 1

    # 生成报告
    python run_full_calibration.py --report-only
"""

import sys
import os
import asyncio
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

# Windows 编码修复
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# 导入配置
from config.calibration_config import (
    SKILL_CALIBRATION_CONFIG,
    SkillConfig,
    Priority,
    SkillCategory,
    PARALLEL_CONFIG,
    get_skills_by_priority,
    get_skills_by_category,
    get_calibration_batches,
    list_all_skills,
)

# 导入 Agents
from agents.base import CALIBRATION_DIR
from agents.literature_agent import LiteratureAgent
from agents.extractor_agent import ExtractorAgent
from agents.gap_analyzer import GapAnalyzer, GapAnalysisInput, GapAnalysisOutput
from agents.formula_agent import FormulaAgent, FormulaInput
from agents.citation_agent import CitationAgent, CitationInput
from agents.validator_agent import ValidatorAgent, ValidationInput
from agents.updater_agent import UpdaterAgent, UpdateInput


# ============================================================================
# 数据结构
# ============================================================================

class CalibrationPhase(Enum):
    """校准阶段"""
    LITERATURE = "literature"
    EXTRACTION = "extraction"
    GAP_ANALYSIS = "gap_analysis"
    VALIDATION = "validation"
    QUALITY_GATE = "quality_gate"
    UPDATE = "update"


@dataclass
class SkillCalibrationResult:
    """单个技能的校准结果"""
    skill_name: str
    phases_completed: List[CalibrationPhase] = field(default_factory=list)
    papers_found: int = 0
    papers_extracted: int = 0
    gaps_identified: int = 0
    formula_score: float = 0.0
    citation_score: float = 0.0
    overall_score: float = 0.0
    passed: bool = False
    updates_generated: int = 0
    updates_applied: int = 0
    errors: List[str] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class CalibrationSummary:
    """校准汇总"""
    total_skills: int = 0
    skills_passed: int = 0
    skills_failed: int = 0
    total_papers: int = 0
    total_gaps: int = 0
    avg_score: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: Dict[str, SkillCalibrationResult] = field(default_factory=dict)


# ============================================================================
# 主校准器
# ============================================================================

class FullCalibrationOrchestrator:
    """完整校准流程协调器"""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        parallel: int = 3,
        auto_update: bool = False,
        verbose: bool = True
    ):
        self.config = config or {}
        self.parallel = parallel
        self.auto_update = auto_update
        self.verbose = verbose

        # 初始化输出目录
        self.output_dir = CALIBRATION_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化 Agents
        self.literature_agent = LiteratureAgent(min_citations=100)
        self.extractor_agent = ExtractorAgent()
        self.gap_analyzer = GapAnalyzer()
        self.formula_agent = FormulaAgent()
        self.citation_agent = CitationAgent()
        self.validator_agent = ValidatorAgent()
        self.updater_agent = UpdaterAgent()

        # 结果存储
        self.results: Dict[str, SkillCalibrationResult] = {}

    def log(self, message: str, level: str = "INFO"):
        """日志输出"""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}")

    async def calibrate_skill(self, skill_name: str) -> SkillCalibrationResult:
        """
        对单个技能执行完整校准

        Parameters
        ----------
        skill_name : str
            技能名称

        Returns
        -------
        SkillCalibrationResult
            校准结果
        """
        result = SkillCalibrationResult(skill_name=skill_name)
        result.started_at = datetime.now()

        config = SKILL_CALIBRATION_CONFIG.get(skill_name)
        if not config:
            result.errors.append(f"未找到技能配置: {skill_name}")
            return result

        self.log(f"开始校准: {skill_name}", "INFO")
        print(f"\n{'='*60}")
        print(f"  技能: {skill_name}")
        print(f"  类别: {config.category.value}")
        print(f"  优先级: P{config.priority.value}")
        print(f"{'='*60}")

        try:
            # ═══════════════════════════════════════════════════════════
            # Phase 1: 文献检索
            # ═══════════════════════════════════════════════════════════
            self.log("Phase 1: 文献检索...", "INFO")
            papers = await self._phase1_literature(skill_name, config)
            result.papers_found = len(papers)
            result.phases_completed.append(CalibrationPhase.LITERATURE)
            self.log(f"  找到 {len(papers)} 篇论文", "INFO")

            if not papers:
                self.log("  无论文，跳过后续阶段", "WARNING")
                result.completed_at = datetime.now()
                return result

            # ═══════════════════════════════════════════════════════════
            # Phase 2: 内容提取
            # ═══════════════════════════════════════════════════════════
            self.log("Phase 2: 内容提取...", "INFO")
            extracted = await self._phase2_extraction(papers, skill_name)
            result.papers_extracted = len(extracted)
            result.phases_completed.append(CalibrationPhase.EXTRACTION)
            self.log(f"  提取了 {len(extracted)} 篇论文内容", "INFO")

            # ═══════════════════════════════════════════════════════════
            # Phase 3: 差距分析
            # ═══════════════════════════════════════════════════════════
            self.log("Phase 3: 差距分析...", "INFO")
            gap_results = await self._phase3_gap_analysis(skill_name, extracted, config)
            total_gaps = sum(len(g.gaps) for g in gap_results.values())
            result.gaps_identified = total_gaps
            result.phases_completed.append(CalibrationPhase.GAP_ANALYSIS)
            self.log(f"  识别了 {total_gaps} 个差距", "INFO")

            # ═══════════════════════════════════════════════════════════
            # Phase 4: 交叉验证
            # ═══════════════════════════════════════════════════════════
            self.log("Phase 4: 交叉验证...", "INFO")
            formula_result, citation_result = await self._phase4_validation(
                skill_name, papers, extracted
            )
            result.formula_score = formula_result.consistency_score if formula_result else 1.0
            result.citation_score = citation_result.coverage_score if citation_result else 1.0
            result.phases_completed.append(CalibrationPhase.VALIDATION)
            self.log(f"  公式一致性: {result.formula_score:.1%}", "INFO")
            self.log(f"  引用覆盖率: {result.citation_score:.1%}", "INFO")

            # ═══════════════════════════════════════════════════════════
            # Phase 5: 质量门控
            # ═══════════════════════════════════════════════════════════
            self.log("Phase 5: 质量门控...", "INFO")
            # 从配置中获取技能特定的必需组件
            required_components = [c.value for c in config.components] if config.components else None
            validation_result = await self._phase5_quality_gate(
                skill_name, gap_results, formula_result, citation_result, len(papers),
                required_components
            )
            result.overall_score = validation_result.final_score
            result.passed = validation_result.passed
            result.phases_completed.append(CalibrationPhase.QUALITY_GATE)
            status = "✅ 通过" if result.passed else "❌ 未通过"
            self.log(f"  综合得分: {result.overall_score:.1%} {status}", "INFO")

            # ═══════════════════════════════════════════════════════════
            # Phase 6: 自动更新 (如果启用)
            # ═══════════════════════════════════════════════════════════
            if self.auto_update and total_gaps > 0:
                self.log("Phase 6: 生成更新...", "INFO")
                update_result = await self._phase6_update(skill_name, gap_results)
                result.updates_generated = update_result.patches_generated
                result.updates_applied = update_result.patches_applied
                result.phases_completed.append(CalibrationPhase.UPDATE)
                self.log(f"  生成 {result.updates_generated} 个补丁", "INFO")
            else:
                self.log("Phase 6: 跳过 (auto_update=False 或无差距)", "INFO")

            # 保存结果
            await self._save_skill_results(skill_name, result, gap_results, validation_result)

        except Exception as e:
            result.errors.append(str(e))
            self.log(f"校准失败: {e}", "ERROR")
            import traceback
            traceback.print_exc()

        result.completed_at = datetime.now()
        self.results[skill_name] = result

        return result

    async def _phase1_literature(
        self,
        skill_name: str,
        config: SkillConfig
    ) -> List[Dict[str, Any]]:
        """Phase 1: 文献检索"""
        all_papers = []
        seen_ids = set()

        for component, query_config in config.queries.items():
            for query in query_config.queries:
                papers = await self.literature_agent.search(
                    query,
                    limit=5
                )
                for paper in papers:
                    paper_id = paper.get('paper_id')
                    if paper_id and paper_id not in seen_ids:
                        seen_ids.add(paper_id)
                        paper['skill_name'] = skill_name
                        paper['component'] = component
                        all_papers.append(paper)

        # 按引用数排序，取前 N 篇
        all_papers.sort(key=lambda x: x.get('citations', 0), reverse=True)
        return all_papers[:15]

    async def _phase2_extraction(
        self,
        papers: List[Dict[str, Any]],
        skill_name: str
    ) -> List[Dict[str, Any]]:
        """Phase 2: 内容提取"""
        extracted = []

        for paper in papers[:10]:  # 限制数量
            try:
                content = await self.extractor_agent.extract(paper)
                if content:
                    extracted.append({
                        'paper_id': paper.get('paper_id'),
                        'title': content.title,
                        'year': content.year,
                        'venue': content.venue,
                        'citations': content.citations,
                        'abstract': content.abstract,
                        'methodology': content.methodology,
                        'estimation': content.estimation,
                        'identification': content.identification,
                        'assumptions': content.assumptions,
                        'formulas': content.formulas,
                    })
            except Exception as e:
                self.log(f"  提取失败 {paper.get('title', '')[:30]}: {e}", "WARNING")

        return extracted

    async def _phase3_gap_analysis(
        self,
        skill_name: str,
        extracted: List[Dict[str, Any]],
        config: SkillConfig
    ) -> Dict[str, GapAnalysisOutput]:
        """Phase 3: 差距分析"""
        results = {}

        for component in config.components:
            component_name = component.value
            input_data = GapAnalysisInput(
                skill_name=skill_name,
                component=component_name,
                paper_contents=extracted,
                existing_doc=""
            )

            try:
                output = await self.gap_analyzer.run(input_data)
                results[component_name] = output
            except Exception as e:
                self.log(f"  分析 {component_name} 失败: {e}", "WARNING")

        return results

    async def _phase4_validation(
        self,
        skill_name: str,
        papers: List[Dict[str, Any]],
        extracted: List[Dict[str, Any]]
    ):
        """Phase 4: 交叉验证"""
        # 公式验证
        all_formulas = []
        for content in extracted:
            all_formulas.extend(content.get('formulas', []))

        formula_result = None
        if all_formulas:
            formula_input = FormulaInput(
                skill_name=skill_name,
                paper_formulas=all_formulas[:20],
                paper_title="Combined Papers"
            )
            try:
                formula_result = await self.formula_agent.run(formula_input)
            except Exception as e:
                self.log(f"  公式验证失败: {e}", "WARNING")

        # 引用验证
        from agents.base import PaperInfo
        expected_papers = [
            PaperInfo.from_dict(p) for p in papers
        ]

        citation_result = None
        citation_input = CitationInput(
            skill_name=skill_name,
            expected_citations=expected_papers
        )
        try:
            citation_result = await self.citation_agent.run(citation_input)
        except Exception as e:
            self.log(f"  引用验证失败: {e}", "WARNING")

        return formula_result, citation_result

    async def _phase5_quality_gate(
        self,
        skill_name: str,
        gap_results: Dict[str, GapAnalysisOutput],
        formula_result,
        citation_result,
        papers_count: int,
        required_components: list = None
    ):
        """Phase 5: 质量门控"""
        validation_input = ValidationInput(
            skill_name=skill_name,
            gap_results=gap_results,
            formula_result=formula_result,
            citation_result=citation_result,
            papers_count=papers_count,
            required_components=required_components
        )

        return await self.validator_agent.run(validation_input)

    async def _phase6_update(
        self,
        skill_name: str,
        gap_results: Dict[str, GapAnalysisOutput]
    ):
        """Phase 6: 生成更新"""
        update_input = UpdateInput(
            skill_name=skill_name,
            gap_results=gap_results,
            auto_apply=False,  # 生成补丁但不自动应用
            backup=True
        )

        return await self.updater_agent.run(update_input)

    async def _save_skill_results(
        self,
        skill_name: str,
        result: SkillCalibrationResult,
        gap_results: Dict[str, GapAnalysisOutput],
        validation_result
    ):
        """保存技能校准结果"""
        skill_dir = self.output_dir / "results" / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)

        # 保存结果 JSON
        result_file = skill_dir / "calibration_result.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                "skill_name": result.skill_name,
                "phases_completed": [p.value for p in result.phases_completed],
                "papers_found": result.papers_found,
                "papers_extracted": result.papers_extracted,
                "gaps_identified": result.gaps_identified,
                "formula_score": result.formula_score,
                "citation_score": result.citation_score,
                "overall_score": result.overall_score,
                "passed": result.passed,
                "updates_generated": result.updates_generated,
                "errors": result.errors,
                "started_at": result.started_at.isoformat() if result.started_at else None,
                "completed_at": result.completed_at.isoformat() if result.completed_at else None,
            }, f, ensure_ascii=False, indent=2)

        # 保存验证报告
        if validation_result:
            report_file = skill_dir / "validation_report.md"
            report_file.write_text(validation_result.report, encoding='utf-8')

        # 保存差距分析
        self.gap_analyzer.save_results(skill_name, gap_results, skill_dir / "gaps")

    async def calibrate_batch(self, skill_names: List[str]) -> List[SkillCalibrationResult]:
        """批量校准"""
        results = []

        for skill_name in skill_names:
            result = await self.calibrate_skill(skill_name)
            results.append(result)

        return results

    async def calibrate_all(
        self,
        priority: Optional[int] = None,
        category: Optional[str] = None
    ) -> CalibrationSummary:
        """
        校准所有技能

        Parameters
        ----------
        priority : int, optional
            只校准指定优先级的技能
        category : str, optional
            只校准指定类别的技能

        Returns
        -------
        CalibrationSummary
            校准汇总
        """
        summary = CalibrationSummary()
        summary.started_at = datetime.now()

        # 确定技能列表
        if priority:
            skill_names = get_skills_by_priority(Priority(priority))
        elif category:
            skill_names = get_skills_by_category(SkillCategory(category))
        else:
            skill_names = list_all_skills()

        summary.total_skills = len(skill_names)

        self.log(f"开始校准 {len(skill_names)} 个技能", "INFO")
        print(f"\n{'#'*60}")
        print(f"#  完整校准流程")
        print(f"#  技能数: {len(skill_names)}")
        print(f"#  并行度: {self.parallel}")
        print(f"{'#'*60}")

        # 分批执行
        batches = [
            skill_names[i:i + self.parallel]
            for i in range(0, len(skill_names), self.parallel)
        ]

        for batch_idx, batch in enumerate(batches, 1):
            self.log(f"批次 {batch_idx}/{len(batches)}: {batch}", "INFO")

            # 顺序执行 (可改为并行)
            for skill_name in batch:
                result = await self.calibrate_skill(skill_name)
                summary.results[skill_name] = result

                if result.passed:
                    summary.skills_passed += 1
                else:
                    summary.skills_failed += 1

                summary.total_papers += result.papers_found
                summary.total_gaps += result.gaps_identified

        # 计算平均分
        scores = [r.overall_score for r in summary.results.values() if r.overall_score > 0]
        summary.avg_score = sum(scores) / len(scores) if scores else 0.0

        summary.completed_at = datetime.now()

        # 生成汇总报告
        self._generate_summary_report(summary)

        return summary

    def _generate_summary_report(self, summary: CalibrationSummary):
        """生成汇总报告"""
        report_path = self.output_dir / "reports"
        report_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d")
        report_file = report_path / f"CALIBRATION_REPORT_{timestamp}.md"

        duration = (summary.completed_at - summary.started_at).total_seconds() if summary.started_at and summary.completed_at else 0

        report = f"""# 校准报告

> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> 耗时: {duration:.1f} 秒

## 概览

| 指标 | 值 |
|------|-----|
| 总技能数 | {summary.total_skills} |
| 通过数 | {summary.skills_passed} ✅ |
| 未通过数 | {summary.skills_failed} ❌ |
| 处理论文数 | {summary.total_papers} |
| 识别差距数 | {summary.total_gaps} |
| 平均得分 | {summary.avg_score:.1%} |

## 各技能详情

| 技能 | 论文 | 差距 | 得分 | 状态 |
|------|------|------|------|------|
"""
        for skill_name, result in sorted(summary.results.items()):
            status = "✅" if result.passed else "❌"
            report += f"| {skill_name} | {result.papers_found} | {result.gaps_identified} | {result.overall_score:.1%} | {status} |\n"

        report += """

## 下一步行动

"""
        # 列出需要关注的技能
        failed_skills = [
            (name, r) for name, r in summary.results.items()
            if not r.passed
        ]
        if failed_skills:
            report += "### 需要改进的技能\n\n"
            for name, result in failed_skills:
                report += f"- **{name}** (得分: {result.overall_score:.1%})\n"
                for error in result.errors[:3]:
                    report += f"  - {error}\n"

        report_file.write_text(report, encoding='utf-8')
        self.log(f"报告已保存: {report_file}", "INFO")

        # 同时打印摘要
        print(f"\n{'='*60}")
        print("校准完成!")
        print(f"{'='*60}")
        print(f"  通过: {summary.skills_passed}")
        print(f"  未通过: {summary.skills_failed}")
        print(f"  平均得分: {summary.avg_score:.1%}")
        print(f"  报告: {report_file}")


# ============================================================================
# CLI 入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="完整校准流程脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 校准所有技能
  python run_full_calibration.py --all --parallel 5

  # 校准单个技能
  python run_full_calibration.py --skill estimator-did

  # 按优先级校准 (P1 = 最高优先级)
  python run_full_calibration.py --priority 1

  # 按类别校准
  python run_full_calibration.py --category classic-methods

  # 仅生成报告
  python run_full_calibration.py --report-only
        """
    )

    parser.add_argument(
        "--skill",
        type=str,
        help="校准单个技能"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="校准所有技能"
    )
    parser.add_argument(
        "--priority",
        type=int,
        choices=[1, 2, 3, 4],
        help="按优先级校准 (1=最高)"
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=["classic-methods", "causal-ml", "ml-foundation", "infrastructure"],
        help="按类别校准"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=3,
        help="并行度 (默认: 3)"
    )
    parser.add_argument(
        "--auto-update",
        action="store_true",
        help="自动应用更新"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="仅生成报告 (不执行校准)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有技能"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="安静模式"
    )

    args = parser.parse_args()

    if args.list:
        print("技能列表:")
        for skill_name in list_all_skills():
            config = SKILL_CALIBRATION_CONFIG[skill_name]
            print(f"  [{config.priority.value}] {skill_name} ({config.category.value})")
        return

    if args.report_only:
        print("生成报告模式 - 暂未实现")
        return

    # 创建协调器
    orchestrator = FullCalibrationOrchestrator(
        parallel=args.parallel,
        auto_update=args.auto_update,
        verbose=not args.quiet
    )

    # 执行校准
    async def run():
        if args.skill:
            await orchestrator.calibrate_skill(args.skill)
        elif args.all:
            await orchestrator.calibrate_all()
        elif args.priority:
            await orchestrator.calibrate_all(priority=args.priority)
        elif args.category:
            await orchestrator.calibrate_all(category=args.category)
        else:
            parser.print_help()

    asyncio.run(run())


if __name__ == "__main__":
    main()

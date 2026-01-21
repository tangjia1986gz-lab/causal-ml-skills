#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ValidatorAgent - 质量验证智能体

职责:
1. 实现四级质量门控
2. 计算综合校准得分
3. 判断通过/失败
4. 生成质量报告
"""

import sys
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

# Windows 编码修复
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from .base import (
    BaseAgent, CalibrationScore, ValidationResult,
    PROJECT_ROOT, CALIBRATION_DIR,
    QUALITY_GATE_THRESHOLDS, QUALITY_GATE_WEIGHTS
)
from .gap_analyzer import GapAnalysisOutput
from .formula_agent import FormulaOutput
from .citation_agent import CitationOutput


@dataclass
class ValidationInput:
    """验证输入"""
    skill_name: str
    gap_results: Dict[str, GapAnalysisOutput]  # component -> gaps
    formula_result: Optional[FormulaOutput] = None
    citation_result: Optional[CitationOutput] = None
    papers_count: int = 0
    required_components: Optional[List[str]] = None  # 技能特定的必需组件列表


@dataclass
class ValidationOutput:
    """验证输出"""
    skill_name: str
    passed: bool
    final_score: float
    gate_results: Dict[str, ValidationResult]
    calibration_score: CalibrationScore
    recommendations: List[str]
    report: str


class ValidatorAgent(BaseAgent[ValidationInput, ValidationOutput]):
    """
    质量验证智能体

    实现四级质量门控:
    1. 文献覆盖度 (Literature Coverage)
    2. 内容完整性 (Content Completeness)
    3. 公式一致性 (Formula Consistency)
    4. 引用有效性 (Citation Validity)
    """

    # 必需的组件
    REQUIRED_COMPONENTS = [
        "identification_assumptions",
        "estimation_methods",
        "diagnostic_tests",
    ]

    # 可选但推荐的组件
    RECOMMENDED_COMPONENTS = [
        "reporting_standards",
        "common_errors",
    ]

    # 每个组件的必需元素
    REQUIRED_ELEMENTS = {
        "identification_assumptions": [
            "formal_definition",
            "intuition",
            "testability",
        ],
        "estimation_methods": [
            "formula",
            "algorithm_steps",
            "standard_errors",
        ],
        "diagnostic_tests": [
            "test_statistic",
            "null_hypothesis",
            "interpretation",
        ],
        "reporting_standards": [
            "required_table_elements",
        ],
        "common_errors": [
            "error_description",
            "correct_approach",
        ],
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("ValidatorAgent", config)
        self.thresholds = config.get("thresholds", QUALITY_GATE_THRESHOLDS) if config else QUALITY_GATE_THRESHOLDS
        self.weights = config.get("weights", QUALITY_GATE_WEIGHTS) if config else QUALITY_GATE_WEIGHTS

    def _gate1_literature_coverage(
        self,
        gap_results: Dict[str, GapAnalysisOutput],
        papers_count: int
    ) -> ValidationResult:
        """
        Gate 1: 文献覆盖度检查

        检查每个组件是否有足够数量的高引论文支持
        """
        issues = []
        details = {}

        # 计算各组件的覆盖度
        component_scores = {}
        for component, output in gap_results.items():
            score = output.coverage_score
            component_scores[component] = score

            if score < 0.6:
                issues.append(f"{component}: 覆盖度过低 ({score:.1%})")
            elif score < 0.8:
                issues.append(f"{component}: 覆盖度需改进 ({score:.1%})")

        # 检查论文数量
        min_papers = 3
        if papers_count < min_papers:
            issues.append(f"论文数量不足: {papers_count} < {min_papers}")

        # 计算总体得分
        if component_scores:
            avg_coverage = sum(component_scores.values()) / len(component_scores)
        else:
            avg_coverage = 0.0

        # 论文数量惩罚
        papers_penalty = min(1.0, papers_count / min_papers)
        final_score = avg_coverage * papers_penalty

        threshold = self.thresholds.get("literature_coverage", 0.8)

        details = {
            "component_scores": component_scores,
            "papers_count": papers_count,
            "avg_coverage": avg_coverage,
        }

        return ValidationResult(
            gate_name="literature_coverage",
            passed=final_score >= threshold,
            score=final_score,
            threshold=threshold,
            issues=issues,
            details=details
        )

    def _gate2_content_completeness(
        self,
        gap_results: Dict[str, GapAnalysisOutput],
        required_components: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Gate 2: 内容完整性检查

        检查所有必需元素是否存在

        Parameters
        ----------
        gap_results : Dict[str, GapAnalysisOutput]
            差距分析结果
        required_components : Optional[List[str]]
            技能特定的必需组件列表，如果为 None 则使用默认的 REQUIRED_COMPONENTS
        """
        issues = []
        details = {}

        # 使用技能特定的组件列表，如果未提供则使用默认
        required = required_components if required_components else self.REQUIRED_COMPONENTS

        # 如果技能没有配置任何组件，直接通过
        if not required:
            return ValidationResult(
                gate_name="content_completeness",
                passed=True,
                score=1.0,
                threshold=self.thresholds.get("content_completeness", 1.0),
                issues=["无必需组件要求 (跳过)"],
                details={"skipped": True, "reason": "no_required_components"}
            )

        # 检查必需组件
        present_components = set(gap_results.keys())
        missing_required = set(required) - present_components

        if missing_required:
            for comp in missing_required:
                issues.append(f"缺少必需组件: {comp}")

        # 检查关键差距
        critical_gaps = []
        major_gaps = []

        for component, output in gap_results.items():
            for gap in output.gaps:
                if gap.severity == "critical":
                    critical_gaps.append((component, gap))
                elif gap.severity == "major":
                    major_gaps.append((component, gap))

        # 计算得分
        required_met = len(required) - len(missing_required)
        required_score = required_met / len(required) if required else 1.0

        # 关键差距惩罚
        critical_penalty = len(critical_gaps) * 0.2
        major_penalty = len(major_gaps) * 0.05

        final_score = max(0, required_score - critical_penalty - major_penalty)

        if critical_gaps:
            for comp, gap in critical_gaps[:5]:
                issues.append(f"[CRITICAL] {comp}: {gap.description[:50]}")

        if major_gaps:
            for comp, gap in major_gaps[:5]:
                issues.append(f"[MAJOR] {comp}: {gap.description[:50]}")

        threshold = self.thresholds.get("content_completeness", 1.0)

        details = {
            "present_components": list(present_components),
            "missing_required": list(missing_required),
            "critical_gaps_count": len(critical_gaps),
            "major_gaps_count": len(major_gaps),
        }

        return ValidationResult(
            gate_name="content_completeness",
            passed=final_score >= threshold and not missing_required,
            score=final_score,
            threshold=threshold,
            issues=issues,
            details=details
        )

    def _gate3_formula_consistency(
        self,
        formula_result: Optional[FormulaOutput]
    ) -> ValidationResult:
        """
        Gate 3: 公式一致性检查

        检查技能文档公式与论文公式的一致性
        """
        issues = []
        details = {}

        if formula_result is None:
            return ValidationResult(
                gate_name="formula_consistency",
                passed=True,
                score=1.0,
                threshold=self.thresholds.get("formula_consistency", 0.9),
                issues=["未进行公式验证 (跳过)"],
                details={"skipped": True}
            )

        score = formula_result.consistency_score

        # 收集问题
        if formula_result.missing_in_doc > 0:
            issues.append(f"文档缺失 {formula_result.missing_in_doc} 个论文公式")

        # 检查匹配质量
        low_quality_matches = [
            m for m in formula_result.matches
            if m.match_type not in ["exact", "similar"]
        ]
        if low_quality_matches:
            issues.append(f"有 {len(low_quality_matches)} 个公式匹配质量较低")

        threshold = self.thresholds.get("formula_consistency", 0.9)

        details = {
            "total_paper_formulas": formula_result.total_paper_formulas,
            "total_doc_formulas": formula_result.total_doc_formulas,
            "exact_matches": formula_result.exact_matches,
            "similar_matches": formula_result.similar_matches,
            "missing_in_doc": formula_result.missing_in_doc,
        }

        return ValidationResult(
            gate_name="formula_consistency",
            passed=score >= threshold,
            score=score,
            threshold=threshold,
            issues=issues,
            details=details
        )

    def _gate4_citation_validity(
        self,
        citation_result: Optional[CitationOutput]
    ) -> ValidationResult:
        """
        Gate 4: 引用有效性检查

        检查所有引用的格式正确性和可追溯性
        """
        issues = []
        details = {}

        if citation_result is None:
            return ValidationResult(
                gate_name="citation_validity",
                passed=True,
                score=1.0,
                threshold=self.thresholds.get("citation_validity", 1.0),
                issues=["未进行引用验证 (跳过)"],
                details={"skipped": True}
            )

        # 综合得分 (覆盖度和有效性的组合)
        score = 0.6 * citation_result.coverage_score + 0.4 * citation_result.validity_score

        # 收集问题
        if citation_result.invalid_citations > 0:
            issues.append(f"有 {citation_result.invalid_citations} 个引用格式无效")

        if citation_result.missing_citations > 0:
            issues.append(f"缺少 {citation_result.missing_citations} 个期望引用")

        # 检查核心引用
        if citation_result.coverage_score < 0.8:
            issues.append(f"核心引用覆盖不足: {citation_result.coverage_score:.1%}")

        threshold = self.thresholds.get("citation_validity", 1.0)

        details = {
            "total_citations": citation_result.total_citations_in_doc,
            "coverage_score": citation_result.coverage_score,
            "validity_score": citation_result.validity_score,
            "invalid_count": citation_result.invalid_citations,
            "missing_count": citation_result.missing_citations,
        }

        return ValidationResult(
            gate_name="citation_validity",
            passed=score >= threshold,
            score=score,
            threshold=threshold,
            issues=issues,
            details=details
        )

    def _calculate_final_score(
        self,
        gate_results: Dict[str, ValidationResult]
    ) -> Tuple[float, bool]:
        """
        计算综合校准得分

        Returns
        -------
        Tuple[float, bool]
            (最终得分, 是否通过)
        """
        weighted_score = 0.0

        for gate_name, result in gate_results.items():
            weight = self.weights.get(gate_name, 0.25)
            weighted_score += result.score * weight

        # 判断通过: 所有门控都通过 OR 综合得分 >= 0.85
        all_passed = all(r.passed for r in gate_results.values())
        score_passed = weighted_score >= 0.85

        passed = all_passed or score_passed

        return round(weighted_score, 4), passed

    def _generate_recommendations(
        self,
        gate_results: Dict[str, ValidationResult],
        gap_results: Dict[str, GapAnalysisOutput]
    ) -> List[str]:
        """生成改进建议"""
        recommendations = []

        # 基于门控结果生成建议
        for gate_name, result in gate_results.items():
            if not result.passed:
                if gate_name == "literature_coverage":
                    recommendations.append("扩展文献检索，增加高引论文覆盖")
                elif gate_name == "content_completeness":
                    recommendations.append("补充缺失的关键组件和内容")
                elif gate_name == "formula_consistency":
                    recommendations.append("校验并更新技能文档中的数学公式")
                elif gate_name == "citation_validity":
                    recommendations.append("完善引用格式，补充核心文献引用")

            for issue in result.issues[:3]:
                recommendations.append(f"修复: {issue}")

        # 基于差距分析生成建议
        for component, output in gap_results.items():
            critical_gaps = [g for g in output.gaps if g.severity == "critical"]
            if critical_gaps:
                recommendations.append(f"紧急: 修复 {component} 中的 {len(critical_gaps)} 个关键差距")

        return recommendations[:15]  # 限制数量

    def _generate_report(
        self,
        skill_name: str,
        gate_results: Dict[str, ValidationResult],
        final_score: float,
        passed: bool,
        recommendations: List[str]
    ) -> str:
        """生成质量验证报告"""
        overall_status = "✅ 通过" if passed else "❌ 未通过"

        report = f"""# {skill_name} 质量验证报告

> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 综合评估

| 指标 | 值 |
|------|-----|
| **最终得分** | **{final_score:.1%}** |
| **状态** | **{overall_status}** |

## 质量门控详情

"""
        # 各门控详情
        for gate_name, result in gate_results.items():
            status = "✅" if result.passed else "❌"
            report += f"""### Gate: {gate_name}

| 指标 | 值 |
|------|-----|
| 状态 | {status} |
| 得分 | {result.score:.1%} |
| 阈值 | {result.threshold:.1%} |

"""
            if result.issues:
                report += "**问题**:\n"
                for issue in result.issues[:5]:
                    report += f"- {issue}\n"
                report += "\n"

        # 建议
        report += "## 改进建议\n\n"
        for i, rec in enumerate(recommendations, 1):
            report += f"{i}. {rec}\n"

        # 门控得分可视化
        report += "\n## 得分分布\n\n"
        report += "```\n"
        for gate_name, result in gate_results.items():
            bar_len = int(result.score * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            status = "✓" if result.passed else "✗"
            report += f"{gate_name:25} [{bar}] {result.score:.1%} {status}\n"
        report += "```\n"

        return report

    async def process(self, input_data: ValidationInput) -> ValidationOutput:
        """
        执行质量验证

        Parameters
        ----------
        input_data : ValidationInput
            验证输入

        Returns
        -------
        ValidationOutput
            验证结果
        """
        skill_name = input_data.skill_name

        self.logger.info(f"开始验证 {skill_name}")

        # 执行四个质量门控
        gate1 = self._gate1_literature_coverage(
            input_data.gap_results,
            input_data.papers_count
        )
        self.logger.info(f"Gate 1 (文献覆盖): {gate1.score:.1%} {'✓' if gate1.passed else '✗'}")

        gate2 = self._gate2_content_completeness(
            input_data.gap_results,
            input_data.required_components
        )
        self.logger.info(f"Gate 2 (内容完整): {gate2.score:.1%} {'✓' if gate2.passed else '✗'}")

        gate3 = self._gate3_formula_consistency(input_data.formula_result)
        self.logger.info(f"Gate 3 (公式一致): {gate3.score:.1%} {'✓' if gate3.passed else '✗'}")

        gate4 = self._gate4_citation_validity(input_data.citation_result)
        self.logger.info(f"Gate 4 (引用有效): {gate4.score:.1%} {'✓' if gate4.passed else '✗'}")

        gate_results = {
            "literature_coverage": gate1,
            "content_completeness": gate2,
            "formula_consistency": gate3,
            "citation_validity": gate4,
        }

        # 计算最终得分
        final_score, passed = self._calculate_final_score(gate_results)

        # 构建校准得分对象
        calibration_score = CalibrationScore(
            skill_name=skill_name,
            component_scores={
                name: result.score for name, result in gate_results.items()
            },
            final_score=final_score,
            passed=passed,
            timestamp=datetime.now(),
            details={
                "gates_passed": sum(1 for r in gate_results.values() if r.passed),
                "total_gates": len(gate_results),
            }
        )

        # 生成建议
        recommendations = self._generate_recommendations(
            gate_results, input_data.gap_results
        )

        # 生成报告
        report = self._generate_report(
            skill_name, gate_results, final_score, passed, recommendations
        )

        self.logger.info(f"验证完成: {final_score:.1%} {'通过' if passed else '未通过'}")

        return ValidationOutput(
            skill_name=skill_name,
            passed=passed,
            final_score=final_score,
            gate_results=gate_results,
            calibration_score=calibration_score,
            recommendations=recommendations,
            report=report
        )

    def save_results(
        self,
        output: ValidationOutput,
        output_dir: Optional[Path] = None
    ) -> Path:
        """保存验证结果"""
        output_dir = output_dir or CALIBRATION_DIR / "scores"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存报告
        report_path = output_dir / f"{output.skill_name}_validation.md"
        report_path.write_text(output.report, encoding='utf-8')

        # 保存得分 JSON
        score_path = output_dir / f"{output.skill_name}_score.json"
        with open(score_path, 'w', encoding='utf-8') as f:
            json.dump({
                "skill_name": output.skill_name,
                "passed": output.passed,
                "final_score": output.final_score,
                "calibration_score": output.calibration_score.to_dict(),
                "gate_results": {
                    name: result.to_dict()
                    for name, result in output.gate_results.items()
                },
                "recommendations": output.recommendations,
            }, f, ensure_ascii=False, indent=2)

        self.logger.info(f"结果已保存到: {output_dir}")
        return output_dir


if __name__ == "__main__":
    from .gap_analyzer import GapAnalysisOutput, GapInfo

    async def test():
        validator = ValidatorAgent()

        # 模拟输入
        test_input = ValidationInput(
            skill_name="estimator-did",
            gap_results={
                "identification_assumptions": GapAnalysisOutput(
                    skill_name="estimator-did",
                    component="identification_assumptions",
                    gaps=[
                        GapInfo(
                            gap_id="1",
                            category="assumption",
                            severity="minor",
                            description="测试差距",
                            source_paper="Test",
                            source_section="test"
                        )
                    ],
                    coverage_score=0.85,
                    papers_analyzed=5,
                    summary=""
                ),
                "estimation_methods": GapAnalysisOutput(
                    skill_name="estimator-did",
                    component="estimation_methods",
                    gaps=[],
                    coverage_score=0.9,
                    papers_analyzed=5,
                    summary=""
                ),
                "diagnostic_tests": GapAnalysisOutput(
                    skill_name="estimator-did",
                    component="diagnostic_tests",
                    gaps=[],
                    coverage_score=0.8,
                    papers_analyzed=5,
                    summary=""
                ),
            },
            papers_count=5
        )

        result = await validator.run(test_input)
        print(f"最终得分: {result.final_score:.1%}")
        print(f"通过: {result.passed}")
        print(f"\n{result.report}")

    asyncio.run(test())

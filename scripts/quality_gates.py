#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quality Gates - 质量门控模块

提供:
1. 四级质量门控检查
2. 覆盖矩阵生成
3. 引用完整性验证
4. 校准得分计算
5. CLI 接口
"""

import sys
import asyncio
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Windows 编码修复
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
SKILLS_DIR = PROJECT_ROOT / "skills"
CALIBRATION_DIR = PROJECT_ROOT / "calibration_v2"


# ============================================================================
# 数据结构
# ============================================================================

class GateStatus(Enum):
    """门控状态"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class GateResult:
    """门控检查结果"""
    gate_name: str
    status: GateStatus
    score: float
    threshold: float
    issues: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillQualityScore:
    """技能质量得分"""
    skill_name: str
    overall_score: float
    gate_results: Dict[str, GateResult]
    passed: bool
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CoverageMatrixEntry:
    """覆盖矩阵条目"""
    skill_name: str
    has_identification: bool
    has_estimation: bool
    has_diagnostics: bool
    has_reporting: bool
    has_errors: bool
    papers_count: int
    citations_avg: float


# ============================================================================
# 质量门控配置
# ============================================================================

DEFAULT_THRESHOLDS = {
    "literature_coverage": 0.8,
    "content_completeness": 1.0,
    "formula_consistency": 0.9,
    "citation_validity": 1.0,
}

DEFAULT_WEIGHTS = {
    "literature_coverage": 0.3,
    "content_completeness": 0.3,
    "formula_consistency": 0.2,
    "citation_validity": 0.2,
}

PASS_THRESHOLD = 0.85


# ============================================================================
# 质量门控实现
# ============================================================================

class QualityGates:
    """质量门控检查器"""

    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.weights = weights or DEFAULT_WEIGHTS

    def check_literature_coverage(
        self,
        skill_name: str,
        papers_count: int,
        component_coverage: Dict[str, float]
    ) -> GateResult:
        """
        Gate 1: 文献覆盖度检查

        检查:
        - 论文数量是否充足 (>=3)
        - 各组件是否有文献覆盖
        """
        issues = []
        details = {
            "papers_count": papers_count,
            "component_coverage": component_coverage,
        }

        # 论文数量检查
        min_papers = 3
        if papers_count < min_papers:
            issues.append(f"论文数量不足: {papers_count} < {min_papers}")

        # 组件覆盖检查
        low_coverage = []
        for comp, score in component_coverage.items():
            if score < 0.6:
                low_coverage.append(f"{comp}: {score:.1%}")
                issues.append(f"组件 {comp} 覆盖度过低: {score:.1%}")

        # 计算得分
        if component_coverage:
            avg_coverage = sum(component_coverage.values()) / len(component_coverage)
        else:
            avg_coverage = 0.0

        papers_penalty = min(1.0, papers_count / min_papers)
        score = avg_coverage * papers_penalty

        threshold = self.thresholds["literature_coverage"]
        status = GateStatus.PASSED if score >= threshold else GateStatus.FAILED
        if 0.6 <= score < threshold:
            status = GateStatus.WARNING

        details["low_coverage_components"] = low_coverage

        return GateResult(
            gate_name="literature_coverage",
            status=status,
            score=score,
            threshold=threshold,
            issues=issues,
            details=details
        )

    def check_content_completeness(
        self,
        skill_name: str,
        present_components: List[str],
        critical_gaps: int,
        major_gaps: int
    ) -> GateResult:
        """
        Gate 2: 内容完整性检查

        检查:
        - 必需组件是否存在
        - 是否有关键差距
        """
        issues = []
        required_components = [
            "identification_assumptions",
            "estimation_methods",
            "diagnostic_tests",
        ]

        # 组件检查
        missing = set(required_components) - set(present_components)
        if missing:
            for comp in missing:
                issues.append(f"缺少必需组件: {comp}")

        # 差距惩罚
        if critical_gaps > 0:
            issues.append(f"有 {critical_gaps} 个关键差距")
        if major_gaps > 0:
            issues.append(f"有 {major_gaps} 个重要差距")

        # 计算得分
        required_met = len(required_components) - len(missing)
        base_score = required_met / len(required_components)

        penalty = critical_gaps * 0.2 + major_gaps * 0.05
        score = max(0, base_score - penalty)

        threshold = self.thresholds["content_completeness"]
        status = GateStatus.PASSED if score >= threshold and not missing else GateStatus.FAILED
        if score >= 0.8 and missing:
            status = GateStatus.WARNING

        return GateResult(
            gate_name="content_completeness",
            status=status,
            score=score,
            threshold=threshold,
            issues=issues,
            details={
                "present_components": present_components,
                "missing_components": list(missing),
                "critical_gaps": critical_gaps,
                "major_gaps": major_gaps,
            }
        )

    def check_formula_consistency(
        self,
        skill_name: str,
        exact_matches: int,
        similar_matches: int,
        total_formulas: int
    ) -> GateResult:
        """
        Gate 3: 公式一致性检查

        检查:
        - 公式精确匹配率
        - 符号一致性
        """
        issues = []

        if total_formulas == 0:
            return GateResult(
                gate_name="formula_consistency",
                status=GateStatus.SKIPPED,
                score=1.0,
                threshold=self.thresholds["formula_consistency"],
                issues=["无公式需要验证"],
                details={"skipped": True}
            )

        # 计算匹配率
        matched = exact_matches + similar_matches * 0.7
        score = matched / total_formulas

        if exact_matches < total_formulas * 0.5:
            issues.append(f"精确匹配率较低: {exact_matches}/{total_formulas}")

        missing = total_formulas - exact_matches - similar_matches
        if missing > 0:
            issues.append(f"有 {missing} 个公式未匹配")

        threshold = self.thresholds["formula_consistency"]
        status = GateStatus.PASSED if score >= threshold else GateStatus.FAILED
        if 0.7 <= score < threshold:
            status = GateStatus.WARNING

        return GateResult(
            gate_name="formula_consistency",
            status=status,
            score=score,
            threshold=threshold,
            issues=issues,
            details={
                "exact_matches": exact_matches,
                "similar_matches": similar_matches,
                "total_formulas": total_formulas,
            }
        )

    def check_citation_validity(
        self,
        skill_name: str,
        total_citations: int,
        valid_citations: int,
        core_coverage: float
    ) -> GateResult:
        """
        Gate 4: 引用有效性检查

        检查:
        - 引用格式有效性
        - 核心引用覆盖
        """
        issues = []

        if total_citations == 0:
            return GateResult(
                gate_name="citation_validity",
                status=GateStatus.WARNING,
                score=0.5,
                threshold=self.thresholds["citation_validity"],
                issues=["文档中未找到引用"],
                details={"no_citations": True}
            )

        # 有效性得分
        validity_score = valid_citations / total_citations if total_citations > 0 else 0

        invalid = total_citations - valid_citations
        if invalid > 0:
            issues.append(f"有 {invalid} 个无效引用")

        if core_coverage < 0.8:
            issues.append(f"核心引用覆盖不足: {core_coverage:.1%}")

        # 综合得分
        score = 0.6 * core_coverage + 0.4 * validity_score

        threshold = self.thresholds["citation_validity"]
        status = GateStatus.PASSED if score >= threshold else GateStatus.FAILED
        if 0.7 <= score < threshold:
            status = GateStatus.WARNING

        return GateResult(
            gate_name="citation_validity",
            status=status,
            score=score,
            threshold=threshold,
            issues=issues,
            details={
                "total_citations": total_citations,
                "valid_citations": valid_citations,
                "core_coverage": core_coverage,
            }
        )

    def calculate_overall_score(
        self,
        gate_results: Dict[str, GateResult]
    ) -> Tuple[float, bool]:
        """计算综合得分和通过状态"""
        weighted_score = 0.0

        for gate_name, result in gate_results.items():
            weight = self.weights.get(gate_name, 0.25)
            weighted_score += result.score * weight

        # 判断通过
        all_passed = all(
            r.status in [GateStatus.PASSED, GateStatus.SKIPPED]
            for r in gate_results.values()
        )
        score_passed = weighted_score >= PASS_THRESHOLD

        passed = all_passed or score_passed

        return round(weighted_score, 4), passed

    def run_all_gates(
        self,
        skill_name: str,
        papers_count: int = 0,
        component_coverage: Optional[Dict[str, float]] = None,
        present_components: Optional[List[str]] = None,
        critical_gaps: int = 0,
        major_gaps: int = 0,
        exact_formula_matches: int = 0,
        similar_formula_matches: int = 0,
        total_formulas: int = 0,
        total_citations: int = 0,
        valid_citations: int = 0,
        core_citation_coverage: float = 0.0,
    ) -> SkillQualityScore:
        """运行所有质量门控"""
        component_coverage = component_coverage or {}
        present_components = present_components or []

        gate1 = self.check_literature_coverage(
            skill_name, papers_count, component_coverage
        )
        gate2 = self.check_content_completeness(
            skill_name, present_components, critical_gaps, major_gaps
        )
        gate3 = self.check_formula_consistency(
            skill_name, exact_formula_matches, similar_formula_matches, total_formulas
        )
        gate4 = self.check_citation_validity(
            skill_name, total_citations, valid_citations, core_citation_coverage
        )

        gate_results = {
            "literature_coverage": gate1,
            "content_completeness": gate2,
            "formula_consistency": gate3,
            "citation_validity": gate4,
        }

        overall_score, passed = self.calculate_overall_score(gate_results)

        return SkillQualityScore(
            skill_name=skill_name,
            overall_score=overall_score,
            gate_results=gate_results,
            passed=passed
        )


# ============================================================================
# 覆盖矩阵生成
# ============================================================================

def generate_coverage_matrix(
    calibration_results: Optional[Dict[str, Any]] = None
) -> List[CoverageMatrixEntry]:
    """
    生成技能覆盖矩阵

    Parameters
    ----------
    calibration_results : Dict
        校准结果数据

    Returns
    -------
    List[CoverageMatrixEntry]
        覆盖矩阵
    """
    matrix = []

    # 如果没有传入结果，扫描技能目录
    if calibration_results is None:
        for category_dir in SKILLS_DIR.iterdir():
            if not category_dir.is_dir():
                continue

            for skill_dir in category_dir.iterdir():
                if not skill_dir.is_dir():
                    continue

                refs_dir = skill_dir / "references"
                entry = CoverageMatrixEntry(
                    skill_name=skill_dir.name,
                    has_identification=(refs_dir / "identification_assumptions.md").exists(),
                    has_estimation=(refs_dir / "estimation_methods.md").exists(),
                    has_diagnostics=(refs_dir / "diagnostic_tests.md").exists(),
                    has_reporting=(refs_dir / "reporting_standards.md").exists(),
                    has_errors=(refs_dir / "common_errors.md").exists(),
                    papers_count=0,
                    citations_avg=0.0,
                )
                matrix.append(entry)

    else:
        for skill_name, result in calibration_results.items():
            entry = CoverageMatrixEntry(
                skill_name=skill_name,
                has_identification=result.get("identification", False),
                has_estimation=result.get("estimation", False),
                has_diagnostics=result.get("diagnostics", False),
                has_reporting=result.get("reporting", False),
                has_errors=result.get("errors", False),
                papers_count=result.get("papers_count", 0),
                citations_avg=result.get("citations_avg", 0.0),
            )
            matrix.append(entry)

    return matrix


def print_coverage_matrix(matrix: List[CoverageMatrixEntry]) -> str:
    """生成覆盖矩阵的 Markdown 表格"""
    output = """# 技能覆盖矩阵

| 技能 | 识别假设 | 估计方法 | 诊断测试 | 报告标准 | 常见错误 | 论文数 |
|------|:--------:|:--------:|:--------:|:--------:|:--------:|:------:|
"""
    for entry in sorted(matrix, key=lambda x: x.skill_name):
        id_status = "✅" if entry.has_identification else "❌"
        est_status = "✅" if entry.has_estimation else "❌"
        diag_status = "✅" if entry.has_diagnostics else "❌"
        rep_status = "✅" if entry.has_reporting else "❌"
        err_status = "✅" if entry.has_errors else "❌"

        output += f"| {entry.skill_name} | {id_status} | {est_status} | {diag_status} | {rep_status} | {err_status} | {entry.papers_count} |\n"

    return output


# ============================================================================
# 引用完整性验证
# ============================================================================

def validate_citations(skill_name: str) -> Dict[str, Any]:
    """
    验证技能的引用完整性

    Parameters
    ----------
    skill_name : str
        技能名称

    Returns
    -------
    Dict
        验证结果
    """
    from .config.calibration_config import SKILL_CALIBRATION_CONFIG

    config = SKILL_CALIBRATION_CONFIG.get(skill_name)
    if not config:
        return {"error": f"未找到技能配置: {skill_name}"}

    # 查找技能目录
    skill_path = None
    for category_dir in SKILLS_DIR.iterdir():
        candidate = category_dir / skill_name
        if candidate.exists():
            skill_path = candidate
            break

    if not skill_path:
        return {"error": f"未找到技能目录: {skill_name}"}

    # 读取所有文档内容
    all_content = ""
    for md_file in skill_path.rglob("*.md"):
        try:
            all_content += md_file.read_text(encoding='utf-8') + "\n"
        except Exception:
            pass

    # 检查核心引用
    core_refs = config.core_citations
    found_refs = []
    missing_refs = []

    for ref in core_refs:
        authors = ref.get("authors", [])
        year = ref.get("year", 0)

        # 简单搜索
        found = False
        for author in authors:
            if author.lower() in all_content.lower() and str(year) in all_content:
                found = True
                break

        if found:
            found_refs.append(ref)
        else:
            missing_refs.append(ref)

    coverage = len(found_refs) / len(core_refs) if core_refs else 1.0

    return {
        "skill_name": skill_name,
        "total_core_refs": len(core_refs),
        "found_refs": found_refs,
        "missing_refs": missing_refs,
        "coverage": coverage,
    }


# ============================================================================
# CLI 接口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="质量门控检查工具"
    )
    parser.add_argument(
        "--skill",
        type=str,
        help="检查单个技能"
    )
    parser.add_argument(
        "--coverage-matrix",
        action="store_true",
        help="生成覆盖矩阵"
    )
    parser.add_argument(
        "--validate-citations",
        action="store_true",
        help="验证引用完整性"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="检查所有技能"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="输出文件路径"
    )

    args = parser.parse_args()

    gates = QualityGates()

    if args.coverage_matrix:
        print("生成覆盖矩阵...")
        matrix = generate_coverage_matrix()
        output = print_coverage_matrix(matrix)
        print(output)

        if args.output:
            Path(args.output).write_text(output, encoding='utf-8')
            print(f"\n已保存到: {args.output}")

    elif args.validate_citations:
        if args.skill:
            print(f"验证 {args.skill} 引用...")
            result = validate_citations(args.skill)
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        elif args.all:
            from .config.calibration_config import list_all_skills

            results = {}
            for skill_name in list_all_skills():
                print(f"验证 {skill_name}...")
                results[skill_name] = validate_citations(skill_name)

            print("\n汇总:")
            for skill_name, result in results.items():
                coverage = result.get("coverage", 0)
                status = "✅" if coverage >= 0.8 else "⚠️" if coverage >= 0.5 else "❌"
                print(f"  {status} {skill_name}: {coverage:.1%}")

    elif args.skill:
        print(f"检查技能: {args.skill}")

        # 模拟数据 (实际使用时应从校准结果加载)
        score = gates.run_all_gates(
            skill_name=args.skill,
            papers_count=5,
            component_coverage={
                "identification_assumptions": 0.85,
                "estimation_methods": 0.90,
                "diagnostic_tests": 0.80,
            },
            present_components=[
                "identification_assumptions",
                "estimation_methods",
                "diagnostic_tests",
            ],
            critical_gaps=0,
            major_gaps=2,
            total_citations=10,
            valid_citations=9,
            core_citation_coverage=0.75,
        )

        print(f"\n{'='*50}")
        print(f"技能: {score.skill_name}")
        print(f"综合得分: {score.overall_score:.1%}")
        print(f"状态: {'✅ 通过' if score.passed else '❌ 未通过'}")
        print(f"{'='*50}")

        for gate_name, result in score.gate_results.items():
            status_symbol = {
                GateStatus.PASSED: "✅",
                GateStatus.FAILED: "❌",
                GateStatus.WARNING: "⚠️",
                GateStatus.SKIPPED: "⏭️",
            }[result.status]

            print(f"\n{status_symbol} {gate_name}")
            print(f"   得分: {result.score:.1%} (阈值: {result.threshold:.1%})")
            if result.issues:
                for issue in result.issues[:3]:
                    print(f"   - {issue}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

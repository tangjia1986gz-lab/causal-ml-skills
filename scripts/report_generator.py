#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Report Generator - 校准报告生成器

生成:
1. 综合校准报告 (CALIBRATION_REPORT_{date}.md)
2. 技能覆盖矩阵 (SKILL_COVERAGE_MATRIX.md)
3. 引用数据库 (CITATION_DATABASE.json)
4. 差距汇总报告
5. 进度仪表板
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

# Windows 编码修复
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
SKILLS_DIR = PROJECT_ROOT / "skills"
CALIBRATION_DIR = PROJECT_ROOT / "calibration_v2"
REPORTS_DIR = CALIBRATION_DIR / "reports"

# 确保目录存在
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class SkillMetrics:
    """技能指标"""
    skill_name: str
    category: str
    priority: int
    papers_count: int = 0
    gaps_count: int = 0
    coverage_score: float = 0.0
    formula_score: float = 0.0
    citation_score: float = 0.0
    overall_score: float = 0.0
    passed: bool = False
    components_present: List[str] = field(default_factory=list)
    critical_gaps: int = 0
    major_gaps: int = 0
    minor_gaps: int = 0


@dataclass
class ReportData:
    """报告数据"""
    generated_at: datetime
    total_skills: int = 0
    skills_passed: int = 0
    skills_failed: int = 0
    total_papers: int = 0
    total_gaps: int = 0
    avg_score: float = 0.0
    skill_metrics: Dict[str, SkillMetrics] = field(default_factory=dict)
    category_breakdown: Dict[str, Dict[str, int]] = field(default_factory=dict)
    priority_breakdown: Dict[int, Dict[str, int]] = field(default_factory=dict)


# ============================================================================
# 报告生成器
# ============================================================================

class ReportGenerator:
    """报告生成器"""

    def __init__(self, calibration_dir: Optional[Path] = None):
        self.calibration_dir = calibration_dir or CALIBRATION_DIR
        self.results_dir = self.calibration_dir / "results"
        self.data = ReportData(generated_at=datetime.now())

    def load_results(self) -> None:
        """加载所有校准结果"""
        if not self.results_dir.exists():
            print(f"结果目录不存在: {self.results_dir}")
            return

        for skill_dir in self.results_dir.iterdir():
            if not skill_dir.is_dir():
                continue

            result_file = skill_dir / "calibration_result.json"
            if not result_file.exists():
                continue

            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)

                skill_name = result.get("skill_name", skill_dir.name)

                # 加载差距数据
                gaps_dir = skill_dir / "gaps"
                critical = 0
                major = 0
                minor = 0

                if gaps_dir.exists():
                    for gap_file in gaps_dir.glob("*_gaps.json"):
                        with open(gap_file, 'r', encoding='utf-8') as f:
                            gap_data = json.load(f)
                            for gap in gap_data.get("gaps", []):
                                severity = gap.get("severity", "minor")
                                if severity == "critical":
                                    critical += 1
                                elif severity == "major":
                                    major += 1
                                else:
                                    minor += 1

                # 从配置获取类别和优先级
                from config.calibration_config import SKILL_CALIBRATION_CONFIG
                config = SKILL_CALIBRATION_CONFIG.get(skill_name)
                category = config.category.value if config else "unknown"
                priority = config.priority.value if config else 4

                metrics = SkillMetrics(
                    skill_name=skill_name,
                    category=category,
                    priority=priority,
                    papers_count=result.get("papers_found", 0),
                    gaps_count=result.get("gaps_identified", 0),
                    formula_score=result.get("formula_score", 0.0),
                    citation_score=result.get("citation_score", 0.0),
                    overall_score=result.get("overall_score", 0.0),
                    passed=result.get("passed", False),
                    critical_gaps=critical,
                    major_gaps=major,
                    minor_gaps=minor,
                )

                self.data.skill_metrics[skill_name] = metrics
                self.data.total_skills += 1
                self.data.total_papers += metrics.papers_count
                self.data.total_gaps += metrics.gaps_count

                if metrics.passed:
                    self.data.skills_passed += 1
                else:
                    self.data.skills_failed += 1

            except Exception as e:
                print(f"加载 {skill_dir.name} 失败: {e}")

        # 计算平均分
        if self.data.skill_metrics:
            scores = [m.overall_score for m in self.data.skill_metrics.values() if m.overall_score > 0]
            self.data.avg_score = sum(scores) / len(scores) if scores else 0.0

        # 按类别/优先级汇总
        self._calculate_breakdowns()

    def _calculate_breakdowns(self) -> None:
        """计算类别和优先级分布"""
        category_stats = defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0})
        priority_stats = defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0})

        for metrics in self.data.skill_metrics.values():
            # 类别
            category_stats[metrics.category]["total"] += 1
            if metrics.passed:
                category_stats[metrics.category]["passed"] += 1
            else:
                category_stats[metrics.category]["failed"] += 1

            # 优先级
            priority_stats[metrics.priority]["total"] += 1
            if metrics.passed:
                priority_stats[metrics.priority]["passed"] += 1
            else:
                priority_stats[metrics.priority]["failed"] += 1

        self.data.category_breakdown = dict(category_stats)
        self.data.priority_breakdown = dict(priority_stats)

    def generate_main_report(self) -> str:
        """生成主校准报告"""
        report = f"""# 系统性方法校准报告

> **生成时间**: {self.data.generated_at.strftime('%Y-%m-%d %H:%M:%S')}
> **框架版本**: v2.0

---

## 1. 执行摘要

本报告汇总了对 causal-ml-skills 所有技能的系统校准结果，确保每个方法论声明都有顶刊/高引文献依据。

### 1.1 关键指标

| 指标 | 值 |
|------|-----|
| **总技能数** | {self.data.total_skills} |
| **通过技能** | {self.data.skills_passed} ✅ |
| **未通过技能** | {self.data.skills_failed} ❌ |
| **通过率** | {self.data.skills_passed / max(self.data.total_skills, 1):.1%} |
| **平均得分** | {self.data.avg_score:.1%} |
| **处理论文数** | {self.data.total_papers} |
| **识别差距数** | {self.data.total_gaps} |

### 1.2 状态分布

```
通过 [{'█' * int(self.data.skills_passed / max(self.data.total_skills, 1) * 20)}{'░' * (20 - int(self.data.skills_passed / max(self.data.total_skills, 1) * 20))}] {self.data.skills_passed}/{self.data.total_skills}
```

---

## 2. 按类别分析

| 类别 | 总数 | 通过 | 未通过 | 通过率 |
|------|------|------|--------|--------|
"""
        for category, stats in sorted(self.data.category_breakdown.items()):
            total = stats["total"]
            passed = stats["passed"]
            rate = passed / max(total, 1)
            report += f"| {category} | {total} | {passed} | {stats['failed']} | {rate:.1%} |\n"

        report += """

---

## 3. 按优先级分析

| 优先级 | 总数 | 通过 | 未通过 | 通过率 |
|--------|------|------|--------|--------|
"""
        for priority in sorted(self.data.priority_breakdown.keys()):
            stats = self.data.priority_breakdown[priority]
            total = stats["total"]
            passed = stats["passed"]
            rate = passed / max(total, 1)
            priority_label = {1: "P1 (Critical)", 2: "P2 (High)", 3: "P3 (Medium)", 4: "P4 (Low)"}.get(priority, f"P{priority}")
            report += f"| {priority_label} | {total} | {passed} | {stats['failed']} | {rate:.1%} |\n"

        report += """

---

## 4. 各技能详情

"""
        # 按优先级排序
        sorted_skills = sorted(
            self.data.skill_metrics.values(),
            key=lambda x: (x.priority, -x.overall_score)
        )

        for metrics in sorted_skills:
            status = "✅" if metrics.passed else "❌"
            score_bar = "█" * int(metrics.overall_score * 10) + "░" * (10 - int(metrics.overall_score * 10))

            report += f"""### {status} {metrics.skill_name}

| 指标 | 值 |
|------|-----|
| 类别 | {metrics.category} |
| 优先级 | P{metrics.priority} |
| 论文数 | {metrics.papers_count} |
| 差距数 | {metrics.gaps_count} (🔴{metrics.critical_gaps} 🟡{metrics.major_gaps} 🟢{metrics.minor_gaps}) |
| 公式一致性 | {metrics.formula_score:.1%} |
| 引用覆盖 | {metrics.citation_score:.1%} |
| **综合得分** | [{score_bar}] {metrics.overall_score:.1%} |

"""

        report += """
---

## 5. 差距热力图

"""
        # 生成简单的差距热力图
        report += "| 技能 | 关键 | 重要 | 次要 |\n"
        report += "|------|:----:|:----:|:----:|\n"

        for metrics in sorted_skills[:15]:  # 只显示前15个
            critical = "🔴" * min(metrics.critical_gaps, 5) or "-"
            major = "🟡" * min(metrics.major_gaps, 5) or "-"
            minor = "🟢" * min(metrics.minor_gaps, 5) or "-"
            report += f"| {metrics.skill_name} | {critical} | {major} | {minor} |\n"

        report += """

---

## 6. 改进建议

### 6.1 紧急处理 (关键差距 > 0)

"""
        urgent = [m for m in sorted_skills if m.critical_gaps > 0]
        if urgent:
            for m in urgent[:5]:
                report += f"- **{m.skill_name}**: {m.critical_gaps} 个关键差距需要修复\n"
        else:
            report += "无紧急事项 ✅\n"

        report += """

### 6.2 重点关注 (得分 < 70%)

"""
        low_score = [m for m in sorted_skills if m.overall_score < 0.7]
        if low_score:
            for m in low_score[:5]:
                report += f"- **{m.skill_name}**: 得分 {m.overall_score:.1%}，需要改进\n"
        else:
            report += "无重点关注事项 ✅\n"

        report += f"""

---

## 7. 附录

### 7.1 数据来源

- 校准结果目录: `{self.results_dir}`
- 配置文件: `scripts/config/calibration_config.py`
- 生成脚本: `scripts/report_generator.py`

### 7.2 质量门控标准

| 门控 | 阈值 | 权重 |
|------|------|------|
| 文献覆盖度 | ≥80% | 30% |
| 内容完整性 | 100% | 30% |
| 公式一致性 | ≥90% | 20% |
| 引用有效性 | 100% | 20% |

---

*报告由多智能体校准框架自动生成*
"""
        return report

    def generate_coverage_matrix(self) -> str:
        """生成技能覆盖矩阵"""
        report = """# 技能覆盖矩阵

> 此矩阵显示每个技能的组件完整性状态

| 技能 | 识别假设 | 估计方法 | 诊断测试 | 报告标准 | 常见错误 | 得分 |
|------|:--------:|:--------:|:--------:|:--------:|:--------:|:----:|
"""
        # 扫描技能目录检查组件
        for category_dir in SKILLS_DIR.iterdir():
            if not category_dir.is_dir():
                continue

            for skill_dir in category_dir.iterdir():
                if not skill_dir.is_dir():
                    continue

                refs_dir = skill_dir / "references"

                has_id = "✅" if (refs_dir / "identification_assumptions.md").exists() else "❌"
                has_est = "✅" if (refs_dir / "estimation_methods.md").exists() else "❌"
                has_diag = "✅" if (refs_dir / "diagnostic_tests.md").exists() else "❌"
                has_rep = "✅" if (refs_dir / "reporting_standards.md").exists() else "❌"
                has_err = "✅" if (refs_dir / "common_errors.md").exists() else "❌"

                # 获取得分
                metrics = self.data.skill_metrics.get(skill_dir.name)
                score = f"{metrics.overall_score:.1%}" if metrics else "-"

                report += f"| {skill_dir.name} | {has_id} | {has_est} | {has_diag} | {has_rep} | {has_err} | {score} |\n"

        return report

    def generate_citation_database(self) -> Dict[str, Any]:
        """生成引用数据库"""
        from config.calibration_config import SKILL_CALIBRATION_CONFIG

        database = {
            "generated_at": self.data.generated_at.isoformat(),
            "skills": {}
        }

        for skill_name, config in SKILL_CALIBRATION_CONFIG.items():
            database["skills"][skill_name] = {
                "category": config.category.value,
                "priority": config.priority.value,
                "core_citations": config.core_citations,
                "queries": {
                    comp: {
                        "queries": qc.queries,
                        "min_citations": qc.min_citations,
                    }
                    for comp, qc in config.queries.items()
                }
            }

        return database

    def generate_gap_summary(self) -> str:
        """生成差距汇总报告"""
        report = """# 差距汇总报告

> 汇总所有技能的差距分析结果

## 1. 差距统计

| 严重程度 | 数量 | 百分比 |
|----------|------|--------|
"""
        total_critical = sum(m.critical_gaps for m in self.data.skill_metrics.values())
        total_major = sum(m.major_gaps for m in self.data.skill_metrics.values())
        total_minor = sum(m.minor_gaps for m in self.data.skill_metrics.values())
        total = total_critical + total_major + total_minor

        if total > 0:
            report += f"| 🔴 关键 | {total_critical} | {total_critical/total:.1%} |\n"
            report += f"| 🟡 重要 | {total_major} | {total_major/total:.1%} |\n"
            report += f"| 🟢 次要 | {total_minor} | {total_minor/total:.1%} |\n"
        else:
            report += "| 无差距 | 0 | - |\n"

        report += f"""

## 2. 按技能分布

| 技能 | 关键 | 重要 | 次要 | 总计 |
|------|------|------|------|------|
"""
        for metrics in sorted(self.data.skill_metrics.values(), key=lambda x: -(x.critical_gaps + x.major_gaps)):
            total = metrics.critical_gaps + metrics.major_gaps + metrics.minor_gaps
            report += f"| {metrics.skill_name} | {metrics.critical_gaps} | {metrics.major_gaps} | {metrics.minor_gaps} | {total} |\n"

        return report

    def generate_all_reports(self) -> None:
        """生成所有报告"""
        print("正在加载校准结果...")
        self.load_results()

        if not self.data.skill_metrics:
            print("警告: 未找到校准结果")

        timestamp = datetime.now().strftime("%Y-%m-%d")

        # 1. 主报告
        print("生成主校准报告...")
        main_report = self.generate_main_report()
        main_path = REPORTS_DIR / f"CALIBRATION_REPORT_{timestamp}.md"
        main_path.write_text(main_report, encoding='utf-8')
        print(f"  -> {main_path}")

        # 2. 覆盖矩阵
        print("生成覆盖矩阵...")
        coverage = self.generate_coverage_matrix()
        coverage_path = REPORTS_DIR / "SKILL_COVERAGE_MATRIX.md"
        coverage_path.write_text(coverage, encoding='utf-8')
        print(f"  -> {coverage_path}")

        # 3. 引用数据库
        print("生成引用数据库...")
        citations = self.generate_citation_database()
        citations_path = REPORTS_DIR / "CITATION_DATABASE.json"
        with open(citations_path, 'w', encoding='utf-8') as f:
            json.dump(citations, f, ensure_ascii=False, indent=2)
        print(f"  -> {citations_path}")

        # 4. 差距汇总
        print("生成差距汇总...")
        gaps = self.generate_gap_summary()
        gaps_path = REPORTS_DIR / "GAP_SUMMARY.md"
        gaps_path.write_text(gaps, encoding='utf-8')
        print(f"  -> {gaps_path}")

        print(f"\n所有报告已生成到: {REPORTS_DIR}")


# ============================================================================
# CLI 入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="校准报告生成器"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="生成所有报告"
    )
    parser.add_argument(
        "--main",
        action="store_true",
        help="生成主报告"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="生成覆盖矩阵"
    )
    parser.add_argument(
        "--citations",
        action="store_true",
        help="生成引用数据库"
    )
    parser.add_argument(
        "--gaps",
        action="store_true",
        help="生成差距汇总"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="输出目录"
    )

    args = parser.parse_args()

    generator = ReportGenerator()

    if args.output:
        generator.calibration_dir = Path(args.output)

    if args.all or not any([args.main, args.coverage, args.citations, args.gaps]):
        generator.generate_all_reports()
    else:
        generator.load_results()

        if args.main:
            report = generator.generate_main_report()
            print(report)

        if args.coverage:
            matrix = generator.generate_coverage_matrix()
            print(matrix)

        if args.citations:
            db = generator.generate_citation_database()
            print(json.dumps(db, ensure_ascii=False, indent=2))

        if args.gaps:
            summary = generator.generate_gap_summary()
            print(summary)


if __name__ == "__main__":
    main()

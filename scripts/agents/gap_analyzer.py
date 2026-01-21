#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GapAnalyzer Agent - 差距分析智能体

职责:
1. 组件级差距分析 (假设/方法/诊断/报告/错误)
2. 论文内容与技能文档对比
3. 差距分类和优先级排序
4. 生成详细差距报告
"""

import sys
import re
import uuid
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import json

# Windows 编码修复
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from .base import (
    BaseAgent, GapInfo, PaperInfo,
    PROJECT_ROOT, SKILLS_DIR, CALIBRATION_DIR,
    COMPONENT_TYPES, CITATION_THRESHOLDS
)


@dataclass
class GapAnalysisInput:
    """差距分析输入"""
    skill_name: str
    component: str  # identification, estimation, diagnostics, reporting, errors
    paper_contents: List[Dict[str, Any]]  # 提取的论文内容
    existing_doc: str  # 现有文档内容


@dataclass
class GapAnalysisOutput:
    """差距分析输出"""
    skill_name: str
    component: str
    gaps: List[GapInfo]
    coverage_score: float
    papers_analyzed: int
    summary: str


class GapAnalyzer(BaseAgent[GapAnalysisInput, GapAnalysisOutput]):
    """
    差距分析智能体

    功能:
    - 加载技能文档和论文内容
    - 逐组件对比分析
    - 识别缺失和不一致
    - 按严重程度分类差距
    """

    # 组件关键术语映射
    COMPONENT_TERM_PATTERNS = {
        "identification_assumptions": {
            "patterns": [
                r"(?:assumption|identifying\s+assumption|condition)\s*[\d\.:]+\s*(.*?)(?=assumption|\n\n|$)",
                r"under\s+the\s+(.*?assumption.*?)(?:\.|,)",
                r"require[s]?\s+(?:that\s+)?(.*?)(?:\.|,)",
            ],
            "key_terms": {
                "did": ["parallel trends", "no anticipation", "common support", "SUTVA"],
                "rd": ["continuity", "local randomization", "no manipulation", "exclusion"],
                "iv": ["relevance", "exogeneity", "exclusion restriction", "monotonicity"],
                "psm": ["unconfoundedness", "CIA", "overlap", "SUTVA"],
                "ddml": ["Neyman orthogonality", "rate conditions", "cross-fitting"],
            },
            "severity_map": {
                "missing_core_assumption": "critical",
                "incomplete_definition": "major",
                "missing_variant": "minor",
            }
        },
        "estimation_methods": {
            "patterns": [
                r"estimat(?:or|e)\s*[:\=]\s*(.*?)(?:\n\n|$)",
                r"(?:the|our)\s+(\w+(?:\s+\w+)?)\s+estimator",
                r"we\s+(?:use|employ|apply)\s+(.*?)(?:to|for)",
            ],
            "key_terms": {
                "did": ["group-time ATT", "CS estimator", "doubly robust", "IPW", "TWFE"],
                "rd": ["local polynomial", "MSE-optimal", "bias-corrected", "rdrobust"],
                "iv": ["2SLS", "LIML", "GMM", "Anderson-Rubin"],
                "psm": ["NN matching", "kernel matching", "AIPW", "DR"],
                "ddml": ["PLR", "IRM", "orthogonal score", "cross-fitting"],
            },
            "severity_map": {
                "missing_primary_method": "critical",
                "missing_alternative": "major",
                "incomplete_algorithm": "minor",
            }
        },
        "diagnostic_tests": {
            "patterns": [
                r"test\s+(?:for|of)\s+(.*?)(?:\.|,|\n)",
                r"(?:we|to)\s+test\s+(.*?)(?:\.|,)",
                r"diagnos(?:tic|e)\s*(.*?)(?:\.|,|\n)",
            ],
            "key_terms": {
                "did": ["pre-trends", "placebo", "event study", "Bacon decomposition"],
                "rd": ["McCrary", "density test", "covariate balance", "donut hole"],
                "iv": ["weak instrument", "Stock-Yogo", "Sargan", "Anderson-Rubin"],
                "psm": ["balance test", "SMD", "Rosenbaum bounds", "sensitivity"],
                "ddml": ["cross-validation", "ML diagnostics", "rate verification"],
            },
            "severity_map": {
                "missing_core_test": "major",
                "missing_robustness_test": "minor",
                "outdated_test": "enhancement",
            }
        },
        "reporting_standards": {
            "patterns": [
                r"report\s+(.*?)(?:\.|,|\n)",
                r"table\s+\d+\s+(?:shows?|presents?|reports?)\s*(.*?)(?:\.|,)",
                r"standard\s+error[s]?\s*(.*?)(?:\.|,)",
            ],
            "key_terms": {
                "all": ["standard errors", "confidence intervals", "p-values",
                        "sample size", "R-squared", "first-stage F"]
            },
            "severity_map": {
                "missing_required_stat": "major",
                "missing_recommended": "minor",
                "format_improvement": "enhancement",
            }
        },
        "common_errors": {
            "patterns": [
                r"(?:common|typical)\s+(?:error|mistake|pitfall)\s*(.*?)(?:\.|,|\n)",
                r"(?:incorrect|wrong)\s+(?:to|when)\s*(.*?)(?:\.|,)",
                r"(?:should\s+not|avoid)\s*(.*?)(?:\.|,)",
            ],
            "key_terms": {
                "did": ["forbidden comparisons", "negative weights", "heterogeneity bias"],
                "rd": ["bandwidth manipulation", "donut hole abuse", "extrapolation"],
                "iv": ["weak instruments", "over-identification", "exclusion violations"],
                "psm": ["propensity paradox", "model dependence", "hidden bias"],
                "ddml": ["overfitting", "rate violations", "invalid cross-fitting"],
            },
            "severity_map": {
                "critical_pitfall": "critical",
                "common_mistake": "major",
                "best_practice": "minor",
            }
        }
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("GapAnalyzer", config)
        self.skill_docs: Dict[str, str] = {}

    def _get_skill_type(self, skill_name: str) -> str:
        """获取技能类型简称"""
        type_map = {
            "estimator-did": "did",
            "estimator-rd": "rd",
            "estimator-iv": "iv",
            "estimator-psm": "psm",
            "causal-ddml": "ddml",
            "causal-forest": "forest",
            "structural-equation-modeling": "sem",
        }
        return type_map.get(skill_name, "generic")

    def _load_skill_document(self, skill_name: str, component: str) -> str:
        """加载技能文档"""
        # 查找技能目录
        search_paths = [
            SKILLS_DIR / "classic-methods" / skill_name,
            SKILLS_DIR / "causal-ml" / skill_name,
            SKILLS_DIR / "ml-foundation" / skill_name,
            SKILLS_DIR / "infrastructure" / skill_name,
        ]

        skill_path = None
        for path in search_paths:
            if path.exists():
                skill_path = path
                break

        if not skill_path:
            self.logger.warning(f"技能目录未找到: {skill_name}")
            return ""

        # 加载相关文档
        doc_files = {
            "identification_assumptions": "references/identification_assumptions.md",
            "estimation_methods": "references/estimation_methods.md",
            "diagnostic_tests": "references/diagnostic_tests.md",
            "reporting_standards": "references/reporting_standards.md",
            "common_errors": "references/common_errors.md",
        }

        doc_path = skill_path / doc_files.get(component, "SKILL.md")
        if doc_path.exists():
            return doc_path.read_text(encoding='utf-8')

        # 尝试加载 SKILL.md
        skill_md = skill_path / "SKILL.md"
        if skill_md.exists():
            return skill_md.read_text(encoding='utf-8')

        return ""

    def _extract_terms_from_paper(
        self,
        paper_content: Dict[str, Any],
        component: str,
        skill_type: str
    ) -> List[Tuple[str, str, str]]:
        """
        从论文内容提取术语

        Returns
        -------
        List[Tuple[str, str, str]]
            (术语, 所在章节, 原文片段)
        """
        extracted = []
        comp_config = self.COMPONENT_TERM_PATTERNS.get(component, {})
        patterns = comp_config.get("patterns", [])
        key_terms_map = comp_config.get("key_terms", {})

        # 获取该技能类型的关键术语
        key_terms = key_terms_map.get(skill_type, key_terms_map.get("all", []))

        # 从各部分提取
        text_sections = {
            "abstract": paper_content.get("abstract", ""),
            "methodology": paper_content.get("methodology", ""),
            "estimation": paper_content.get("estimation", ""),
            "identification": paper_content.get("identification", ""),
            "assumptions": " ".join(paper_content.get("assumptions", [])),
        }

        for section_name, text in text_sections.items():
            if not text:
                continue

            text_lower = text.lower()

            # 检查关键术语
            for term in key_terms:
                term_lower = term.lower()
                if term_lower in text_lower:
                    # 提取上下文
                    idx = text_lower.find(term_lower)
                    start = max(0, idx - 100)
                    end = min(len(text), idx + len(term) + 100)
                    context = text[start:end].strip()
                    extracted.append((term, section_name, context))

            # 使用正则模式提取
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
                for match in matches[:5]:  # 限制数量
                    if isinstance(match, str) and len(match) > 10:
                        extracted.append((match[:100], section_name, match[:200]))

        return extracted

    def _compare_with_document(
        self,
        extracted_terms: List[Tuple[str, str, str]],
        doc_content: str,
        component: str,
        skill_type: str
    ) -> List[GapInfo]:
        """
        对比提取的术语和现有文档

        Returns
        -------
        List[GapInfo]
            识别的差距
        """
        gaps = []
        doc_lower = doc_content.lower()
        comp_config = self.COMPONENT_TERM_PATTERNS.get(component, {})
        severity_map = comp_config.get("severity_map", {})

        # 检查关键术语覆盖
        key_terms = comp_config.get("key_terms", {}).get(
            skill_type,
            comp_config.get("key_terms", {}).get("all", [])
        )

        for term in key_terms:
            term_lower = term.lower()
            term_found_in_paper = any(
                term_lower in ext[0].lower() or term_lower in ext[2].lower()
                for ext in extracted_terms
            )
            term_found_in_doc = term_lower in doc_lower

            if term_found_in_paper and not term_found_in_doc:
                # 确定严重程度
                if term in key_terms[:2]:  # 前两个是核心术语
                    severity = "critical"
                elif term in key_terms[2:5]:
                    severity = "major"
                else:
                    severity = "minor"

                gaps.append(GapInfo(
                    gap_id=str(uuid.uuid4())[:8],
                    category=component.replace("_", " "),
                    severity=severity,
                    description=f"关键术语缺失: {term}",
                    source_paper="文献综合",
                    source_section="multiple",
                    existing_content="",
                    suggested_addition=f"添加 '{term}' 的描述和形式化定义",
                    target_file=f"references/{component}.md",
                    confidence=0.9
                ))

        # 检查提取内容的覆盖
        for term, section, context in extracted_terms:
            term_lower = term.lower()[:50]

            # 简单的相似度检查
            if len(term) < 10:
                continue

            # 检查是否在文档中
            if term_lower not in doc_lower:
                # 尝试模糊匹配
                similar = self._find_similar_in_doc(term, doc_content)
                if not similar:
                    gaps.append(GapInfo(
                        gap_id=str(uuid.uuid4())[:8],
                        category=component.replace("_", " "),
                        severity="minor",
                        description=f"可能缺失的内容: {term[:100]}",
                        source_paper="文献",
                        source_section=section,
                        existing_content="",
                        suggested_addition=context[:500],
                        target_file=f"references/{component}.md",
                        confidence=0.7
                    ))

        return gaps

    def _find_similar_in_doc(self, term: str, doc: str, threshold: float = 0.6) -> bool:
        """查找文档中是否有相似内容"""
        term_words = set(term.lower().split())
        doc_words = set(doc.lower().split())

        # 计算词汇重叠
        overlap = len(term_words & doc_words) / max(len(term_words), 1)
        return overlap > threshold

    def _calculate_coverage_score(
        self,
        gaps: List[GapInfo],
        total_papers: int,
        component: str,
        skill_type: str
    ) -> float:
        """计算覆盖度分数"""
        comp_config = self.COMPONENT_TERM_PATTERNS.get(component, {})
        key_terms = comp_config.get("key_terms", {}).get(
            skill_type,
            comp_config.get("key_terms", {}).get("all", [])
        )

        if not key_terms:
            return 1.0

        # 统计缺失的关键术语
        missing_terms = sum(1 for g in gaps if "关键术语缺失" in g.description)
        total_terms = len(key_terms)

        # 基础覆盖率
        term_coverage = 1.0 - (missing_terms / max(total_terms, 1))

        # 根据严重程度调整
        critical_count = sum(1 for g in gaps if g.severity == "critical")
        major_count = sum(1 for g in gaps if g.severity == "major")

        penalty = critical_count * 0.2 + major_count * 0.1
        final_score = max(0, term_coverage - penalty)

        return round(min(1.0, final_score), 4)

    def _deduplicate_gaps(self, gaps: List[GapInfo]) -> List[GapInfo]:
        """去重差距"""
        seen = set()
        unique = []

        for gap in gaps:
            key = (gap.category, gap.description[:50])
            if key not in seen:
                seen.add(key)
                unique.append(gap)

        return unique

    def _generate_summary(
        self,
        skill_name: str,
        component: str,
        gaps: List[GapInfo],
        coverage_score: float
    ) -> str:
        """生成差距摘要"""
        critical = sum(1 for g in gaps if g.severity == "critical")
        major = sum(1 for g in gaps if g.severity == "major")
        minor = sum(1 for g in gaps if g.severity == "minor")

        status = "✅ 良好" if coverage_score >= 0.8 else "⚠️ 需改进" if coverage_score >= 0.6 else "❌ 需紧急处理"

        summary = f"""
## {skill_name} - {component} 差距分析

**覆盖度**: {coverage_score:.1%} {status}

**差距统计**:
- 关键 (Critical): {critical}
- 重要 (Major): {major}
- 次要 (Minor): {minor}

**主要发现**:
"""
        for gap in gaps[:5]:
            summary += f"- [{gap.severity.upper()}] {gap.description}\n"

        return summary

    async def process(self, input_data: GapAnalysisInput) -> GapAnalysisOutput:
        """
        执行差距分析

        Parameters
        ----------
        input_data : GapAnalysisInput
            包含技能名称、组件类型、论文内容列表

        Returns
        -------
        GapAnalysisOutput
            差距分析结果
        """
        skill_name = input_data.skill_name
        component = input_data.component
        paper_contents = input_data.paper_contents

        self.logger.info(f"分析 {skill_name}/{component}, 论文数: {len(paper_contents)}")

        # 获取技能类型
        skill_type = self._get_skill_type(skill_name)

        # 加载现有文档
        doc_content = input_data.existing_doc
        if not doc_content:
            doc_content = self._load_skill_document(skill_name, component)

        if not doc_content:
            self.logger.warning(f"未找到文档: {skill_name}/{component}")

        # 从所有论文提取术语
        all_extracted_terms = []
        for paper in paper_contents:
            terms = self._extract_terms_from_paper(paper, component, skill_type)
            all_extracted_terms.extend(terms)

        self.logger.info(f"提取了 {len(all_extracted_terms)} 个术语")

        # 对比分析
        gaps = self._compare_with_document(
            all_extracted_terms,
            doc_content,
            component,
            skill_type
        )

        # 去重和排序
        gaps = self._deduplicate_gaps(gaps)
        gaps = sorted(gaps, key=lambda x: (
            0 if x.severity == "critical" else
            1 if x.severity == "major" else
            2 if x.severity == "minor" else 3
        ))

        # 计算覆盖度
        coverage_score = self._calculate_coverage_score(
            gaps, len(paper_contents), component, skill_type
        )

        # 生成摘要
        summary = self._generate_summary(skill_name, component, gaps, coverage_score)

        self.logger.info(f"识别 {len(gaps)} 个差距, 覆盖度: {coverage_score:.1%}")

        return GapAnalysisOutput(
            skill_name=skill_name,
            component=component,
            gaps=gaps,
            coverage_score=coverage_score,
            papers_analyzed=len(paper_contents),
            summary=summary
        )

    async def analyze_all_components(
        self,
        skill_name: str,
        paper_contents: List[Dict[str, Any]]
    ) -> Dict[str, GapAnalysisOutput]:
        """
        分析技能的所有组件

        Parameters
        ----------
        skill_name : str
            技能名称
        paper_contents : List[Dict]
            论文内容列表

        Returns
        -------
        Dict[str, GapAnalysisOutput]
            各组件的分析结果
        """
        results = {}

        for component in COMPONENT_TYPES:
            input_data = GapAnalysisInput(
                skill_name=skill_name,
                component=component,
                paper_contents=paper_contents,
                existing_doc=""
            )

            try:
                output = await self.run(input_data)
                results[component] = output
            except Exception as e:
                self.logger.error(f"分析 {component} 失败: {e}")

        return results

    def save_results(
        self,
        skill_name: str,
        results: Dict[str, GapAnalysisOutput],
        output_dir: Optional[Path] = None
    ) -> Path:
        """保存分析结果"""
        output_dir = output_dir or CALIBRATION_DIR / "gaps" / skill_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存各组件结果
        for component, output in results.items():
            # JSON 格式
            json_path = output_dir / f"{component}_gaps.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "skill_name": output.skill_name,
                    "component": output.component,
                    "coverage_score": output.coverage_score,
                    "papers_analyzed": output.papers_analyzed,
                    "gaps": [g.to_dict() for g in output.gaps]
                }, f, ensure_ascii=False, indent=2)

            # Markdown 摘要
            md_path = output_dir / f"{component}_summary.md"
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(output.summary)

        # 保存汇总
        summary_path = output_dir / "summary.md"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"# {skill_name} 差距分析汇总\n\n")
            f.write(f"> 生成时间: {__import__('datetime').datetime.now().isoformat()}\n\n")

            for component, output in results.items():
                f.write(output.summary)
                f.write("\n---\n")

        self.logger.info(f"结果已保存到: {output_dir}")
        return output_dir


if __name__ == "__main__":
    async def test():
        analyzer = GapAnalyzer()

        # 测试数据
        test_input = GapAnalysisInput(
            skill_name="estimator-did",
            component="identification_assumptions",
            paper_contents=[
                {
                    "title": "Test Paper",
                    "abstract": "We use difference-in-differences with parallel trends assumption.",
                    "methodology": "The key identifying assumption is parallel trends in the absence of treatment.",
                    "assumptions": ["Parallel trends", "No anticipation", "SUTVA"],
                }
            ],
            existing_doc="# Identification\n\nThe parallel trends assumption..."
        )

        result = await analyzer.run(test_input)
        print(f"覆盖度: {result.coverage_score:.1%}")
        print(f"差距数: {len(result.gaps)}")
        print(result.summary)

    asyncio.run(test())

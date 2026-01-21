#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CalibrationAgent - 校准分析智能体

职责:
1. 对比论文内容与现有技能文档
2. 识别差距和缺失内容
3. 生成更新建议
"""

import os
import sys
import re
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from difflib import SequenceMatcher

# Windows 编码修复
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
SKILLS_DIR = PROJECT_ROOT / "skills"


class GapSeverity(Enum):
    """差距严重程度"""
    CRITICAL = "critical"      # 关键方法论缺失
    MAJOR = "major"           # 重要内容缺失
    MINOR = "minor"           # 次要内容缺失
    ENHANCEMENT = "enhancement"  # 可优化项


class GapCategory(Enum):
    """差距类别"""
    ASSUMPTION = "assumption"        # 假设缺失
    FORMULA = "formula"              # 公式缺失
    METHOD = "method"                # 方法缺失
    DIAGNOSTIC = "diagnostic"        # 诊断测试缺失
    REFERENCE = "reference"          # 参考文献缺失
    IMPLEMENTATION = "implementation"  # 实现细节缺失
    NOTATION = "notation"            # 符号不一致


@dataclass
class Gap:
    """差距定义"""
    category: GapCategory
    severity: GapSeverity
    description: str
    source_paper: str
    source_section: str
    existing_content: str = ""
    suggested_addition: str = ""
    target_file: str = ""


@dataclass
class CalibrationReport:
    """校准报告"""
    skill_name: str
    papers_analyzed: int
    gaps: List[Gap]
    coverage_score: float  # 0-1 覆盖度分数
    recommendations: List[str]
    files_to_update: List[str]


class CalibrationAgent:
    """
    校准分析智能体

    功能:
    - 加载现有技能文档
    - 与论文内容对比
    - 识别差距
    - 生成更新建议
    """

    # 关键术语映射（用于匹配检测）
    TERM_MAPPINGS = {
        'estimator-did': {
            'core_concepts': [
                'parallel trends', 'common trends',
                'treatment effect', 'ATT', 'ATE',
                'staggered adoption', 'staggered treatment',
                'two-way fixed effects', 'TWFE',
                'group-time', 'cohort',
                'never treated', 'not yet treated',
                'event study', 'dynamic effects',
                'pre-trends', 'pre-treatment',
            ],
            'methods': [
                'Callaway Sant\'Anna', 'Callaway-Sant\'Anna',
                'Goodman-Bacon', 'Sun Abraham',
                'de Chaisemartin', 'D\'Haultfoeuille',
                'Borusyak', 'doubly robust',
                'inverse probability weighting', 'IPW',
            ],
            'diagnostics': [
                'placebo test', 'falsification',
                'balance test', 'covariate balance',
                'sensitivity analysis',
            ],
        },
        'estimator-rd': {
            'core_concepts': [
                'running variable', 'forcing variable',
                'cutoff', 'threshold',
                'bandwidth', 'optimal bandwidth',
                'sharp RD', 'fuzzy RD',
                'local linear', 'local polynomial',
                'continuity', 'discontinuity',
            ],
            'methods': [
                'Cattaneo', 'Imbens Kalyanaraman',
                'MSE-optimal', 'CER-optimal',
                'bias-corrected', 'robust inference',
            ],
            'diagnostics': [
                'McCrary', 'density test', 'manipulation',
                'placebo cutoffs', 'donut hole',
            ],
        },
        'structural-equation-modeling': {
            'core_concepts': [
                'latent variable', 'latent construct',
                'manifest variable', 'indicator',
                'factor loading', 'path coefficient',
                'measurement model', 'structural model',
                'CFA', 'confirmatory factor analysis',
                'SEM', 'structural equation',
                'covariance structure',
            ],
            'methods': [
                'maximum likelihood', 'ML estimation',
                'weighted least squares', 'WLS', 'WLSMV',
                'robust standard errors',
                'lavaan', 'semopy', 'Mplus', 'LISREL',
            ],
            'diagnostics': [
                'chi-square', 'chi-squared',
                'CFI', 'comparative fit index',
                'RMSEA', 'root mean square error',
                'SRMR', 'standardized root mean',
                'TLI', 'Tucker-Lewis',
                'modification indices',
            ],
        },
    }

    def __init__(self, skill_name: str):
        self.skill_name = skill_name
        self.skill_path = self._find_skill_path(skill_name)
        self.existing_docs = {}
        self.term_set: Set[str] = set()

        if self.skill_path:
            self._load_existing_docs()
            self._build_term_set()

    def _find_skill_path(self, skill_name: str) -> Optional[Path]:
        """查找技能目录路径"""
        # 搜索可能的位置
        search_paths = [
            SKILLS_DIR / "classic-methods" / skill_name,
            SKILLS_DIR / "ml-methods" / skill_name,
            SKILLS_DIR / "causal-ml" / skill_name,
            SKILLS_DIR / "support" / skill_name,
            SKILLS_DIR / skill_name,
        ]

        for path in search_paths:
            if path.exists():
                return path

        print(f"  [CalibrationAgent] 警告: 未找到技能目录 {skill_name}")
        return None

    def _load_existing_docs(self) -> None:
        """加载现有技能文档"""
        if not self.skill_path:
            return

        # 加载 SKILL.md
        skill_md = self.skill_path / "SKILL.md"
        if skill_md.exists():
            with open(skill_md, 'r', encoding='utf-8') as f:
                self.existing_docs['SKILL.md'] = f.read()

        # 加载 references 目录
        refs_dir = self.skill_path / "references"
        if refs_dir.exists():
            for ref_file in refs_dir.glob("*.md"):
                with open(ref_file, 'r', encoding='utf-8') as f:
                    self.existing_docs[f"references/{ref_file.name}"] = f.read()

        # 加载 Python 实现
        for py_file in self.skill_path.glob("*_estimator.py"):
            with open(py_file, 'r', encoding='utf-8') as f:
                self.existing_docs[py_file.name] = f.read()

        print(f"  [CalibrationAgent] 加载了 {len(self.existing_docs)} 个文档")

    def _build_term_set(self) -> None:
        """构建现有术语集合"""
        all_text = " ".join(self.existing_docs.values()).lower()
        self.term_set = set(re.findall(r'\b\w+\b', all_text))

    async def analyze_gaps(
        self,
        extracted_contents: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        分析内容差距

        Parameters
        ----------
        extracted_contents : List[ExtractedContent]
            从论文提取的内容

        Returns
        -------
        List[Dict]
            识别的差距列表
        """
        gaps = []

        for content in extracted_contents:
            # 检查假设覆盖
            assumption_gaps = self._check_assumptions(content)
            gaps.extend(assumption_gaps)

            # 检查方法覆盖
            method_gaps = self._check_methods(content)
            gaps.extend(method_gaps)

            # 检查公式覆盖
            formula_gaps = self._check_formulas(content)
            gaps.extend(formula_gaps)

            # 检查术语覆盖
            term_gaps = self._check_terms(content)
            gaps.extend(term_gaps)

        # 去重和排序
        unique_gaps = self._deduplicate_gaps(gaps)
        sorted_gaps = sorted(
            unique_gaps,
            key=lambda x: (
                0 if x['severity'] == 'critical' else
                1 if x['severity'] == 'major' else
                2 if x['severity'] == 'minor' else 3
            )
        )

        return sorted_gaps

    def _check_assumptions(self, content: Any) -> List[Dict]:
        """检查假设覆盖"""
        gaps = []

        for assumption in getattr(content, 'assumptions', []):
            # 检查假设是否在现有文档中提及
            assumption_lower = assumption.lower()
            found = False

            for doc_name, doc_content in self.existing_docs.items():
                if 'assumption' in doc_name.lower() or 'identification' in doc_name.lower():
                    # 简单相似度检查
                    if self._text_similarity(assumption_lower, doc_content.lower()) > 0.3:
                        found = True
                        break

            if not found:
                gaps.append({
                    'category': 'assumption',
                    'severity': 'major',
                    'description': f"假设可能缺失: {assumption[:200]}...",
                    'source_paper': content.title,
                    'source_section': 'assumptions',
                    'suggested_addition': assumption,
                    'target_file': 'references/identification_assumptions.md',
                })

        return gaps

    def _check_methods(self, content: Any) -> List[Dict]:
        """检查方法覆盖"""
        gaps = []

        methodology = getattr(content, 'methodology', '')
        if not methodology:
            return gaps

        # 提取方法名称
        method_patterns = [
            r'(\w+)\s+estimator',
            r'(\w+)\s+method',
            r'(\w+-\w+)\s+(?:estimator|method|approach)',
        ]

        for pattern in method_patterns:
            matches = re.findall(pattern, methodology, re.IGNORECASE)
            for method_name in matches:
                method_lower = method_name.lower()
                if method_lower not in self.term_set and len(method_name) > 3:
                    gaps.append({
                        'category': 'method',
                        'severity': 'minor',
                        'description': f"方法未提及: {method_name}",
                        'source_paper': content.title,
                        'source_section': 'methodology',
                        'suggested_addition': f"考虑添加 {method_name} 方法的描述",
                        'target_file': 'references/estimation_methods.md',
                    })

        return gaps

    def _check_formulas(self, content: Any) -> List[Dict]:
        """检查公式覆盖"""
        gaps = []

        formulas = getattr(content, 'formulas', [])
        if not formulas:
            return gaps

        # 提取公式中的关键符号
        for formula in formulas[:5]:  # 只检查前5个
            # 检查是否包含关键统计量
            key_symbols = ['ATT', 'ATE', 'LATE', 'ITT', 'CATE']
            for symbol in key_symbols:
                if symbol in formula and symbol.lower() not in self.term_set:
                    gaps.append({
                        'category': 'formula',
                        'severity': 'minor',
                        'description': f"公式符号可能缺失: {symbol}",
                        'source_paper': content.title,
                        'source_section': 'formulas',
                        'suggested_addition': formula[:200],
                        'target_file': 'SKILL.md',
                    })

        return gaps

    def _check_terms(self, content: Any) -> List[Dict]:
        """检查核心术语覆盖"""
        gaps = []

        # 获取该技能的术语映射
        mappings = self.TERM_MAPPINGS.get(self.skill_name, {})
        core_concepts = mappings.get('core_concepts', [])
        methods = mappings.get('methods', [])
        diagnostics = mappings.get('diagnostics', [])

        # 从论文内容中提取文本
        paper_text = " ".join([
            getattr(content, 'abstract', ''),
            getattr(content, 'methodology', ''),
            getattr(content, 'estimation', ''),
        ]).lower()

        # 检查论文中提及但文档中缺失的术语
        all_doc_text = " ".join(self.existing_docs.values()).lower()

        for term in core_concepts + methods:
            term_lower = term.lower()
            if term_lower in paper_text and term_lower not in all_doc_text:
                gaps.append({
                    'category': 'method',
                    'severity': 'major' if term in methods else 'minor',
                    'description': f"核心术语缺失: {term}",
                    'source_paper': content.title,
                    'source_section': 'terminology',
                    'suggested_addition': f"添加 '{term}' 的描述和实现",
                    'target_file': 'SKILL.md',
                })

        for term in diagnostics:
            term_lower = term.lower()
            if term_lower in paper_text and term_lower not in all_doc_text:
                gaps.append({
                    'category': 'diagnostic',
                    'severity': 'minor',
                    'description': f"诊断测试缺失: {term}",
                    'source_paper': content.title,
                    'source_section': 'diagnostics',
                    'suggested_addition': f"添加 '{term}' 诊断测试",
                    'target_file': 'references/diagnostic_tests.md',
                })

        return gaps

    def _text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        return SequenceMatcher(None, text1[:500], text2[:500]).ratio()

    def _deduplicate_gaps(self, gaps: List[Dict]) -> List[Dict]:
        """去重差距"""
        seen = set()
        unique = []

        for gap in gaps:
            key = (gap['category'], gap['description'][:50])
            if key not in seen:
                seen.add(key)
                unique.append(gap)

        return unique

    def generate_recommendations(self, gaps: List[Dict]) -> List[str]:
        """
        生成更新建议

        Parameters
        ----------
        gaps : List[Dict]
            识别的差距

        Returns
        -------
        List[str]
            更新建议列表
        """
        recommendations = []

        # 按目标文件分组
        by_file: Dict[str, List[Dict]] = {}
        for gap in gaps:
            target = gap.get('target_file', 'SKILL.md')
            if target not in by_file:
                by_file[target] = []
            by_file[target].append(gap)

        # 生成每个文件的建议
        for target_file, file_gaps in by_file.items():
            critical = [g for g in file_gaps if g['severity'] == 'critical']
            major = [g for g in file_gaps if g['severity'] == 'major']
            minor = [g for g in file_gaps if g['severity'] == 'minor']

            if critical:
                recommendations.append(
                    f"【紧急】{target_file}: 有 {len(critical)} 个关键差距需要修复"
                )
                for gap in critical:
                    recommendations.append(f"  - {gap['description']}")

            if major:
                recommendations.append(
                    f"【重要】{target_file}: 有 {len(major)} 个重要差距"
                )
                for gap in major[:5]:  # 只列出前5个
                    recommendations.append(f"  - {gap['description']}")

            if minor:
                recommendations.append(
                    f"【次要】{target_file}: 有 {len(minor)} 个次要差距"
                )

        return recommendations

    def generate_report(
        self,
        gaps: List[Dict],
        papers_analyzed: int
    ) -> CalibrationReport:
        """生成校准报告"""
        recommendations = self.generate_recommendations(gaps)

        # 计算覆盖度
        total_checks = papers_analyzed * 10  # 假设每篇论文10个检查点
        gaps_found = len(gaps)
        coverage = max(0, 1 - gaps_found / max(total_checks, 1))

        # 确定需要更新的文件
        files_to_update = list(set(g.get('target_file', '') for g in gaps))

        return CalibrationReport(
            skill_name=self.skill_name,
            papers_analyzed=papers_analyzed,
            gaps=gaps,
            coverage_score=coverage,
            recommendations=recommendations,
            files_to_update=files_to_update,
        )


if __name__ == "__main__":
    # 测试
    agent = CalibrationAgent("estimator-did")
    print(f"加载文档: {list(agent.existing_docs.keys())}")

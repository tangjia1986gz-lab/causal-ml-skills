#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CitationAgent - 引用检查智能体

职责:
1. 验证引用格式正确性
2. 检查引用完整性
3. 验证引用可追溯性
4. 生成引用覆盖报告
"""

import sys
import re
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
import json

# Windows 编码修复
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from .base import (
    BaseAgent, PaperInfo, PROJECT_ROOT, SKILLS_DIR, CALIBRATION_DIR, TOP_JOURNALS
)


@dataclass
class Citation:
    """引用数据结构"""
    raw_text: str
    authors: List[str]
    year: Optional[int]
    title: Optional[str]
    venue: Optional[str]
    citation_style: str  # author-year, numeric, footnote
    location: str  # 在哪个文件中
    line_number: int
    is_valid: bool = True
    validation_issues: List[str] = field(default_factory=list)


@dataclass
class CitationMatch:
    """引用匹配结果"""
    doc_citation: Citation
    matched_paper: Optional[PaperInfo]
    match_confidence: float
    is_verified: bool
    notes: str = ""


@dataclass
class CitationInput:
    """引用检查输入"""
    skill_name: str
    expected_citations: List[PaperInfo]  # 期望引用的论文
    doc_content: Optional[str] = None


@dataclass
class CitationOutput:
    """引用检查输出"""
    skill_name: str
    total_citations_in_doc: int
    total_expected_citations: int
    matched_citations: int
    missing_citations: int
    invalid_citations: int
    coverage_score: float
    validity_score: float
    matches: List[CitationMatch]
    missing_papers: List[PaperInfo]
    report: str


class CitationAgent(BaseAgent[CitationInput, CitationOutput]):
    """
    引用检查智能体

    功能:
    - 提取文档中的引用
    - 验证引用格式
    - 匹配期望论文
    - 计算覆盖度
    """

    # 引用格式模式
    CITATION_PATTERNS = {
        # Author-Year 格式
        "author_year": [
            # (Author, Year) 或 Author (Year)
            r'([A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+)?(?:\s+et\s+al\.?)?)\s*[\(\[]?\s*(\d{4})\s*[\)\]]?',
            # Author et al. (Year)
            r'([A-Z][a-z]+(?:\s+et\s+al\.?))\s*[\(\[](\d{4})[\)\]]',
            # 多作者: Author, Author, and Author (Year)
            r'([A-Z][a-z]+(?:,\s+[A-Z][a-z]+)*(?:,?\s+(?:and|&)\s+[A-Z][a-z]+))\s*[\(\[](\d{4})[\)\]]',
        ],
        # Markdown 链接格式
        "markdown_link": [
            r'\[([^\]]+)\]\(([^\)]+)\)',
        ],
        # 参考文献列表格式
        "reference_list": [
            r'^[-\*•]\s*([A-Z][a-z]+(?:,\s*[A-Z]\.?)+(?:\s+(?:and|&)\s+[A-Z][a-z]+)?)\s*[\(\[]?(\d{4})[\)\]]?[:\.]?\s*(.*?)$',
            r'^(\d+)\.\s*([A-Z][a-z]+(?:,\s*[A-Z]\.?)+)\s*[\(\[]?(\d{4})[\)\]]?',
        ],
    }

    # 核心方法论引用 (按技能类型)
    CORE_CITATIONS = {
        "estimator-did": [
            {"authors": ["Callaway", "Sant'Anna"], "year": 2021, "weight": 1.0},
            {"authors": ["Goodman-Bacon"], "year": 2021, "weight": 1.0},
            {"authors": ["Sun", "Abraham"], "year": 2021, "weight": 0.9},
            {"authors": ["de Chaisemartin", "D'Haultfoeuille"], "year": 2020, "weight": 0.9},
            {"authors": ["Borusyak", "Jaravel", "Spiess"], "year": 2024, "weight": 0.8},
            {"authors": ["Roth", "Sant'Anna"], "year": 2023, "weight": 0.8},
        ],
        "estimator-rd": [
            {"authors": ["Cattaneo", "Idrobo", "Titiunik"], "year": 2020, "weight": 1.0},
            {"authors": ["Imbens", "Kalyanaraman"], "year": 2012, "weight": 1.0},
            {"authors": ["Calonico", "Cattaneo", "Titiunik"], "year": 2014, "weight": 0.9},
            {"authors": ["McCrary"], "year": 2008, "weight": 0.8},
            {"authors": ["Lee", "Lemieux"], "year": 2010, "weight": 0.8},
        ],
        "estimator-iv": [
            {"authors": ["Angrist", "Imbens"], "year": 1996, "weight": 1.0},
            {"authors": ["Stock", "Yogo"], "year": 2005, "weight": 1.0},
            {"authors": ["Andrews", "Stock", "Sun"], "year": 2019, "weight": 0.9},
            {"authors": ["Lee", "McCrary", "Moreira"], "year": 2022, "weight": 0.8},
        ],
        "estimator-psm": [
            {"authors": ["Rosenbaum", "Rubin"], "year": 1983, "weight": 1.0},
            {"authors": ["Imbens", "Rubin"], "year": 2015, "weight": 1.0},
            {"authors": ["Abadie", "Imbens"], "year": 2006, "weight": 0.9},
            {"authors": ["King", "Nielsen"], "year": 2019, "weight": 0.8},
            {"authors": ["Rosenbaum"], "year": 2002, "weight": 0.8},
        ],
        "causal-ddml": [
            {"authors": ["Chernozhukov"], "year": 2018, "weight": 1.0},
            {"authors": ["Chernozhukov", "Chetverikov", "Demirer"], "year": 2018, "weight": 1.0},
            {"authors": ["Bach", "Chernozhukov", "Kurz"], "year": 2024, "weight": 0.9},
        ],
        "causal-forest": [
            {"authors": ["Athey", "Wager"], "year": 2019, "weight": 1.0},
            {"authors": ["Wager", "Athey"], "year": 2018, "weight": 1.0},
            {"authors": ["Athey", "Imbens"], "year": 2016, "weight": 0.9},
        ],
        "structural-equation-modeling": [
            {"authors": ["Bollen"], "year": 1989, "weight": 1.0},
            {"authors": ["Kline"], "year": 2016, "weight": 1.0},
            {"authors": ["Rosseel"], "year": 2012, "weight": 0.9},
            {"authors": ["Hu", "Bentler"], "year": 1999, "weight": 0.8},
        ],
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("CitationAgent", config)

    def _extract_citations(self, text: str, location: str) -> List[Citation]:
        """
        从文本中提取引用

        Parameters
        ----------
        text : str
            输入文本
        location : str
            文件位置

        Returns
        -------
        List[Citation]
            提取的引用列表
        """
        citations = []
        seen = set()

        lines = text.split('\n')

        for line_num, line in enumerate(lines, 1):
            # Author-Year 格式
            for pattern in self.CITATION_PATTERNS["author_year"]:
                for match in re.finditer(pattern, line):
                    author_text = match.group(1)
                    year = int(match.group(2))

                    # 解析作者
                    authors = self._parse_authors(author_text)

                    key = (tuple(authors), year)
                    if key not in seen:
                        seen.add(key)
                        citations.append(Citation(
                            raw_text=match.group(0),
                            authors=authors,
                            year=year,
                            title=None,
                            venue=None,
                            citation_style="author_year",
                            location=location,
                            line_number=line_num
                        ))

            # 参考文献列表格式
            for pattern in self.CITATION_PATTERNS["reference_list"]:
                match = re.match(pattern, line.strip(), re.MULTILINE)
                if match:
                    groups = match.groups()
                    if len(groups) >= 2:
                        author_text = groups[0] if not groups[0].isdigit() else groups[1]
                        year_idx = 1 if not groups[0].isdigit() else 2
                        year = int(groups[year_idx]) if year_idx < len(groups) else None

                        authors = self._parse_authors(author_text)

                        if authors and year:
                            key = (tuple(authors), year)
                            if key not in seen:
                                seen.add(key)
                                citations.append(Citation(
                                    raw_text=line.strip()[:200],
                                    authors=authors,
                                    year=year,
                                    title=groups[-1] if len(groups) > 2 else None,
                                    venue=None,
                                    citation_style="reference_list",
                                    location=location,
                                    line_number=line_num
                                ))

        return citations

    def _parse_authors(self, text: str) -> List[str]:
        """解析作者姓名"""
        authors = []

        # 清理文本
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'\s*et\s+al\.?\s*', '', text)

        # 分割作者
        parts = re.split(r'\s*(?:,|and|&)\s*', text)

        for part in parts:
            part = part.strip()
            if part and len(part) > 1:
                # 提取姓氏
                surname_match = re.match(r'^([A-Z][a-z]+)', part)
                if surname_match:
                    authors.append(surname_match.group(1))

        return authors

    def _validate_citation(self, citation: Citation) -> Citation:
        """验证引用格式"""
        issues = []

        # 检查年份范围
        if citation.year:
            if citation.year < 1950 or citation.year > 2030:
                issues.append(f"年份异常: {citation.year}")

        # 检查作者
        if not citation.authors:
            issues.append("缺少作者信息")
        elif len(citation.authors[0]) < 2:
            issues.append("作者名太短")

        citation.is_valid = len(issues) == 0
        citation.validation_issues = issues

        return citation

    def _match_citation_to_paper(
        self,
        citation: Citation,
        papers: List[PaperInfo]
    ) -> Tuple[Optional[PaperInfo], float]:
        """
        匹配引用到论文

        Parameters
        ----------
        citation : Citation
            文档中的引用
        papers : List[PaperInfo]
            期望的论文列表

        Returns
        -------
        Tuple[Optional[PaperInfo], float]
            (匹配的论文, 置信度)
        """
        best_match = None
        best_confidence = 0.0

        citation_authors = set(a.lower() for a in citation.authors)

        for paper in papers:
            # 作者匹配
            paper_authors = set()
            for author in paper.authors:
                # 提取姓氏
                parts = author.split()
                if parts:
                    paper_authors.add(parts[-1].lower())

            author_overlap = len(citation_authors & paper_authors)
            author_score = author_overlap / max(len(citation_authors), 1)

            # 年份匹配
            year_score = 1.0 if citation.year == paper.year else 0.0

            # 综合置信度
            confidence = 0.7 * author_score + 0.3 * year_score

            if confidence > best_confidence:
                best_confidence = confidence
                best_match = paper

        # 阈值过滤
        if best_confidence < 0.5:
            return None, 0.0

        return best_match, best_confidence

    def _check_core_citations(
        self,
        citations: List[Citation],
        skill_name: str
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        检查核心引用覆盖

        Returns
        -------
        Tuple[List[Dict], List[Dict]]
            (已覆盖, 缺失)
        """
        core_refs = self.CORE_CITATIONS.get(skill_name, [])
        covered = []
        missing = []

        for core_ref in core_refs:
            core_authors = set(a.lower() for a in core_ref["authors"])
            core_year = core_ref["year"]

            found = False
            for citation in citations:
                cit_authors = set(a.lower() for a in citation.authors)

                # 检查作者重叠
                overlap = len(core_authors & cit_authors)
                if overlap > 0 and citation.year == core_year:
                    found = True
                    covered.append(core_ref)
                    break

            if not found:
                missing.append(core_ref)

        return covered, missing

    def _load_skill_content(self, skill_name: str) -> str:
        """加载技能文档内容"""
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
            return ""

        content_parts = []

        # 读取所有 .md 文件
        for md_file in skill_path.rglob("*.md"):
            try:
                content_parts.append(md_file.read_text(encoding='utf-8'))
            except Exception:
                pass

        return "\n\n".join(content_parts)

    def _calculate_scores(
        self,
        citations: List[Citation],
        matches: List[CitationMatch],
        covered_core: List[Dict],
        total_core: List[Dict]
    ) -> Tuple[float, float]:
        """
        计算覆盖度和有效性得分

        Returns
        -------
        Tuple[float, float]
            (覆盖度, 有效性)
        """
        # 有效性得分
        valid_count = sum(1 for c in citations if c.is_valid)
        validity_score = valid_count / max(len(citations), 1)

        # 覆盖度得分 (加权)
        if not total_core:
            coverage_score = 1.0
        else:
            total_weight = sum(r.get("weight", 1.0) for r in total_core)
            covered_weight = sum(r.get("weight", 1.0) for r in covered_core)
            coverage_score = covered_weight / max(total_weight, 1)

        return round(coverage_score, 4), round(validity_score, 4)

    def _generate_report(
        self,
        skill_name: str,
        citations: List[Citation],
        matches: List[CitationMatch],
        missing_core: List[Dict],
        coverage_score: float,
        validity_score: float
    ) -> str:
        """生成引用报告"""
        status_cov = "✅" if coverage_score >= 0.8 else "⚠️" if coverage_score >= 0.6 else "❌"
        status_val = "✅" if validity_score >= 0.9 else "⚠️" if validity_score >= 0.7 else "❌"

        report = f"""# {skill_name} 引用检查报告

## 概览

| 指标 | 值 | 状态 |
|------|-----|------|
| 文档引用数 | {len(citations)} | - |
| 核心引用覆盖 | {coverage_score:.1%} | {status_cov} |
| 引用有效性 | {validity_score:.1%} | {status_val} |

## 核心引用检查

"""
        # 核心引用覆盖情况
        core_refs = self.CORE_CITATIONS.get(skill_name, [])
        for ref in core_refs:
            authors = ", ".join(ref["authors"])
            year = ref["year"]
            is_covered = ref not in missing_core
            status = "✅" if is_covered else "❌"
            report += f"- {status} {authors} ({year})\n"

        if missing_core:
            report += f"\n### 缺失的核心引用 ({len(missing_core)})\n\n"
            for ref in missing_core:
                authors = ", ".join(ref["authors"])
                year = ref["year"]
                weight = ref.get("weight", 1.0)
                report += f"- **{authors} ({year})** [权重: {weight}]\n"

        report += "\n## 引用格式问题\n\n"

        invalid = [c for c in citations if not c.is_valid]
        if invalid:
            for cit in invalid[:10]:
                report += f"- L{cit.line_number}: `{cit.raw_text[:50]}...`\n"
                for issue in cit.validation_issues:
                    report += f"  - {issue}\n"
        else:
            report += "无格式问题 ✅\n"

        report += "\n## 已识别引用\n\n"
        for cit in citations[:20]:
            authors = ", ".join(cit.authors)
            report += f"- {authors} ({cit.year}) @ {cit.location}:L{cit.line_number}\n"

        return report

    async def process(self, input_data: CitationInput) -> CitationOutput:
        """
        执行引用检查

        Parameters
        ----------
        input_data : CitationInput
            输入数据

        Returns
        -------
        CitationOutput
            检查结果
        """
        skill_name = input_data.skill_name

        self.logger.info(f"检查 {skill_name} 引用")

        # 加载文档内容
        doc_content = input_data.doc_content
        if not doc_content:
            doc_content = self._load_skill_content(skill_name)

        if not doc_content:
            self.logger.warning(f"未找到文档内容: {skill_name}")

        # 提取引用
        citations = self._extract_citations(doc_content, skill_name)
        self.logger.info(f"提取了 {len(citations)} 个引用")

        # 验证引用格式
        citations = [self._validate_citation(c) for c in citations]
        invalid_count = sum(1 for c in citations if not c.is_valid)

        # 匹配到期望论文
        matches = []
        matched_papers = set()

        for cit in citations:
            paper, confidence = self._match_citation_to_paper(cit, input_data.expected_citations)
            matches.append(CitationMatch(
                doc_citation=cit,
                matched_paper=paper,
                match_confidence=confidence,
                is_verified=paper is not None and confidence > 0.7
            ))
            if paper:
                matched_papers.add(paper.paper_id)

        # 检查核心引用
        covered_core, missing_core = self._check_core_citations(citations, skill_name)

        # 计算得分
        coverage_score, validity_score = self._calculate_scores(
            citations, matches, covered_core,
            self.CORE_CITATIONS.get(skill_name, [])
        )

        # 计算缺失的期望论文
        missing_papers = [
            p for p in input_data.expected_citations
            if p.paper_id not in matched_papers
        ]

        # 生成报告
        report = self._generate_report(
            skill_name, citations, matches, missing_core,
            coverage_score, validity_score
        )

        self.logger.info(f"覆盖度: {coverage_score:.1%}, 有效性: {validity_score:.1%}")

        return CitationOutput(
            skill_name=skill_name,
            total_citations_in_doc=len(citations),
            total_expected_citations=len(input_data.expected_citations),
            matched_citations=len(matched_papers),
            missing_citations=len(missing_papers),
            invalid_citations=invalid_count,
            coverage_score=coverage_score,
            validity_score=validity_score,
            matches=matches,
            missing_papers=missing_papers,
            report=report
        )

    def save_results(
        self,
        output: CitationOutput,
        output_dir: Optional[Path] = None
    ) -> Path:
        """保存检查结果"""
        output_dir = output_dir or CALIBRATION_DIR / "validation" / output.skill_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存报告
        report_path = output_dir / "citation_validation.md"
        report_path.write_text(output.report, encoding='utf-8')

        # 保存 JSON 结果
        json_path = output_dir / "citation_validation.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "skill_name": output.skill_name,
                "coverage_score": output.coverage_score,
                "validity_score": output.validity_score,
                "total_citations_in_doc": output.total_citations_in_doc,
                "matched_citations": output.matched_citations,
                "missing_citations": output.missing_citations,
                "invalid_citations": output.invalid_citations,
                "missing_papers": [p.to_dict() for p in output.missing_papers[:20]],
            }, f, ensure_ascii=False, indent=2)

        self.logger.info(f"结果已保存到: {output_dir}")
        return output_dir


if __name__ == "__main__":
    async def test():
        agent = CitationAgent()

        test_input = CitationInput(
            skill_name="estimator-did",
            expected_citations=[
                PaperInfo(
                    paper_id="1",
                    title="Difference-in-Differences with Multiple Time Periods",
                    authors=["Brantly Callaway", "Pedro H.C. Sant'Anna"],
                    year=2021,
                    venue="Journal of Econometrics",
                    citations=1000
                )
            ],
            doc_content="""
# DID Estimator

Following Callaway and Sant'Anna (2021), we implement the group-time ATT.

## References
- Callaway, B. and Sant'Anna, P. (2021). Difference-in-Differences with Multiple Time Periods.
- Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment timing.
            """
        )

        result = await agent.run(test_input)
        print(f"覆盖度: {result.coverage_score:.1%}")
        print(f"有效性: {result.validity_score:.1%}")

    asyncio.run(test())

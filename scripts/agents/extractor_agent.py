#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ExtractorAgent - 论文内容提取智能体

职责:
1. 从 PDF 提取文本
2. 识别和提取关键章节
3. 提取公式和算法
4. 生成结构化校准笔记
"""

import os
import sys
import re
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

# Windows 编码修复
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
PAPERS_DIR = PROJECT_ROOT / "papers"
NOTES_DIR = PROJECT_ROOT / "calibration_notes"
NOTES_DIR.mkdir(exist_ok=True)


@dataclass
class ExtractedContent:
    """提取的内容结构"""
    paper_id: str
    title: str
    year: int
    venue: str
    citations: int
    abstract: str = ""
    assumptions: List[str] = field(default_factory=list)
    methodology: str = ""
    estimation: str = ""
    identification: str = ""
    formulas: List[str] = field(default_factory=list)
    algorithms: List[str] = field(default_factory=list)
    key_findings: List[str] = field(default_factory=list)
    references_to_other_methods: List[str] = field(default_factory=list)


class ExtractorAgent:
    """
    论文内容提取智能体

    功能:
    - 从 PDF 提取全文
    - 智能分割章节
    - 提取关键方法论内容
    - 生成校准笔记
    """

    # 章节识别模式
    SECTION_PATTERNS = {
        'abstract': [
            r'Abstract[:\s]*\n(.*?)(?=\n\s*(?:1\.?\s*)?Introduction|Keywords|JEL)',
            r'ABSTRACT[:\s]*\n(.*?)(?=\n\s*(?:1\.?\s*)?INTRODUCTION|Keywords|JEL)',
        ],
        'introduction': [
            r'(?:1\.?\s*)?Introduction\s*\n(.*?)(?=\n\s*(?:2\.?\s*)?(?:Model|Method|Framework|Setup|Literature|Background|Data))',
        ],
        'methodology': [
            r'(?:\d\.?\s*)?(?:Methodology|Method|Identification|Setup|Framework|Model)[:\s]*\n(.*?)(?=\n\s*(?:\d\.?\s*)?(?:Data|Empirical|Application|Results|Estimation))',
            r'(?:II\.?\s*)?(?:METHODOLOGY|METHOD|IDENTIFICATION|SETUP|FRAMEWORK|MODEL)[:\s]*\n(.*?)(?=\n\s*(?:III\.?\s*)?)',
        ],
        'estimation': [
            r'(?:\d\.?\s*)?(?:Estimation|Estimator|Inference)[:\s]*\n(.*?)(?=\n\s*(?:\d\.?\s*)?(?:Asymptotic|Standard Error|Bootstrap|Application|Results|Simulation))',
        ],
        'assumptions': [
            r'Assumption[s]?\s*[\d\.]*[:\s]*(.*?)(?=\n\s*(?:Theorem|Proposition|Lemma|Definition|\d+\.\d+|Assumption))',
            r'(?:Key |Main |Identifying )?Assumptions?(.*?)(?=\n\s*(?:Under |Given |Theorem))',
        ],
        'identification': [
            r'(?:\d\.?\s*)?(?:Identification|Identifying)[:\s]*\n(.*?)(?=\n\s*(?:\d\.?\s*)?(?:Estimation|Inference|Results))',
        ],
    }

    def __init__(self, max_pages: int = 30):
        self.max_pages = max_pages

    async def extract(self, paper: Dict[str, Any]) -> Optional[ExtractedContent]:
        """
        提取论文内容

        Parameters
        ----------
        paper : Dict
            论文信息（包含 local_path 或 pdf_url）

        Returns
        -------
        ExtractedContent or None
        """
        # 获取 PDF 路径
        local_path = paper.get('local_path')
        if local_path:
            pdf_path = Path(local_path)
        else:
            # 如果没有本地路径，尝试下载
            from .literature_agent import LiteratureAgent
            agent = LiteratureAgent()
            pdf_path = await agent.download_pdf(paper)

        if not pdf_path or not pdf_path.exists():
            print(f"  [ExtractorAgent] 无法获取 PDF: {paper.get('title', 'Unknown')[:50]}...")
            return None

        # 提取文本
        text = await self._extract_text_from_pdf(pdf_path)
        if not text:
            return None

        # 提取各章节
        sections = self._extract_sections(text)
        formulas = self._extract_formulas(text)
        assumptions = self._extract_assumptions(text)

        content = ExtractedContent(
            paper_id=paper.get('paper_id', ''),
            title=paper.get('title', ''),
            year=paper.get('year', 0),
            venue=paper.get('venue', ''),
            citations=paper.get('citations', 0),
            abstract=sections.get('abstract', paper.get('abstract', '')),
            assumptions=assumptions,
            methodology=sections.get('methodology', ''),
            estimation=sections.get('estimation', ''),
            identification=sections.get('identification', ''),
            formulas=formulas,
        )

        # 生成校准笔记
        await self._generate_note(content, paper.get('skill_name', 'unknown'))

        print(f"  [ExtractorAgent] 提取完成: {paper.get('title', 'Unknown')[:50]}...")
        return content

    async def _extract_text_from_pdf(self, pdf_path: Path) -> Optional[str]:
        """从 PDF 提取文本"""
        text = None

        # 方法 1: PyMuPDF (fitz)
        try:
            import fitz
            doc = await asyncio.to_thread(fitz.open, pdf_path)
            pages = []
            for i, page in enumerate(doc):
                if i >= self.max_pages:
                    break
                pages.append(page.get_text())
            text = "\n\n".join(pages)
            doc.close()
            return text
        except ImportError:
            pass
        except Exception as e:
            print(f"    [PyMuPDF 失败] {e}")

        # 方法 2: pdfplumber
        try:
            import pdfplumber

            def extract_with_pdfplumber():
                with pdfplumber.open(pdf_path) as pdf:
                    pages = []
                    for i, page in enumerate(pdf.pages):
                        if i >= self.max_pages:
                            break
                        page_text = page.extract_text()
                        if page_text:
                            pages.append(page_text)
                    return "\n\n".join(pages)

            text = await asyncio.to_thread(extract_with_pdfplumber)
            return text
        except ImportError:
            pass
        except Exception as e:
            print(f"    [pdfplumber 失败] {e}")

        # 方法 3: pypdf
        try:
            from pypdf import PdfReader

            def extract_with_pypdf():
                reader = PdfReader(pdf_path)
                pages = []
                for i, page in enumerate(reader.pages):
                    if i >= self.max_pages:
                        break
                    pages.append(page.extract_text())
                return "\n\n".join(pages)

            text = await asyncio.to_thread(extract_with_pypdf)
            return text
        except ImportError:
            pass
        except Exception as e:
            print(f"    [pypdf 失败] {e}")

        print("    [错误] 未安装 PDF 解析库")
        return None

    def _extract_sections(self, text: str) -> Dict[str, str]:
        """提取各章节内容"""
        sections = {}

        for section_name, patterns in self.SECTION_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                if match:
                    content = match.group(1).strip()
                    # 限制长度
                    max_len = 5000 if section_name in ['methodology', 'estimation'] else 3000
                    sections[section_name] = content[:max_len]
                    break

        return sections

    def _extract_formulas(self, text: str) -> List[str]:
        """提取数学公式"""
        formulas = []

        patterns = [
            r'\$\$(.*?)\$\$',  # Display math
            r'\$(.*?)\$',      # Inline math (longer ones)
            r'\\begin\{equation\}(.*?)\\end\{equation\}',
            r'\\begin\{align\}(.*?)\\end\{align\}',
            r'\\begin\{eqnarray\}(.*?)\\end\{eqnarray\}',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for m in matches:
                m = m.strip()
                if len(m) > 15:  # 过滤太短的
                    formulas.append(m[:500])  # 限制长度

        # 去重
        unique_formulas = list(dict.fromkeys(formulas))
        return unique_formulas[:30]

    def _extract_assumptions(self, text: str) -> List[str]:
        """提取假设"""
        assumptions = []

        # 匹配 "Assumption X" 格式
        pattern = r'Assumption\s*(\d+)[:\s.]*(.*?)(?=Assumption\s*\d+|Theorem|Proposition|Lemma|\n\n\n)'
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

        for num, content in matches:
            content = content.strip()[:1000]
            assumptions.append(f"Assumption {num}: {content}")

        return assumptions[:10]

    async def _generate_note(
        self,
        content: ExtractedContent,
        skill_name: str
    ) -> Path:
        """生成校准笔记"""

        # 生成文件名
        safe_title = re.sub(r'[^\w\s-]', '', content.title[:50])
        filename = f"{skill_name}_{content.year}_{safe_title}.md"
        note_path = NOTES_DIR / filename

        note = f"""# 校准笔记: {content.title}

> **技能**: {skill_name}
> **论文 ID**: {content.paper_id}
> **年份**: {content.year}
> **期刊**: {content.venue}
> **引用数**: {content.citations}

---

## 摘要

{content.abstract or '未提取到摘要'}

---

## 核心假设

"""
        if content.assumptions:
            for assumption in content.assumptions:
                note += f"- {assumption[:500]}\n\n"
        else:
            note += "未提取到假设部分\n"

        note += f"""
---

## 方法论/识别策略

{content.methodology[:3000] if content.methodology else '未提取到方法论部分'}

---

## 估计方法

{content.estimation[:3000] if content.estimation else '未提取到估计部分'}

---

## 关键公式

"""
        if content.formulas:
            for i, formula in enumerate(content.formulas[:10], 1):
                note += f"{i}. `{formula[:200]}`\n\n"
        else:
            note += "未提取到公式\n"

        note += """
---

## 校准检查清单

- [ ] 识别假设是否完整覆盖
- [ ] 估计方法是否准确描述
- [ ] 诊断检验是否包含
- [ ] 代码实现是否一致
- [ ] 参考文献是否引用

---

## 与现有文档的差异

<!-- 由 CalibrationAgent 自动填写 -->

"""

        with open(note_path, 'w', encoding='utf-8') as f:
            f.write(note)

        return note_path

    async def batch_extract(
        self,
        papers: List[Dict[str, Any]],
        skill_name: str = "unknown"
    ) -> List[ExtractedContent]:
        """
        批量提取多篇论文

        Parameters
        ----------
        papers : List[Dict]
            论文列表
        skill_name : str
            技能名称（用于标记笔记）

        Returns
        -------
        List[ExtractedContent]
        """
        results = []

        for paper in papers:
            paper['skill_name'] = skill_name
            content = await self.extract(paper)
            if content:
                results.append(content)

        return results


if __name__ == "__main__":
    async def test():
        agent = ExtractorAgent()

        # 测试论文
        test_paper = {
            'paper_id': 'test',
            'title': 'Test Paper',
            'year': 2021,
            'venue': 'Test Journal',
            'citations': 100,
            'abstract': 'This is a test abstract.',
            'local_path': str(PAPERS_DIR / 'test.pdf')  # 需要存在的文件
        }

        # 如果有测试 PDF
        if Path(test_paper['local_path']).exists():
            content = await agent.extract(test_paper)
            if content:
                print(f"提取成功: {content.title}")
                print(f"  假设数: {len(content.assumptions)}")
                print(f"  公式数: {len(content.formulas)}")

    asyncio.run(test())

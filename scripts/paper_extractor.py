#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
论文内容提取工具
从 PDF 中提取关键内容用于技能校准

功能：
1. 提取摘要、方法论、关键公式
2. 生成结构化的校准笔记
3. 与现有技能文档对比
"""

import os
import sys
import re
import json
from pathlib import Path
from typing import Optional, Dict, List, Any

# 加载 .env
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

# Windows 编码修复
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# 目录配置
PAPERS_DIR = Path(__file__).parent.parent / "papers"
NOTES_DIR = Path(__file__).parent.parent / "calibration_notes"
PAPERS_DIR.mkdir(exist_ok=True)
NOTES_DIR.mkdir(exist_ok=True)


def extract_text_from_pdf(pdf_path: Path, max_pages: int = 20) -> Optional[str]:
    """
    从 PDF 提取文本

    尝试多种方法：pymupdf > pdfplumber > pypdf
    """
    text = None

    # 方法 1: PyMuPDF (fitz) - 最快最准确
    try:
        import fitz
        doc = fitz.open(pdf_path)
        pages = []
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            pages.append(page.get_text())
        text = "\n\n".join(pages)
        doc.close()
        return text
    except ImportError:
        pass
    except Exception as e:
        print(f"  [PyMuPDF 失败] {e}")

    # 方法 2: pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            pages = []
            for i, page in enumerate(pdf.pages):
                if i >= max_pages:
                    break
                page_text = page.extract_text()
                if page_text:
                    pages.append(page_text)
            text = "\n\n".join(pages)
        return text
    except ImportError:
        pass
    except Exception as e:
        print(f"  [pdfplumber 失败] {e}")

    # 方法 3: pypdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        pages = []
        for i, page in enumerate(reader.pages):
            if i >= max_pages:
                break
            pages.append(page.extract_text())
        text = "\n\n".join(pages)
        return text
    except ImportError:
        pass
    except Exception as e:
        print(f"  [pypdf 失败] {e}")

    print("  [错误] 未安装 PDF 解析库，请运行: pip install pymupdf pdfplumber pypdf")
    return None


def extract_sections(text: str) -> Dict[str, str]:
    """
    从论文文本中提取关键章节
    """
    sections = {
        'title': '',
        'abstract': '',
        'introduction': '',
        'methodology': '',
        'identification': '',
        'estimation': '',
        'assumptions': '',
        'conclusion': ''
    }

    if not text:
        return sections

    # 提取摘要
    abstract_patterns = [
        r'Abstract[:\s]*\n(.*?)(?=\n\s*(?:1\.?\s*)?Introduction|Keywords|JEL)',
        r'ABSTRACT[:\s]*\n(.*?)(?=\n\s*(?:1\.?\s*)?INTRODUCTION|Keywords|JEL)',
    ]
    for pattern in abstract_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            sections['abstract'] = match.group(1).strip()[:3000]
            break

    # 提取方法论/识别部分
    method_patterns = [
        r'(?:2\.?\s*)?(?:Methodology|Method|Identification|Setup|Framework)[:\s]*\n(.*?)(?=\n\s*(?:3\.?\s*)?(?:Data|Empirical|Application|Results))',
        r'(?:II\.?\s*)?(?:METHODOLOGY|METHOD|IDENTIFICATION|SETUP|FRAMEWORK)[:\s]*\n(.*?)(?=\n\s*(?:III\.?\s*)?)',
    ]
    for pattern in method_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            sections['methodology'] = match.group(1).strip()[:5000]
            break

    # 提取假设部分
    assumption_patterns = [
        r'Assumption[s]?\s*[:\n](.*?)(?=\n\s*(?:Theorem|Proposition|Lemma|Definition|\d+\.\d+))',
        r'(?:Key |Main |Identifying )?Assumptions?(.*?)(?=\n\s*(?:Under |Given |Theorem))',
    ]
    assumptions_found = []
    for pattern in assumption_patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        assumptions_found.extend(matches)
    if assumptions_found:
        sections['assumptions'] = "\n---\n".join([a.strip()[:1500] for a in assumptions_found[:5]])

    # 提取估计部分
    estimation_patterns = [
        r'(?:Estimation|Estimator)[:\s]*\n(.*?)(?=\n\s*(?:Inference|Asymptotic|Standard Error|Bootstrap))',
    ]
    for pattern in estimation_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            sections['estimation'] = match.group(1).strip()[:3000]
            break

    return sections


def extract_formulas(text: str) -> List[str]:
    """
    提取数学公式（简化版）
    """
    formulas = []

    # LaTeX 公式模式
    patterns = [
        r'\$\$(.*?)\$\$',  # Display math
        r'\$(.*?)\$',      # Inline math
        r'\\begin\{equation\}(.*?)\\end\{equation\}',
        r'\\begin\{align\}(.*?)\\end\{align\}',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        formulas.extend([m.strip() for m in matches if len(m.strip()) > 10])

    # 去重并限制数量
    unique_formulas = list(dict.fromkeys(formulas))
    return unique_formulas[:20]


def generate_calibration_note(
    paper_info: Dict[str, Any],
    sections: Dict[str, str],
    formulas: List[str],
    skill_name: str
) -> str:
    """
    生成校准笔记 Markdown
    """
    note = f"""# 校准笔记: {paper_info.get('title', 'Unknown')}

> **技能**: {skill_name}
> **论文 ID**: {paper_info.get('paper_id', 'N/A')}
> **年份**: {paper_info.get('year', 'N/A')}
> **期刊**: {paper_info.get('venue', 'N/A')}
> **引用数**: {paper_info.get('citations', 0)}

---

## 摘要

{sections.get('abstract', '未提取到摘要')}

---

## 核心假设

{sections.get('assumptions', '未提取到假设部分')}

---

## 方法论/识别策略

{sections.get('methodology', '未提取到方法论部分')}

---

## 估计方法

{sections.get('estimation', '未提取到估计部分')}

---

## 关键公式

"""

    if formulas:
        for i, formula in enumerate(formulas[:10], 1):
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

<!-- 手动填写或自动对比后填写 -->

"""

    return note


def process_paper(pdf_path: Path, paper_info: Dict, skill_name: str) -> Optional[Path]:
    """
    处理单篇论文，生成校准笔记
    """
    print(f"\n处理: {pdf_path.name}")

    # 提取文本
    print("  提取文本...")
    text = extract_text_from_pdf(pdf_path)
    if not text:
        print("  [跳过] 无法提取文本")
        return None

    print(f"  提取到 {len(text)} 字符")

    # 提取章节
    print("  分析章节...")
    sections = extract_sections(text)

    # 提取公式
    print("  提取公式...")
    formulas = extract_formulas(text)
    print(f"  找到 {len(formulas)} 个公式")

    # 生成笔记
    print("  生成校准笔记...")
    note = generate_calibration_note(paper_info, sections, formulas, skill_name)

    # 保存笔记
    safe_title = re.sub(r'[^\w\s-]', '', paper_info.get('title', 'unknown')[:50])
    note_filename = f"{skill_name}_{paper_info.get('year', 'XXXX')}_{safe_title}.md"
    note_path = NOTES_DIR / note_filename

    with open(note_path, 'w', encoding='utf-8') as f:
        f.write(note)

    print(f"  [OK] 保存到: {note_path.name}")
    return note_path


def batch_process_for_skill(skill_name: str, search_queries: List[str], limit_per_query: int = 3):
    """
    为特定技能批量处理论文
    """
    from paper_api import search_papers, download_pdf, PAPERS_DIR

    print(f"\n{'='*60}")
    print(f"技能校准: {skill_name}")
    print(f"{'='*60}")

    all_notes = []

    for query in search_queries:
        print(f"\n搜索: {query}")
        results = search_papers(query, limit=limit_per_query)

        if not results or 'data' not in results:
            print("  无结果")
            continue

        for paper in results['data']:
            title = paper.get('title', 'Unknown')
            year = paper.get('year', 'XXXX')
            paper_id = paper.get('paperId', '')

            # 检查是否有 PDF
            pdf_info = paper.get('openAccessPdf')
            if not pdf_info or not pdf_info.get('url'):
                print(f"  [跳过] {title[:50]}... (无 PDF)")
                continue

            pdf_url = pdf_info['url']

            # 生成文件名
            safe_title = re.sub(r'[^\w\s-]', '', title[:40])
            pdf_filename = f"{year}_{safe_title}.pdf"
            pdf_path = PAPERS_DIR / pdf_filename

            # 下载 PDF
            if not pdf_path.exists():
                print(f"  下载: {title[:50]}...")
                from paper_api import download_pdf
                if not download_pdf(pdf_url, pdf_path):
                    continue

            # 处理论文
            paper_info = {
                'title': title,
                'year': year,
                'paper_id': paper_id,
                'venue': paper.get('venue', ''),
                'citations': paper.get('citationCount', 0)
            }

            note_path = process_paper(pdf_path, paper_info, skill_name)
            if note_path:
                all_notes.append(note_path)

    print(f"\n完成! 生成 {len(all_notes)} 份校准笔记")
    return all_notes


# 预定义的技能校准查询
SKILL_QUERIES = {
    'estimator-did': [
        'Callaway Sant\'Anna difference-in-differences multiple time periods',
        'Goodman-Bacon decomposition difference-in-differences',
        'Sun Abraham event study heterogeneous treatment',
        'de Chaisemartin D\'Haultfoeuille two-way fixed effects',
    ],
    'estimator-rd': [
        'Cattaneo Idrobo Titiunik regression discontinuity practical',
        'Imbens Kalyanaraman optimal bandwidth regression discontinuity',
        'McCrary density test manipulation',
    ],
    'estimator-iv': [
        'Andrews Stock Sun weak instruments many',
        'Staiger Stock instrumental variables weak',
        'Lee McCrary Moreira Vytlacil robust weak IV',
    ],
    'estimator-psm': [
        'Imbens Rubin propensity score matching causal',
        'King Nielsen propensity score matching',
        'Rosenbaum bounds sensitivity analysis matching',
    ],
    'causal-ddml': [
        'Chernozhukov double debiased machine learning',
        'Chernozhukov Victor causal inference machine learning',
    ],
    'causal-forest': [
        'Athey Wager causal forest heterogeneous',
        'Athey Imbens recursive partitioning causal effects',
    ],
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='论文内容提取与校准笔记生成')
    parser.add_argument('--skill', '-s', type=str, help='技能名称')
    parser.add_argument('--pdf', '-p', type=str, help='单个 PDF 文件路径')
    parser.add_argument('--all', '-a', action='store_true', help='处理所有技能')
    parser.add_argument('--limit', '-l', type=int, default=2, help='每个查询的论文数量')
    parser.add_argument('--list-skills', action='store_true', help='列出可用技能')

    args = parser.parse_args()

    if args.list_skills:
        print("可用技能:")
        for skill in SKILL_QUERIES:
            print(f"  - {skill}")
        sys.exit(0)

    if args.pdf:
        # 处理单个 PDF
        pdf_path = Path(args.pdf)
        if pdf_path.exists():
            paper_info = {
                'title': pdf_path.stem,
                'year': 'XXXX',
                'paper_id': '',
                'venue': '',
                'citations': 0
            }
            process_paper(pdf_path, paper_info, args.skill or 'unknown')
        else:
            print(f"文件不存在: {pdf_path}")

    elif args.all:
        # 处理所有技能
        for skill, queries in SKILL_QUERIES.items():
            batch_process_for_skill(skill, queries, limit_per_query=args.limit)

    elif args.skill:
        # 处理特定技能
        if args.skill in SKILL_QUERIES:
            batch_process_for_skill(args.skill, SKILL_QUERIES[args.skill], limit_per_query=args.limit)
        else:
            print(f"未知技能: {args.skill}")
            print(f"可用技能: {', '.join(SKILL_QUERIES.keys())}")

    else:
        parser.print_help()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技能校准主脚本

运行完整的校准流程:
1. 搜索高引文献
2. 提取论文内容
3. 与现有文档对比
4. 生成差距分析报告
5. 输出更新建议

用法:
    python run_calibration.py --skill estimator-did
    python run_calibration.py --all
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

from agents.literature_agent import LiteratureAgent
from agents.extractor_agent import ExtractorAgent
from agents.calibration_agent import CalibrationAgent

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
CALIBRATION_DIR = PROJECT_ROOT / "calibration_notes"
SKILLS_DIR = PROJECT_ROOT / "skills"

# 技能校准配置
SKILL_CALIBRATION_CONFIG = {
    "estimator-did": {
        "queries": [
            "Callaway Sant'Anna difference-in-differences multiple time periods",
            "Goodman-Bacon decomposition difference-in-differences staggered",
            "Sun Abraham event study heterogeneous treatment effects",
            "de Chaisemartin D'Haultfoeuille two-way fixed effects negative weights",
            "Roth pre-trends testing parallel trends",
        ],
        "skill_path": "classic-methods/estimator-did",
        "min_citations": 200,
    },
    "estimator-rd": {
        "queries": [
            "Cattaneo Idrobo Titiunik regression discontinuity practical introduction",
            "Imbens Kalyanaraman optimal bandwidth regression discontinuity",
            "Calonico Cattaneo Titiunik robust nonparametric inference RD",
            "McCrary density test manipulation running variable",
            "Lee Lemieux regression discontinuity design economic applications",
        ],
        "skill_path": "classic-methods/estimator-rd",
        "min_citations": 200,
    },
    "estimator-iv": {
        "queries": [
            "Andrews Stock Sun weak instruments many instruments",
            "Staiger Stock instrumental variables weak identification",
            "Lee McCrary Moreira Porter Valid two-stage adjustment",
            "Angrist Krueger instrumental variables empirical strategies",
            "Bound Jaeger Baker problems weak instruments",
        ],
        "skill_path": "classic-methods/estimator-iv",
        "min_citations": 300,
    },
    "estimator-psm": {
        "queries": [
            "Imbens Rubin causal inference propensity score",
            "Rosenbaum sensitivity analysis matching observational studies",
            "Abadie Imbens matching estimators average treatment effects",
            "King Nielsen propensity score matching paradox",
            "Caliendo Kopeinig practical guidance propensity score matching",
        ],
        "skill_path": "classic-methods/estimator-psm",
        "min_citations": 200,
    },
    "causal-ddml": {
        "queries": [
            "Chernozhukov double debiased machine learning treatment effects",
            "Chernozhukov Victor inference heterogeneous treatment effects",
            "DoubleML Neyman orthogonal score high-dimensional",
            "Belloni Chernozhukov Hansen inference treatment effects",
        ],
        "skill_path": "causal-ml/causal-ddml",
        "min_citations": 200,
    },
    "causal-forest": {
        "queries": [
            "Athey Wager generalized random forests causal",
            "Athey Imbens recursive partitioning causal effects",
            "Wager Athey estimation inference heterogeneous treatment",
            "Athey Tibshirani Wager solving heterogeneous causal",
        ],
        "skill_path": "causal-ml/causal-forest",
        "min_citations": 200,
    },
    "structural-equation-modeling": {
        "queries": [
            "Rosseel lavaan structural equation modeling R",
            "Hu Bentler cutoff criteria fit indexes SEM",
            "Kline principles practice structural equation modeling",
            "Bollen structural equations latent variables",
        ],
        "skill_path": "causal-ml/structural-equation-modeling",
        "min_citations": 500,
    },
}


async def calibrate_skill(skill_name: str, config: dict, download_pdfs: bool = False):
    """
    对单个技能进行校准
    """
    print(f"\n{'='*70}")
    print(f"校准技能: {skill_name}")
    print(f"{'='*70}")

    # 创建输出目录
    output_dir = CALIBRATION_DIR / skill_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: 搜索文献
    print(f"\n[Step 1] 搜索高引文献...")
    literature_agent = LiteratureAgent(
        min_citations=config.get("min_citations", 100)
    )

    all_papers = []
    for query in config["queries"]:
        print(f"  查询: {query[:50]}...")
        papers = await literature_agent.search(query, limit=5)
        all_papers.extend(papers)

    # 去重并排序
    seen_ids = set()
    unique_papers = []
    for p in all_papers:
        pid = p.get('paper_id')
        if pid and pid not in seen_ids:
            seen_ids.add(pid)
            unique_papers.append(p)

    unique_papers.sort(key=lambda x: x.get('citations', 0), reverse=True)
    top_papers = unique_papers[:10]

    print(f"\n  找到 {len(unique_papers)} 篇论文，取 Top 10:")
    for i, p in enumerate(top_papers, 1):
        print(f"    {i}. [{p.get('year')}] {p.get('title', 'N/A')[:50]}... (引用: {p.get('citations', 0):,})")

    # 保存论文列表
    papers_file = output_dir / "papers.json"
    with open(papers_file, 'w', encoding='utf-8') as f:
        json.dump(top_papers, f, indent=2, ensure_ascii=False)

    # Step 2: 提取关键信息（从摘要和元数据）
    print(f"\n[Step 2] 分析论文内容...")

    paper_summaries = []
    for paper in top_papers:
        abstract = paper.get('abstract') or ''
        summary = {
            'title': paper.get('title'),
            'year': paper.get('year'),
            'citations': paper.get('citations'),
            'venue': paper.get('venue'),
            'abstract': abstract[:1000] if abstract else '',
            'key_terms': extract_key_terms(abstract, skill_name),
        }
        paper_summaries.append(summary)

    # Step 3: 差距分析
    print(f"\n[Step 3] 进行差距分析...")
    calibration_agent = CalibrationAgent(skill_name)

    # 加载现有文档
    skill_path = SKILLS_DIR / config["skill_path"]
    existing_content = load_skill_content(skill_path)

    # 分析差距
    gaps = analyze_gaps(paper_summaries, existing_content, skill_name)

    print(f"  识别到 {len(gaps)} 个差距")
    for gap in gaps[:5]:
        print(f"    - [{gap['severity']}] {gap['description'][:60]}...")

    # Step 4: 生成报告
    print(f"\n[Step 4] 生成校准报告...")

    report = generate_calibration_report(
        skill_name=skill_name,
        papers=top_papers,
        gaps=gaps,
        existing_content=existing_content
    )

    report_file = output_dir / "calibration_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n  报告已保存: {report_file}")

    # Step 5: 生成更新建议
    print(f"\n[Step 5] 生成更新建议...")

    updates = generate_update_suggestions(gaps, paper_summaries, skill_name)

    updates_file = output_dir / "suggested_updates.md"
    with open(updates_file, 'w', encoding='utf-8') as f:
        f.write(updates)

    print(f"  更新建议已保存: {updates_file}")

    return {
        'skill': skill_name,
        'papers_found': len(top_papers),
        'gaps_identified': len(gaps),
        'report_path': str(report_file),
        'updates_path': str(updates_file),
    }


def extract_key_terms(text: str, skill_name: str) -> list:
    """从文本中提取与技能相关的关键术语"""
    import re

    # 技能特定术语
    skill_terms = {
        'estimator-did': [
            'parallel trends', 'staggered adoption', 'two-way fixed effects',
            'treatment effect', 'ATT', 'heterogeneous', 'pre-trends',
            'never treated', 'not yet treated', 'group-time',
        ],
        'estimator-rd': [
            'running variable', 'cutoff', 'bandwidth', 'local polynomial',
            'sharp RD', 'fuzzy RD', 'manipulation', 'McCrary',
        ],
        'estimator-iv': [
            'instrument', 'endogenous', 'first stage', 'weak instrument',
            'LATE', '2SLS', 'just identified', 'over identified',
        ],
        'estimator-psm': [
            'propensity score', 'matching', 'balance', 'overlap',
            'common support', 'ATT', 'ATE', 'sensitivity',
        ],
        'causal-ddml': [
            'double machine learning', 'Neyman orthogonal', 'cross-fitting',
            'regularization', 'high-dimensional', 'CATE',
        ],
        'causal-forest': [
            'heterogeneous', 'CATE', 'splitting', 'honest',
            'confidence interval', 'treatment effect',
        ],
        'structural-equation-modeling': [
            'latent variable', 'factor loading', 'CFI', 'RMSEA',
            'measurement model', 'structural model', 'fit indices',
        ],
    }

    terms_to_find = skill_terms.get(skill_name, [])
    found_terms = []

    text_lower = text.lower()
    for term in terms_to_find:
        if term.lower() in text_lower:
            found_terms.append(term)

    return found_terms


def load_skill_content(skill_path: Path) -> dict:
    """加载技能现有内容"""
    content = {
        'skill_md': '',
        'references': {},
        'scripts': [],
    }

    # 读取 SKILL.md
    skill_md = skill_path / "SKILL.md"
    if skill_md.exists():
        with open(skill_md, 'r', encoding='utf-8') as f:
            content['skill_md'] = f.read()

    # 读取 references
    refs_dir = skill_path / "references"
    if refs_dir.exists():
        for ref_file in refs_dir.glob("*.md"):
            with open(ref_file, 'r', encoding='utf-8') as f:
                content['references'][ref_file.name] = f.read()

    # 列出 scripts
    scripts_dir = skill_path / "scripts"
    if scripts_dir.exists():
        content['scripts'] = [f.name for f in scripts_dir.glob("*.py")]

    return content


def analyze_gaps(papers: list, existing: dict, skill_name: str) -> list:
    """分析论文与现有文档的差距"""
    gaps = []

    all_text = existing['skill_md'] + ' '.join(existing['references'].values())
    all_text_lower = all_text.lower()

    # 从论文摘要中提取可能缺失的内容
    for paper in papers:
        abstract = paper.get('abstract', '').lower()
        title = paper.get('title', '').lower()
        key_terms = paper.get('key_terms', [])

        # 检查作者方法是否被引用
        # 提取作者名（简化处理）
        if paper.get('citations', 0) > 1000:
            # 高引论文应该被提及
            year = paper.get('year', '')
            # 检查是否在参考文献中
            if str(year) not in all_text and paper.get('title', '')[:20].lower() not in all_text_lower:
                gaps.append({
                    'category': 'reference',
                    'severity': 'major',
                    'description': f"高引论文未在参考文献中提及: {paper.get('title', '')[:60]}... ({paper.get('citations', 0):,} 引用)",
                    'source': paper.get('title'),
                    'suggestion': f"添加引用: {paper.get('title')} ({year})",
                })

        # 检查关键术语
        for term in key_terms:
            if term.lower() not in all_text_lower:
                gaps.append({
                    'category': 'terminology',
                    'severity': 'minor',
                    'description': f"术语 '{term}' 在论文中出现但文档中缺失",
                    'source': paper.get('title'),
                    'suggestion': f"考虑添加关于 '{term}' 的说明",
                })

    # 检查最新方法（2020年后的论文）
    recent_papers = [p for p in papers if p.get('year', 0) >= 2020]
    if recent_papers:
        recent_titles = [p.get('title', '')[:30].lower() for p in recent_papers]
        for paper in recent_papers:
            if paper.get('citations', 0) > 500:
                year = paper.get('year')
                if str(year) not in all_text:
                    gaps.append({
                        'category': 'method',
                        'severity': 'major',
                        'description': f"近期高引方法可能缺失: {paper.get('title', '')[:60]}... ({year})",
                        'source': paper.get('title'),
                        'suggestion': "检查是否需要添加此方法的介绍",
                    })

    # 去重
    seen = set()
    unique_gaps = []
    for gap in gaps:
        key = gap['description'][:50]
        if key not in seen:
            seen.add(key)
            unique_gaps.append(gap)

    return unique_gaps


def generate_calibration_report(skill_name: str, papers: list, gaps: list, existing_content: dict) -> str:
    """生成校准报告"""

    report = f"""# 校准报告: {skill_name}

> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 概述

本报告通过检索高引学术文献，对 `{skill_name}` 技能文档进行校准分析。

### 现有文档结构

- SKILL.md: {'✓ 存在' if existing_content['skill_md'] else '✗ 缺失'}
- references/: {len(existing_content['references'])} 个文件
- scripts/: {len(existing_content['scripts'])} 个文件

---

## 2. 高引文献检索结果

共检索到 {len(papers)} 篇高引论文:

| # | 年份 | 标题 | 引用数 | 期刊 |
|---|------|------|--------|------|
"""

    for i, p in enumerate(papers, 1):
        title = p.get('title', 'N/A')[:50] + '...' if len(p.get('title', '')) > 50 else p.get('title', 'N/A')
        venue = p.get('venue', 'N/A')[:30] if p.get('venue') else 'N/A'
        report += f"| {i} | {p.get('year', 'N/A')} | {title} | {p.get('citations', 0):,} | {venue} |\n"

    report += f"""

---

## 3. 差距分析

共识别到 **{len(gaps)}** 个差距:

### 按严重程度分类

"""

    critical = [g for g in gaps if g['severity'] == 'critical']
    major = [g for g in gaps if g['severity'] == 'major']
    minor = [g for g in gaps if g['severity'] == 'minor']

    if critical:
        report += f"#### 关键差距 ({len(critical)})\n\n"
        for g in critical:
            report += f"- **{g['description']}**\n  - 建议: {g['suggestion']}\n\n"

    if major:
        report += f"#### 重要差距 ({len(major)})\n\n"
        for g in major:
            report += f"- {g['description']}\n  - 建议: {g['suggestion']}\n\n"

    if minor:
        report += f"#### 次要差距 ({len(minor)})\n\n"
        for g in minor[:10]:  # 只显示前10个
            report += f"- {g['description']}\n"
        if len(minor) > 10:
            report += f"\n*...及其他 {len(minor) - 10} 个次要差距*\n"

    report += """

---

## 4. 核心文献清单

以下论文应确保在技能文档中被正确引用:

"""

    for p in papers[:5]:
        if p.get('citations', 0) > 500:
            report += f"- [ ] **{p.get('title')}** ({p.get('year')}) - {p.get('citations', 0):,} 引用\n"

    report += """

---

## 5. 下一步行动

1. 审查上述差距分析
2. 更新 SKILL.md 和 references/ 文档
3. 确保高引论文被正确引用
4. 补充缺失的方法论内容

"""

    return report


def generate_update_suggestions(gaps: list, papers: list, skill_name: str) -> str:
    """生成具体的更新建议"""

    updates = f"""# 更新建议: {skill_name}

> 基于校准分析生成的具体更新建议

## 参考文献更新

以下引用应添加到 `references/` 或 SKILL.md:

```bibtex
"""

    for p in papers[:5]:
        if p.get('citations', 0) > 500:
            # 生成简化的 BibTeX
            authors = "Author et al."
            title = p.get('title', 'Unknown')
            year = p.get('year', 'XXXX')
            venue = p.get('venue', 'Journal')

            updates += f"""
@article{{{skill_name}_{year},
  title = {{{title}}},
  author = {{{authors}}},
  year = {{{year}}},
  journal = {{{venue}}}
}}
"""

    updates += """```

## 内容更新建议

"""

    # 按类别组织建议
    by_category = {}
    for gap in gaps:
        cat = gap.get('category', 'other')
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(gap)

    category_names = {
        'reference': '参考文献',
        'method': '方法论',
        'terminology': '术语说明',
        'diagnostic': '诊断测试',
        'other': '其他',
    }

    for cat, cat_gaps in by_category.items():
        updates += f"### {category_names.get(cat, cat)}\n\n"
        for g in cat_gaps[:5]:
            updates += f"- [ ] {g['suggestion']}\n"
        updates += "\n"

    updates += """
## 文件修改清单

- [ ] `SKILL.md` - 更新参考文献部分
- [ ] `references/identification_assumptions.md` - 检查假设完整性
- [ ] `references/estimation_methods.md` - 添加新方法
- [ ] `references/diagnostic_tests.md` - 补充诊断测试

"""

    return updates


async def main():
    parser = argparse.ArgumentParser(description='技能校准脚本')
    parser.add_argument('--skill', '-s', type=str, help='要校准的技能名称')
    parser.add_argument('--all', '-a', action='store_true', help='校准所有技能')
    parser.add_argument('--list', '-l', action='store_true', help='列出可用技能')
    parser.add_argument('--download', '-d', action='store_true', help='下载 PDF（耗时）')

    args = parser.parse_args()

    if args.list:
        print("可用技能:")
        for skill in SKILL_CALIBRATION_CONFIG:
            print(f"  - {skill}")
        return

    if args.all:
        results = []
        for skill_name, config in SKILL_CALIBRATION_CONFIG.items():
            try:
                result = await calibrate_skill(skill_name, config, args.download)
                results.append(result)
            except Exception as e:
                print(f"[错误] {skill_name}: {e}")

        print("\n" + "="*70)
        print("校准完成摘要")
        print("="*70)
        for r in results:
            print(f"  {r['skill']}: {r['papers_found']} 论文, {r['gaps_identified']} 差距")

    elif args.skill:
        if args.skill in SKILL_CALIBRATION_CONFIG:
            await calibrate_skill(args.skill, SKILL_CALIBRATION_CONFIG[args.skill], args.download)
        else:
            print(f"未知技能: {args.skill}")
            print(f"可用技能: {', '.join(SKILL_CALIBRATION_CONFIG.keys())}")

    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())

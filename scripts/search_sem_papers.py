#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
搜索 SEM 高引论文脚本

用于校准 structural-equation-modeling 技能
"""

import asyncio
import sys
from pathlib import Path

# 添加 agents 目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from agents.literature_agent import LiteratureAgent, SEM_QUERIES


async def main():
    """搜索 SEM 高引论文"""
    print("=" * 70)
    print("搜索 SEM 方法论高引论文")
    print("=" * 70)

    # 创建文献搜索 agent
    agent = LiteratureAgent(min_citations=500)  # 高引论文

    all_papers = []

    for query in SEM_QUERIES[:6]:  # 前 6 个查询
        print(f"\n搜索: {query}")
        papers = await agent.search(query, limit=5)

        for paper in papers:
            print(f"  [{paper.get('year')}] {paper.get('title', 'N/A')[:60]}...")
            print(f"       引用: {paper.get('citations', 0)} | 期刊: {paper.get('venue', 'N/A')[:40]}")
            print(f"       PDF: {'有' if paper.get('pdf_url') else '无'}")

        all_papers.extend(papers)

    # 去重
    seen = set()
    unique_papers = []
    for p in all_papers:
        pid = p.get('paper_id')
        if pid and pid not in seen:
            seen.add(pid)
            unique_papers.append(p)

    # 按引用数排序
    unique_papers.sort(key=lambda x: x.get('citations', 0), reverse=True)

    print("\n" + "=" * 70)
    print(f"共找到 {len(unique_papers)} 篇去重后的高引论文")
    print("=" * 70)

    print("\nTop 10 高引论文:")
    for i, paper in enumerate(unique_papers[:10], 1):
        print(f"{i:2}. [{paper.get('year')}] {paper.get('title', 'N/A')[:70]}...")
        print(f"    引用: {paper.get('citations', 0):,} | {paper.get('venue', 'N/A')[:50]}")

    # 保存结果
    import json
    output_path = Path(__file__).parent.parent / "calibration_notes" / "sem_papers.json"
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unique_papers, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())

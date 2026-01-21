#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LiteratureAgent - 文献搜索智能体

职责:
1. 搜索高引学术论文
2. 筛选顶刊文献
3. 下载 PDF 文件
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Windows 编码修复
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
except ImportError:
    pass

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
PAPERS_DIR = PROJECT_ROOT / "papers"
PAPERS_DIR.mkdir(exist_ok=True)


@dataclass
class Paper:
    """论文数据结构"""
    paper_id: str
    title: str
    authors: List[str]
    year: int
    venue: str
    citations: int
    abstract: str
    pdf_url: Optional[str] = None
    local_path: Optional[Path] = None


class LiteratureAgent:
    """
    文献搜索智能体

    功能:
    - 通过 API 搜索学术论文
    - 按引用数和期刊筛选
    - 下载 PDF 文件
    """

    # 高质量期刊列表
    TOP_JOURNALS = {
        # 经济学顶刊
        "American Economic Review",
        "Econometrica",
        "Journal of Political Economy",
        "Quarterly Journal of Economics",
        "Review of Economic Studies",
        # 计量经济学顶刊
        "Journal of Econometrics",
        "Review of Economics and Statistics",
        "Journal of Business & Economic Statistics",
        "Econometric Theory",
        # 统计学顶刊
        "Journal of the American Statistical Association",
        "Annals of Statistics",
        "Biometrika",
        "Journal of the Royal Statistical Society",
        # 机器学习/因果
        "Journal of Machine Learning Research",
        "Machine Learning",
        # 心理学方法 (SEM 相关)
        "Psychological Methods",
        "Structural Equation Modeling",
        "Multivariate Behavioral Research",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        min_citations: int = 100,
        target_journals: Optional[List[str]] = None
    ):
        self.api_key = api_key or os.environ.get("AI4SCHOLAR_API_KEY", "")
        self.min_citations = min_citations
        self.target_journals = set(target_journals) if target_journals else self.TOP_JOURNALS
        self.base_url = "https://ai4scholar.net"

    async def search(
        self,
        query: str,
        limit: int = 10,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索论文

        Parameters
        ----------
        query : str
            搜索关键词
        limit : int
            返回数量上限
        year_from : int, optional
            起始年份
        year_to : int, optional
            结束年份

        Returns
        -------
        List[Dict]
            符合条件的论文列表
        """
        import requests

        url = f"{self.base_url}/graph/v1/paper/search"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        params = {
            'query': query,
            'limit': limit * 3,  # 多获取一些用于筛选
            'fields': 'paperId,title,authors,year,abstract,citationCount,venue,openAccessPdf,externalIds'
        }

        if year_from:
            params['year'] = f"{year_from}-"
        if year_to:
            params['year'] = f"-{year_to}" if not year_from else f"{year_from}-{year_to}"

        try:
            # 使用同步请求（可以后续改为 aiohttp）
            response = await asyncio.to_thread(
                requests.get, url, headers=headers, params=params, timeout=30
            )
            response.raise_for_status()
            data = response.json()

            papers = data.get('data', [])

            # 筛选高引和顶刊论文
            filtered = self._filter_papers(papers)

            print(f"  [LiteratureAgent] 搜索 '{query[:50]}...' -> {len(papers)} 篇, 筛选后 {len(filtered)} 篇")

            return filtered[:limit]

        except Exception as e:
            print(f"  [LiteratureAgent] 搜索失败: {e}")
            return []

    def _filter_papers(self, papers: List[Dict]) -> List[Dict]:
        """筛选论文"""
        filtered = []

        for paper in papers:
            citations = paper.get('citationCount', 0) or 0
            venue = paper.get('venue', '') or ''

            # 引用数筛选
            if citations < self.min_citations:
                continue

            # 期刊筛选（如果有期刊信息）
            if venue and self.target_journals:
                venue_match = any(
                    journal.lower() in venue.lower()
                    for journal in self.target_journals
                )
                # 高引论文即使不是顶刊也保留
                if not venue_match and citations < 500:
                    continue

            filtered.append({
                'paper_id': paper.get('paperId'),
                'title': paper.get('title'),
                'authors': [a.get('name') for a in paper.get('authors', [])],
                'year': paper.get('year'),
                'venue': venue,
                'citations': citations,
                'abstract': paper.get('abstract', ''),
                'pdf_url': paper.get('openAccessPdf', {}).get('url') if paper.get('openAccessPdf') else None,
                'external_ids': paper.get('externalIds', {})
            })

        # 按引用数排序
        filtered.sort(key=lambda x: x.get('citations', 0), reverse=True)

        return filtered

    async def download_pdf(self, paper: Dict) -> Optional[Path]:
        """
        下载论文 PDF

        Parameters
        ----------
        paper : Dict
            论文信息字典

        Returns
        -------
        Path or None
            下载的文件路径
        """
        import requests
        import re

        pdf_url = paper.get('pdf_url')
        if not pdf_url:
            print(f"  [LiteratureAgent] 无 PDF: {paper.get('title', 'Unknown')[:50]}...")
            return None

        # 生成文件名
        title = paper.get('title', 'unknown')
        year = paper.get('year', 'XXXX')
        safe_title = re.sub(r'[^\w\s-]', '', title[:50])
        filename = f"{year}_{safe_title}.pdf"
        save_path = PAPERS_DIR / filename

        if save_path.exists():
            print(f"  [LiteratureAgent] 已存在: {filename}")
            return save_path

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = await asyncio.to_thread(
                requests.get, pdf_url, headers=headers, timeout=120, stream=True
            )
            response.raise_for_status()

            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"  [LiteratureAgent] 下载成功: {filename}")
            return save_path

        except Exception as e:
            print(f"  [LiteratureAgent] 下载失败: {e}")
            return None

    async def search_and_download(
        self,
        query: str,
        limit: int = 5,
        download: bool = True
    ) -> List[Dict]:
        """
        搜索并下载论文

        Parameters
        ----------
        query : str
            搜索关键词
        limit : int
            下载数量上限
        download : bool
            是否下载 PDF

        Returns
        -------
        List[Dict]
            论文列表（包含本地路径）
        """
        papers = await self.search(query, limit=limit)

        if download:
            for paper in papers:
                local_path = await self.download_pdf(paper)
                paper['local_path'] = str(local_path) if local_path else None

        return papers

    async def batch_search(
        self,
        queries: List[str],
        limit_per_query: int = 3
    ) -> List[Dict]:
        """
        批量搜索多个查询

        Parameters
        ----------
        queries : List[str]
            查询列表
        limit_per_query : int
            每个查询的结果数量

        Returns
        -------
        List[Dict]
            所有查询的结果（去重后）
        """
        all_papers = []
        seen_ids = set()

        for query in queries:
            papers = await self.search(query, limit=limit_per_query)
            for paper in papers:
                paper_id = paper.get('paper_id')
                if paper_id and paper_id not in seen_ids:
                    seen_ids.add(paper_id)
                    all_papers.append(paper)

        # 按引用数排序
        all_papers.sort(key=lambda x: x.get('citations', 0), reverse=True)

        return all_papers


# SEM 相关搜索查询
SEM_QUERIES = [
    # 经典 SEM
    "Bollen structural equation modeling latent variables",
    "Jöreskog LISREL confirmatory factor analysis",
    # R 实现
    "Rosseel lavaan structural equation modeling R",
    "semPlot visualization SEM R",
    # Python 实现
    "semopy structural equation modeling Python",
    # 高级方法
    "Muthén Muthén mixture modeling latent class",
    "multilevel structural equation modeling",
    # 应用方法论
    "Kline principles practice structural equation modeling",
    "MacCallum model specification SEM",
    # 模型评估
    "Hu Bentler fit indices structural equation",
    "CFI RMSEA SRMR model fit",
]


if __name__ == "__main__":
    async def test():
        agent = LiteratureAgent(min_citations=50)

        # 测试搜索
        papers = await agent.search(
            "structural equation modeling lavaan",
            limit=5
        )

        print(f"\n找到 {len(papers)} 篇论文:")
        for p in papers:
            print(f"  - [{p['year']}] {p['title'][:60]}... (引用: {p['citations']})")

    asyncio.run(test())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
论文检索与PDF下载工具
基于 ai4scholar.net API

用于获取顶刊论文以校正 causal-ml-skills
"""

import requests
import json
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

# 修复 Windows 终端编码问题
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

API_KEY = os.environ.get("AI4SCHOLAR_API_KEY", "your-api-key-here")
BASE_URL = "https://ai4scholar.net"

# 论文保存目录
PAPERS_DIR = Path(__file__).parent.parent / "papers"
PAPERS_DIR.mkdir(exist_ok=True)


def search_papers(
    query: str,
    limit: int = 10,
    fields: str = "paperId,title,authors,year,abstract,citationCount,venue,openAccessPdf,externalIds"
) -> Optional[Dict[str, Any]]:
    """
    搜索论文

    Parameters
    ----------
    query : str
        搜索关键词
    limit : int
        返回数量限制
    fields : str
        返回字段

    Returns
    -------
    dict or None
        搜索结果
    """
    url = f"{BASE_URL}/graph/v1/paper/search"
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    params = {
        'query': query,
        'limit': limit,
        'fields': fields
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"响应内容: {e.response.text}")
        return None


def get_paper_by_id(paper_id: str) -> Optional[Dict[str, Any]]:
    """
    通过 paper ID 获取论文详情
    """
    url = f"{BASE_URL}/graph/v1/paper/{paper_id}"
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    params = {
        'fields': 'paperId,title,authors,year,abstract,citationCount,venue,openAccessPdf,externalIds'
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return None


def download_pdf(pdf_url: str, save_path: Path) -> bool:
    """
    下载 PDF 文件
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(pdf_url, timeout=120, stream=True, headers=headers)
        response.raise_for_status()

        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"[OK] PDF 已保存: {save_path}")
        return True
    except Exception as e:
        print(f"[FAIL] 下载失败: {e}")
        return False


def search_and_download(query: str, limit: int = 5, download: bool = False) -> List[Dict]:
    """
    搜索论文并可选下载 PDF
    """
    print(f"\n{'='*70}")
    print(f"搜索: {query}")
    print(f"{'='*70}")

    results = search_papers(query, limit=limit)

    if not results:
        print("搜索失败或无结果")
        return []

    papers = results.get('data', [])
    total = results.get('total', 0)

    print(f"\n找到 {total} 篇论文，显示前 {len(papers)} 篇:\n")

    downloaded = []

    for i, paper in enumerate(papers, 1):
        title = paper.get('title', 'N/A')
        year = paper.get('year', 'N/A')
        venue = paper.get('venue', 'N/A')
        citations = paper.get('citationCount', 0)
        paper_id = paper.get('paperId', 'N/A')

        print(f"[{i}] {title}")
        print(f"    年份: {year} | 期刊: {venue} | 引用: {citations}")

        # 检查是否有 PDF
        pdf_info = paper.get('openAccessPdf')
        if pdf_info and pdf_info.get('url'):
            pdf_url = pdf_info.get('url')
            print(f"    PDF:  {pdf_url}")

            if download:
                # 生成文件名
                safe_title = "".join(c if c.isalnum() or c in ' -_' else '_' for c in title[:50])
                filename = f"{year}_{safe_title}.pdf"
                save_path = PAPERS_DIR / filename

                if save_path.exists():
                    print(f"    [SKIP] 已存在: {save_path}")
                else:
                    if download_pdf(pdf_url, save_path):
                        downloaded.append({
                            'title': title,
                            'year': year,
                            'path': str(save_path),
                            'paper_id': paper_id
                        })
        else:
            print(f"    PDF:  无开放获取")

        print(f"    ID:   {paper_id}")
        print()

    if download:
        print(f"\n下载完成: {len(downloaded)} 篇")

    return papers


def get_methodology_papers():
    """
    获取因果推断方法论核心论文
    """
    methodology_queries = {
        'DID': [
            'Callaway Sant\'Anna difference-in-differences multiple time periods',
            'Sun Abraham event study heterogeneous treatment',
            'Goodman-Bacon decomposition difference-in-differences',
            'de Chaisemartin D\'Haultfoeuille two-way fixed effects',
        ],
        'RD': [
            'Cattaneo Idrobo Titiunik regression discontinuity',
            'Lee Lemieux regression discontinuity design',
            'Imbens Kalyanaraman optimal bandwidth regression discontinuity',
        ],
        'IV': [
            'Andrews Stock Sun weak instruments',
            'Angrist Pischke instrumental variables',
            'Staiger Stock instrumental variables weak',
        ],
        'PSM': [
            'Imbens matching propensity score causal',
            'King Nielsen propensity score matching',
            'Rosenbaum Rubin propensity score',
        ],
        'DDML': [
            'Chernozhukov double machine learning econometrics',
            'DoubleML Neyman orthogonal',
        ],
        'Causal Forest': [
            'Athey Wager causal forest',
            'Athey Imbens recursive partitioning causal',
        ],
    }

    all_papers = {}

    for method, queries in methodology_queries.items():
        print(f"\n{'#'*70}")
        print(f"# {method} 方法论论文")
        print(f"{'#'*70}")

        method_papers = []
        for query in queries:
            papers = search_and_download(query, limit=3, download=True)
            method_papers.extend(papers)

        all_papers[method] = method_papers

    return all_papers


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='论文检索与下载工具')
    parser.add_argument('--query', '-q', type=str, help='搜索关键词')
    parser.add_argument('--limit', '-l', type=int, default=5, help='返回数量')
    parser.add_argument('--download', '-d', action='store_true', help='下载PDF')
    parser.add_argument('--methodology', '-m', action='store_true', help='获取方法论核心论文')

    args = parser.parse_args()

    if args.methodology:
        get_methodology_papers()
    elif args.query:
        search_and_download(args.query, limit=args.limit, download=args.download)
    else:
        # 默认测试
        print("测试搜索 DID 方法论论文:")
        search_and_download("Callaway Sant'Anna difference-in-differences", limit=3, download=False)

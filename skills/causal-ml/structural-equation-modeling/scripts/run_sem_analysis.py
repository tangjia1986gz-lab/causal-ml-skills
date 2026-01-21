#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SEM 分析脚本

用法:
    python run_sem_analysis.py data.csv --model model.txt --output results/

功能:
    1. 数据预处理和描述统计
    2. 测量模型 (CFA) 分析
    3. 结构模型分析
    4. 生成报告和图表
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sem_estimator import fit_sem, fit_cfa, calculate_reliability
    SEM_AVAILABLE = True
except ImportError:
    SEM_AVAILABLE = False
    print("警告: sem_estimator 未找到，将使用简化功能")


def load_data(file_path: str) -> pd.DataFrame:
    """加载数据文件"""
    path = Path(file_path)

    if path.suffix == '.csv':
        return pd.read_csv(path)
    elif path.suffix in ['.xlsx', '.xls']:
        return pd.read_excel(path)
    elif path.suffix == '.dta':
        return pd.read_stata(path)
    elif path.suffix == '.sav':
        import pyreadstat
        df, _ = pyreadstat.read_sav(path)
        return df
    else:
        raise ValueError(f"不支持的文件格式: {path.suffix}")


def descriptive_statistics(data: pd.DataFrame, variables: list) -> pd.DataFrame:
    """计算描述统计"""
    stats = data[variables].describe().T
    stats['skewness'] = data[variables].skew()
    stats['kurtosis'] = data[variables].kurtosis()
    return stats


def correlation_matrix(data: pd.DataFrame, variables: list) -> pd.DataFrame:
    """计算相关矩阵"""
    return data[variables].corr()


def check_normality(data: pd.DataFrame, variables: list) -> dict:
    """检验多元正态性"""
    from scipy import stats as scipy_stats

    results = {
        'univariate': {},
        'multivariate': {}
    }

    # 单变量正态性检验 (Shapiro-Wilk)
    for var in variables:
        if len(data[var].dropna()) <= 5000:  # Shapiro-Wilk 限制
            stat, p = scipy_stats.shapiro(data[var].dropna())
            results['univariate'][var] = {
                'statistic': stat,
                'p_value': p,
                'normal': p > 0.05
            }

    # 偏度和峰度
    skewness = data[variables].skew()
    kurtosis = data[variables].kurtosis()

    results['multivariate']['max_skewness'] = skewness.abs().max()
    results['multivariate']['max_kurtosis'] = kurtosis.abs().max()
    results['multivariate']['severe_nonnormal'] = (
        skewness.abs().max() > 2 or kurtosis.abs().max() > 7
    )

    return results


def run_cfa(data: pd.DataFrame, model: str, estimator: str = "ML") -> dict:
    """运行验证性因子分析"""
    if not SEM_AVAILABLE:
        return {"error": "semopy 未安装"}

    result = fit_cfa(data, model, estimator=estimator)

    output = {
        'fit_indices': {
            'chi_square': result.fit_indices.chi_square,
            'df': result.fit_indices.df,
            'p_value': result.fit_indices.p_value,
            'cfi': result.fit_indices.cfi,
            'tli': result.fit_indices.tli,
            'rmsea': result.fit_indices.rmsea,
            'srmr': result.fit_indices.srmr,
        },
        'factor_loadings': result.factor_loadings.to_dict() if not result.factor_loadings.empty else {},
        'converged': result.converged,
    }

    # 计算信度
    if not result.factor_loadings.empty:
        factors = result.factor_loadings['Factor'].unique()
        output['reliability'] = {}
        for factor in factors:
            rel = calculate_reliability(result, factor)
            output['reliability'][factor] = rel

    return output


def run_sem(data: pd.DataFrame, model: str, estimator: str = "ML") -> dict:
    """运行结构方程模型"""
    if not SEM_AVAILABLE:
        return {"error": "semopy 未安装"}

    result = fit_sem(data, model, estimator=estimator)

    output = {
        'fit_indices': {
            'chi_square': result.fit_indices.chi_square,
            'df': result.fit_indices.df,
            'p_value': result.fit_indices.p_value,
            'cfi': result.fit_indices.cfi,
            'tli': result.fit_indices.tli,
            'rmsea': result.fit_indices.rmsea,
            'srmr': result.fit_indices.srmr,
            'aic': result.fit_indices.aic,
            'bic': result.fit_indices.bic,
        },
        'parameters': [
            {
                'lhs': p.lhs,
                'op': p.op,
                'rhs': p.rhs,
                'estimate': p.estimate,
                'se': p.se,
                'z_value': p.z_value,
                'p_value': p.p_value,
                'std_estimate': p.std_estimate,
            }
            for p in result.parameters
        ],
        'r_squared': result.r_squared,
        'converged': result.converged,
        'summary': result.summary(),
    }

    return output


def generate_report(
    data: pd.DataFrame,
    model: str,
    output_dir: Path,
    estimator: str = "ML"
) -> None:
    """生成完整分析报告"""

    output_dir.mkdir(parents=True, exist_ok=True)

    # 解析模型获取变量列表
    variables = extract_variables_from_model(model)

    # 1. 描述统计
    print("生成描述统计...")
    desc_stats = descriptive_statistics(data, variables)
    desc_stats.to_csv(output_dir / "descriptive_statistics.csv")

    # 2. 相关矩阵
    print("计算相关矩阵...")
    corr = correlation_matrix(data, variables)
    corr.to_csv(output_dir / "correlation_matrix.csv")

    # 3. 正态性检验
    print("检验正态性...")
    normality = check_normality(data, variables)
    with open(output_dir / "normality_test.json", 'w') as f:
        json.dump(normality, f, indent=2, default=str)

    # 4. SEM 分析
    print("运行 SEM 分析...")
    sem_results = run_sem(data, model, estimator=estimator)

    with open(output_dir / "sem_results.json", 'w', encoding='utf-8') as f:
        json.dump(sem_results, f, indent=2, ensure_ascii=False, default=str)

    # 5. 生成 Markdown 报告
    print("生成报告...")
    report = generate_markdown_report(desc_stats, corr, normality, sem_results)
    with open(output_dir / "analysis_report.md", 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"报告已保存到: {output_dir}")


def extract_variables_from_model(model: str) -> list:
    """从模型语法中提取变量名"""
    import re

    # 匹配所有可能的变量名
    # 排除操作符和保留字
    operators = {'=~', '~', '~~', ':=', '*', '+'}

    tokens = re.findall(r'[\w]+', model)
    variables = [t for t in tokens if t not in operators and not t.isdigit()]

    return list(set(variables))


def generate_markdown_report(
    desc_stats: pd.DataFrame,
    corr: pd.DataFrame,
    normality: dict,
    sem_results: dict
) -> str:
    """生成 Markdown 格式报告"""

    report = """# 结构方程模型分析报告

## 1. 描述统计

"""
    report += desc_stats.to_markdown()

    report += """

## 2. 相关矩阵

"""
    report += corr.round(3).to_markdown()

    report += """

## 3. 正态性检验

"""
    if normality['multivariate']['severe_nonnormal']:
        report += "**警告**: 数据存在严重非正态性，建议使用稳健估计方法 (MLR)。\n\n"
    else:
        report += "数据正态性可接受。\n\n"

    report += f"- 最大偏度: {normality['multivariate']['max_skewness']:.3f}\n"
    report += f"- 最大峰度: {normality['multivariate']['max_kurtosis']:.3f}\n"

    report += """

## 4. 模型拟合

"""
    if 'error' in sem_results:
        report += f"错误: {sem_results['error']}\n"
    else:
        fit = sem_results['fit_indices']
        report += f"""
| 指标 | 值 | 标准 | 评估 |
|------|-----|------|------|
| χ² | {fit['chi_square']:.2f} | - | - |
| df | {fit['df']} | - | - |
| p | {fit['p_value']:.4f} | > .05 | {'✓' if fit['p_value'] > 0.05 else '✗'} |
| CFI | {fit['cfi']:.3f} | ≥ .95 | {'✓' if fit['cfi'] >= 0.95 else '○' if fit['cfi'] >= 0.90 else '✗'} |
| TLI | {fit['tli']:.3f} | ≥ .95 | {'✓' if fit['tli'] >= 0.95 else '○' if fit['tli'] >= 0.90 else '✗'} |
| RMSEA | {fit['rmsea']:.3f} | ≤ .06 | {'✓' if fit['rmsea'] <= 0.06 else '○' if fit['rmsea'] <= 0.08 else '✗'} |
| SRMR | {fit['srmr']:.3f} | ≤ .08 | {'✓' if fit['srmr'] <= 0.08 else '✗'} |

注: ✓ = 良好, ○ = 可接受, ✗ = 不佳
"""

        if sem_results.get('summary'):
            report += f"""

## 5. 详细结果

```
{sem_results['summary']}
```
"""

    report += """

---

*报告由 SEM 分析脚本自动生成*
"""

    return report


def generate_lavaan_code(model: str) -> str:
    """生成 R lavaan 代码"""
    code = f'''
# R lavaan 代码
library(lavaan)

# 模型定义
model <- '
{model}
'

# 拟合模型
fit <- sem(model, data = data, estimator = "ML")

# 查看结果
summary(fit, fit.measures = TRUE, standardized = TRUE)

# 修正指数
modindices(fit, sort = TRUE, minimum.value = 10)

# 信度
library(semTools)
reliability(fit)

# 路径图
library(semPlot)
semPaths(fit, "std", layout = "tree")
'''
    return code


def main():
    parser = argparse.ArgumentParser(
        description='SEM 分析脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
    # 基本使用
    python run_sem_analysis.py data.csv --model model.txt

    # 指定输出目录和估计方法
    python run_sem_analysis.py data.csv --model model.txt --output results/ --estimator MLR

    # 仅运行 CFA
    python run_sem_analysis.py data.csv --model cfa_model.txt --cfa-only
'''
    )

    parser.add_argument('data', help='数据文件路径 (CSV, Excel, Stata, SPSS)')
    parser.add_argument('--model', '-m', required=True, help='模型定义文件或字符串')
    parser.add_argument('--output', '-o', default='sem_results', help='输出目录')
    parser.add_argument('--estimator', '-e', default='ML',
                       choices=['ML', 'MLR', 'WLSMV', 'ULS'],
                       help='估计方法')
    parser.add_argument('--cfa-only', action='store_true', help='仅运行 CFA')
    parser.add_argument('--generate-r', action='store_true', help='生成 R lavaan 代码')

    args = parser.parse_args()

    # 加载数据
    print(f"加载数据: {args.data}")
    data = load_data(args.data)
    print(f"样本量: {len(data)}")

    # 加载模型
    model_path = Path(args.model)
    if model_path.exists():
        with open(model_path, 'r', encoding='utf-8') as f:
            model = f.read()
    else:
        model = args.model

    print(f"模型:\n{model}")

    # 生成 R 代码
    if args.generate_r:
        r_code = generate_lavaan_code(model)
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "lavaan_code.R", 'w', encoding='utf-8') as f:
            f.write(r_code)
        print(f"R 代码已保存到: {output_dir / 'lavaan_code.R'}")

    # 运行分析
    output_dir = Path(args.output)

    if args.cfa_only:
        print("运行 CFA...")
        results = run_cfa(data, model, estimator=args.estimator)
        with open(output_dir / "cfa_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    else:
        generate_report(data, model, output_dir, estimator=args.estimator)


if __name__ == "__main__":
    main()

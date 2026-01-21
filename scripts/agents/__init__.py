"""
多智能体校准系统 (Multi-Agent Calibration System) v2.0

扩展架构:
- BaseAgent: 所有 Agent 的公共基类
- OrchestratorAgent: 协调所有 Agent，管理工作流
- LiteratureAgent: 搜索高引文献
- ExtractorAgent: 提取论文关键内容
- CalibrationAgent: 对比现有文档，识别差距 (旧版)
- GapAnalyzer: 组件级差距分析 (新版)
- FormulaAgent: 公式一致性验证
- CitationAgent: 引用完整性检查
- ValidatorAgent: 四级质量门控
- UpdaterAgent: 文档自动更新

使用方法:
```python
import asyncio
from agents import (
    OrchestratorAgent,
    LiteratureAgent,
    GapAnalyzer,
    FormulaAgent,
    CitationAgent,
    ValidatorAgent,
    UpdaterAgent,
)

async def main():
    # 1. 文献搜索
    lit_agent = LiteratureAgent(min_citations=200)
    papers = await lit_agent.search("difference-in-differences parallel trends")

    # 2. 差距分析
    gap_agent = GapAnalyzer()
    gaps = await gap_agent.analyze_all_components("estimator-did", paper_contents)

    # 3. 公式验证
    formula_agent = FormulaAgent()
    formula_result = await formula_agent.run(formula_input)

    # 4. 质量验证
    validator = ValidatorAgent()
    validation = await validator.run(validation_input)

asyncio.run(main())
```
"""

# Base
from .base import (
    BaseAgent,
    AgentStatus,
    AgentMetrics,
    PaperInfo,
    GapInfo,
    CalibrationScore,
    ValidationResult,
    TOP_JOURNALS,
    CITATION_THRESHOLDS,
    COMPONENT_TYPES,
    QUALITY_GATE_THRESHOLDS,
    QUALITY_GATE_WEIGHTS,
    PROJECT_ROOT,
    SKILLS_DIR,
    CALIBRATION_DIR,
)

# Original Agents
from .orchestrator import OrchestratorAgent, SKILL_CALIBRATION_CONFIG as ORCHESTRATOR_CONFIG
from .literature_agent import LiteratureAgent, SEM_QUERIES
from .extractor_agent import ExtractorAgent, ExtractedContent
from .calibration_agent import CalibrationAgent, CalibrationReport

# New Agents (v2.0)
from .gap_analyzer import GapAnalyzer, GapAnalysisInput, GapAnalysisOutput
from .formula_agent import FormulaAgent, FormulaInput, FormulaOutput, Formula
from .citation_agent import CitationAgent, CitationInput, CitationOutput, Citation
from .validator_agent import ValidatorAgent, ValidationInput, ValidationOutput
from .updater_agent import UpdaterAgent, UpdateInput, UpdateOutput, UpdatePatch

__all__ = [
    # Base
    'BaseAgent',
    'AgentStatus',
    'AgentMetrics',
    'PaperInfo',
    'GapInfo',
    'CalibrationScore',
    'ValidationResult',
    'TOP_JOURNALS',
    'CITATION_THRESHOLDS',
    'COMPONENT_TYPES',
    'QUALITY_GATE_THRESHOLDS',
    'QUALITY_GATE_WEIGHTS',
    'PROJECT_ROOT',
    'SKILLS_DIR',
    'CALIBRATION_DIR',

    # Original Agents
    'OrchestratorAgent',
    'ORCHESTRATOR_CONFIG',
    'LiteratureAgent',
    'SEM_QUERIES',
    'ExtractorAgent',
    'ExtractedContent',
    'CalibrationAgent',
    'CalibrationReport',

    # New Agents (v2.0)
    'GapAnalyzer',
    'GapAnalysisInput',
    'GapAnalysisOutput',
    'FormulaAgent',
    'FormulaInput',
    'FormulaOutput',
    'Formula',
    'CitationAgent',
    'CitationInput',
    'CitationOutput',
    'Citation',
    'ValidatorAgent',
    'ValidationInput',
    'ValidationOutput',
    'UpdaterAgent',
    'UpdateInput',
    'UpdateOutput',
    'UpdatePatch',
]

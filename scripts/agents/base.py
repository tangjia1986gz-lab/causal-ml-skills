#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Agent - 智能体基类

所有校准智能体的公共基类，提供:
1. 统一的日志记录
2. 状态管理
3. 配置处理
4. 异常处理
5. 性能追踪
"""

import os
import sys
import asyncio
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import traceback

# Windows 编码修复
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
SKILLS_DIR = PROJECT_ROOT / "skills"
CALIBRATION_DIR = PROJECT_ROOT / "calibration_v2"
LOGS_DIR = CALIBRATION_DIR / "logs"

# 确保目录存在
CALIBRATION_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


class AgentStatus(Enum):
    """Agent 状态"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class AgentMetrics:
    """Agent 性能指标"""
    tasks_processed: int = 0
    tasks_successful: int = 0
    tasks_failed: int = 0
    total_time_seconds: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    errors: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """计算成功率"""
        if self.tasks_processed == 0:
            return 0.0
        return self.tasks_successful / self.tasks_processed

    @property
    def avg_time_per_task(self) -> float:
        """平均每任务耗时"""
        if self.tasks_processed == 0:
            return 0.0
        return self.total_time_seconds / self.tasks_processed

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "tasks_processed": self.tasks_processed,
            "tasks_successful": self.tasks_successful,
            "tasks_failed": self.tasks_failed,
            "success_rate": f"{self.success_rate:.2%}",
            "total_time_seconds": round(self.total_time_seconds, 2),
            "avg_time_per_task": round(self.avg_time_per_task, 2),
            "errors": self.errors[-10:] if self.errors else []  # 只保留最近10个错误
        }


# 泛型类型用于输入输出
T_Input = TypeVar('T_Input')
T_Output = TypeVar('T_Output')


class BaseAgent(ABC, Generic[T_Input, T_Output]):
    """
    智能体基类

    所有校准智能体继承此类，提供统一的:
    - 日志记录
    - 状态管理
    - 配置处理
    - 性能追踪
    """

    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        log_level: int = logging.INFO
    ):
        """
        初始化 Agent

        Parameters
        ----------
        name : str
            Agent 名称
        config : Dict, optional
            配置字典
        log_level : int
            日志级别
        """
        self.name = name
        self.config = config or {}
        self.status = AgentStatus.IDLE
        self.metrics = AgentMetrics()

        # 设置日志
        self.logger = self._setup_logger(log_level)

        # 状态存储
        self._state: Dict[str, Any] = {}

    def _setup_logger(self, level: int) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(f"Agent.{self.name}")
        logger.setLevel(level)

        # 避免重复添加 handler
        if not logger.handlers:
            # 控制台输出
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_format = logging.Formatter(
                f'[{self.name}] %(asctime)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_format)
            logger.addHandler(console_handler)

            # 文件输出
            log_file = LOGS_DIR / f"{self.name.lower()}_{datetime.now().strftime('%Y%m%d')}.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)

        return logger

    @abstractmethod
    async def process(self, input_data: T_Input) -> T_Output:
        """
        处理输入数据 (子类必须实现)

        Parameters
        ----------
        input_data : T_Input
            输入数据

        Returns
        -------
        T_Output
            处理结果
        """
        pass

    async def run(self, input_data: T_Input) -> T_Output:
        """
        运行 Agent (带性能追踪)

        Parameters
        ----------
        input_data : T_Input
            输入数据

        Returns
        -------
        T_Output
            处理结果
        """
        self.status = AgentStatus.RUNNING
        self.metrics.start_time = datetime.now()
        self.metrics.tasks_processed += 1

        self.logger.info(f"开始处理任务...")

        try:
            start_time = asyncio.get_event_loop().time()

            result = await self.process(input_data)

            elapsed = asyncio.get_event_loop().time() - start_time
            self.metrics.total_time_seconds += elapsed
            self.metrics.tasks_successful += 1
            self.status = AgentStatus.COMPLETED

            self.logger.info(f"任务完成，耗时 {elapsed:.2f}s")

            return result

        except Exception as e:
            self.metrics.tasks_failed += 1
            self.metrics.errors.append(f"{datetime.now().isoformat()}: {str(e)}")
            self.status = AgentStatus.FAILED

            self.logger.error(f"任务失败: {e}")
            self.logger.debug(traceback.format_exc())

            raise

        finally:
            self.metrics.end_time = datetime.now()

    async def run_batch(
        self,
        inputs: List[T_Input],
        max_concurrent: int = 3
    ) -> List[T_Output]:
        """
        批量运行任务

        Parameters
        ----------
        inputs : List[T_Input]
            输入数据列表
        max_concurrent : int
            最大并发数

        Returns
        -------
        List[T_Output]
            结果列表
        """
        self.logger.info(f"批量处理 {len(inputs)} 个任务 (并发: {max_concurrent})")

        results = []
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(input_data: T_Input) -> Optional[T_Output]:
            async with semaphore:
                try:
                    return await self.run(input_data)
                except Exception as e:
                    self.logger.warning(f"任务失败，跳过: {e}")
                    return None

        tasks = [process_with_semaphore(inp) for inp in inputs]
        batch_results = await asyncio.gather(*tasks)

        # 过滤失败的任务
        results = [r for r in batch_results if r is not None]

        self.logger.info(f"批量处理完成: {len(results)}/{len(inputs)} 成功")

        return results

    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return self.metrics.to_dict()

    def reset_metrics(self) -> None:
        """重置性能指标"""
        self.metrics = AgentMetrics()
        self.status = AgentStatus.IDLE

    def save_state(self, key: str, value: Any) -> None:
        """保存状态"""
        self._state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        """获取状态"""
        return self._state.get(key, default)

    def clear_state(self) -> None:
        """清除状态"""
        self._state.clear()

    def export_state(self, filepath: Path) -> None:
        """导出状态到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self._state, f, ensure_ascii=False, indent=2, default=str)

    def import_state(self, filepath: Path) -> None:
        """从文件导入状态"""
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                self._state = json.load(f)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name}, status={self.status.value})>"


# ============================================================================
# 常用数据结构
# ============================================================================

@dataclass
class PaperInfo:
    """论文信息"""
    paper_id: str
    title: str
    authors: List[str]
    year: int
    venue: str
    citations: int
    abstract: str = ""
    pdf_url: Optional[str] = None
    local_path: Optional[str] = None
    external_ids: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PaperInfo":
        """从字典创建"""
        return cls(
            paper_id=data.get('paper_id', ''),
            title=data.get('title', ''),
            authors=data.get('authors', []),
            year=data.get('year', 0),
            venue=data.get('venue', ''),
            citations=data.get('citations', 0),
            abstract=data.get('abstract', ''),
            pdf_url=data.get('pdf_url'),
            local_path=data.get('local_path'),
            external_ids=data.get('external_ids', {})
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'paper_id': self.paper_id,
            'title': self.title,
            'authors': self.authors,
            'year': self.year,
            'venue': self.venue,
            'citations': self.citations,
            'abstract': self.abstract,
            'pdf_url': self.pdf_url,
            'local_path': self.local_path,
            'external_ids': self.external_ids
        }


@dataclass
class GapInfo:
    """差距信息"""
    gap_id: str
    category: str  # assumption, formula, method, diagnostic, reference, implementation, notation
    severity: str  # critical, major, minor, enhancement
    description: str
    source_paper: str
    source_section: str
    existing_content: str = ""
    suggested_addition: str = ""
    target_file: str = ""
    confidence: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'gap_id': self.gap_id,
            'category': self.category,
            'severity': self.severity,
            'description': self.description,
            'source_paper': self.source_paper,
            'source_section': self.source_section,
            'existing_content': self.existing_content,
            'suggested_addition': self.suggested_addition,
            'target_file': self.target_file,
            'confidence': self.confidence
        }


@dataclass
class CalibrationScore:
    """校准得分"""
    skill_name: str
    component_scores: Dict[str, float]
    final_score: float
    passed: bool
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'skill_name': self.skill_name,
            'component_scores': self.component_scores,
            'final_score': round(self.final_score, 4),
            'passed': self.passed,
            'timestamp': self.timestamp.isoformat(),
            'details': self.details
        }


@dataclass
class ValidationResult:
    """验证结果"""
    gate_name: str
    passed: bool
    score: float
    threshold: float
    issues: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'gate_name': self.gate_name,
            'passed': self.passed,
            'score': round(self.score, 4),
            'threshold': self.threshold,
            'issues': self.issues,
            'details': self.details
        }


# ============================================================================
# 常量定义
# ============================================================================

# 顶刊列表
TOP_JOURNALS = {
    # 经济学五大刊
    "American Economic Review",
    "Econometrica",
    "Journal of Political Economy",
    "Quarterly Journal of Economics",
    "Review of Economic Studies",
    # 因果推断/计量
    "Journal of Econometrics",
    "Journal of Causal Inference",
    "Review of Economics and Statistics",
    "Journal of Business & Economic Statistics",
    "Econometric Theory",
    # 统计学
    "Journal of the American Statistical Association",
    "Annals of Statistics",
    "Biometrika",
    "Journal of the Royal Statistical Society",
    # ML/因果ML
    "Journal of Machine Learning Research",
    "Machine Learning",
    "NeurIPS",
    "ICML",
    # 应用领域
    "Management Science",
    "Marketing Science",
    "American Journal of Epidemiology",
    # 方法论
    "Psychological Methods",
    "Structural Equation Modeling",
    "Multivariate Behavioral Research",
}

# 引用数阈值 (按组件类型)
CITATION_THRESHOLDS = {
    "identification": 500,   # 识别假设: 理论稳定
    "estimation": 300,       # 估计方法: 方法论核心
    "diagnostics": 200,      # 诊断测试: 实用工具
    "reporting": 100,        # 报告标准: 格式指南
    "errors": 50,            # 常见错误: 评审经验
}

# 组件类型
COMPONENT_TYPES = [
    "identification_assumptions",
    "estimation_methods",
    "diagnostic_tests",
    "reporting_standards",
    "common_errors",
]

# 质量门控阈值
QUALITY_GATE_THRESHOLDS = {
    "literature_coverage": 0.8,
    "content_completeness": 1.0,
    "formula_consistency": 0.9,
    "citation_validity": 1.0,
}

# 质量门控权重
QUALITY_GATE_WEIGHTS = {
    "literature_coverage": 0.3,
    "content_completeness": 0.3,
    "formula_consistency": 0.2,
    "citation_validity": 0.2,
}


if __name__ == "__main__":
    # 测试基类功能
    class TestAgent(BaseAgent[str, str]):
        async def process(self, input_data: str) -> str:
            await asyncio.sleep(0.1)  # 模拟处理
            return f"Processed: {input_data}"

    async def test():
        agent = TestAgent("TestAgent")
        result = await agent.run("hello")
        print(f"Result: {result}")
        print(f"Metrics: {agent.get_metrics()}")

    asyncio.run(test())

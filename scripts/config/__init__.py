"""
Calibration Configuration Module

包含:
- 完整技能配置 (27个技能)
- 组件校准模板
- 并行执行配置
- 质量门控配置

使用方法:
```python
from config import (
    SKILL_CALIBRATION_CONFIG,
    get_skills_by_priority,
    get_skills_by_category,
    get_calibration_batches,
    list_all_skills,
)

# 获取所有 P1 优先级技能
p1_skills = get_skills_by_priority(Priority.CRITICAL)

# 获取校准批次
batches = get_calibration_batches(batch_size=3)
```
"""

from .calibration_config import (
    # 枚举
    SkillCategory,
    Priority,
    ComponentType,

    # 数据结构
    QueryConfig,
    SkillConfig,

    # 配置
    SKILL_CALIBRATION_CONFIG,
    TOP_JOURNALS,
    CITATION_THRESHOLDS,
    COMPONENT_CALIBRATION_TEMPLATES,
    PARALLEL_CONFIG,
    QUALITY_GATE_CONFIG,

    # 辅助函数
    get_skills_by_category,
    get_skills_by_priority,
    get_all_queries,
    get_skill_config,
    list_all_skills,
    get_calibration_batches,
)

__all__ = [
    # 枚举
    'SkillCategory',
    'Priority',
    'ComponentType',

    # 数据结构
    'QueryConfig',
    'SkillConfig',

    # 配置
    'SKILL_CALIBRATION_CONFIG',
    'TOP_JOURNALS',
    'CITATION_THRESHOLDS',
    'COMPONENT_CALIBRATION_TEMPLATES',
    'PARALLEL_CONFIG',
    'QUALITY_GATE_CONFIG',

    # 辅助函数
    'get_skills_by_category',
    'get_skills_by_priority',
    'get_all_queries',
    'get_skill_config',
    'list_all_skills',
    'get_calibration_batches',
]

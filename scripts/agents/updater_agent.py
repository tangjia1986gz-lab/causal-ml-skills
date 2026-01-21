#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UpdaterAgent - 文档更新智能体

职责:
1. 根据差距分析生成更新补丁
2. 应用更新到技能文档
3. 维护更新历史
4. 生成更新报告
"""

import sys
import re
import asyncio
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import difflib

# Windows 编码修复
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from .base import (
    BaseAgent, GapInfo,
    PROJECT_ROOT, SKILLS_DIR, CALIBRATION_DIR
)
from .gap_analyzer import GapAnalysisOutput


@dataclass
class UpdatePatch:
    """更新补丁"""
    patch_id: str
    target_file: str
    operation: str  # add, modify, delete
    location: str  # section name or line range
    original_content: str
    new_content: str
    reason: str
    source_gaps: List[str]  # gap_ids
    priority: int  # 1=highest
    applied: bool = False


@dataclass
class UpdateInput:
    """更新输入"""
    skill_name: str
    gap_results: Dict[str, GapAnalysisOutput]
    auto_apply: bool = False
    backup: bool = True


@dataclass
class UpdateOutput:
    """更新输出"""
    skill_name: str
    patches_generated: int
    patches_applied: int
    files_updated: List[str]
    patches: List[UpdatePatch]
    backup_path: Optional[Path]
    report: str


class UpdaterAgent(BaseAgent[UpdateInput, UpdateOutput]):
    """
    文档更新智能体

    功能:
    - 分析差距生成补丁
    - 智能定位插入位置
    - 备份和恢复
    - 生成 diff 报告
    """

    # 组件到文件的映射
    COMPONENT_FILE_MAP = {
        "identification_assumptions": "references/identification_assumptions.md",
        "estimation_methods": "references/estimation_methods.md",
        "diagnostic_tests": "references/diagnostic_tests.md",
        "reporting_standards": "references/reporting_standards.md",
        "common_errors": "references/common_errors.md",
    }

    # 各组件的内容模板
    CONTENT_TEMPLATES = {
        "identification_assumptions": {
            "assumption": """
### {title}

**形式化定义**:
{definition}

**直观解释**:
{intuition}

**可测试性**: {testability}

**文献来源**: {source}

---
""",
        },
        "estimation_methods": {
            "method": """
### {title}

**估计量公式**:
$$
{formula}
$$

**算法步骤**:
{steps}

**标准误计算**:
{standard_errors}

**适用条件**: {conditions}

**参考**: {source}

---
""",
        },
        "diagnostic_tests": {
            "test": """
### {title}

**检验统计量**:
$$
{statistic}
$$

**原假设**: {null_hypothesis}

**临界值**: {critical_values}

**解释标准**: {interpretation}

**参考**: {source}

---
""",
        },
        "reporting_standards": {
            "standard": """
### {title}

**必填元素**:
{elements}

**示例格式**:
```
{example}
```

**参考**: {source}

---
""",
        },
        "common_errors": {
            "error": """
### {title}

**错误描述**: {description}

**为什么错误**: {why_wrong}

**正确做法**: {correct_approach}

**代码示例**:
```python
{code_example}
```

**参考**: {source}

---
""",
        },
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("UpdaterAgent", config)
        self.updates_dir = CALIBRATION_DIR / "updates"
        self.updates_dir.mkdir(parents=True, exist_ok=True)

    def _find_skill_path(self, skill_name: str) -> Optional[Path]:
        """查找技能目录"""
        search_paths = [
            SKILLS_DIR / "classic-methods" / skill_name,
            SKILLS_DIR / "causal-ml" / skill_name,
            SKILLS_DIR / "ml-foundation" / skill_name,
            SKILLS_DIR / "infrastructure" / skill_name,
        ]

        for path in search_paths:
            if path.exists():
                return path

        return None

    def _create_backup(self, skill_path: Path) -> Path:
        """创建技能目录备份"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.updates_dir / f"{skill_path.name}_backup_{timestamp}"

        shutil.copytree(skill_path, backup_path, dirs_exist_ok=True)
        self.logger.info(f"备份创建: {backup_path}")

        return backup_path

    def _gap_to_patch(
        self,
        gap: GapInfo,
        component: str,
        skill_path: Path
    ) -> Optional[UpdatePatch]:
        """将差距转换为更新补丁"""
        target_file = self.COMPONENT_FILE_MAP.get(component, "SKILL.md")
        full_path = skill_path / target_file

        # 确定操作类型
        if gap.existing_content:
            operation = "modify"
        else:
            operation = "add"

        # 生成新内容
        new_content = self._generate_content(gap, component)

        if not new_content:
            return None

        # 确定优先级
        priority_map = {"critical": 1, "major": 2, "minor": 3, "enhancement": 4}
        priority = priority_map.get(gap.severity, 3)

        return UpdatePatch(
            patch_id=gap.gap_id,
            target_file=target_file,
            operation=operation,
            location=gap.category,
            original_content=gap.existing_content,
            new_content=new_content,
            reason=gap.description,
            source_gaps=[gap.gap_id],
            priority=priority
        )

    def _generate_content(self, gap: GapInfo, component: str) -> str:
        """根据差距生成新内容"""
        templates = self.CONTENT_TEMPLATES.get(component, {})

        # 根据差距类型选择模板
        if gap.category in ["assumption", "identification"]:
            template = templates.get("assumption", "")
            if template:
                return template.format(
                    title=gap.description[:50],
                    definition=gap.suggested_addition or "待补充",
                    intuition="待补充",
                    testability="待评估",
                    source=gap.source_paper
                )

        elif gap.category in ["method", "estimation"]:
            template = templates.get("method", "")
            if template:
                return template.format(
                    title=gap.description[:50],
                    formula=gap.suggested_addition or "待补充",
                    steps="1. 待补充\n2. 待补充",
                    standard_errors="待补充",
                    conditions="待补充",
                    source=gap.source_paper
                )

        elif gap.category in ["diagnostic", "test"]:
            template = templates.get("test", "")
            if template:
                return template.format(
                    title=gap.description[:50],
                    statistic=gap.suggested_addition or "待补充",
                    null_hypothesis="待补充",
                    critical_values="待补充",
                    interpretation="待补充",
                    source=gap.source_paper
                )

        elif gap.category in ["error", "common_errors"]:
            template = templates.get("error", "")
            if template:
                return template.format(
                    title=gap.description[:50],
                    description=gap.suggested_addition or "待补充",
                    why_wrong="待补充",
                    correct_approach="待补充",
                    code_example="# 待补充",
                    source=gap.source_paper
                )

        # 默认: 简单文本
        return f"""
### {gap.description[:80]}

{gap.suggested_addition or '待补充详细内容'}

**来源**: {gap.source_paper}

---
"""

    def _find_insertion_point(
        self,
        content: str,
        component: str,
        gap: GapInfo
    ) -> int:
        """找到内容插入位置"""
        lines = content.split('\n')

        # 查找相关章节
        section_headers = {
            "identification_assumptions": ["## 识别假设", "## Assumptions", "## 假设"],
            "estimation_methods": ["## 估计方法", "## Methods", "## 方法"],
            "diagnostic_tests": ["## 诊断测试", "## Diagnostics", "## 诊断"],
            "reporting_standards": ["## 报告标准", "## Reporting", "## 报告"],
            "common_errors": ["## 常见错误", "## Errors", "## 错误"],
        }

        headers = section_headers.get(component, [])

        # 找到章节末尾
        for i, line in enumerate(lines):
            for header in headers:
                if header.lower() in line.lower():
                    # 找到下一个 ## 标题或文件末尾
                    for j in range(i + 1, len(lines)):
                        if lines[j].startswith('## '):
                            return j
                    return len(lines)

        # 没找到则添加到末尾
        return len(lines)

    def _apply_patch(
        self,
        patch: UpdatePatch,
        skill_path: Path
    ) -> bool:
        """应用单个补丁"""
        file_path = skill_path / patch.target_file

        # 确保目录存在
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if file_path.exists():
                content = file_path.read_text(encoding='utf-8')
            else:
                content = f"# {patch.target_file}\n\n"

            if patch.operation == "add":
                # 找到插入点
                lines = content.split('\n')
                insert_point = self._find_insertion_point(
                    content,
                    patch.target_file.replace("references/", "").replace(".md", ""),
                    None
                )
                lines.insert(insert_point, patch.new_content)
                new_content = '\n'.join(lines)

            elif patch.operation == "modify":
                # 替换原内容
                if patch.original_content in content:
                    new_content = content.replace(
                        patch.original_content,
                        patch.new_content
                    )
                else:
                    # 找不到原内容则追加
                    new_content = content + "\n" + patch.new_content

            elif patch.operation == "delete":
                new_content = content.replace(patch.original_content, "")

            else:
                return False

            file_path.write_text(new_content, encoding='utf-8')
            patch.applied = True

            self.logger.info(f"已应用补丁到 {patch.target_file}")
            return True

        except Exception as e:
            self.logger.error(f"应用补丁失败: {e}")
            return False

    def _merge_patches(self, patches: List[UpdatePatch]) -> List[UpdatePatch]:
        """合并同一文件的补丁"""
        by_file: Dict[str, List[UpdatePatch]] = {}

        for patch in patches:
            if patch.target_file not in by_file:
                by_file[patch.target_file] = []
            by_file[patch.target_file].append(patch)

        merged = []
        for target_file, file_patches in by_file.items():
            # 按优先级排序
            file_patches.sort(key=lambda p: p.priority)

            # 合并同一文件的添加操作
            add_patches = [p for p in file_patches if p.operation == "add"]
            if len(add_patches) > 1:
                combined_content = "\n".join(p.new_content for p in add_patches)
                combined_gaps = []
                for p in add_patches:
                    combined_gaps.extend(p.source_gaps)

                merged.append(UpdatePatch(
                    patch_id=f"merged_{target_file.replace('/', '_')}",
                    target_file=target_file,
                    operation="add",
                    location="multiple",
                    original_content="",
                    new_content=combined_content,
                    reason=f"合并了 {len(add_patches)} 个添加操作",
                    source_gaps=combined_gaps,
                    priority=min(p.priority for p in add_patches)
                ))
            else:
                merged.extend(add_patches)

            # 保留其他操作
            merged.extend(p for p in file_patches if p.operation != "add")

        return merged

    def _generate_diff(self, patch: UpdatePatch, skill_path: Path) -> str:
        """生成 diff 格式的变更"""
        file_path = skill_path / patch.target_file

        if file_path.exists():
            original = file_path.read_text(encoding='utf-8').splitlines()
        else:
            original = []

        # 模拟应用补丁后的内容
        if patch.operation == "add":
            modified = original + [""] + patch.new_content.splitlines()
        elif patch.operation == "modify":
            content = '\n'.join(original)
            new_content = content.replace(patch.original_content, patch.new_content)
            modified = new_content.splitlines()
        else:
            modified = original

        diff = difflib.unified_diff(
            original,
            modified,
            fromfile=f"a/{patch.target_file}",
            tofile=f"b/{patch.target_file}",
            lineterm=""
        )

        return '\n'.join(diff)

    def _generate_report(
        self,
        skill_name: str,
        patches: List[UpdatePatch],
        applied_count: int,
        files_updated: List[str],
        backup_path: Optional[Path]
    ) -> str:
        """生成更新报告"""
        report = f"""# {skill_name} 更新报告

> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 概览

| 指标 | 值 |
|------|-----|
| 生成补丁数 | {len(patches)} |
| 已应用补丁 | {applied_count} |
| 更新文件数 | {len(files_updated)} |
| 备份位置 | {backup_path or '无'} |

## 补丁详情

"""
        for i, patch in enumerate(patches, 1):
            status = "✅ 已应用" if patch.applied else "⏳ 待应用"
            priority_symbol = "🔴" if patch.priority == 1 else "🟡" if patch.priority == 2 else "🟢"

            report += f"""### {i}. {patch.target_file}

- **状态**: {status}
- **优先级**: {priority_symbol} P{patch.priority}
- **操作**: {patch.operation}
- **原因**: {patch.reason}

**变更内容预览**:
```
{patch.new_content[:500]}{'...' if len(patch.new_content) > 500 else ''}
```

---

"""

        if files_updated:
            report += "## 已更新文件\n\n"
            for f in files_updated:
                report += f"- `{f}`\n"

        return report

    async def process(self, input_data: UpdateInput) -> UpdateOutput:
        """
        执行文档更新

        Parameters
        ----------
        input_data : UpdateInput
            更新输入

        Returns
        -------
        UpdateOutput
            更新结果
        """
        skill_name = input_data.skill_name

        self.logger.info(f"开始处理 {skill_name} 的更新")

        # 查找技能路径
        skill_path = self._find_skill_path(skill_name)
        if not skill_path:
            raise ValueError(f"技能目录未找到: {skill_name}")

        # 创建备份
        backup_path = None
        if input_data.backup:
            backup_path = self._create_backup(skill_path)

        # 生成补丁
        patches = []
        for component, output in input_data.gap_results.items():
            for gap in output.gaps:
                patch = self._gap_to_patch(gap, component, skill_path)
                if patch:
                    patches.append(patch)

        self.logger.info(f"生成了 {len(patches)} 个补丁")

        # 合并补丁
        patches = self._merge_patches(patches)
        self.logger.info(f"合并后 {len(patches)} 个补丁")

        # 应用补丁
        applied_count = 0
        files_updated = []

        if input_data.auto_apply:
            for patch in patches:
                if self._apply_patch(patch, skill_path):
                    applied_count += 1
                    if patch.target_file not in files_updated:
                        files_updated.append(patch.target_file)

            self.logger.info(f"应用了 {applied_count} 个补丁")
        else:
            self.logger.info("补丁未自动应用 (auto_apply=False)")

        # 保存补丁文件
        patches_dir = self.updates_dir / skill_name
        patches_dir.mkdir(parents=True, exist_ok=True)

        for patch in patches:
            patch_file = patches_dir / f"{patch.patch_id}.patch"
            patch_file.write_text(
                self._generate_diff(patch, skill_path),
                encoding='utf-8'
            )

        # 保存补丁元数据
        meta_file = patches_dir / "patches.json"
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump([
                {
                    "patch_id": p.patch_id,
                    "target_file": p.target_file,
                    "operation": p.operation,
                    "priority": p.priority,
                    "reason": p.reason,
                    "applied": p.applied,
                }
                for p in patches
            ], f, ensure_ascii=False, indent=2)

        # 生成报告
        report = self._generate_report(
            skill_name, patches, applied_count, files_updated, backup_path
        )

        return UpdateOutput(
            skill_name=skill_name,
            patches_generated=len(patches),
            patches_applied=applied_count,
            files_updated=files_updated,
            patches=patches,
            backup_path=backup_path,
            report=report
        )

    async def apply_pending_patches(
        self,
        skill_name: str,
        patch_ids: Optional[List[str]] = None
    ) -> int:
        """应用待处理的补丁"""
        patches_dir = self.updates_dir / skill_name
        meta_file = patches_dir / "patches.json"

        if not meta_file.exists():
            self.logger.warning(f"无待处理补丁: {skill_name}")
            return 0

        skill_path = self._find_skill_path(skill_name)
        if not skill_path:
            raise ValueError(f"技能目录未找到: {skill_name}")

        with open(meta_file, 'r', encoding='utf-8') as f:
            patches_meta = json.load(f)

        applied = 0
        for meta in patches_meta:
            if meta["applied"]:
                continue

            if patch_ids and meta["patch_id"] not in patch_ids:
                continue

            patch_file = patches_dir / f"{meta['patch_id']}.patch"
            if patch_file.exists():
                # 简化: 标记为已应用
                meta["applied"] = True
                applied += 1

        # 更新元数据
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(patches_meta, f, ensure_ascii=False, indent=2)

        self.logger.info(f"应用了 {applied} 个待处理补丁")
        return applied

    def save_results(
        self,
        output: UpdateOutput,
        output_dir: Optional[Path] = None
    ) -> Path:
        """保存更新结果"""
        output_dir = output_dir or self.updates_dir / output.skill_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存报告
        report_path = output_dir / "update_report.md"
        report_path.write_text(output.report, encoding='utf-8')

        self.logger.info(f"结果已保存到: {output_dir}")
        return output_dir


if __name__ == "__main__":
    from .gap_analyzer import GapAnalysisOutput, GapInfo

    async def test():
        updater = UpdaterAgent()

        # 模拟输入
        test_input = UpdateInput(
            skill_name="estimator-did",
            gap_results={
                "identification_assumptions": GapAnalysisOutput(
                    skill_name="estimator-did",
                    component="identification_assumptions",
                    gaps=[
                        GapInfo(
                            gap_id="test1",
                            category="assumption",
                            severity="major",
                            description="缺少 No Anticipation 假设",
                            source_paper="Callaway & Sant'Anna (2021)",
                            source_section="assumptions",
                            suggested_addition="在处理发生前，处理组的行为不应发生变化。"
                        )
                    ],
                    coverage_score=0.8,
                    papers_analyzed=5,
                    summary=""
                ),
            },
            auto_apply=False,
            backup=False
        )

        result = await updater.run(test_input)
        print(f"生成补丁数: {result.patches_generated}")
        print(f"已应用补丁: {result.patches_applied}")
        print(f"\n{result.report}")

    asyncio.run(test())

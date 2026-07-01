"""Agent 记忆系统 — 仿 cc-haha agentMemory.ts 设计。

持久化 agent 的偏好和行为模式到 MEMORY.md 文件。
支持三种作用域:
  - user:    用户级 (~/.zhiban/agent-memory/{agent_type}/MEMORY.md)
  - project: 项目级 (工作区/.context/agent-memory/{agent_type}/MEMORY.md)
  - local:   本地级 (项目/.claude/agent-memory-local/{agent_type}/MEMORY.md)

cc-haha 对应: agentMemory.ts + agentMemorySnapshot.ts
"""

from __future__ import annotations

from pathlib import Path


def _get_user_memory_dir() -> Path:
    """用户级 agent memory 目录。"""
    return Path.home() / ".zhiban" / "agent-memory"


def _get_project_memory_dir(project_root: Path | None = None) -> Path:
    """项目级 agent memory 目录。

    project_root 默认为当前工作目录。
    """
    root = project_root or Path.cwd()
    return root / ".context" / "agent-memory"


def get_agent_memory_path(
    agent_type: str,
    scope: str = "project",
    project_root: Path | None = None,
) -> Path:
    """获取 agent 的 MEMORY.md 路径。

    Args:
        agent_type: agent 类型名，如 "paper_assistant"
        scope: 作用域 "user" | "project" | "local"
        project_root: 项目根目录（scope=project/local 时使用）

    Returns:
        MEMORY.md 文件的 Path 对象。
    """
    if scope == "user":
        base = _get_user_memory_dir()
    elif scope == "local":
        root = project_root or Path.cwd()
        base = root / ".claude" / "agent-memory-local"
    else:
        base = _get_project_memory_dir(project_root)

    return base / agent_type / "MEMORY.md"


def load_agent_memory(
    agent_type: str,
    scope: str = "project",
    project_root: Path | None = None,
) -> str:
    """加载 agent 的持久记忆。

    Returns:
        MEMORY.md 内容字符串，文件不存在时返回空字符串。
    """
    path = get_agent_memory_path(agent_type, scope, project_root)
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def save_agent_memory(
    agent_type: str,
    content: str,
    scope: str = "project",
    project_root: Path | None = None,
) -> None:
    """保存 agent 的持久记忆到 MEMORY.md。

    自动创建父目录。
    """
    path = get_agent_memory_path(agent_type, scope, project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_memory_prompt(memory_text: str) -> str:
    """将 memory 内容格式化为可注入 system prompt 的片段。

    cc-haha 对应: loadAgentMemoryPrompt() 中的格式化逻辑
    """
    if not memory_text or not memory_text.strip():
        return ""
    return f"""【持久记忆 — 跨会话保留的行为偏好】
{memory_text.strip()}
—— 记忆结束 ——"""


def clear_agent_memory(
    agent_type: str,
    scope: str = "project",
    project_root: Path | None = None,
) -> None:
    """清除 agent 的持久记忆（删除 MEMORY.md）。"""
    path = get_agent_memory_path(agent_type, scope, project_root)
    if path.exists():
        path.unlink()

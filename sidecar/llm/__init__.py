"""知伴 LLM 模块 — Provider 抽象 + 本地推理引擎 + KV Cache 管理"""

# 顶层入口不做 relative import，避免 tests/ 等非 package 上下文导入失败。
# 各子模块通过绝对路径导入: from llm.xxx import ...

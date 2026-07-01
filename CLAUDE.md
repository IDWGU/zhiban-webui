# 知伴 (ZhiBan) — AI Architecture Notes

## AI 伴读引擎架构 (V12 — 2026-06-09, cc-haha Agent 模式重构)

### 顶层流程

```
用户输入 → WebSocket → handle_user_query
  ├─ _detect_question_type() → question_type
  ├─ _resolve_agent_type() → agent_type (paper_assistant/paper_explorer/paper_summarizer)
  ├─ AgentRegistry.get(agent_type) → AgentDefinition
  │   ├─ definition.allows_tool() → 过滤工具列表
  │   ├─ definition.get_system_prompt() → 角色 prompt
  │   └─ load_agent_memory() → 持久记忆注入
  ├─ AgentLoop (while step < max_steps):
  │   ├─ 流式调用 LLM → 优先 API function calling，回退 XML 解析
  │   ├─ reasoning_text 回退: thinking 中嵌入的 <tool_call> 也会被解析执行
  │   ├─ search(query) → 向量搜索 → MMR 重排 → 章节多样性平衡
  │   └─ paper_section/reading_context 工具
  └─ 最终回答: 流式推送 → 前端
```

### V12 vs V11 关键变化

| 方面 | V11 (旧) | V12 (新) |
|------|----------|----------|
| Agent 定义 | 硬编码在 engine._run_agent() | AgentDefinition + AgentRegistry |
| 工具过滤 | 全部工具可用 | 按 AgentDefinition.tools 白名单过滤 |
| System prompt | 单一 build_agent_system_prompt() | AgentDefinition.get_system_prompt() 按 agent 定制 |
| Agent 记忆 | 无 | MEMORY.md 跨会话持久化 |
| 工具调用解析 | structured > text | structured > text(content) > text(reasoning回退) |
| 内置 agent | 无 | paper_assistant, paper_explorer, paper_summarizer |

### Agent 定义系统 (`agent/definitions.py`)

每个 agent 是声明式的：
- `agent_type`: 唯一名称
- `get_system_prompt()`: 返回 agent 专属 system prompt
- `tools`/`disallowed_tools`: 两级工具过滤
- `memory_scope`: user/project/local
- `max_turns`: 最大工具调用轮数

```python
PAPER_EXPLORER_AGENT = AgentDefinition(
    agent_type="paper_explorer",
    tools=["search_knowledge_base", "get_paper_section", "get_reading_context"],
    max_turns=4,
    ...
)
```

### 内置 Agent (`agent/builtin/`)

| Agent | 工具 | 用途 |
|-------|------|------|
| `paper_assistant` | 全部 | 默认通用论文助手 |
| `paper_explorer` | search, section, reading_context | 只读论文搜索 |
| `paper_summarizer` | overview, section | 结构化论文总结 |

### 工具调用解析三层回退 (V12 关键修复)

```
structured_tool_calls (API native function calling)
  ↓ 空
parse_tool_call(full_text) (响应内容中的 <tool_call> XML)
  ↓ 空
parse_tool_call(reasoning_text) (推理文本中的 <tool_call>，V12 新增)
```

这解决了 DeepSeek 模型在 reasoning 中嵌入工具调用但不触发 native function call 的问题。

```
用户输入 → WebSocket → handle_user_query
  ├─ _detect_question_type() → concise/pedagogical/conversational/analytical/translation/summary
  │   (仅用于决定 l2_text 策略和是否跳过 RAG)
  ├─ _run_agent() → AgentLoop (Agent 自主工具调用)
  │   ├─ while step < max_steps:
  │   │   ├─ 构建消息 (system_prompt + active_slice/history + user_context)
  │   │   ├─ 流式调用 LLM → 解析 <tool_call>{JSON}</tool_call>
  │   │   ├─ search(query) → 向量搜索 → MMR 重排 → 章节多样性平衡
  │   │   ├─ paper_section(paper_id, section) → 读取论文特定章节
  │   │   └─ reading_context() → 获取当前阅读上下文
  │   └─ 回答质量自检 → 返回最终回答
  └─ 最终回答: L0+L1+L2+RAG+Agent结果 → LLM 流式 → 前端
```

### 架构演进

**V9 (旧):** 固定 Call1/Call2 流程 — Call1 分类 → RAG → Call2 回答
**V10 (旧):** 两级分类 — 启发式优先 (≈95%) → LLM 分类兜底 (≈5%) → 分流处理器
**V11 (当前):** Agent 自主决策 — AI 自主决定搜索关键词和搜索次数, 跳过分类步骤直接进入 Agent Loop

已移除的部分:
- `_classify_heuristic()` (8 级规则分类器) — 代码仍在 `engine/__init__.py:235` 但已不被调用 (死代码)
- `_llm_decision_loop()` (LLM 有限决断循环) — 已被 `_run_agent()` → `AgentLoop` 替代
- `ref_num` 参考文献编号提取 — 随 LLM 分类路径一起移除

---

### Agent Loop 实现细节 (`agent/agent_loop.py`)

```
while step < max_steps:
  1. 构建消息 (system_prompt + active_slice/history + user_context)
  2. 流式调用 LLM
  3. 解析响应:
     - <tool_call>{JSON}</tool_call> → 执行工具, 追加结果, continue
     - 文本(无工具调用) → 进入回答质量自检
  4. 回答质量自检:
     - detect_tool_intent() → 正则检测"我需要搜索"但无XML → 注入格式纠正, continue
     - is_final_answer() → 通过质量评估 → 返回
     - 太短(<40字)但有工具调用 → 强制重新总结(只有>2x长度才替换)
  5. 后处理: detect_thinking_leak → _strip_thinking_by_regex 四级剥离
```

**Agent 防护机制:**
- **同参数重复检测** (`agent_loop.py:318-332`): 维护 `_prev_tool_calls` 列表 `[(tool_name, args_json)]`, 同一对出现3次 → 注入"你已经用相同参数搜索了N次, 立即停止搜索直接回答"
- **工具调用格式纠正** (`agent_loop.py:449-475`): `detect_tool_intent()` 11 种正则 (我需要搜索/让我搜索/需要检索/应该使用工具/我先用...) → 重新注入显式格式示例要求输出 `<tool_call>`
- **回答质量自检** (`agent_loop.py:491-529`): 工具调用后回答<40字或等于"暂无/暂无相关信息" → 注入"你已经拿到搜索结果但回答太简单，重新撰写完整回答"
- **最终回答思考剥离** (`_strip_thinking_by_regex`): 四级策略 — 1)找 `**粗体标题**` 作为答案起点; 2)找"答案是/结论是/综上所述"标记; 3)找数学算式提取答案; 4)"用户问/根据我的/让我"开头则取最后一个非编号段落
- **流式循环检测** (`detect_stream_loop` / `detect_chinese_loop`): 三层 — 长句≥3次重复; 50字符窗口≥4次; 中文犹豫短语密度>1.5/百字

**Agent 消息组装:**
- 有 `active_slice` 直接插入 (不含 system prompt, system 已前置)
- 无 `active_slice` 用 `history_str` 文本
- `build_user_context()` 只在 `screen_changed=True` 时自动注入阅读位置; 否则 LLM 需通过 `get_reading_context` 工具主动获取

---

### System Prompt 变体与工具描述 (`agent/prompts.py`)

**4 种变体** (注册在 `PROMPT_VARIANTS`):
| variant | 来源函数 | 特点 |
|---------|----------|------|
| `concise` (默认) | `_variant_concise` | 最少规则，正向引导，适合小模型 |
| `strict` | `_variant_strict` | 强制先搜再答，减少自由发挥 |
| `academic` | `_variant_academic` | 严谨引用，区分"论文结论"和"学界共识" |
| `verbose` | `_variant_default` | 旧默认，偏啰嗦，保留用于对比 |

**工具描述构建** (`_build_tool_prompt`): 用第一个真实 tool 名和真实参数构造示例, 不写 `{"name":"工具名","arguments":{"参数":"值"}}` 这种占位符 — 避免小模型逐字复制模板。

**tool_call 格式解析** (`parse_tool_call`): 三级回退 — 1) XML `<tool_call>{JSON}</tool_call>`; 2) 未闭合 `<tool_call>`; 3) YAML `name: xxx\n参数: xxx`; 4) `[工具调用: xxx]`; 5) `_parse_kv_args` 兜底自动推断 int/float/string

**指令背诵检测** (`detect_instruction_echo`): 22个正则模式检测模型是否在背诵 prompt 指令文本 (用户问的是/你是知伴/知识库没有的信息/引用格式【来源/不要逐条罗列...), 流式前200字符内命中触发重试

---

### question_type 检测 (`_detect_question_type`)

6 种类型决定回答风格和是否跑 RAG:

| type | 触发条件 | RAG |
|------|----------|-----|
| `translation` | "翻译/译成/translate" | 路由到翻译模块 |
| `summary` | "总结全文/全文总结/概括全文/总结这篇文章/总结一下全文" | 不走 RAG, 传完整论文全文 |
| `concise` | yes/no确认 ("是吗/是不是/能否") 或 短定义 ("是什么/指什么/全称") | 不走 RAG |
| `pedagogical` | "初学者/入门/怎么学/基本概念/新手" | 不走 RAG |
| `conversational` | "你觉得/你认为/我应该/建议/怎么看" (排除含学术关键词的 "为什么/机理/DFT/电子结构") | 不走 RAG |
| `analytical` (默认) | 学术问题 | 完整 RAG |

---

### KV Cache 前缀对齐

```
分类 prompt: [L0(SP)][L1(历史)][L2(阅读内容)][分类指令]
回答 prompt: [L0(SP)][L1(历史)][L2(阅读内容)][RAG+回答指令]
                  └──────── 前缀对齐 ────────┘ → KV cache 自动复用
```

- L0: 恒定 System Prompt (不变)
- L1: `MessageStore.get_active_slice()` (活跃对话切片)
- L2: 论文阅读内容, 在 `build_classify_prompt_l3` 和 `build_answer_prompt_l3` 中同位置
- `build_decision_prompt_l3` 也是同位置 L2 — V11 决策+回答链路同样受益
- 不手动操作 KV cache (llama.cpp 原生 prefix caching 处理)
- Provider 实例以 `(api_key, base_url)` 为 key 缓存, 同轮 classify→answer 复用同一实例

---

### RAG 检索细节

**向量搜索** (`search_utils.py:vector_search`):
- 先用 paper_ids 做范围搜索 (top_k=max(20, top_k*3))
- 范围搜索结果最高分 < 0.60 → 自动回退全库搜索
- 全库结果比范围结果好 0.05+ → 采用全库结果
- 防止 frontend paperId 与 ChromaDB doc_id 不匹配

**MMR 重排** (`mmr_rerank`): 用 `paper_id` 计数做多样性惩罚 (非文本相似度), 同一篇论文出现次数越多惩罚越大

**章节多样性** (`select_with_section_diversity`): 如果 MMR 选出的 chunks 全来自论文前40% (引言), 强制替换2个为后50% (实验/结论) 的 chunks

**混合检索** (`rag/engine.py:search`): 向量搜索 + 知识图谱邻居遍历并行 (asyncio.gather), 图谱结果 `score * 0.8`, 不同 key 前缀 (`v_{id}` vs `g_{pid}`) 防键冲突

**JSON 提取** (`extract_json`): 五级回退 — 1)去 markdown code fence; 2)去小模型常见前后缀; 3)全量 JSON; 4)嵌套2层大括号; 5)扁平 JSON 兜底

---

### 翻译引擎实现细节 (`translation/translator.py`)

**本地模式 (GGUF/MLX):**
```
稳定前缀: {TRANSLATION_SYSTEM_PROMPT}\n\n{全文原文}\n\n
逐句追加: 翻译第N句：\n{sentence}
→ llama.cpp 原生 prefix caching 自动复用前缀部分 KV cache
→ 首句翻译自然完成 prefill, 无需独立 warmup
→ 超出 85% 上下文时退化为简洁 prompt (无全文上下文)
```

**关键设计决策:**
- 不走 chat template → 避免 `truncate_cache` + chat template 组合导致的 RoPE 位置错位
- 用 `raw_generate_stream()` 直接拼接原始文本
- API 模式: 每句携带完整原文, 利用 DeepSeek Context Caching 跨请求缓存前缀
- thinking 强制禁用
- 翻译取消信号跨请求传递: 新 Event 创建前检测旧 Event 是否 set
- 提取进度每 5% 推送一次 WebSocket
- 选区翻译: 坐标级重叠判定 (pageIndex/x/y/w/h + sentence rects 逐句比对)

---

### 模型管理器 (`llm/model_manager.py`)

**内存压力双检测:**
- `psutil.virtual_memory().available` (硬指标): <2GB → critical, <5GB → warning
- macOS `memory_pressure` 命令 (系统级): free% <5 → critical, <15 → warning
- 两个阈值交叉验证, psutil 优先级更高

**低内存策略 (≤16GB 总内存):**
- `_is_low_memory` 检测: `psutil.virtual_memory().total <= 16GB` (不受动态 available 波动影响)
- 强制互斥: 翻译时无条件卸载 embedding + 伴读模型, 跳过动态判断

**切换流程:**
1. `wait_for_idle()` 等待伴读查询完成 (30s 超时)
2. `check_memory_pressure()` 检测压力
3. 低内存机型: 强制 `_unload_embedding()` + `_unload_companion(save_kv=False)`
4. 正常机型: `free_memory(level)` — warning 卸载 embedding+伴读, critical 加保存 KV savepoint
5. 加载前二次确认内存
6. 加载翻译模型 → 翻译 → `switch_to_companion()` 恢复伴读+embedding

---

### KV Cache 配置 (`llm/kv_cache_config.py`)

**n_ctx 解析优先级:**
1. `LLM_MAX_CONTEXT` 环境变量 > 0 → 直接使用 (超硬件安全上限时警告并限制)
2. GGUF 二进制解析 `llama.context_length` → `min(detected, effective_max)`
3. 默认 `DEFAULT_N_CTX=65536`

**GGUF 二进制解析** (`_read_gguf_context_length`): 直接 `struct.unpack` 读文件头 — magic(4字节)→version(4)→tensor_count(8)→kv_count(8)→遍历 KV pairs 找 `llama.context_length`。实现了完整 GGUF v3 类型系统 (12种类型, ARRAY 递归跳过)。不加载模型, 0 token消耗, 0 VRAM

**内存分档** (`_CTX_TIERS`): ≥64GB→256K, ≥32GB→128K, ≥16GB→64K, <16GB→32K

**压缩阈值:** `COMPRESS_TRIGGER_RATIO=0.8`, `COMPRESS_BUFFER=2000`, `WARN_BUFFER=3000`, `BLOCK_BUFFER=1000`

---

### 本地 LLM 推理 (`llm/local_chat_engine.py`)

**双后端架构:**
- `_LlamaCppBackend`: 直接 `llama_cpp.Llama()` 实例, 支持 `kv_cache_seq_rm`/`truncate_cache`/`kv_cache_clear`
- `_MLXBackend`: `mlx_lm.load()`, Apple Silicon 原生加速, 不支持 seq_rm (MLX 不暴露部分序列移除)
- 自动检测: `.gguf`文件→llama_cpp, 目录含`model.safetensors`→MLX

**双后端统一接口:** `chat()/chat_stream()/raw_generate()/raw_generate_stream()/kv_cache_clear()/truncate_cache()/kv_cache_savepoint()/kv_cache_restore()`

**`truncate_cache` (已修复):** 统一使用 `kv_cache_clear()` 全清, 不再用 `seq_rm` 精确截断 (会导致 RoPE 位置错位)。依赖 llama.cpp 下次调用时 prefix caching 自动恢复

**Thinking 控制:** `_inject_thinking_prompt()` — 向 system prompt 追加 `<think>` 格式指令, 向最后一条 user message 追加 `\n\n<think>` 开头提示 (近因效应)

**HealthSnapshot:** 每次调用记录 prefill/decoded tokens, cache hit/miss, VRAM before/after, cache hit rate

---

### Thinking 流式过滤器

**ThinkingStreamFilter** (`llm_utils.py`): 状态机 — `token_count<80且未检测到think`→缓冲等待; `检测到<think>`→抑制输出; `检测到</think>`→开始透传; `缓冲>2000字符未结束`→强制放行原文不剥离

**ThinkTagStreamSplitter** (`llm_utils.py`): 更精确版本 — `_safe_end()` 检查缓冲区末尾是否有部分标签(如`</thi`), 与已知标签做 `startswith` 前缀匹配, 是则不完整→保留等下一个token; 不是则输出

**`_extract_answer_from_reasoning`**: 段落级分类 — 思考行(9种模式:"用户问的是/我需要/应该/根据规则"...), 正文标题行(7种:"摘要与引言：/结果与讨论：/核心发现："含冒号+后续文字)。找第一个正文标题→最后一个非思考行

---

### MessageStore 双视图 (`llm/message_store.py`)

- `conv.messages`: 完整对话历史 (UI 展示用, 只增不减)
- `get_active_slice()`: 最后一个 compact_boundary 之后的活跃切片 (LLM 上下文)
  - L0 (system prompt) 始终保留在切片头部
  - 过滤掉 boundary 标记本身, 保留摘要和后续轮次
- `clean_old_rounds()`: 超 500 条时物理删除第一个 boundary 之前的原始轮次 (保留 L0)
- `estimate_tokens()`: 混合估算 — 中文≈1.5 char/tok, 英文≈4 char/tok

---

### SessionState 状态机 (`llm/session_state.py`)

```
IDLE → THINKING → IDLE
           ↘ COMPRESSING → IDLE
```

- `can_accept()`: IDLE→直接接受; THINKING→拒绝("正在生成回答中"); COMPRESSING→接受但返回"queued"
- pending queue: `asyncio.Queue` 存 async callable 闭包 (非序列化数据), 压缩完成 `process_pending()` 逐个 await
- 压缩失败时不丢消息 (pending queue 在 except 也处理)

---

### 论文导入去重 (三层)

1. SHA256 文件哈希 → 相同字节直接跳过 (大文件>100MB 采样: 前64KB+中间64KB+后64KB)
2. 内容指纹 (SHA256(前2000字+总长+后500字+文件大小)) → 格式转换但内容相同跳过
3. doc_id 冲突 → 删旧 chunk 再写新

身份追踪: `paper_identities.json` (ChromaDB 目录下)
维度不匹配检测: ChromaDB collection metadata 的 `embedding_dim` vs 当前 `embedding_engine.dim`

---

### 嵌入引擎 (`rag/embeddings.py`)

**5 国内镜像** (按优先级): hf-mirror.com → huggingface.modelscope.cn → mirrors.tuna.tsinghua.edu.cn → ai.gitcode.com → aliendao.cn

**Jina 特殊补丁** (`_apply_jina_patch`): 修复 EuroBERT 相对导入、添加 `config_class` 兼容 transformers v5.x、修复 adapter `snapshot_download` 缺少 `cache_dir`

**设备检测**: 默认 CPU (`USE_GPU=1` 才能 GPU), jina PEFT 在 GPU 上长期批量推理存在内存碎片化

---

### 会话持久化 (`persistence.py`)

- SQLite WAL 模式 + 外键 CASCADE
- 自动迁移: `PRAGMA table_info` 动态检测缺失列 → `ALTER TABLE ADD COLUMN`
- 原子写入: BEGIN→DELETE+INSERT→COMMIT, 失败 ROLLBACK
- `save_full_conversation` 原子保存 conv+messages+papers

---

### 前端→后端 WebSocket 消息

| 消息 type | payload | 触发动作 |
|-----------|---------|---------|
| `user_query` | {queryText, context:{conversationId, activeDoc, activeParagraph}, openPapers, apiKey, model, baseUrl, thinking} | 主流程 |
| `import_paper` | {filePath} | 单论文向量化 |
| `new_conversation` | {name} | 创建会话 |
| `switch_conversation` | {conversationId} | 切换会话 |
| `delete_conversation` | {conversationId} | 删除会话 |
| `model_config` | {action, path, ...} | 模型配置 |
| `swap_model` | {modelName, ...} | 切换模型 |
| `build_control` | {action: pause/resume/cancel} | 构建控制 |
| `translation_request` | {filePath, scope, selectionRects, ...} | 翻译请求 |
| `cancel_translation` | {} | 取消翻译 |

### 后端→前端 WebSocket 消息

| type | 阶段 |
|------|------|
| `workflow_status` | classifying→searching→evaluating→generating |
| `llm_token` | LLM 流式输出 |
| `llm_health` | 每次 LLM 调用后 (call, prefill, output, cache_hit) |
| `llm_done` | 回答完成 (totalTokens, mode, refused, loopDetected) |
| `agent_step` | Agent 工具调用步骤 (thinking/tool_call/tool_result) |
| `agent_thinking` | Agent 思考流式输出 |
| `import_paper_progress` | 导入进度 |
| `import_paper_result` | 导入结果 (success, duplicate, chunks, sha256) |
| `build_index_progress` | 构建进度 (支持 pause/resume/cancel) |
| `translation_token` | 逐句翻译 token 流 |
| `conversation_list` | 会话列表 (连接时自动推送) |
| `status` | 状态广播 (压缩进度、会话切换等) |

---

### 论文结构解析 (零 LLM)

`_build_paper_outline()` 和 `classifier._build_outline()` — 纯正则扫描章节标题 (Abstract/Introduction/Methods/Results/Discussion/Conclusion 中英文), 每章取前150字拼成 <2000 字"论文地图"。零 token 消耗。

---

### @Paper 引用解析 (`handlers/query_utils.py`)

`parse_citation_refs()` — 正则 `@Paper\s*#(\d+)(?:\(([^)]*)\))?(?:\s*:\s*[""]([^""]+)[""])?`, 三种粒度可选:
- `@Paper#1` — 纯引用
- `@Paper#1(methods)` — 指定章节
- `@Paper#1(results): "the catalyst shows..."` — 精确引用原文

`build_citation_context()`: 用户提供了 quoted_text 就直接用 (不查 ChromaDB), 否则 fallback 向量检索

---

### Sidecar 进程管理 (`electron/sidecar.ts`)

**LLM 加载链:** llama-server 子进程优先 (端口 18923, 30s 健康检查轮询, 原始 HTTP GET /health) → 失败回退 LocalChatEngine 直连

**健康检查:** 5s 间隔, 120s 宽限期 (模型加载期只检测进程存活不因端口不通误杀), spawn 后 1s 即时端口探测

**崩溃恢复:** 最多 5 次自动重启, 3s 间隔。非零退出码写 crash.json (退出码/运行时长/重启次数/版本/平台), 启动成功删除

**环境变量安全:** 子进程环境白名单, 不透传父进程密钥

**macOS Gatekeeper:** `xattr -cr sidecar-dist` 清除隔离标记

**64MB 线程栈:** `threading.stack_size(64MB)` — macOS 默认 512KB, llama.cpp 深度递归会 SIGBUS 栈溢出

---

### WebUI 发布版

**启动流程:** `启动知伴.command` → `start-zhiban.sh` → 检测 macOS arm64 → 查找 Python 3.12+ → pip install from 清华源 → 交互选择下载模型 (4个国内镜像轮换, 15s超时自动切换) → python3 scripts/serve.py → 打开浏览器

**模型下载:** curl 断点续传 (`-C -`), 4 镜像依次尝试, 文件≥1MB 视为有效

**静态文件:** FastAPI `app.mount("/", StaticFiles(html=True))` 同端口 serve 前端

---

## LLM Provider 架构 (2026-05-21)

**Provider 抽象层**位于 `sidecar/llm/providers/`:
- `base.py` → `LLMProvider` Protocol
- `openai_compatible.py` → `OpenAICompatibleProvider` (覆盖 DeepSeek/Ollama/vLLM/LM Studio)
- `__init__.py` → `get_provider()` 工厂函数

**数据流**: Settings UI → localStorage/Zustand → WebSocket message → handler → engine → `build_provider()` → Provider

**关键设计决策**:
- Thinking/Reasoning 是一级功能 (多模型通用), Provider 负责翻译为具体 API 参数
- `build_provider()` 合并 per-request 参数与全局 config, per-request 优先
- `config.py` 中的 `DEEPSEEK_*` 变量是 deprecated 别名, 新代码统一用 `LLM_*`
- Extra Headers / Extra Body 在 frontend 用 JSON string 传递, backend 解析
- 本地模型 (Ollama 等) api_key 留空或填任意值, Provider 自动填 "ollama" 占位

## 相关文件

| 层级 | 文件 |
|------|------|
| Provider 协议 | `sidecar/llm/providers/base.py` |
| Provider 实现 | `sidecar/llm/providers/openai_compatible.py` |
| 工厂函数 | `sidecar/llm/providers/__init__.py` |
| 全局配置 | `sidecar/config.py` |
| LLM Proxy | `sidecar/llm/deepseek_proxy.py` |
| 引擎工具 | `sidecar/engine/llm_utils.py` |
| Classifier | `sidecar/engine/classifier.py` |
| Agent Loop | `sidecar/agent/agent_loop.py` |
| Agent Tools | `sidecar/agent/tools.py` |
| Agent Prompts | `sidecar/agent/prompts.py` |
| KV Cache 配置 | `sidecar/llm/kv_cache_config.py` |
| 本地推理引擎 | `sidecar/llm/local_chat_engine.py` |
| 模型管理器 | `sidecar/llm/model_manager.py` |
| 会话状态 | `sidecar/llm/session_state.py` |
| 消息存储 | `sidecar/llm/message_store.py` |
| 压缩引擎 | `sidecar/llm/compress_engine.py` |
| 嵌入引擎 | `sidecar/rag/embeddings.py` |
| 向量存储 | `sidecar/rag/vector_store.py` |
| 知识图谱 | `sidecar/rag/graph_store.py` |
| RAG 引擎 | `sidecar/rag/engine.py` |
| 搜索工具 | `sidecar/engine/search_utils.py` |
| 会话持久化 | `sidecar/persistence.py` |
| Handlers | `sidecar/handlers/llm_handlers.py`, `index_handlers.py`, `query_utils.py` |
| 翻译 | `sidecar/translation/translator.py`, `handler.py`, `extractor.py` |
| WebUI 启动器 | `sidecar/webui_launcher.py` |
| 前端设置 UI | `src/components/settings/LLMSection.tsx` |
| 前端类型 | `src/types/state.ts`, `websocket.ts` |
| Settings Store | `src/stores/slices/settingsSlice.ts` |
| Electron Sidecar | `electron/sidecar.ts` |
| Electron Main | `electron/main.ts` |
| 构建脚本 | `scripts/build-sidecar.sh` |
| WebUI 启动脚本 | `start-zhiban.sh`, `启动知伴.command` |
| 模型下载器 | `scripts/download-models.sh` |
| 镜像配置 | `config/mirrors.json` |

## 技术栈

- Frontend: React 18 + TypeScript 5.9 + Zustand 4 + Vite 6
- Backend: Python 3.14 + FastAPI + WebSocket (port 18921)
- LLM: llama-cpp-python 直连 / MLX (Apple Silicon) / llama-server 子进程 / OpenAI-compatible API
- Embeddings: jinaai/jina-embeddings-v5-text-nano (768-dim)
- Vector: ChromaDB (cosine distance)
- Knowledge Graph: NetworkX from YAML
- Desktop: Electron 33
- Translation: 腾讯混元 Hy-MT2-1.8B (Q4_K_M GGUF, 1.1GB)
- Companion LLM: Qwopus3.5-9B/4B-v3 GGUF
- Portable Python: cpython-3.14.4 astral-sh/python-build-standalone

## 不变量 (改动前必读)

这些行为是有意设计的, 修改会破坏现有功能:

1. **LLM 调用共享 L0+L1 KV cache 前缀** — 不能给分类或回答单独加 system 消息
2. **L2 内容在分类/回答/决策 prompt 中同位置插入** — 顺序: 问题→日期→阅读内容→上下文→指令
3. **`_term` 失败必须回落 RAG** — refused=True 时不返回, 继续走 RAG+LLM 回答
4. **`truncate_cache` 用全清不用 seq_rm** — seq_rm 精确截断会导致 RoPE 位置错位
5. **翻译不走 chat template** — 用 `raw_generate_stream` + 稳定前缀, 避免 chat template+truncate_cache 的 RoPE 问题
6. **chroma_doc_id 从文件名推导** — `conversation.py` `add_paper()` 里的逻辑与 `vector_store._extract_doc_id()` 保持一致
7. **导入去重三层不调换顺序** — SHA256 → 内容指纹 → doc_id 冲突
8. **分类器 prompt 不硬编码例子** — LLM 自主判断空间
9. **`WorkflowError` 按环节报错** — 前端按 step 展示
10. **启发式分类优先于 LLM 分类** — `_classify_heuristic()` 在 LLM 分类之前执行, 规则命中直接路由
11. **启发式分类绝不返回 `ref_num`** — 参考文献编号只能由 LLM 提取
12. **`l2_text` (论文全文) 统一在分类前获取** — 所有 LLM 回答路径都注入
13. **Agent Loop 替代固定 Call1/Call2** — AI 自主决定搜索策略, 不再硬编码搜索轮次
14. **低内存机型 (≤16GB) 强制互斥** — 翻译时跳过动态内存检测, 无条件卸载伴读+embedding
15. **`ThinkingStreamFilter` 80 token 阈值** — 无标签模型的前 80 token 被缓冲, 之后才判定为无 thinking 模式
16. **Sidecar 子进程环境白名单** — 不透传父进程密钥 (API keys/session tokens)
17. **`sanitize_api_key` 去除非 ASCII** — 用户复制粘贴可能带入零宽字符, 直接发 HTTP 会 401

## DMG 打包 & Bug 维护优化 (2026-06-01)

### 构建清洁度

`build-sidecar.sh` 在 PyInstaller 构建后自动执行清理:
- 删除 sidecar 源码中的测试文件 (`*_test.py`, `verify.py`, `acceptance.py`)
- 删除 Python stdlib 非必要目录 (`test/`, `idlelib/`, `turtledemo/`, `ensurepip/`) — 节省 ~40MB
- 删除 `.cache/`, `.chroma/`, `.conversations/` (运行时数据不应打包)
- 删除 `__pycache__/`, `.DS_Store`, 断开的符号链接, `.env` 文件
- 节省约 150MB (从 ~4.2G → ~4.0G)

### electron-builder.yml DMG 配置

- 支持 arm64 + x64 universal 构建
- `extraResources.filter` 增加排除规则: `!**/tests/**`, `!**/__pycache__/**`, `!**/.cache/**` 等
- `hardenedRuntime: true` + `disable-library-validation` (因捆绑 Python 无 Team ID)

### 文件日志系统 (Bug 事后分析)

位于 `electron/sidecar.ts`，双写策略:
- **内存缓冲区** → IPC 推送前端调试面板 (实时)
- **文件日志** → `~/Library/Application Support/ZhiBan/logs/zhiban.log` (持久化)
  - 日志轮转: 最多 10 个文件, 每个 5MB
  - 每次启动时轮转, 旧文件: `zhiban.0.log` ~ `zhiban.9.log`
- **Crash 状态** → `crash.json` (记录最后崩溃原因、退出码、运行时长)
  - 启动成功时自动清除
  - 启动时检查上次 crash 记录并写入日志

### start-sidecar.sh 启动脚本优化

- `set -euo pipefail` 严格错误处理
- 带时间戳的结构化日志输出 (`log()` 函数)
- Rpath 修复和签名使用 marker 文件跳过重复执行 (首次启动后秒级启动)
- 启动耗时诊断

### Sidecar 进程管理健壮性

- 进程崩溃时写入 `crash.json` 持久化状态
- `stopSidecar()` 同时关闭文件日志流
- 启动时检测上次 crash 记录并报告
- `_starting` 标志在所有错误路径正确重置
- 清理 debug 日志缓冲区上限 (1000 条)

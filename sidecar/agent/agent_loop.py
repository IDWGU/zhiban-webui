"""Agent 自主循环 — 仿 cc-haha queryLoop 设计。

核心流程:
  while step < max_steps:
    1. 构建消息: system_prompt + history + user_query [+ tool_results]
    2. 调用 LLM (流式)
    3. 解析响应:
       - 工具调用 → 执行工具，追加结果，继续循环
       - 文本回答 → 返回最终答案
    4. KV Cache: 复用 provider 实例 + 前缀对齐
    5. 检查停止条件

与 cc-haha 的对齐:
  - cc-haha: queryLoop() generator, while(true), callModel → runTools → continue
  - 知伴: AgentLoop.run() async, while step < max_steps, callLLM → execTools → continue

KV Cache 复用（关键优化）:
  - 同一轮内多次 LLM 调用共享 provider 实例
  - L0 (system prompt) 恒定 → 始终缓存
  - 历史前缀不变 → 前缀缓存命中
  - 仅最后一条 user message (含 tool results) 变化
  - 对于 llama.cpp/LM Studio: 使用 cache_prompt=True 参数
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .tools import AgentTool, ToolResult
from .prompts import (
    build_agent_system_prompt,
    build_user_context,
    parse_tool_call,
    detect_loop,
    is_final_answer,
    detect_tool_intent,
    detect_thinking_leak,
    build_answer_eval_prompt,
    parse_eval_json,
)

logger = logging.getLogger("zhiban.agent")


# ── 配置 ──

@dataclass
class AgentConfig:
    """Agent 行为配置"""
    max_steps: int = 5                # 最大工具调用轮数
    max_search_rounds: int = 3        # 最大搜索次数
    tool_call_timeout: float = 15.0   # 单次工具调用超时（秒）
    thinking_budget: int = 1024       # thinking token 预算
    answer_max_tokens: int = 0        # 回答最大 token（0 = 不限制，由模型自然停止）
    # tiny_model 已废弃，保留仅用于旧代码兼容
    verbose: bool = False             # 调试输出
    kv_cache_reuse: bool = True       # 是否复用 provider 实例（KV cache）
    answer_evaluation: bool = True    # 回答质量自检（正则+LLM评估）


@dataclass
class AgentStep:
    """单步执行记录"""
    step_index: int
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    llm_raw_output: str = ""
    llm_usage: dict = field(default_factory=dict)
    duration_ms: float = 0


@dataclass
class AgentResult:
    """Agent 运行结果"""
    final_answer: str
    steps: list[AgentStep] = field(default_factory=list)
    total_duration_ms: float = 0
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    loop_detected: bool = False
    truncated: bool = False


# ── Agent 循环 ──

class AgentLoop:
    """知伴自主 Agent 循环。

    用法:
        loop = AgentLoop(config, tools, callbacks)
        result = await loop.run(question, context, llm_params)
    """

    def __init__(
        self,
        config: AgentConfig,
        tools: list[AgentTool] | None = None,
        on_status: Callable | None = None,
        on_token: Callable | None = None,
        on_health: Callable | None = None,
    ):
        self.config = config
        self.tools: dict[str, AgentTool] = {}
        for t in (tools or []):
            self.tools[t.name] = t

        self.on_status = on_status
        self.on_token = on_token
        self.on_health = on_health

        # 运行时状态
        self._cancel_event: asyncio.Event | None = None
        self._steps: list[AgentStep] = []
        self._recent_outputs: list[str] = []  # 用于循环检测

    # ── 公共 API ──

    async def run(
        self,
        question: str,
        *,
        screen_ctx: str = "",
        screen_changed: bool = False,
        l2_text: str = "",
        current_topic: str = "",
        history_str: str = "",
        wake_records_text: str = "",
        question_type: str = "",
        system_prompt_override: str | None = None,
        active_slice: list[dict] | None = None,
        knowledge_brief: str = "",
        llm_call_fn: Callable | None = None,
        llm_params: dict | None = None,
        cancel_event: asyncio.Event | None = None,
        on_agent_step: Callable | None = None,
        eval_llm_call_fn: Callable | None = None,
        native_tools: bool = False,
    ) -> AgentResult:
        """运行 Agent 循环，返回最终答案。

        llm_call_fn: async (messages, max_tokens, **params) -> (text, usage_dict)
          messages 格式: [{"role": "...", "content": "..."}, ...]
        native_tools=True: API 模式，系统提示词不含 XML 工具格式
          （由 API 原生 tools 参数提供 function calling）
        """
        t_start = time.time()
        self._steps = []
        self._recent_outputs = []
        self._cancel_event = cancel_event or asyncio.Event()
        llm_params = llm_params or {}

        # 1. 构建 system prompt
        tools_list = list(self.tools.values())
        if system_prompt_override:
            system_prompt = system_prompt_override
        else:
            system_prompt = build_agent_system_prompt(
                tools=tools_list,
                knowledge_brief=knowledge_brief,
                native_tools=native_tools,
            )

        # 2. 构建初始消息
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
        ]

        # 添加历史（如果有 active_slice 直接用，否则用 history_str）
        # 必须保留完整消息字段：role/content/tool_calls/reasoning_content/tool_call_id
        # DeepSeek 多轮 API 要求工具调用轮次的 reasoning_content 和 tool_calls 完整回传
        if active_slice:
            for m in active_slice:
                role = m.get("role", "")
                if role not in ("user", "assistant", "tool"):
                    continue
                restored: dict = {"role": role, "content": m.get("content", "")}
                if role == "assistant":
                    if m.get("tool_calls"):
                        restored["tool_calls"] = m["tool_calls"]
                    if m.get("reasoning_content"):
                        restored["reasoning_content"] = m["reasoning_content"]
                elif role == "tool":
                    if m.get("tool_call_id"):
                        restored["tool_call_id"] = m["tool_call_id"]
                messages.append(restored)
        elif history_str:
            messages.append({"role": "user", "content": f"【对话历史】\n{history_str}"})

        # 构建用户上下文消息
        user_msg = build_user_context(
            question=question,
            screen_ctx=screen_ctx,
            screen_changed=screen_changed,
            l2_text=l2_text,
            current_topic=current_topic,
            history_str="" if active_slice else history_str,
            wake_records_text=wake_records_text,
            question_type=question_type,
        )
        messages.append({"role": "user", "content": user_msg})

        # 3. Agent 主循环
        step_count = 0
        search_count = 0
        final_answer = ""
        _force_stop = False      # 同参数重复检测标志
        _prev_tool_calls: list[tuple[str, str]] = []  # (name, args_json) 历史

        if self.config.verbose:
            _vprint(f"\n{'='*50}")
            _vprint(f"🤖 Agent 启动 | max_steps={self.config.max_steps}")
            _vprint(f"   问题: {question[:200]}")
            _vprint(f"   工具: {list(self.tools.keys())}")

        while step_count < self.config.max_steps:
            self._check_cancel()

            step = AgentStep(step_index=step_count)
            step_start = time.time()

            if self.on_status:
                if step_count == 0:
                    self.on_status("thinking", "正在分析问题...")
                else:
                    self.on_status("thinking", f"正在评估搜索结果 (第 {step_count + 1} 轮)...")

            # 3a. 调用 LLM (returns 2-tuple or 3-tuple with structured tool_calls)
            if llm_call_fn is None:
                raise RuntimeError("llm_call_fn is required")

            try:
                result = await llm_call_fn(
                    messages=messages,
                    max_tokens=self.config.answer_max_tokens
                    if step_count == 0
                    else max(2048, self.config.answer_max_tokens // 2) if self.config.answer_max_tokens > 0 else 0,
                    **llm_params,
                )
                llm_text, usage, structured_tool_calls, reasoning_text = (
                    result if isinstance(result, tuple) and len(result) >= 4
                    else (result, {}, None, "")
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("LLM call failed at step %d: %s", step_count, e)
                if step_count == 0:
                    raise  # 第一步就失败，向上抛
                # 非第一步失败：用已收集的信息回答
                final_answer = f"（AI 调用出错: {e}）"
                break

            step.llm_raw_output = llm_text
            step.llm_usage = usage
            step.duration_ms = (time.time() - step_start) * 1000
            self._recent_outputs.append(llm_text)

            if self.config.verbose:
                _vprint(f"   📝 Step {step_count}: {len(llm_text)} chars, "
                        f"{step.duration_ms:.0f}ms")

            # 3b. 工具调用：直接使用 API 返回的结构化 tool_calls（cc-haha 风格）
            tool_calls = structured_tool_calls
            # 思考内容已通过 agent_step(phase="thinking") 发送到前端思考面板，
            # 且通过 <think> 标签嵌在 content 流中。不再单独用 isThinking flag 发送。
            if tool_calls:
                # 有工具调用 → 执行工具
                step.tool_calls = tool_calls

                # ── 发送 agent_step: thinking ──
                if on_agent_step:
                    reasoning = _clean_thinking_text(
                        reasoning_text if isinstance(reasoning_text, str) else ""
                    )
                    if reasoning.strip():
                        on_agent_step({
                            "stepIndex": step_count,
                            "phase": "thinking",
                            "content": reasoning.strip(),
                        })
                    else:
                        # 模型未产出推理文本时，基于工具调用生成占位描述
                        tc_names = [tc.get("name", "") for tc in tool_calls if tc.get("name")]
                        tc_queries = [
                            (tc.get("arguments") or {}).get("query", "")
                            for tc in tool_calls
                            if (tc.get("arguments") or {}).get("query")
                        ]
                        placeholder = f"决定搜索知识库: {', '.join(tc_queries)}" if tc_queries \
                            else f"调用工具: {', '.join(tc_names)}" if tc_names \
                            else "分析问题并决定调用工具"
                        on_agent_step({
                            "stepIndex": step_count,
                            "phase": "thinking",
                            "content": placeholder,
                        })

                for tc in tool_calls:
                    self._check_cancel()
                    tool_name = tc.get("name", "")
                    tool_args = (tc.get("arguments") or {})

                    # ── 发送 agent_step: tool_call ──
                    if on_agent_step:
                        on_agent_step({
                            "stepIndex": step_count,
                            "phase": "tool_call",
                            "toolName": tool_name,
                            "toolArgs": json.dumps(tool_args, ensure_ascii=False),
                        })

                    if tool_name not in self.tools:
                        logger.warning("Unknown tool: %s", tool_name)
                        step.tool_results.append(ToolResult(
                            tool_name=tool_name,
                            success=False,
                            content=f"未知工具: {tool_name}。可用工具: {list(self.tools.keys())}",
                            error=f"unknown tool: {tool_name}",
                        ))
                        if on_agent_step:
                            on_agent_step({
                                "stepIndex": step_count,
                                "phase": "tool_result",
                                "toolResult": f"未知工具: {tool_name}",
                                "toolName": tool_name,
                                "success": False,
                                "error": f"unknown tool: {tool_name}",
                            })
                        continue

                    if tool_name == "search_knowledge_base":
                        search_count += 1
                        if search_count > self.config.max_search_rounds:
                            logger.info("Max search rounds reached (%d)", self.config.max_search_rounds)

                    # 同工具+同参数重复检测
                    _this_call = (tool_name, json.dumps(tool_args, ensure_ascii=False, sort_keys=True))
                    _same_count = sum(1 for pc in _prev_tool_calls if pc == _this_call)
                    if _same_count >= 2:
                        logger.warning("Same tool+args repeated %d times, forcing stop", _same_count + 1)
                        messages.append({"role": "user", "content": (
                            f"你已经用相同参数搜索了 {_same_count + 1} 次，结果都一样。"
                            "请立即停止搜索，直接基于已有信息回答。不要再调用工具。"
                        )})
                        if on_agent_step:
                            on_agent_step({
                                "stepIndex": step_count, "phase": "thinking",
                                "content": f"重复搜索 {_same_count + 1} 次，强制要求直接回答。",
                            })
                        _force_stop = True
                        break  # 跳出工具执行循环

                    _prev_tool_calls.append(_this_call)

                    if self.on_status:
                        self.on_status("searching", f"搜索: {tool_args.get('query', '')[:30]}")

                    try:
                        tool = self.tools[tool_name]
                        t0 = time.time()
                        result = await asyncio.wait_for(
                            tool.handler(**tool_args),
                            timeout=self.config.tool_call_timeout,
                        )
                        duration_ms = int((time.time() - t0) * 1000)
                        step.tool_results.append(result)
                        # ── 发送 agent_step: tool_result (丰富格式) ──
                        if on_agent_step:
                            on_agent_step({
                                "stepIndex": step_count,
                                "phase": "tool_result",
                                "toolResult": result.content[:2000],
                                "toolName": tool_name,
                                "success": result.success,
                                "error": result.error or "",
                                "durationMs": duration_ms,
                            })
                    except asyncio.TimeoutError:
                        step.tool_results.append(ToolResult(
                            tool_name=tool_name,
                            success=False,
                            content=f"工具调用超时（{self.config.tool_call_timeout}s）",
                            error="timeout",
                        ))
                        if on_agent_step:
                            on_agent_step({
                                "stepIndex": step_count,
                                "phase": "tool_result",
                                "toolResult": f"工具调用超时（{self.config.tool_call_timeout}s）",
                                "toolName": tool_name,
                                "success": False,
                                "error": "timeout",
                                "durationMs": int(self.config.tool_call_timeout * 1000),
                            })
                    except Exception as e:
                        logger.error("Tool %s failed: %s", tool_name, e)
                        step.tool_results.append(ToolResult(
                            tool_name=tool_name,
                            success=False,
                            content=f"工具调用失败: {e}",
                            error=str(e),
                        ))
                        if on_agent_step:
                            on_agent_step({
                                "stepIndex": step_count,
                                "phase": "tool_result",
                                "toolResult": f"工具调用失败: {e}",
                                "toolName": tool_name,
                                "success": False,
                                "error": str(e),
                            })

                # 同工具重复 → 跳过工具结果，直接让模型看到疲劳提示
                if _force_stop:
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": reasoning_text.strip() if reasoning_text else None,
                    })
                    self._steps.append(step)
                    step_count += 1
                    continue

                # 将工具调用和结果追加到消息 — DeepSeek 思考模式规范格式
                # Assistant 消息：content 必须为 null（有 tool_calls 时），
                # reasoning_content 独立承载思考文本。多轮对话中工具调用轮次的
                # reasoning_content 必须完整回传，否则 API 返回 400。
                reasoning_str = reasoning_text.strip() if reasoning_text else ""
                assistant_msg: dict = {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": reasoning_str or None,
                }
                formatted_tool_calls = []
                for tc in tool_calls:
                    tc_id = tc.get("id", "")
                    tc_name = tc.get("name", "")
                    tc_args = tc.get("arguments", {})
                    formatted_tool_calls.append({
                        "id": tc_id,
                        "type": "function",
                        "function": {
                            "name": tc_name,
                            "arguments": json.dumps(tc_args, ensure_ascii=False),
                        },
                    })
                assistant_msg["tool_calls"] = formatted_tool_calls
                messages.append(assistant_msg)

                # Tool 结果消息: 每个工具调用对应一条 role="tool" 消息，
                # tool_call_id 必须匹配 assistant 消息中的 tool_call id
                for i, tr in enumerate(step.tool_results):
                    tc_id = tool_calls[i].get("id", "") if i < len(tool_calls) else ""
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": tr.content,
                    })

                self._steps.append(step)
                step_count += 1

                # 检查是否超过最大步数
                if step_count >= self.config.max_steps:
                    # 强制要求模型基于已有信息总结回答
                    messages.append({
                        "role": "user",
                        "content": "你已经收集了足够的信息。现在请基于以上所有搜索结果，"
                                  "直接给出最终回答。不要再调用工具。",
                    })
                    if self.on_status:
                        self.on_status("thinking", "正在整合信息...")

                    try:
                        result = await llm_call_fn(
                            messages=messages,
                            max_tokens=self.config.answer_max_tokens,
                            **llm_params,
                        )
                        llm_text, usage, *_ = (result if isinstance(result, tuple) else (result, {}))
                        final_answer = _extract_answer_text(llm_text)
                        if llm_text and self.on_token:
                            self.on_token(llm_text, False)
                    except Exception:
                        final_answer = "（信息整合失败，请重试）"
                    break

                continue  # 继续循环

            # 3c. 无工具调用 → 最终回答
            final_answer = _extract_answer_text(llm_text)

            # ── 回答质量自检 ──
            if self.config.answer_evaluation and not tool_calls:
                # 正则快速检测：模型表达了搜索意图但不按格式调用工具
                if detect_tool_intent(llm_text):
                    logger.info("Step %d: tool intent detected via regex, re-prompting",
                                step_count)
                    if on_agent_step:
                        on_agent_step({
                            "stepIndex": step_count,
                            "phase": "thinking",
                            "content": "模型表达了搜索意图但未使用正确格式，正在要求重新调用工具...",
                        })
                    if native_tools:
                        _TOOL_FORMAT_RETRY = (
                            "你有可用的函数调用工具（function calling）。"
                            "请使用函数调用来搜索知识库，不要用文字描述搜索意图。"
                            "直接调用 search_knowledge_base 工具，传入合适的查询参数。"
                        )
                    else:
                        # Local model: force EXACT format, forbid any other output
                        _TOOL_FORMAT_RETRY = (
                            "你刚才输出了错误格式。现在请严格按以下格式重新输出：\n\n"
                            '<tool_call>\n'
                            '{"name":"search_knowledge_base","arguments":{"query":"你的搜索关键词"}}\n'
                            '</tool_call>\n\n'
                            "重要：\n"
                            "- 只输出上面这个 <tool_call> 块\n"
                            "- 不要在前面加任何解释、思考、抱歉等文字\n"
                            "- 不要在后面加任何文字\n"
                            "- 不要在 JSON 里面换行\n"
                            "- 如果你已经知道答案，就不要输出 <tool_call>，直接回答"
                        )
                    messages.append({
                        "role": "user", "content": _TOOL_FORMAT_RETRY,
                    })
                    self._steps.append(step)
                    step_count += 1
                    if step_count < self.config.max_steps:
                        continue
                    final_answer = "（模型多次尝试搜索失败，请重新提问）"
                    break

            # 循环检测 (临时关闭以排查截断问题)
            _loop_detected = detect_loop(llm_text, self._recent_outputs[:-1])
            if _loop_detected:
                logger.warning(
                    "Loop detected at step %d (DISABLED — would have set error msg). "
                    "final_answer=%d chars, llm_text=%d chars, recent_outputs=%d",
                    step_count, len(final_answer.strip()), len(llm_text), len(self._recent_outputs)
                )
            # DISABLED: 不再因循环检测而修改 final_answer

            # Token overflow recovery (cc-haha pattern):
            # When model hits max_tokens mid-answer, send resume message instead of truncating.
            _finish_reason = usage.get("finish_reason", "")
            if _finish_reason == "length" and final_answer.strip() and step_count < self.config.max_steps - 1:
                logger.info("Token overflow detected at step %d, resuming", step_count)
                messages.append({"role": "assistant", "content": final_answer.strip()})
                messages.append({"role": "user", "content": (
                    "Output token limit hit. Resume directly — no apology, "
                    "no recap of what you were doing. Pick up mid-thought "
                    "if that is where the cut happened. Continue from exactly where you stopped."
                )})
                self._steps.append(step)
                step_count += 1
                continue

            # 发送带 <think> 标签的原始文本。前端 </think> 解析器会分离
            # 思考面板内容和正文，无需 isThinking flag。
            if llm_text and self.on_token:
                self.on_token(llm_text, False)

            self._steps.append(step)
            break

        # ── 循环后：回答质量后处理 ──
        if self.config.answer_evaluation and final_answer.strip():
            # 步骤 1: 回答过短且有工具调用历史 → 强制重新总结
            _had_tools = any(s.tool_results for s in self._steps)
            _is_trivial = (
                len(final_answer) < 40
                or final_answer.strip() in ('**暂无**', '暂无', '暂无相关信息')  # noqa: FURB
            )
            if _had_tools and _is_trivial and llm_call_fn and step_count < self.config.max_steps:
                logger.info("Answer too short after tools (%d chars), forcing re-summary",
                            len(final_answer))
                messages.append({
                    "role": "user",
                    "content": (
                        "你已经拿到了搜索结果，但回答太简单。"
                        "请基于上面的搜索结果，重新撰写一个完整的回答。"
                        "直接给出结论，不要解释规则、不要复述问题、不要自我对话。"
                    ),
                })
                if self.on_status:
                    self.on_status("thinking", "回答太简短，正在要求重新总结...")
                try:
                    result = await llm_call_fn(
                        messages=messages,
                        max_tokens=0,
                    )
                    retry_text, _, *_ = (result if isinstance(result, tuple) else (result, {}))
                    if retry_text.strip():
                        retry_answer = _extract_answer_text(retry_text)
                        if len(retry_answer) > len(final_answer) * 2:
                            logger.info("Retry produced better answer: %d→%d chars",
                                        len(final_answer), len(retry_answer))
                            final_answer = retry_answer
                            if on_agent_step:
                                on_agent_step({
                                    "stepIndex": -1,
                                    "phase": "thinking",
                                    "content": "回答太简短，已自动重新生成完整回答。",
                                })
                except Exception:
                    pass

            # 步骤 2: 正则检测思考泄漏 → 直接剥离
            # 注意：此剥离只影响持久化版本，流式已推送的 token 无法撤回。
            if detect_thinking_leak(final_answer):
                stripped = _strip_thinking_by_regex(final_answer)
                if stripped and len(stripped) > 0:
                    logger.info("Regex strip: %d→%d chars",
                                len(final_answer), len(stripped))
                    final_answer = stripped

            # 步骤 3: LLM 完整度评估（答案太短时检查是否需要补充搜索）
            # 使用非流式 eval_llm_call_fn 避免评估推理泄漏到前端
            _eval_fn = eval_llm_call_fn or llm_call_fn
            if (_eval_fn
                    and final_answer.strip()
                    and len(final_answer) < 50
                    and not final_answer.startswith("（模型")):
                try:
                    if self.on_status:
                        self.on_status("thinking", "正在评估回答质量...")
                    eval_prompt = build_answer_eval_prompt(
                        question, final_answer)
                    eval_raw = await _eval_fn(
                        messages=[{"role": "user", "content": eval_prompt}],
                        max_tokens=256,
                    )
                    eval_text, _, *_ = (eval_raw if isinstance(eval_raw, tuple) else (str(eval_raw), {}))
                    eval_result = parse_eval_json(eval_text)
                    logger.info("Answer eval result: %s", eval_result)
                except Exception as e:
                    logger.warning("Answer eval failed: %s", e)

        # 4. 构建结果
        total_duration = (time.time() - t_start) * 1000

        if self.config.verbose:
            _vprint(f"   ✅ Agent 完成: {len(self._steps)} steps, "
                    f"{total_duration:.0f}ms, "
                    f"{'truncated' if step_count >= self.config.max_steps else 'complete'}")

        return AgentResult(
            final_answer=final_answer,
            steps=self._steps,
            total_duration_ms=total_duration,
            total_llm_calls=len(self._steps) + (1 if final_answer else 0),
            total_tool_calls=sum(len(s.tool_calls) for s in self._steps),
            loop_detected=False,
            truncated=step_count >= self.config.max_steps and not final_answer,
        )

    def _check_cancel(self):
        if self._cancel_event and self._cancel_event.is_set():
            raise asyncio.CancelledError("Agent loop cancelled")


# ── 辅助函数 ──

def _vprint(*args, **kwargs):
    """调试输出"""
    import sys
    print(*args, **kwargs, file=sys.stderr, flush=True)


def _find_answer_boundary(text: str) -> int:
    """在无 <think> 标签时，检测推理→最终回答的转折点。

    返回分割位置（回答开始的字符索引），-1 表示无法判断。
    与 local_chat_engine._find_thinking_split 使用相同的启发式策略。
    """
    if not text:
        return -1

    # 优先级 1：思考结束标识词
    conclusion_pattern = re.compile(
        r"(?:直接给出答案[即可。]*|现在(?:开始)?(?:正式)?回答[：:]?"
        r"|以下是(?:最终)?(?:正式)?回答[：:]?"
        r"|开始(?:回答|撰写|输出)[^。\n]*[。]?"
        r"|(?:好的|明白了)[，,]?(?:我(?:来|将|会))?(?:直接)?回答[：:]?"
        r"|让我(?:们|直接)?(?:正式)?回答[：:]?"
        r"|回答如下[：:]?"
        r"|正式回复[：:]?"
        r"|基于以上(?:分析|信息|思考)[，,]?\s*(?:我|现在)?(?:来|将)?回答[：:]?)"
        r"\s*\n",
        re.MULTILINE,
    )
    match = conclusion_pattern.search(text)
    if match:
        return match.end()

    # 优先级 2：**较长标题**(≥5 个非 * 字符) 独立成行
    heading = re.search(r'\n\*\*([^\*]{5,80})\*\*\s*\n', text)
    if heading:
        return heading.start()

    # 优先级 3：双换行后紧跟中文回答引导词
    answer_starters = (
        r"(?:现在|好的|根据|以下是|综上|总结|最终|让我|我来|下面|接下来"
        r"|基于以上|据此|因此|所以|那么|OK|好的|本文"
        r"|综合来看|总结一下|简要来说|简而言之|总而言之"
        r"|首先|第一|首先我要|我会从|我们将从)"
    )
    pattern = re.compile(rf"\n\n\s*(?={answer_starters})", re.MULTILINE)
    match = pattern.search(text)
    if match:
        return match.start()

    # 优先级 4：无引导词的双换行（前文至少 80 字）
    dnl_pos = text.find("\n\n")
    if dnl_pos >= 80:
        return dnl_pos

    return -1


def _clean_thinking_text(text: str) -> str:
    """清理流式输出中泄露的 <think> / </think> 标签碎片。

    问题：LLM tokenizer 可能把 &lt;think&gt; 切为两个 token，
    正则 &lt;/think&gt; 匹配不到完整标签，碎片直接显示在 UI。
    """
    if not text:
        return text
    # 完整标签
    text = re.sub(r'</think>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<think>', '', text, flags=re.IGNORECASE)
    # 截断碎片: 开头 （< 或 </ 后跟 think）
    text = re.sub(r'<[/\s]*think\b', '', text, flags=re.IGNORECASE)
    # 截断碎片: 结尾 （think 后跟 >）
    text = re.sub(r'\bthink[/\s]*>', '', text, flags=re.IGNORECASE)
    return text


def _extract_answer_text(text: str) -> str:
    """从模型输出中提取最终回答文本。

    如果输出中含工具调用标记，提取标记之外的纯文本。
    去掉空的/仅含标点的输出。
    """
    if not text:
        return ""

    # 移除 <think>...</think> 块
    cleaned = re.sub(
        r'<think>.*?</think>',
        '',
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # 移除 tool_call 块，保留其余文本
    cleaned = re.sub(
        r'<tool_call>.*?</tool_call>',
        '',
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(
        r'\[工具调用:.*?(?=\[工具调用:|$)',
        '',
        cleaned,
        flags=re.DOTALL,
    )

    # 移除 leading/trailing 空白
    cleaned = cleaned.strip()
    if not cleaned or cleaned in ('{}', '[]', '""', "''"):
        return ""

    # 清理可能残留的 <think> / </think> 标签碎片
    cleaned = _clean_thinking_text(cleaned).strip()
    return cleaned




def _extract_reasoning_text(text: str) -> str:
    """从模型输出中提取推理文本（剥离工具调用块），用于 thinking 展示。"""
    if not text:
        return ""
    # 移除 <tool_call>...</tool_call> 块
    cleaned = re.sub(
        r'<tool_call>.*?</tool_call>',
        '', text, flags=re.DOTALL | re.IGNORECASE,
    )
    # 移除未闭合的 <tool_call> 块
    cleaned = re.sub(
        r'<tool_call>.*$', '', cleaned, flags=re.DOTALL | re.IGNORECASE,
    )
    # 移除 [工具调用: ...] 块
    cleaned = re.sub(
        r'\[工具调用:.*?(?=\[工具调用:|$)',
        '', cleaned, flags=re.DOTALL,
    )
    return cleaned.strip()



def extract_agent_messages(steps: list[AgentStep]) -> list[dict]:
    """从 Agent 步骤中提取 chat 消息列表（用于持久化）。"""
    messages = []
    for step in steps:
        if step.tool_calls:
            # 过滤掉 tool_call 标记，只保留思考部分
            thinking = _extract_answer_text(step.llm_raw_output)
            if thinking:
                messages.append({"role": "assistant", "content": f"🤔 {thinking}"})
            # 工具调用
            for tc in step.tool_calls:
                messages.append({
                    "role": "system",
                    "content": f"🔍 搜索: {(tc.get('arguments') or {}).get('query', '')}",
                })
        else:
            # 纯文本回答
            cleaned = _extract_answer_text(step.llm_raw_output)
            if cleaned:
                messages.append({"role": "assistant", "content": cleaned})
    return messages


def _strip_thinking_by_regex(text: str) -> str:
    """用正则剥离常见的自我对话/思考前缀，保留实质性回答内容。

    处理小模型在回答中附加的"用户问的是""根据我的规则"等元文本。
    策略：找实质性回答标记（粗体标题、答案引导句），返回其后的内容。
    """
    import re
    if not text or len(text) < 20:
        return text

    # 策略 1: 粗体 Markdown 标题在前 60% 位置 → 标题开始是实际回答
    m = re.search(r'\*\*([^*]{3,60})\*\*', text)
    if m and m.start() < len(text) * 0.6:
        return text[m.start():].strip()

    # 策略 2: 找"直接回答/答案是/结论是"等最终回答标记
    answer_markers = [
        r'(?:直接|正式|最终)(?:回答|答案|结论)[:：]?\s*',
        r'答案是[:：]?\s*',
        r'结论是[:：]?\s*',
        r'简而言之[,，]?\s*',
        r'综上所述[,，]?\s*',
    ]
    for pattern in answer_markers:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            after = text[m.end():].strip()
            if len(after) > 20:
                return after

    # 策略 3: 数学算式 → 提取答案
    m = re.search(r'(?:等于|=\s*)(\d+)', text)
    if m:
        return f"答案是 {m.group(1)}"

    # 策略 4: 文本以思考前缀开头 → 取最后一个非编号列表的段落
    if re.match(r'^(?:用户问|根据我的|让我|我需要|我应该)', text):
        parts = text.split('\n\n')
        for p in reversed(parts):
            p = p.strip()
            if p and not re.match(r'^\d+\.\s', p) and len(p) > 15:
                if not re.match(r'^(?:用户|根据|让我|我是|这是|但是|不过|首先)', p):
                    return p
        # 回退：返回最长段落
        if parts:
            longest = max(parts, key=len).strip()
            return longest

    return text

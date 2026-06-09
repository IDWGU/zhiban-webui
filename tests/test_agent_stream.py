"""Agent 模式 WebSocket 流式输出诊断脚本。

验证：
1. agent_step 消息是否到达（thinking/tool_call/tool_result）
2. llm_token 消息是否正确分流（isThinking true/false）
3. 内容令牌是否逐 token 到达（流式），而非一次性批量到达
"""
import asyncio
import json
import time
import sys
import os
import websockets

# Auto-detect port: try common ports, fall back to env
import os as _os
_PORT = _os.getenv("WS_PORT") or "18921"
for _try_port in (_PORT, "18921", "18922", "18923", "18924", "18925"):
    import socket as _socket
    try:
        _s = _socket.socket()
        _s.settimeout(0.2)
        _s.connect(("127.0.0.1", int(_try_port)))
        _s.close()
        _PORT = str(_try_port)
        break
    except Exception:
        continue
BACKEND_URL = f"ws://127.0.0.1:{_PORT}/ws"
HTTP_BASE = f"http://127.0.0.1:{_PORT}"


async def wait_ready(timeout=60):
    """等待后端就绪"""
    import aiohttp
    start = time.time()
    async with aiohttp.ClientSession() as session:
        while time.time() - start < timeout:
            try:
                async with session.get(f"{HTTP_BASE}/ready") as resp:
                    data = await resp.json()
                    if data.get("ready"):
                        return True
            except Exception:
                pass
            await asyncio.sleep(1)
    return False


async def diagnose_agent_streaming():
    print("=== Agent 模式流式输出诊断 ===\n")

    # 等待后端就绪
    print("[1] 等待后端就绪...")
    if not await wait_ready():
        print("❌ 后端未就绪")
        return
    print("   ✅ 后端就绪\n")

    # 连接 WebSocket
    print("[2] 连接 WebSocket...")
    ws = await websockets.connect(BACKEND_URL)
    print(f"   ✅ 已连接\n")

    # 先获取对话列表
    await ws.send(json.dumps({"type": "list_conversations"}))
    conv_msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
    conv_id = "default"
    if conv_msg.get("type") == "conversation_list" and conv_msg.get("conversations"):
        conv_id = conv_msg["conversations"][0]["id"]
        await ws.send(json.dumps({"type": "switch_conversation", "conversationId": conv_id}))
        # 消费 switched 消息
        try:
            await asyncio.wait_for(ws.recv(), timeout=3)
        except Exception:
            pass
    print(f"   对话 ID: {conv_id}\n")

    # 发送查询
    query = "解释一下Transformer的注意力机制"
    print(f"[3] 发送查询: \"{query}\"")

    await ws.send(json.dumps({
        "type": "user_query",
        "queryText": query,
        "context": {
            "activeDoc": "",
            "activeParagraph": "",
            "paragraphIndex": None,
            "conversationId": conv_id,
        },
        "timestamp": int(time.time() * 1000),
    }))

    # 收集消息
    messages = []
    agent_steps = []
    thinking_tokens_count = 0
    content_tokens_count = 0
    content_tokens_timestamps = []
    start_time = time.time()

    print("\n[4] 接收消息（最多等待 120 秒）...")
    print("-" * 60)

    try:
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=120)
            msg = json.loads(raw)
            msg_type = msg.get("type", "?")
            elapsed = time.time() - start_time

            if msg_type == "llm_token":
                is_thinking = msg.get("isThinking", False)
                token = msg.get("token", "")
                if is_thinking:
                    thinking_tokens_count += 1
                else:
                    content_tokens_count += 1
                    content_tokens_timestamps.append(elapsed)
                # 精简打印（避免刷屏）
                if content_tokens_count <= 5 or content_tokens_count % 20 == 0:
                    tag = "🧠" if is_thinking else "📝"
                    preview = token[:60].replace("\n", "\\n")
                    print(f"  [{elapsed:6.1f}s] {tag} llm_token isThinking={is_thinking} | "
                          f"{preview}{'...' if len(token) > 60 else ''}")

            elif msg_type == "agent_step":
                phase = msg.get("phase", "?")
                step_data = {
                    "stepIndex": msg.get("stepIndex"),
                    "phase": phase,
                    "content": msg.get("content", "")[:80],
                    "toolName": msg.get("toolName", ""),
                    "toolResult": msg.get("toolResult", "")[:80],
                }
                agent_steps.append(step_data)
                icon = {"thinking": "💭", "tool_call": "🔧", "tool_result": "📋"}.get(phase, "?")
                print(f"  [{elapsed:6.1f}s] {icon} agent_step phase={phase} "
                      f"stepIdx={msg.get('stepIndex')}")

            elif msg_type == "llm_done":
                print(f"  [{elapsed:6.1f}s] ✅ llm_done")
                messages.append(msg)
                break

            elif msg_type in ("pong",):
                pass  # 忽略心跳

            else:
                detail = json.dumps(msg, ensure_ascii=False)[:200]
                print(f"  [{elapsed:6.1f}s] 🔔 {msg_type} | {detail}")

    except asyncio.TimeoutError:
        print("  ⚠️ 超时 (120s)")

    print("-" * 60)

    # ── 诊断结果 ──
    print("\n[5] 诊断结果")
    print("=" * 60)

    # 检查 1: agent_step 消息
    print(f"\n📋 agent_step 消息数量: {len(agent_steps)}")
    if agent_steps:
        phases = [s["phase"] for s in agent_steps]
        print(f"   阶段序列: {phases}")
        has_thinking = any(s["phase"] == "thinking" for s in agent_steps)
        has_tool_call = any(s["phase"] == "tool_call" for s in agent_steps)
        has_tool_result = any(s["phase"] == "tool_result" for s in agent_steps)
        print(f"   含 thinking: {has_thinking} | tool_call: {has_tool_call} | tool_result: {has_tool_result}")
        if has_thinking and has_tool_call and has_tool_result:
            print("   ✅ 完整的 agent_step 阶段覆盖")
        else:
            print("   ⚠️ 缺少某些阶段")
    else:
        print("   ❌ 未收到任何 agent_step 消息！")

    # 检查 2: 令牌分流
    print(f"\n📝 llm_token 统计:")
    print(f"   思考令牌 (isThinking=true): {thinking_tokens_count}")
    print(f"   内容令牌 (isThinking=false): {content_tokens_count}")

    if thinking_tokens_count > 0 and content_tokens_count > 0:
        print("   ✅ 令牌正确分流")
    elif thinking_tokens_count == 0 and content_tokens_count > 0:
        print("   ⚠️ 没有思考令牌 — 可能是模型不支持 thinking 模式")
    elif thinking_tokens_count > 0 and content_tokens_count == 0:
        print("   ❌ 没有内容令牌 — 这可能是一个 bug")
    else:
        print("   ⚠️ 没有收到 llm_token")

    # 检查 3: 流式输出
    if len(content_tokens_timestamps) >= 3:
        # 检查 token 间的时间间隔
        intervals = []
        for i in range(1, min(len(content_tokens_timestamps), 20)):
            intervals.append(content_tokens_timestamps[i] - content_tokens_timestamps[i - 1])

        avg_interval = sum(intervals) / len(intervals)
        max_interval = max(intervals)
        first_ts = content_tokens_timestamps[0]
        last_ts = content_tokens_timestamps[-1]
        total_span = last_ts - first_ts

        print(f"\n⏱️ 流式检查:")
        print(f"   首个令牌: {first_ts:.1f}s | 末个令牌: {last_ts:.1f}s")
        print(f"   时间跨度: {total_span:.1f}s")
        print(f"   平均间隔: {avg_interval*1000:.0f}ms | 最大间隔: {max_interval*1000:.0f}ms")

        if total_span > 1.0 and avg_interval < 0.5:
            print("   ✅ 令牌逐 token 到达（真正的流式输出）")
        elif total_span < 0.3:
            print("   ❌ 所有令牌在短时间内一次性到达 — 不是流式输出！")
        else:
            print("   ⚠️ 间隔较大，可能是 chunk 模式")
    else:
        print("\n⏱️ 流式检查: 令牌数不足以判断")

    # 检查 4: 最终回答内容
    if content_tokens_count > 0:
        print(f"\n📄 回答令牌总数: {content_tokens_count}")
    else:
        print(f"\n📄 未收到回答内容")

    print("\n" + "=" * 60)

    await ws.close()
    return {
        "agent_steps": agent_steps,
        "thinking_tokens": thinking_tokens_count,
        "content_tokens": content_tokens_count,
        "is_streaming": len(content_tokens_timestamps) >= 3
            and (content_tokens_timestamps[-1] - content_tokens_timestamps[0]) > 1.0,
    }


if __name__ == "__main__":
    asyncio.run(diagnose_agent_streaming())

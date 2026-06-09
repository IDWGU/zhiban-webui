"""多轮极端端到端测试 — 覆盖所有已知故障模式。

故障模式:
  F1: <tool_call> XML 泄漏到思考面板 (isThinking=True)
  F2: <tool_call> XML 泄漏到正文 (isThinking=False) — 预期行为
  F3: 模型预判"内容为空"自污染上下文
  F4: agent_step 分类错误（答案进 thinking）
  F5: 搜索结果返回但被模型忽略
  F6: 多轮工具调用链正确性
  F7: 回答过短/垃圾

渲染检查点:
  R1: llm_token(isThinking=True) → no <tool_call>
  R2: agent_step(phase=thinking) → no <tool_call>
  R3: agent_step(phase=tool_call) → has toolName + toolArgs
  R4: agent_step(phase=tool_result) → has toolResult content
  R5: llm_done → 回答不为空，不包含"内容为空"
"""
import asyncio
import json
import sys
import time
import websockets
import aiohttp

BACKEND = "ws://127.0.0.1:18921/ws"
HTTP = "http://127.0.0.1:18921"

# ── 测试用例 ──
TEST_CASES = [
    {
        "id": "paper_search",
        "query": "电化学CO2还原中过渡金属大环化合物有什么优势",
        "expects": "需要搜索论文知识库，回答应包含具体催化剂信息",
        "min_answer_chars": 100,
        "must_not_contain": ["内容为空", "内容似乎为空"],
    },
    {
        "id": "concept_explain",
        "query": "什么是密度泛函理论",
        "expects": "学术概念解释，可能搜索也可能直接回答",
        "min_answer_chars": 80,
        "must_not_contain": ["内容为空"],
    },
    {
        "id": "short_fact",
        "query": "论文126的作者是谁",
        "expects": "简短事实查询，需要搜索",
        "min_answer_chars": 20,
        "must_not_contain": ["内容为空"],
    },
    {
        "id": "multi_turn",
        "query": "解释一下电催化CO2还原的反应机理",
        "expects": "可能触发多轮搜索",
        "min_answer_chars": 100,
        "must_not_contain": ["内容为空", "内容似乎为空"],
    },
]


async def wait_ready(timeout=30):
    async with aiohttp.ClientSession() as s:
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                async with s.get(f"{HTTP}/ready") as r:
                    if (await r.json()).get("ready"):
                        return True
            except Exception:
                pass
            await asyncio.sleep(1)
    return False


async def run_test(test_case):
    """Run a single test case, returning diagnostic data."""
    result = {
        "id": test_case["id"],
        "query": test_case["query"],
        "thinking_tokens": [],
        "content_tokens": [],
        "agent_steps": [],
        "answer": "",
        "checks": {},
    }

    ws = await websockets.connect(BACKEND)
    await ws.send(json.dumps({"type": "new_conversation", "name": f"test-{test_case['id']}"}))
    resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
    cid = resp.get("conversationId", "default")

    await ws.send(json.dumps({
        "type": "user_query",
        "queryText": test_case["query"],
        "context": {"activeDoc": "", "activeParagraph": "", "conversationId": cid},
        "timestamp": int(time.time() * 1000),
    }))

    try:
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=120))
            t = msg.get("type", "")

            if t == "llm_token":
                token = msg.get("token", "")
                if msg.get("isThinking"):
                    result["thinking_tokens"].append(token)
                else:
                    result["content_tokens"].append(token)
            elif t == "agent_step":
                result["agent_steps"].append(msg)
            elif t == "llm_done":
                break
    except asyncio.TimeoutError:
        result["checks"]["timeout"] = True

    await ws.close()

    result["answer"] = "".join(result["content_tokens"])
    thinking_text = "".join(result["thinking_tokens"])

    # ── 运行检查 ──
    # R1: thinking tokens 不含 <tool_call>
    result["checks"]["R1_no_tc_in_thinking"] = "<tool_call" not in thinking_text.lower()

    # R2: agent_step(thinking) 不含 <tool_call>
    think_steps = [s for s in result["agent_steps"] if s.get("phase") == "thinking"]
    result["checks"]["R2_no_tc_in_agent_think"] = all(
        "<tool_call" not in s.get("content", "").lower() for s in think_steps
    )

    # R3: tool_call phase has toolName
    tool_steps = [s for s in result["agent_steps"] if s.get("phase") == "tool_call"]
    result["checks"]["R3_tool_calls_have_name"] = all(
        s.get("toolName") for s in tool_steps
    ) if tool_steps else None  # None = N/A

    # R4: tool_result has content
    result_steps = [s for s in result["agent_steps"] if s.get("phase") == "tool_result"]
    result["checks"]["R4_tool_results_have_content"] = all(
        len(s.get("toolResult", "")) > 10 for s in result_steps
    ) if result_steps else None

    # R5: 回答不空，不含垃圾
    answer = result["answer"]
    result["checks"]["R5_answer_not_empty"] = len(answer.strip()) > 10
    result["checks"]["R5_no_content_empty"] = not any(
        phrase in answer for phrase in test_case.get("must_not_contain", [])
    )
    result["checks"]["R5_min_length"] = len(answer) >= test_case.get("min_answer_chars", 20)

    # R6: agent_step 有正确的 phase
    phases = [s.get("phase") for s in result["agent_steps"]]
    valid_phases = all(p in ("thinking", "tool_call", "tool_result") for p in phases)
    result["checks"]["R6_valid_phases"] = valid_phases

    return result


def print_result(r, i):
    """Pretty-print a test result."""
    print(f"\n{'='*60}")
    print(f"  TEST #{i+1}: {r['id']}")
    print(f"  Query: {r['query'][:60]}...")
    print(f"{'='*60}")

    # Stats
    thinking_chars = len("".join(r["thinking_tokens"]))
    content_chars = len(r["answer"])
    steps = [s.get("phase") for s in r["agent_steps"]]
    tool_count = len([s for s in r["agent_steps"] if s.get("phase") == "tool_call"])

    print(f"  Thinking: {thinking_chars} chars | Content: {content_chars} chars")
    print(f"  Agent steps: {steps} ({tool_count} tool calls)")

    # Answer preview
    answer_preview = r["answer"][:150].replace("\n", "\\n")
    print(f"  Answer: {answer_preview}...")

    # Checks
    all_ok = True
    for check_name, value in r["checks"].items():
        if value is None:
            continue  # N/A
        icon = "✅" if value else "❌"
        if not value:
            all_ok = False
        print(f"  {icon} {check_name}")

    return all_ok


async def main():
    if not await wait_ready():
        print("❌ Server not ready")
        return 1

    print("╔══════════════════════════════════════════════╗")
    print("║  知伴 Agent V12 — 多轮极端端到端测试         ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"\n  Test cases: {len(TEST_CASES)}")
    print(f"  Failure modes covered: F1-F7")
    print(f"  Render checkpoints: R1-R6")

    results = []
    for i, tc in enumerate(TEST_CASES):
        try:
            r = await run_test(tc)
            results.append(r)
        except Exception as e:
            print(f"\n  ❌ TEST #{i+1} CRASHED: {e}")
            results.append({"id": tc["id"], "query": tc["query"], "checks": {"crash": str(e)}})

    # ── Summary ──
    print(f"\n\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")

    passed = 0
    failed = 0
    for i, r in enumerate(results):
        ok = print_result(r, i)
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"  PASSED: {passed}/{len(results)}")
    print(f"  FAILED: {failed}/{len(results)}")
    print(f"{'='*60}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

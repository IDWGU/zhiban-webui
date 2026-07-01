"""快速端到端测试：验证搜索结果返回实际内容（非空条目）"""
import asyncio
import json
import sys
sys.path.insert(0, "/Users/xiaodu/zhiban-standalone")

import websockets


async def main():
    async with websockets.connect("ws://localhost:18921/ws") as ws:
        msg = {
            "type": "user_query",
            "queryText": "126.pdf论文讨论了几种大环化合物，分别是什么",
            "model": "deepseek-chat",
            "baseUrl": "https://api.deepseek.com",
            "openPapers": [{"paperId": "126", "title": "126.pdf"}],
            "context": {
                "activeDoc": "126.pdf",
                "conversationId": "test-search-content",
            },
            "history": [],
        }
        await ws.send(json.dumps(msg))
        print(f"Sent: {json.dumps(msg, ensure_ascii=False)[:200]}...")

        tool_calls = 0
        tool_results_with_content = 0
        answer_tokens = 0

        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=90)
                msg = json.loads(raw)
                t = msg.get("type", "")

                if t == "agent_step":
                    d = msg.get("data", {})
                    phase = d.get("phase", "")
                    if phase == "tool_call":
                        tool_calls += 1
                        print(f"[TOOL_CALL] {d.get('toolName')} args={d.get('toolArgs')}")
                    elif phase == "tool_result":
                        content = d.get("toolResult", "")
                        if "Paper #" in content and len(content) > 50:
                            tool_results_with_content += 1
                        print(f"[TOOL_RESULT] {len(content)} chars paper_in_content={'Paper' in content}")
                    elif phase == "thinking":
                        txt = str(d.get("content", ""))[:100]
                        print(f"[THINKING] {txt}")
                elif t == "llm_token":
                    answer_tokens += 1
                elif t == "llm_done":
                    usage = msg.get("data", {}).get("usage", {})
                    print(f"[DONE] tokens={usage.get('total_tokens', '?')}")
                    break
                elif t == "error":
                    print(f"[ERROR] {msg}")
                    break
                elif t == "status":
                    print(f"[STATUS] {msg.get('code')}: {msg.get('message', '')}")
        except asyncio.TimeoutError:
            print("[TIMEOUT]")

        print(f"\ntool_calls={tool_calls} tool_results_with_content={tool_results_with_content} answer_tokens={answer_tokens}")
        assert tool_calls > 0, "FAIL: 没有工具调用"
        assert tool_results_with_content > 0, "FAIL: 搜索结果仍为空"
        assert answer_tokens > 0, "FAIL: 没有生成回答"
        print("PASS")


if __name__ == "__main__":
    asyncio.run(main())

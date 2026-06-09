"""完整端到端测试：验证 agent 多轮工具调用和搜索结果内容"""
import asyncio
import json
import os
import sys
sys.path.insert(0, "/Users/xiaodu/zhiban-standalone")
from dotenv import load_dotenv
load_dotenv("/Users/xiaodu/zhiban-standalone/.env")
import websockets


async def main():
    api_key = os.getenv("LLM_API_KEY", "")
    async with websockets.connect("ws://localhost:18921/ws") as ws:
        await ws.send(json.dumps({
            "type": "user_query",
            "queryText": "126.pdf论文讨论了几种大环化合物，分别是什么？请搜索知识库后回答。",
            "model": "deepseek-chat",
            "baseUrl": "https://api.deepseek.com",
            "apiKey": api_key,
            "openPapers": [{"paperId": "126", "title": "126.pdf", "filename": "126.pdf"}],
            "context": {"activeDoc": "126.pdf", "conversationId": "test-e2e"},
            "history": [],
        }))

        tool_calls = []
        tool_results = []
        thinking_texts = []
        answer = ""
        phases = []
        health_reports = []
        result = {"tc": 0, "tr": 0, "think": 0, "ans": 0}

        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=180)
                msg = json.loads(raw)
                t = msg.get("type", "")

                # agent_step 消息：字段在顶层（不是 data 下）
                if t == "agent_step":
                    phase = msg.get("phase", "?")
                    phases.append(phase)
                    if phase == "tool_call":
                        result["tc"] += 1
                        name = msg.get("toolName", "?")
                        args = (msg.get("toolArgs", "") or "")[:80]
                        tool_calls.append({"name": name, "args": args})
                        print(f"  [TC #{result['tc']}] {name}({args})")
                    elif phase == "tool_result":
                        result["tr"] += 1
                        content = msg.get("toolResult", "")
                        ok = msg.get("success", False)
                        has_paper = "Paper #" in content
                        has_body = len(content) > 80
                        tool_results.append({"ok": ok, "len": len(content), "paper": has_paper, "body": has_body})
                        preview = content[:100].replace("\n", " ")
                        status = "OK" if (has_paper and has_body) else "EMPTY"
                        print(f"  [TR #{result['tr']}] {status} success={ok} len={len(content)} {preview}...")
                    elif phase == "thinking":
                        result["think"] += 1
                        text = msg.get("content", "")
                        thinking_texts.append(text[:200])
                        print(f"  [THINK #{result['think']}] {text[:120]}")
                    else:
                        print(f"  [agent_step/{phase}] content={msg.get('content','')[:80]}")
                elif t == "llm_token":
                    result["ans"] += 1
                    answer += msg.get("token", "")
                elif t == "llm_done":
                    d = msg.get("data", {}) or {}
                    usage = d.get("usage", {})
                    print(f"\n[DONE] {usage.get('total_tokens','?')} tokens, {d.get('duration',0)}ms")
                    break
                elif t == "llm_citation":
                    n = len(msg.get("citations", []))
                    print(f"  [CITATION] {n} refs")
                elif t == "llm_health":
                    health_reports.append(msg)
                elif t == "error":
                    print(f"  [ERROR] {msg.get('code')}: {msg.get('message','')[:200]}")
                elif t == "status":
                    code = msg.get("code", "")
                    if code and code != "ready":
                        print(f"  [STATUS] {code}: {msg.get('message','')[:80]}")
        except asyncio.TimeoutError:
            print("  [TIMEOUT after 180s]")

        print(f"\n{'='*60}")
        print(f"PHASES: {' → '.join(phases)}")
        print(f"tool_calls: {result['tc']}, tool_results: {result['tr']}, "
              f"thinking: {result['think']}, answer_tokens: {result['ans']}")
        print(f"health_reports: {len(health_reports)}")
        print(f"Answer ({len(answer)} chars):")
        print(f"  {answer[:400]}")

        # 验证结果
        errors = []
        if result["tc"] == 0:
            errors.append("NO tool calls — model didn't search")
        if result["tr"] == 0:
            errors.append("NO tool results")
        if result["tc"] > 0 and result["tr"] == 0:
            errors.append("CRITICAL: tool calls without results")
        if len(answer) < 20:
            errors.append(f"Answer too short ({len(answer)} chars)")

        # 检查搜索结果是否包含实际内容
        empty_results = [tr for tr in tool_results if not tr["paper"] or not tr["body"]]
        if result["tr"] > 0 and len(empty_results) == result["tr"]:
            errors.append(f"ALL tool results EMPTY ({result['tr']}/{result['tr']})")
        elif result["tr"] > 0 and empty_results:
            print(f"\n  {len(empty_results)}/{result['tr']} tool results appear empty (may be valid if search returned nothing)")

        if errors:
            print(f"\n❌ FAILURES:")
            for e in errors:
                print(f"  - {e}")
            sys.exit(1)
        else:
            print(f"\n✅ ALL CHECKS PASSED — agent is working correctly")
            print(f"   tool_calls={result['tc']} tool_results={result['tr']} answer_chars={len(answer)}")


if __name__ == "__main__":
    asyncio.run(main())

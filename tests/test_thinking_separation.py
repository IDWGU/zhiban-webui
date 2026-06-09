"""验证 `<think>` 标签方案：后端透传标签，前端解析器分离"""
import asyncio, json, os, re, sys
sys.path.insert(0, "/Users/xiaodu/zhiban-standalone")
from dotenv import load_dotenv
load_dotenv("/Users/xiaodu/zhiban-standalone/.env")
import websockets

THINK_RE = re.compile(r'<think>(.*?)</think>', re.DOTALL)

async def main():
    api_key = os.getenv("LLM_API_KEY", "")
    async with websockets.connect("ws://localhost:18921/ws") as ws:
        await ws.send(json.dumps({
            "type": "user_query",
            "queryText": "126.pdf论文讨论了几种大环化合物",
            "model": "deepseek-chat",
            "baseUrl": "https://api.deepseek.com",
            "apiKey": api_key,
            "thinking": True,
            "openPapers": [{"paperId": "126", "title": "126.pdf", "filename": "126.pdf"}],
            "context": {"activeDoc": "126.pdf", "conversationId": "test-think-tag"},
        }))

        answer_text = ""
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=180)
                msg = json.loads(raw)
                t = msg.get("type", "")
                if t == "llm_token" and not msg.get("isThinking"):
                    answer_text += msg.get("token", "")
                elif t == "llm_done":
                    break
                elif t == "error":
                    print(f"[ERROR] {msg.get('code')}: {msg.get('message','')[:200]}")
        except asyncio.TimeoutError:
            pass

        # 解析 <think> 标签
        think_blocks = THINK_RE.findall(answer_text)
        content_after_think = THINK_RE.sub('', answer_text).strip()
        has_think = bool(think_blocks)

        print(f"Answer total: {len(answer_text)} chars")
        print(f"<think> blocks: {len(think_blocks)}")
        for i, block in enumerate(think_blocks):
            print(f"  [{i}] {len(block)} chars: {block[:100]}...")
        print(f"Content after </think>: {len(content_after_think)} chars")
        print(f"  {content_after_think[:200]}")
        print(f"Has <think>: {has_think}")
        print(f"Has content: {bool(content_after_think)}")

        assert has_think, "FAIL: 没有 <think> 标签"
        assert content_after_think, "FAIL: 没有正文内容"
        print(f"\n✅ <think> 标签方案正确 — 后端透传，前端解析器分离")

asyncio.run(main())

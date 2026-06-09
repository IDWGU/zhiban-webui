"""直接测试 DeepSeek API provider 的 chat_stream 逐 token 流式行为。

绕过 WebSocket / agent_loop 等上层代码，直接调用 provider。
"""
import asyncio
import json
import os
import sys
import time

# Setup path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sidecar'))

from llm.providers.openai_compatible import OpenAICompatibleProvider


async def test_provider_streaming():
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY") or ""
    base_url = os.getenv("DEEPSEEK_BASE_URL") or os.getenv("LLM_BASE_URL") or "https://api.deepseek.com"
    model = os.getenv("LLM_MODEL") or "deepseek-chat"

    print(f"API: {base_url}")
    print(f"Model: {model}")
    print(f"API Key: {'***' + api_key[-4:] if api_key and len(api_key) > 4 else '(not set)'}")
    print()

    provider = OpenAICompatibleProvider(
        api_key=api_key,
        base_url=base_url,
        model=model,
    )

    messages = [
        {"role": "user", "content": "用一两句话解释Transformer注意力机制"}
    ]

    print("Calling provider.chat_stream()...")
    print("-" * 60)

    chunks = []
    t0 = time.time()
    async for chunk in provider.chat_stream(messages, max_tokens=200):
        chunks.append(chunk)
        chunk_type = chunk.get("type", "?")
        token = chunk.get("token", "")
        elapsed = time.time() - t0

        if chunk_type == "token":
            preview = token[:80].replace("\n", "\\n")
            print(f"  [{elapsed:7.3f}s] TOKEN  len={len(token)} | {preview}{'...' if len(token) > 80 else ''}")
        elif chunk_type == "reasoning_token":
            preview = token[:80].replace("\n", "\\n")
            print(f"  [{elapsed:7.3f}s] REASON len={len(token)} | {preview}{'...' if len(token) > 80 else ''}")
        elif chunk_type == "done":
            print(f"  [{elapsed:7.3f}s] DONE  total_tokens={chunk.get('total_tokens', '?')}")
        else:
            print(f"  [{elapsed:7.3f}s] {chunk_type} | {json.dumps(chunk, ensure_ascii=False)[:120]}")

    print("-" * 60)

    token_chunks = [c for c in chunks if c.get("type") == "token"]
    reasoning_chunks = [c for c in chunks if c.get("type") == "reasoning_token"]

    print(f"\n诊断结果:")
    print(f"  总 chunks: {len(chunks)}")
    print(f"  token chunks: {len(token_chunks)}")
    print(f"  reasoning chunks: {len(reasoning_chunks)}")

    if token_chunks:
        sizes = [len(c["token"]) for c in token_chunks]
        print(f"  token 大小: min={min(sizes)} max={max(sizes)} avg={sum(sizes)/len(sizes):.1f}")
        if len(token_chunks) >= 2:
            intervals = []
            for i in range(1, min(len(token_chunks), 10)):
                # We don't have per-chunk timestamps, but we can check sizes
                pass
            if all(s < 10 for s in sizes):
                print("  ✅ 逐 token 流式（每 chunk < 10 字符）")
            else:
                print(f"  ❌ 大块输出（最大 chunk {max(sizes)} 字符）— 不是真正的流式")
        else:
            print("  ❌ 只有 1 个 token chunk — 整个回答被当成一个 token 返回！")

    if reasoning_chunks:
        print(f"\n  💭 模型支持 thinking（reasoning_content）")

    return chunks


if __name__ == "__main__":
    asyncio.run(test_provider_streaming())

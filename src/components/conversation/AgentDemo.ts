// Agent 模式离线演示 — 模拟多轮工具调用思考过程
import { useAppStore } from '@/stores/appStore'

const delay = (ms: number) => new Promise<void>(r => setTimeout(r, ms))

export async function runAgentDemo() {
  const store = useAppStore.getState()
  if (store.conversation.isStreaming) return

  const msgId = 'demo-' + Date.now()

  store.addMessage({
    id: 'user-' + Date.now(),
    role: 'user',
    content: 'Transformer 的注意力机制计算复杂度是多少？为什么 O(n²) 会成为瓶颈？',
    timestamp: Date.now(),
  })

  store.startStreaming(msgId)

  // Step 0: 思考
  await delay(600)
  store.appendAgentStep({
    stepIndex: 0, phase: 'thinking',
    content: '用户询问 Transformer 注意力机制的计算复杂度。需要从论文中查找 self-attention 的时间复杂度分析。',
  })

  // Step 1: tool_call
  await delay(500)
  store.appendAgentStep({
    stepIndex: 1, phase: 'tool_call',
    toolName: 'search_papers',
    toolArgs: '{"query": "transformer self-attention computational complexity", "top_k": 5}',
  })

  // Step 2: tool_result
  await delay(700)
  store.appendAgentStep({
    stepIndex: 2, phase: 'tool_result',
    toolResult: '找到 3 篇相关论文:\n1. "Attention Is All You Need" (Vaswani et al., 2017)\n2. "Efficient Transformers: A Survey" (Tay et al., 2022)\n3. "FlashAttention" (Dao et al., 2022)',
  })

  // Step 3: 继续思考
  await delay(500)
  store.appendAgentStep({
    stepIndex: 3, phase: 'thinking',
    content: '需要更具体的复杂度分析和优化方案对比。',
  })

  // Step 4: tool_call
  await delay(500)
  store.appendAgentStep({
    stepIndex: 4, phase: 'tool_call',
    toolName: 'search_papers',
    toolArgs: '{"query": "quadratic complexity self-attention optimization", "top_k": 3}',
  })

  // Step 5: tool_result
  await delay(700)
  store.appendAgentStep({
    stepIndex: 5, phase: 'tool_result',
    toolResult: '找到 2 篇:\n1. "Linformer: Self-Attention with Linear Complexity" (Wang et al., 2020)\n2. "Reformer: The Efficient Transformer" (Kitaev et al., 2020)',
  })

  // 思考 token 流
  await delay(400)
  const thinkingText = '标准 Transformer 的 self-attention 复杂度是 O(n²)，瓶颈在于注意力矩阵 QK^T 的计算和存储。多篇论文提出优化方案：Linformer 低秩分解到 O(nk)，Reformer 用 LSH 到 O(n log n)，FlashAttention 通过 IO-aware 分块优化实际速度。'
  for (const ch of thinkingText) {
    store.appendThinkingToken(ch)
    await delay(8)
  }

  // 回答 token 流
  await delay(300)
  const answerText = [
    '## Transformer 注意力机制的计算复杂度\n\n',
    '### 1. 标准 Self-Attention\n\n',
    '标准 scaled dot-product attention：\n\n',
    '$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$\n\n',
    '- 计算 $QK^T$：$O(n^2 \\cdot d)$\n',
    '- softmax：$O(n^2)$\n',
    '- 乘以 V：$O(n^2 \\cdot d)$\n\n',
    '### 2. 为什么 O(n²) 是瓶颈\n\n',
    '处理长文档时：\n',
    '- 内存：存储 $n \\times n$ 矩阵需 $O(n^2)$ 显存\n',
    '- 计算：矩阵乘法随序列长度平方增长\n\n',
    '### 3. 优化方案\n\n',
    '| 方法 | 复杂度 | 核心思路 |\n',
    '|------|--------|----------|\n',
    '| Linformer | O(nk) | 低秩分解 |\n',
    '| Reformer | O(n log n) | LSH 近似 |\n',
    '| FlashAttention | O(n²)* | IO-aware 分块 |\n',
  ]
  for (const t of answerText) {
    for (const ch of t) {
      store.appendStreamingToken(ch)
      await delay(6)
    }
  }

  store.finishStreaming(msgId, {
    mode: 'agent',
    model: 'deepseek-v4',
    usage: { input: 5200, output: 680 },
  })
}

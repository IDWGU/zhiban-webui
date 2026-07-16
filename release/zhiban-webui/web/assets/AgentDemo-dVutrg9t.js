import{u as i}from"./index-BqbfsXhZ.js";const e=t=>new Promise(n=>setTimeout(n,t));async function l(){const t=i.getState();if(t.conversation.isStreaming)return;const n="demo-"+Date.now();t.addMessage({id:"user-"+Date.now(),role:"user",content:"Transformer 的注意力机制计算复杂度是多少？为什么 O(n²) 会成为瓶颈？",timestamp:Date.now()}),t.startStreaming(n),await e(600),t.appendAgentStep({stepIndex:0,phase:"thinking",content:"用户询问 Transformer 注意力机制的计算复杂度。需要从论文中查找 self-attention 的时间复杂度分析。"}),await e(500),t.appendAgentStep({stepIndex:1,phase:"tool_call",toolName:"search_papers",toolArgs:'{"query": "transformer self-attention computational complexity", "top_k": 5}'}),await e(700),t.appendAgentStep({stepIndex:2,phase:"tool_result",toolResult:`找到 3 篇相关论文:
1. "Attention Is All You Need" (Vaswani et al., 2017)
2. "Efficient Transformers: A Survey" (Tay et al., 2022)
3. "FlashAttention" (Dao et al., 2022)`}),await e(500),t.appendAgentStep({stepIndex:3,phase:"thinking",content:"需要更具体的复杂度分析和优化方案对比。"}),await e(500),t.appendAgentStep({stepIndex:4,phase:"tool_call",toolName:"search_papers",toolArgs:'{"query": "quadratic complexity self-attention optimization", "top_k": 3}'}),await e(700),t.appendAgentStep({stepIndex:5,phase:"tool_result",toolResult:`找到 2 篇:
1. "Linformer: Self-Attention with Linear Complexity" (Wang et al., 2020)
2. "Reformer: The Efficient Transformer" (Kitaev et al., 2020)`}),await e(400);const a="标准 Transformer 的 self-attention 复杂度是 O(n²)，瓶颈在于注意力矩阵 QK^T 的计算和存储。多篇论文提出优化方案：Linformer 低秩分解到 O(nk)，Reformer 用 LSH 到 O(n log n)，FlashAttention 通过 IO-aware 分块优化实际速度。";for(const o of a)t.appendThinkingToken(o),await e(8);await e(300);const r=[`## Transformer 注意力机制的计算复杂度

`,`### 1. 标准 Self-Attention

`,`标准 scaled dot-product attention：

`,`$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$

`,`- 计算 $QK^T$：$O(n^2 \\cdot d)$
`,`- softmax：$O(n^2)$
`,`- 乘以 V：$O(n^2 \\cdot d)$

`,`### 2. 为什么 O(n²) 是瓶颈

`,`处理长文档时：
`,`- 内存：存储 $n \\times n$ 矩阵需 $O(n^2)$ 显存
`,`- 计算：矩阵乘法随序列长度平方增长

`,`### 3. 优化方案

`,`| 方法 | 复杂度 | 核心思路 |
`,`|------|--------|----------|
`,`| Linformer | O(nk) | 低秩分解 |
`,`| Reformer | O(n log n) | LSH 近似 |
`,`| FlashAttention | O(n²)* | IO-aware 分块 |
`];for(const o of r)for(const s of o)t.appendStreamingToken(s),await e(6);t.finishStreaming(n,{mode:"agent",model:"deepseek-v4",usage:{input:5200,output:680}})}export{l as runAgentDemo};

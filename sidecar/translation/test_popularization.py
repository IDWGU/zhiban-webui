"""通俗化翻译 A/B 对比测试脚本。

用法:
    cd /Users/xiaodu/zhiban-webui/sidecar
    python3 -m translation.test_popularization /path/to/paper.pdf [句子数]

先加载 ../.env 中的环境变量（API key 等），再提取 PDF 句子，
逐句用「学术翻译」和「通俗化翻译」两个 system prompt 各翻译一次，
输出并排对比。
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# ── 加载 .env ──
_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"
if _ENV_FILE.exists():
    with open(_ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())
    print(f"[env] Loaded {_ENV_FILE}")

# 现在 import config 才能读到正确的 env 值
# 支持两种运行方式：
#   cd sidecar && python3 translation/test_popularization.py ...
#   cd zhiban-webui && python3 -m sidecar.translation.test_popularization ...
try:
    from .extractor import extract_blocks
    from ..llm.deepseek_proxy import llm_proxy
    from .. import config
except ImportError:
    from translation.extractor import extract_blocks
    from llm.deepseek_proxy import llm_proxy
    import config


# ═══════════════════════════════════════════════════════════
# 两个 system prompt
# ═══════════════════════════════════════════════════════════

PROMPT_ACADEMIC = (
    "你是论文学术翻译器。将任意语言的学术句子翻译为简体中文。"
    "必须只输出中文译文，禁止输出英文或原文，禁止加任何解释、注释、前缀。"
    "禁止输出句子序号（如 [9]、[10]），序号仅用于标记原文位置。"
    "专业术语保留英文原文并括号标注中文，如 \"XRD (X射线衍射)\"。"
    "保留文献引用标记如 [69] 和 LaTeX 公式如 $H_2O_2$ 不变。"
    "如果原文已经是中文，直接输出原文。"
)

PROMPT_POPULAR = (
    # 角色
    "你是中文学术翻译器，任务是将英文学术论文翻译为通俗化的中文版本。"
    # 输出约束
    "必须只输出当前句的中文译文，禁止输出英文或原文，禁止加任何解释、注释、前缀。"
    "禁止输出句子序号（如 [9]、[10]），序号仅用于标记原文位置。"
    # 通俗化核心规则
    "拆解嵌套长句为短句，用直白的主动语态替代被动缠绕。"
    "避免「在……的条件下」「通过……的方式」等冗余结构，直接陈述事实。"
    "it is … that … 强调句改为直接陈述，不译「正是……」。"
    "nevertheless/however/nonetheless 等弱转折词直接删除，让逻辑自然顺承。"
    "remains to be elucidated / remains unclear 类迂回表达直译为「尚不清楚」。"
    "taken together / in summary 类收束词保留但简化为「总之」或「综合来看」。"
    # 风格
    "保持学术严肃性：通俗但不口语化、不玩梗、不网络化。"
    "不加原文没有的评价性语言（如「令人惊叹」「这项研究非常精彩」）。"
    "不使用「我们」「本文」之外的叙事视角。"
    # 术语与格式
    "专业术语首次出现时保留英文缩写并括号标注中文全称，"
    "如 \"XRD (X射线衍射)\"；后续出现直接用缩写。"
    "所有数值、单位、统计量一字不改。"
    "保留文献引用标记如 [69] 和 LaTeX 公式如 $H_2O_2$ 不变。"
    "图表引用（Fig. 1a、Table 1 等）保持原文格式不翻译。"
    # 兜底
    "如果原文已经是中文，直接输出原文。"
)


def extract_sentences(pdf_path: str, max_sentences: int = 15):
    """从 PDF 提取句子并采样。"""
    print(f"[提取] {pdf_path}")
    blocks = extract_blocks(pdf_path)
    print(f"[提取] {len(blocks)} blocks")

    all_sents = []
    for block in blocks:
        for s in block.sentences:
            all_sents.append((s.text.strip(), block.type, block.page_num))

    print(f"[提取] {len(all_sents)} sentences total")

    if len(all_sents) <= max_sentences:
        selected = all_sents
    else:
        # 分层采样：每页取若干句，优先长句（能体现通俗化差异）
        from collections import defaultdict
        by_page = defaultdict(list)
        for text, btype, page in all_sents:
            by_page[page].append((text, btype))

        selected = []
        pages = sorted(by_page.keys())
        per_page = max(1, max_sentences // len(pages))
        for page in pages:
            page_sents = by_page[page]
            page_sents.sort(key=lambda x: len(x[0]), reverse=True)
            selected.extend((text, btype, page) for text, btype in page_sents[:per_page])

    selected = selected[:max_sentences]
    print(f"[提取] 选取 {len(selected)} 句用于对比测试\n")
    return selected, all_sents


async def translate_one(
    system_prompt: str,
    full_original: str,
    sent_text: str,
    sent_idx: int,
) -> str:
    """调用 DeepSeek API 翻译单句。"""
    user_msg = (
        f"以下是待翻译论文原文：\n\n{full_original}\n\n"
        f"请翻译第{sent_idx + 1}句：\n{sent_text}"
    )

    tokens = []
    try:
        async for chunk in llm_proxy.chat_stream(
            query=user_msg,
            context="",
            system_prompt=system_prompt,
            thinking=False,
        ):
            if chunk["type"] == "token":
                tokens.append(chunk["token"])
    except Exception as e:
        return f"[错误: {type(e).__name__}: {e}]"

    return "".join(tokens).strip()


async def main():
    if len(sys.argv) < 2:
        print("用法: python3 -m translation.test_popularization <pdf_path> [句子数]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not Path(pdf_path).exists():
        print(f"文件不存在: {pdf_path}")
        sys.exit(1)

    max_sents = int(sys.argv[2]) if len(sys.argv) > 2 else 15

    # ── 1. 提取 ──
    sample_sents, all_sents = extract_sentences(pdf_path, max_sents)
    # 全文上下文用所有句子（提供完整的术语上下文）
    full_all = "\n\n".join(t[0] for t in all_sents)

    # ── 2. API 配置 ──
    api_key = config.DEEPSEEK_API_KEY or config.LLM_API_KEY
    if not api_key:
        print("[错误] 未检测到 API Key，请检查 .env 文件")
        sys.exit(1)

    model = config.DEEPSEEK_MODEL or config.LLM_MODEL or "deepseek-chat"
    print(f"[API] model={model}")
    print(f"[API] base_url={config.LLM_BASE_URL}")
    print()

    # ── 3. 逐句 A/B 对比 ──
    results = []
    total = len(sample_sents)

    for idx, (text, btype, page) in enumerate(sample_sents):
        short = text[:100].replace("\n", " ")
        print(f"──── [{idx + 1}/{total}] P{page} [{btype}] ────")
        print(f"原文: {text[:300]}")

        # 学术翻译
        t0 = time.time()
        r_academic = await translate_one(PROMPT_ACADEMIC, full_all, text, idx)
        t_a = time.time() - t0

        # 通俗化翻译
        t0 = time.time()
        r_popular = await translate_one(PROMPT_POPULAR, full_all, text, idx)
        t_p = time.time() - t0

        print(f"学术 ({t_a:.1f}s): {r_academic[:200]}")
        print(f"通俗 ({t_p:.1f}s): {r_popular[:200]}")
        print()

        results.append({
            "idx": idx + 1, "page": page, "type": btype,
            "original": text,
            "academic": r_academic, "popular": r_popular,
            "t_academic": t_a, "t_popular": t_p,
        })

    # ── 4. 汇总 ──
    print("\n" + "=" * 80)
    print("                        逐 句 对 比 总 览")
    print("=" * 80)

    for r in results:
        print(f"\n── [{r['idx']}/{total}] P{r['page']} [{r['type']}] ──")
        print(f"原文: {r['original'][:250]}")
        print(f"学术: {r['academic'][:250]}")
        print(f"通俗: {r['popular'][:250]}")

    avg_a = sum(r["t_academic"] for r in results) / len(results)
    avg_p = sum(r["t_popular"] for r in results) / len(results)
    print(f"\n平均耗时: 学术={avg_a:.1f}s, 通俗={avg_p:.1f}s")
    print(f"总耗时: 学术={sum(r['t_academic'] for r in results):.1f}s, "
          f"通俗={sum(r['t_popular'] for r in results):.1f}s")


if __name__ == "__main__":
    asyncio.run(main())

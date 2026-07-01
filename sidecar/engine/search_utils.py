"""Vector search, MMR reranking, and JSON extraction utilities."""
import json
import re

from ..rag.vector_store import vector_store

MIN_CHUNK_LENGTH = 20


def vector_search(query: str, top_k: int = 8, paper_ids: list[str] | None = None) -> list[dict]:
    """Vector search, converting vector_store results to unified V10 format.

    paper_ids 不为空时先按范围搜，无好结果（dist >= 0.40）则自动回退全库。
    防止 frontend paperId 与 ChromaDB doc_id 不匹配导致搜错范围。
    """
    raw = None
    if paper_ids:
        raw = vector_store.search_by_doc_ids(query, paper_ids, top_k=max(20, top_k * 3))
    # 范围搜索无结果或无好结果 → 全库回退
    if not raw or (raw and raw[0].get("score", 0) < 0.60):
        fallback = vector_store.search(query, top_k=max(20, top_k * 3))
        # 全库结果更好时采用全库结果
        if fallback and (not raw or fallback[0].get("score", 0) > raw[0].get("score", 0) + 0.05):
            raw = fallback
    if not raw:
        raw = vector_store.search(query, top_k=max(20, top_k * 3))

    results = []
    for r in raw:
        score = r.get("score", 0)
        dist = 1.0 - score if score > 0 else 0.999  # similarity → distance
        results.append({
            "dist": dist,
            "doc": r.get("content", ""),
            "meta": r.get("metadata", {}),
        })
    results.sort(key=lambda x: x["dist"])
    return results


def mmr_rerank(results: list[dict], top_k: int = 8, lambda_mmr: float = 0.5) -> list[dict]:
    """MMR (Maximal Marginal Relevance) reranking."""
    valid = [r for r in results if len(r.get("doc", "").strip()) >= MIN_CHUNK_LENGTH]
    if len(valid) <= top_k:
        return valid

    selected = []
    candidates = list(range(len(valid)))
    pid_count: dict[int, int] = {}

    max_iters = len(candidates) * 2  # 无限循环守护
    while len(selected) < min(top_k, len(valid)):
        best_score = -999.0
        best_idx = -1
        for i in candidates:
            pid = valid[i]["meta"].get("paper_id", 0)
            try:
                pid = int(pid)
            except (ValueError, TypeError):
                pid = hash(str(pid)) % 100000

            sim = 1.0 - valid[i]["dist"]
            penalty = (1 - lambda_mmr) * pid_count.get(pid, 0)
            score = lambda_mmr * sim - penalty

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx >= 0:
            selected.append(best_idx)
            pid = valid[best_idx]["meta"].get("paper_id", 0)
            try:
                pid = int(pid)
            except (ValueError, TypeError):
                pid = hash(str(pid)) % 100000
            pid_count[pid] = pid_count.get(pid, 0) + 1
            candidates.remove(best_idx)
        else:
            break  # 无法选出新的最佳候选，避免死循环

        if len(selected) >= max_iters:
            break

    return [valid[i] for i in selected]


def extract_json(text: str, context: str = "classifier") -> dict | None:
    """Extract JSON from LLM response text.

    Handles common wrapper patterns from small models (Qwen 4B etc):
    markdown code fences, trailing text, unescaped characters.
    """
    cleaned = text.strip()
    # 1. Strip markdown code fences
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
    if m:
        cleaned = m.group(1).strip()
    # 2. Strip common prefixes/suffixes small models add
    cleaned = re.sub(r'^.*?(\{)', r'\1', cleaned, count=1, flags=re.DOTALL)
    cleaned = re.sub(r'(\})[^}]*$', r'\1', cleaned, flags=re.DOTALL)
    # 3. Try full text as JSON
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        pass
    # 4. Regex: nested JSON with up to 2 levels of braces
    for m in re.finditer(r"\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}", cleaned):
        try:
            return json.loads(m.group())
        except (json.JSONDecodeError, TypeError):
            continue
    # 5. Flat JSON (last resort)
    for m in re.finditer(r"\{[^{}]*\}", cleaned):
        try:
            return json.loads(m.group())
        except (json.JSONDecodeError, TypeError):
            continue
    print(f"  [engine] {context}: no valid JSON structure found, raw={text[:200]}")
    return None


def select_with_section_diversity(ranked: list[dict], max_chunks: int) -> list[int]:
    """从 MMR 排序结果中选 chunks，确保章节多样性。

    核心思路：chunks 按 chunk_index 升序排列（引言→实验→结论）。
    如果前 max_chunks 全部来自文档前 40%，强制替换 2 个为后 50% 的 chunks。
    这样保证 AI 能看到实验数据和结论，而不只是引言。
    """
    if len(ranked) <= 3 or max_chunks <= 3:
        return list(range(min(max_chunks, len(ranked))))

    # 获取所有 chunk 的 chunk_index 范围
    indices = [r["meta"].get("chunk_index", -1) for r in ranked]
    valid = [ci for ci in indices if ci >= 0]
    if not valid:
        return list(range(min(max_chunks, len(ranked))))

    max_ci = max(valid)
    if max_ci < 5:  # 文档太小，不需要多样性
        return list(range(min(max_chunks, len(ranked))))

    # 分三段：前 40% (引言) / 中 40% (实验) / 后 20% (结论)
    intro_boundary = int(max_ci * 0.4)
    later_boundary = int(max_ci * 0.5)

    selected = []
    intro_count = 0
    later_indices = []

    for i in range(len(ranked)):
        ci = ranked[i]["meta"].get("chunk_index", -1)
        if len(selected) >= max_chunks:
            # 记录后方 chunks 的索引供替换
            if ci >= later_boundary:
                later_indices.append(i)
            continue

        selected.append(i)
        if ci >= 0 and ci <= intro_boundary:
            intro_count += 1

    # 如果前排全是引言 → 用后方 chunks 替换最后 intro_count 中的一部分
    if intro_count >= max_chunks - 1 and later_indices:
        n_swap = min(2, len(later_indices))
        # 替换 selected 中靠后的 intro chunks
        swap_targets = [j for j in reversed(selected)
                        if ranked[j]["meta"].get("chunk_index", -1) <= intro_boundary]
        for k in range(min(n_swap, len(swap_targets))):
            if k < len(later_indices):
                # Replace the last intro chunk with a later chunk
                old_idx = selected.index(swap_targets[k])
                selected[old_idx] = later_indices[k]

    return sorted(selected)[:max_chunks]


def build_history_from_messages(messages: list[dict], max_rounds: int = 3) -> str:
    """Build recent N-round history summary from conversation messages.

    Filters out system messages (L0 prompt, summaries, boundaries)
    to avoid garbled history like "assistant: 你是知伴...".
    """
    user_asst = [m for m in messages if m.get("role") in ("user", "assistant")]
    recent = user_asst[-(max_rounds * 2):]
    parts = []
    for m in recent:
        role = m.get("role", "user")
        content = str(m.get("content", ""))[:300]
        parts.append(f"{role}: {content}")
    return "\n".join(parts)

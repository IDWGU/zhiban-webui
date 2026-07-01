"""PDF 结构化文本提取：分块分类 + 分句 + 坐标包围盒"""

import re
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

# Pre-download NLTK punkt_tab at import time to avoid blocking first translation
# Socket-level timeout to prevent hanging on network issues
import socket as _nltk_socket
_nltk_default_timeout = _nltk_socket.getdefaulttimeout()
_nltk_socket.setdefaulttimeout(10)
try:
    from nltk import sent_tokenize, download, data
    try:
        sent_tokenize("test.")
    except LookupError:
        try:
            download("punkt_tab", quiet=True)
        except Exception:
            pass  # network failure is non-fatal; sent_tokenize will retry at runtime
    except OSError:
        pass  # nltk data path issues
except ImportError:
    pass
finally:
    _nltk_socket.setdefaulttimeout(_nltk_default_timeout)


@dataclass
class WordFragment:
    """A word with its bounding box (PyMuPDF coords: origin top-left, in points)."""
    x0: float
    y0: float
    x1: float
    y1: float
    text: str


@dataclass
class BBox:
    x: float   # normalized 0-1 relative to page width
    y: float   # normalized 0-1 relative to page height
    w: float
    h: float


@dataclass
class Sentence:
    id: str
    text: str
    rects: list[BBox]  # one per line (multi-line sentences have multiple rects)
    translation: str = ""  # filled during translation phase


@dataclass
class Block:
    id: str
    type: str  # "heading" | "paragraph" | "table" | "formula"
    level: int | None
    sentences: list[Sentence]
    page_num: int
    bbox: BBox | None = None  # block-level bounding box (page-normalized 0-1)


# ---- Public API ----

def extract_blocks(pdf_path: str, on_progress=None) -> list[Block]:
    """Extract structured blocks with sentence-level bounding boxes.

    on_progress(current_page, total_pages) is called for each page processed.
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is not available — translation unavailable in frozen build")
    doc = fitz.open(pdf_path)
    blocks: list[Block] = []
    block_idx = 0
    total_pages = doc.page_count

    for page_num in range(total_pages):
        t0 = time.time()
        if on_progress:
            on_progress(page_num + 1, total_pages)

        page = doc[page_num]
        pw = page.rect.width
        ph = page.rect.height

        text_blocks = _get_text_blocks_with_spans(page)

        for tb in text_blocks:
            text = tb["text"].strip()
            if not text or len(text) < 2:
                continue

            btype = _classify_block(tb, text)
            level = _heading_level(tb, text) if btype == "heading" else None
            # Convert block bbox to normalized 0-1 coordinates
            raw_bbox = tb["bbox"]  # (x0, y0, x1, y1) in page px
            block_bbox = BBox(
                x=raw_bbox[0] / pw, y=raw_bbox[1] / ph,
                w=(raw_bbox[2] - raw_bbox[0]) / pw,
                h=(raw_bbox[3] - raw_bbox[1]) / ph,
            )
            # Use span-based extraction: spans already have text+bbox, no cross-path matching
            sentences = _split_block_sentences_with_spans(
                text, block_idx, tb["spans"], tb["lines"], pw, ph, block_bbox,
            )

            if not sentences:
                continue

            blocks.append(Block(
                id=f"B{block_idx}",
                type=btype,
                level=level,
                sentences=sentences,
                page_num=page_num,
                bbox=block_bbox,
            ))
            block_idx += 1

        elapsed = time.time() - t0
        if elapsed > 3:
            print(f"  [extract] page {page_num+1}/{total_pages}: {len(text_blocks)} blocks, {elapsed:.1f}s")

    doc.close()
    return blocks


# ---- Word extraction ----

def _get_page_words(page: fitz.Page) -> list[WordFragment]:
    """Extract words with bounding boxes from a page."""
    # get_text("words") returns: (x0,y0,x1,y1, word, block_no, line_no, word_no)
    raw = page.get_text("words")
    words = []
    for w in raw:
        words.append(WordFragment(
            x0=w[0], y0=w[1], x1=w[2], y1=w[3],
            text=w[4],
        ))
    return words


# ---- Sentence splitting with bbox ----

def _match_sentence_by_words(
    sent_text: str, page_words: list[WordFragment],
    page_w: float, page_h: float,
) -> tuple[list[BBox], float]:
    """
    Fallback: match sentence against page word stream at word level.
    Finds significant words, clusters them by reading order, keeps the
    largest cluster, then expands to include ALL page words between the
    cluster's first and last word for a contiguous highlight region.
    Returns (rects, match_ratio).
    """
    sent_norm = normalize_dehyphen(sent_text)
    sent_words = [w for w in sent_norm.split() if len(w) >= 3]
    if not sent_words:
        return [], 0.0

    # Match sentence words to page words, tracking page-stream indices
    matched_indices: list[int] = []
    for sw in sent_words:
        for pi, pw in enumerate(page_words):
            pw_norm = normalize_dehyphen(pw.text)
            if pw_norm == sw:
                matched_indices.append(pi)
                break
            if len(sw) >= 4 and len(pw_norm) >= 4:
                if sw in pw_norm or pw_norm in sw:
                    matched_indices.append(pi)
                    break

    if not matched_indices:
        return [], 0.0

    # Cluster by reading-order proximity, keep largest cluster
    matched_indices = _largest_contiguous_cluster_indices(matched_indices, page_words)

    # Expand: include ALL page words between first and last matched word
    lo = min(matched_indices)
    hi = max(matched_indices) + 1
    span_words = page_words[lo:hi]

    rects = _words_to_line_rects(span_words, page_w, page_h)
    return rects, len(matched_indices) / len(sent_words)


def _largest_contiguous_cluster_indices(
    indices: list[int], page_words: list[WordFragment],
) -> list[int]:
    """
    Sort matched word indices by reading order (y, x), split into
    clusters at large Y gaps, return indices of the largest cluster.
    """
    if len(indices) <= 1:
        return indices

    sorted_idx = sorted(indices, key=lambda i: (page_words[i].y0, page_words[i].x0))

    # Compute typical line height from matched words
    heights = [page_words[i].y1 - page_words[i].y0 for i in sorted_idx]
    avg_h = sum(heights) / len(heights) if heights else 10

    clusters: list[list[int]] = []
    cur = [sorted_idx[0]]

    for i in sorted_idx[1:]:
        prev_w = page_words[cur[-1]]
        curr_w = page_words[i]
        y_gap = curr_w.y0 - prev_w.y0
        if y_gap > avg_h * 2.5:
            clusters.append(cur)
            cur = [i]
        else:
            cur.append(i)

    clusters.append(cur)
    return max(clusters, key=len)


def _split_sentences_with_bbox(
    text: str, block_idx: int, words: list[WordFragment],
    page_w: float, page_h: float,
    block_bbox: BBox | None = None,
) -> list[Sentence]:
    """Split text into sentences and map each to bounding rectangles."""
    norm_text = normalize(text)
    if not words:
        return _split_sentences_fallback(text, block_idx, block_bbox)

    # Build word stream with character offsets from the page-level word list
    word_stream: list[tuple[int, int, WordFragment]] = []
    offset = 0
    for w in words:
        wt = w.text.strip()
        if not wt:
            continue
        word_stream.append((offset, offset + len(wt), w))
        offset += len(wt) + 1  # +1 for space

    # Find which words belong to THIS block by matching the block text
    # against the concatenated word stream
    full_word_text = " ".join(w.text for _, _, w in word_stream)
    full_norm = normalize(full_word_text)
    stream_start = full_norm.find(norm_text)

    if stream_start >= 0:
        stream_end = stream_start + len(norm_text)
        matched_words = [
            w for cs, ce, w in word_stream
            if ce > stream_start and cs < stream_end
        ]
    else:
        # Try with de-hyphenation on BOTH sides
        norm_text_dh = normalize_dehyphen(text)
        full_norm_dh = normalize_dehyphen(" ".join(w.text for _, _, w in word_stream))
        stream_start = full_norm_dh.find(norm_text_dh)
        if stream_start >= 0:
            stream_end = stream_start + len(norm_text_dh)
            matched_words = [
                w for cs, ce, w in word_stream
                if ce > stream_start and cs < stream_end
            ]
        else:
            # Last resort: match by first/last words
            text_words = norm_text.split()
            if len(text_words) >= 3:
                prefix = " ".join(text_words[:3])
                suffix = " ".join(text_words[-3:])
                pfx_pos = full_norm.find(prefix)
                sfx_pos = full_norm.rfind(suffix)
                if pfx_pos >= 0 and sfx_pos >= pfx_pos:
                    stream_start = pfx_pos
                    stream_end = sfx_pos + len(suffix)
                    matched_words = [
                        w for cs, ce, w in word_stream
                        if ce > stream_start and cs < stream_end
                    ]
                else:
                    matched_words = [w for _, _, w in word_stream]
            else:
                matched_words = [w for _, _, w in word_stream]

    if not matched_words:
        return _split_sentences_fallback(text, block_idx)

    # Build the concatenated text of matched words for sentence-level matching
    matched_full = " ".join(w.text for w in matched_words)
    matched_norm = normalize(matched_full)
    matched_norm_dh = normalize_dehyphen(matched_full)
    full_norm_dh = normalize_dehyphen(" ".join(w.text for _, _, w in word_stream))

    # Split into sentences
    try:
        from nltk import sent_tokenize, download
        try:
            sents = sent_tokenize(text)
        except LookupError:
            download("punkt_tab", quiet=True)
            sents = sent_tokenize(text)
    except ImportError:
        sents = _regex_split_sentences(text)

    # Map each sentence to bounding rects using matched words
    result: list[Sentence] = []

    for si, sent in enumerate(sents):
        sent = sent.strip()
        if not sent:
            continue
        sent_norm = normalize(sent)
        # Try to find sentence: exact → dehyphenated → full word stream → dehyphenated full
        sent_norm_dh = normalize_dehyphen(sent)
        idx = matched_norm.find(sent_norm)
        use_words = matched_words
        if idx < 0:
            idx = matched_norm_dh.find(sent_norm_dh)
        if idx < 0:
            sent_rects, _ = _match_sentence_by_words(sent, matched_words, page_w, page_h)
            if sent_rects:
                result.append(Sentence(id=f"B{block_idx}-S{si}", text=sent, rects=sent_rects))
                continue
            idx = full_norm.find(sent_norm)
            use_words = [w for _, _, w in word_stream]
        if idx < 0 and use_words is matched_words:
            idx = full_norm.find(sent_norm)
            use_words = [w for _, _, w in word_stream]
        if idx < 0:
            idx = full_norm_dh.find(sent_norm_dh)
            use_words = [w for _, _, w in word_stream]

        if idx >= 0:
            sent_end = idx + len(sent_norm)
            sent_words = _words_in_char_range(use_words, idx, sent_end)
            if sent_words:
                rects = _words_to_line_rects(sent_words, page_w, page_h)
            else:
                rects = []
        else:
            # Level 5: word-level fuzzy matching against full page word stream
            rects, _ = _match_sentence_by_words(sent, words, page_w, page_h)

        # Fallback: use block bbox estimate when word-level rects are empty
        if not rects and block_bbox:
            rects = [BBox(
                x=block_bbox.x, y=block_bbox.y + (si / max(len(sents), 1)) * block_bbox.h,
                w=block_bbox.w, h=block_bbox.h / max(len(sents), 1) * 0.9,
            )]

        result.append(Sentence(id=f"B{block_idx}-S{si}", text=sent, rects=rects))

    return result


def _words_in_char_range(words: list[WordFragment], start: int, end: int) -> list[WordFragment]:
    """Find words whose char range overlaps [start, end) in the concatenated text of words."""
    result = []
    char_cursor = 0
    for w in words:
        w_start = char_cursor
        w_end = char_cursor + len(w.text)
        if w_end > start and w_start < end:
            result.append(w)
        char_cursor += len(w.text) + 1  # +1 for space between words
    return result


def _words_to_line_rects(words: list[WordFragment], pw: float, ph: float) -> list[BBox]:
    """Group words by line (y-coordinate proximity) and produce a BBox per line."""
    if not words:
        return []
    # Sort by y, then x
    sorted_words = sorted(words, key=lambda w: (w.y0, w.x0))
    lines: list[list[WordFragment]] = []
    cur = [sorted_words[0]]
    cur_y = sorted_words[0].y0
    for w in sorted_words[1:]:
        if abs(w.y0 - cur_y) < 4:  # same line
            cur.append(w)
        else:
            lines.append(cur)
            cur = [w]
            cur_y = w.y0
    lines.append(cur)

    rects = []
    for ln in lines:
        xs = [w.x0 for w in ln]
        ys = [w.y0 for w in ln]
        x1s = [w.x1 for w in ln]
        y1s = [w.y1 for w in ln]
        rects.append(BBox(
            x=min(xs) / pw,
            y=min(ys) / ph,
            w=(max(x1s) - min(xs)) / pw,
            h=(max(y1s) - min(ys)) / ph,
        ))
    return rects


def _split_sentences_fallback(text: str, block_idx: int, block_bbox: BBox | None = None) -> list[Sentence]:
    """Fallback: split without coordinates. Uses block_bbox to estimate positions."""
    sents = _regex_split_sentences(text)
    result = []
    total = len([s for s in sents if s.strip()])
    idx = 0
    for s in sents:
        s = s.strip()
        if not s:
            continue
        if block_bbox:
            # Distribute sentences within the block's bounding box
            h = block_bbox.h / total
            rects = [BBox(x=block_bbox.x, y=block_bbox.y + idx * h, w=block_bbox.w, h=h * 0.9)]
        else:
            rects = []
        result.append(Sentence(id=f"B{block_idx}-S{idx}", text=s, rects=rects))
        idx += 1
    return result


# ---- Internal helpers (unchanged) ----

def _get_text_blocks_with_spans(page: fitz.Page) -> list[dict]:
    """Extract text blocks with their spans (each span has text + bbox).

    Returns [{text, avg_size, is_bold, bbox, spans}]
    where spans is [{text, x0, y0, x1, y1}] in page pixel coords.
    """
    blocks = page.get_text("dict")["blocks"]
    result = []
    for b in blocks:
        if b.get("type") != 0:
            continue
        lines = []
        all_spans: list[dict] = []
        for line in b.get("lines", []):
            line_text = "".join(s["text"] for s in line["spans"])
            lines.append(line_text)
            for s in line["spans"]:
                sbbox = s.get("bbox", (0, 0, 0, 0))
                all_spans.append({
                    "text": s.get("text", ""),
                    "x0": sbbox[0], "y0": sbbox[1],
                    "x1": sbbox[2], "y1": sbbox[3],
                })
        full_text = " ".join(lines)
        if not full_text.strip():
            continue
        fonts = [s.get("font", "") for s in all_spans]
        sizes = [s.get("size", 0) for s in all_spans]
        flags = [s.get("flags", 0) for s in all_spans]
        avg_size = sum(sizes) / len(sizes) if sizes else 10
        is_bold = any(f & 2 for f in flags)
        result.append({
            "text": full_text,
            "avg_size": avg_size,
            "is_bold": is_bold,
            "bbox": b["bbox"],
            "spans": all_spans,
            "lines": lines,  # per-line text (for offset reconstruction)
        })
    return result


def _split_block_sentences_with_spans(
    text: str, block_idx: int, spans: list[dict], lines: list[str],
    page_w: float, page_h: float, block_bbox: BBox,
) -> list[Sentence]:
    """Split block text into sentences, mapping each to rectangle(s) using span bboxes.

    Uses the block text DIRECTLY for sentence matching (not span text),
    then maps matched char positions to spans via a precise offset stream.
    The offset stream is built from lines: "".join(spans_in_line) + " " between lines.
    """
    if not text.strip():
        return []

    # Build span stream with offsets matching the block text EXACTLY:
    # block_text = " ".join(line0_spans_concat, line1_spans_concat, ...)
    # So: line0 spans at offset 0..len0, then +1 for space, line1 spans at len0+1.., etc.
    span_stream: list[tuple[int, int, dict]] = []
    offset = 0
    span_idx = 0
    for li, line in enumerate(lines):
        line_spans: list[dict] = []
        # Collect spans belonging to this line
        while span_idx < len(spans):
            sp = spans[span_idx]
            st = sp["text"]
            if not st:
                span_idx += 1
                continue
            line_spans.append(sp)
            span_idx += 1
            # Check if we've reached end of this line
            # (spans within same line have close y0; next line has different y0)
            if span_idx < len(spans):
                next_sp = spans[span_idx]
                if next_sp.get("y0", 0) - sp.get("y0", 0) > 4:  # new line detected
                    break
        # Add spans for this line at current offset
        for sp in line_spans:
            st = sp["text"]
            end = offset + len(st)
            span_stream.append((offset, end, sp))
            offset = end
        # Add space between lines (if not last line)
        if li < len(lines) - 1:
            offset += 1  # " " between lines in block text

    # Match sentences against block text directly
    norm_text = normalize(text)
    try:
        from nltk import sent_tokenize, download
        try:
            sents = sent_tokenize(text)
        except LookupError:
            download("punkt_tab", quiet=True)
            sents = sent_tokenize(text)
    except ImportError:
        sents = _regex_split_sentences(text)

    result: list[Sentence] = []
    for si, sent in enumerate(sents):
        sent = sent.strip()
        if not sent:
            continue
        sent_norm = normalize(sent)

        # Find sentence in block text (exact match — text came from same source)
        idx = norm_text.find(sent_norm)
        if idx < 0:
            # Fuzzy: match first 3+ significant words
            sent_words = [w for w in sent_norm.split() if len(w) >= 3]
            if sent_words:
                pfx = " ".join(sent_words[:min(3, len(sent_words))])
                idx = norm_text.find(pfx)
        if idx < 0:
            rects = [BBox(
                x=block_bbox.x, y=block_bbox.y + (si / max(len(sents), 1)) * block_bbox.h,
                w=block_bbox.w, h=block_bbox.h / max(len(sents), 1) * 0.9,
            )]
            result.append(Sentence(id=f"B{block_idx}-S{si}", text=sent, rects=rects))
            continue

        sent_end = idx + len(sent_norm)
        matched_spans = []
        for cs, ce, sp in span_stream:
            if ce > idx and cs < sent_end:
                matched_spans.append(sp)

        if matched_spans:
            rects = _spans_to_line_rects(matched_spans, page_w, page_h)
        else:
            rects = [BBox(
                x=block_bbox.x, y=block_bbox.y + (si / max(len(sents), 1)) * block_bbox.h,
                w=block_bbox.w, h=block_bbox.h / max(len(sents), 1) * 0.9,
            )]

        result.append(Sentence(id=f"B{block_idx}-S{si}", text=sent, rects=rects))

    return result


def _spans_to_line_rects(spans: list[dict], pw: float, ph: float) -> list[BBox]:
    """Convert a list of span dicts to normalized line-rect BBoxes.

    Groups spans by line (similar y0), creates one rect per line.
    """
    if not spans:
        return []
    # Group by line — spans on same line have similar y0
    lines: list[list[dict]] = []
    for sp in sorted(spans, key=lambda s: (s["y0"], s["x0"])):
        placed = False
        for line in lines:
            last = line[-1]
            if abs(sp["y0"] - last["y0"]) < 2:  # same line (within 2pt)
                line.append(sp)
                placed = True
                break
        if not placed:
            lines.append([sp])
    rects = []
    for line in lines:
        x0 = min(s["x0"] for s in line)
        y0 = min(s["y0"] for s in line)
        x1 = max(s["x1"] for s in line)
        y1 = max(s["y1"] for s in line)
        rects.append(BBox(
            x=x0 / pw, y=y0 / ph,
            w=(x1 - x0) / pw, h=(y1 - y0) / ph,
        ))
    return rects


def _classify_block(tb: dict, text: str) -> str:
    size = tb.get("avg_size", 10)
    bold = tb.get("is_bold", False)
    if len(text) < 200 and _formula_score(text) > 0.4:
        return "formula"
    if _looks_like_table(text):
        return "table"
    lines = text.split("\n")
    avg_line_len = sum(len(l) for l in lines) / len(lines) if lines else 0
    if avg_line_len < 120 and (bold or size > 10.5):
        return "heading"
    return "paragraph"


def _heading_level(tb: dict, text: str) -> int:
    size = tb.get("avg_size", 10)
    m = re.match(r"^(\d+(\.\d+)*|[IVX]+)\.?\s+", text.strip())
    if size >= 16: return 1
    elif size >= 14: return 2 if m else 3
    elif size >= 12: return 3 if m else 4
    else: return 4 if m else 5


def _regex_split_sentences(text: str) -> list[str]:
    cleaned = text.replace("\n", " ")
    for abbr in ["e.g.", "i.e.", "et al.", "Fig.", "Eq.", "vol.", "pp.", "Dr.", "Prof."]:
        cleaned = cleaned.replace(abbr, abbr.replace(".", "§DOT§"))
    raw = re.split(r'(?<=[.!?])\s+(?=[A-Z])', cleaned)
    parts = []
    for p in raw:
        p = p.replace("§DOT§", ".").strip()
        if p:
            parts.append(p)
    return parts if parts else [text]


def _formula_score(text: str) -> float:
    symbols = sum(1 for c in text if c in r"\{}[]∫∑∏√∞∂∇∈∉⊂⊃∪∩∧∨→⇒⇔∀∃∄∠⊥θαβγδελμπσφωΓΔΛΠΣΦΩ")
    if len(text) == 0: return 0.0
    return symbols / len(text)


def _looks_like_table(text: str) -> bool:
    lines = text.strip().split("\n")
    if len(lines) < 2: return False
    col_counts = [len([c for c in re.split(r"\s{2,}|\t", l) if c.strip()]) for l in lines]
    if not col_counts: return False
    median = sorted(col_counts)[len(col_counts) // 2]
    if median < 2: return False
    return sum(1 for c in col_counts if c >= 2) / len(col_counts) > 0.6


def normalize(s: str) -> str:
    # Strip control characters (U+0000-U+001F, U+0080-U+009F)
    s = re.sub(r'[\x00-\x1f\x80-\x9f]', '', s)
    return re.sub(r'\s+', ' ', s).strip().lower()


def normalize_dehyphen(s: str) -> str:
    """Normalize and merge line-break hyphens as a fallback for matching."""
    s = normalize(s)
    return re.sub(r'-\s+', '', s)

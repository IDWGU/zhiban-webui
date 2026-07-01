import { useMemo, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkBreaks from 'remark-breaks'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'
import { useAppStore } from '@/stores/appStore'

// ── 增量正则处理缓存：避免每帧对全量文本重跑 4 次 replace ──
let _lastProcessed = ''
let _lastResult = ''

function processCitationsIncremental(text: string): string {
  if (text.startsWith(_lastProcessed) && text.length > _lastProcessed.length) {
    const delta = text.slice(_lastProcessed.length)
    const processed = preprocessCitations(delta)
    _lastProcessed = text
    _lastResult += processed
    return _lastResult
  }
  _lastProcessed = text
  _lastResult = preprocessCitations(text)
  return _lastResult
}

export function resetCitationCache() {
  _lastProcessed = ''
  _lastResult = ''
}

// ── 增量 safe-cut 扫描状态（模块级，跨 render 保持）──
let _lastScanEnd = 0
let _lastSafeCut = 0

export function resetSafeCutCache() {
  _lastScanEnd = 0
  _lastSafeCut = 0
}

// ── Markdown 重渲染节流 ──
// ReactMarkdown + rehypeKatex 全量解析大段文本可能 >50ms，
// 根据内容长度自适应调整渲染间隔：
//   < 5000 字: 100ms（流畅）
//   5000-10000 字: 200ms
//   > 10000 字: 300ms（长总结场景）
function _getMdRerenderInterval(stableLen: number): number {
  if (stableLen > 10000) return 300
  if (stableLen > 5000) return 200
  return 100
}
let _lastMdRenderTime = 0

export default function StreamingMarkdown() {
  const streamingText = useAppStore(s => s.conversation.streamingText)
  const citations = useAppStore(s => s.conversation.streamingCitations)
  const prevTextLen = useRef(0)

  // 重置检测：文本被清空（新消息开始）
  if (streamingText.length < prevTextLen.current) {
    resetCitationCache()
    resetSafeCutCache()
  }
  prevTextLen.current = streamingText.length

  // ── 增量 safe-cut 扫描：只扫描新增部分 ──
  const { stable, pending } = useMemo(() => {
    // 从上次扫描位置继续，只扫描新增文本
    if (_lastScanEnd < streamingText.length) {
      const result = scanNewText(streamingText, _lastScanEnd, _lastSafeCut)
      _lastSafeCut = result.lastSafe
      _lastScanEnd = streamingText.length
    }
    // 如果文本被截断（不应该发生但做防御），回退
    if (_lastSafeCut > streamingText.length) {
      _lastSafeCut = 0
      _lastScanEnd = 0
      const result = scanNewText(streamingText, 0, 0)
      _lastSafeCut = result.lastSafe
      _lastScanEnd = streamingText.length
    }
    const cut = streamingText.length === 0 ? 0 : (_lastSafeCut > 0 ? _lastSafeCut : 0)
    return {
      stable: streamingText.slice(0, cut),
      pending: streamingText.slice(cut),
    }
  }, [streamingText])

  const streamingComponents = useMemo(() => ({
    a: ({ href, title, children, node, ...props }: any) => {
      if (href && /^\d+$/.test(href)) {
        const paperId = href
        const idx = citations.findIndex(c => c.paperId === parseInt(paperId))
        const num = idx >= 0 ? idx + 1 : paperId
        return (
          <sup
            style={{ color: 'hsl(var(--citation-accent))', cursor: 'pointer' }}
            title={title ? `Paper #${paperId}, ${title}` : `Paper #${paperId}`}
          >
            [{num}]
          </sup>
        )
      }
      return <a href={href} target="_blank" rel="noopener noreferrer" {...props}>{children}</a>
    },
  }), [citations])

  // ── ReactMarkdown 缓存：stable 未变则复用上次结果，避免重复 KaTeX 解析 ──
  const cachedMdRef = useRef<{ stable: string; jsx: React.ReactNode }>({ stable: '', jsx: null })

  const stableContent = useMemo(() => {
    if (!stable) {
      cachedMdRef.current = { stable: '', jsx: null }
      return null
    }
    // 文本没变 → 直接返回缓存
    if (stable === cachedMdRef.current.stable) {
      return cachedMdRef.current.jsx
    }
    // 自适应节流：长内容减少重渲染频率
    const now = performance.now()
    const interval = _getMdRerenderInterval(stable.length)
    if (_lastMdRenderTime > 0 && now - _lastMdRenderTime < interval) {
      return cachedMdRef.current.jsx
    }
    _lastMdRenderTime = now
    const jsx = (
      <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath, remarkBreaks]} rehypePlugins={[rehypeKatex]} components={streamingComponents}>
        {processCitationsIncremental(stable)}
      </ReactMarkdown>
    )
    cachedMdRef.current = { stable, jsx }
    return jsx
  }, [stable, streamingComponents])

  return (
    <div className="streaming-markdown" style={{ lineHeight: 1.7, fontSize: 13, overflowWrap: 'break-word', wordBreak: 'break-word' }}>
      {stableContent || (
        <span style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>思考中...</span>
      )}
      {pending && (
        <span style={{ color: 'var(--text-soft)' }}>{stripPendingCitations(pending)}</span>
      )}
      <span className="cursor-blink" style={{
        display: 'inline-block', width: 1, height: 14, background: 'var(--accent)',
        marginLeft: 2, verticalAlign: 'middle',
      }} />
    </div>
  )
}

// ── 增量扫描：从 lastScanEnd 位置开始，只扫描新增文本 ──
function scanNewText(text: string, scanStart: number, prevSafe: number): { lastSafe: number } {
  let lastSafe = prevSafe
  let inCodeBlock = false
  // 需要从开头确定 inCodeBlock 状态，但仅当 scanStart === 0 时从头扫
  if (scanStart === 0) {
    lastSafe = 0
  }
  // 从上次 safe 点之前最近的换行开始，确保 inCodeBlock 状态正确
  // 策略：向前回退到最近的 \n\n，从那里开始重新确定状态
  let i = scanStart > 0 ? Math.max(0, scanStart - 200) : 0 // 回退 200 字符以内确保 code block 状态正确
  // 从头快速扫描到 scanStart 以确定 inCodeBlock 状态
  for (let j = 0; j < i && j < text.length; j++) {
    if (text[j] === '`' && text[j + 1] === '`' && text[j + 2] === '`') {
      inCodeBlock = !inCodeBlock
      j += 2
    }
  }
  // 扫描新增部分
  for (; i < text.length - 1; i++) {
    if (text[i] === '`' && text[i + 1] === '`' && text[i + 2] === '`') {
      inCodeBlock = !inCodeBlock
      i += 2
      continue
    }
    if (!inCodeBlock && text[i] === '\n' && text[i + 1] === '\n') {
      lastSafe = i
    }
    if (!inCodeBlock && (text[i] === '。' || text[i] === '.' || text[i] === '！' || text[i] === '?') && text[i + 1] === '\n') {
      lastSafe = i + 1
    }
  }
  // pending 区有未闭合的 【来源: 标记 → 回退 cut point
  if (lastSafe > 0) {
    const pendingText = text.slice(lastSafe)
    const openMarker = pendingText.indexOf('【来源:')
    if (openMarker >= 0 && !pendingText.slice(openMarker).includes('】')) {
      const prevCut = text.lastIndexOf('\n\n', lastSafe - 2)
      if (prevCut >= 0) lastSafe = prevCut
    }
  }
  return { lastSafe: lastSafe > 0 ? lastSafe : 0 }
}

function preprocessCitations(text: string): string {
  return text
    .replace(
      /【来源:\s*Paper\s*#(\d+)[，,]\s*([^，,]+)[，,]\s*([^】]+)】/g,
      (_, n, fname, section) => `[Paper #${n} · ${section.trim().slice(0, 30)}](${n} "${fname.trim()}§${section.trim()}")`
    )
    .replace(
      /【来源:\s*Paper\s*#(\d+)[，,]\s*([^】]+?\.[a-zA-Z0-9]{2,5})】/g,
      (_, n, fname) => `[Paper #${n} · ${fname.trim().slice(0, 30)}](${n} "${fname.trim()}")`
    )
    .replace(
      /【来源:\s*Paper\s*#(\d+)[，,]\s*([^】]+)】/g,
      (_, n, section) => `[Paper #${n} · ${section.trim().slice(0, 30)}](${n} "${section.trim()}")`
    )
    .replace(
      /【来源:\s*Paper\s*#(\d+)】/g,
      (_, n) => `[Paper #${n}](${n})`
    )
}

// pending 区原文中的 【来源:...】 会被 stable 区已处理的引用徽章重复显示，直接剥离
function stripPendingCitations(text: string): string {
  return text
    .replace(/【来源:\s*Paper\s*#\d+[，,][^】]*】/g, '')
    .replace(/【来源:\s*Paper\s*#\d+】/g, '')
    .replace(/【来源:[\s\S]*?[】$]?/g, '')  // 截断残留
}

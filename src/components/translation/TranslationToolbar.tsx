import { useState, useEffect, useRef } from 'react'
import { useAppStore } from '@/stores/appStore'

interface Props {
  filePath?: string
  onCancel: () => void
}

export default function TranslationToolbar({ filePath, onCancel }: Props) {
  const isTranslating = useAppStore(s => s.translation.isTranslating)
  const phase = useAppStore(s => s.translation.phase)
  const progress = useAppStore(s => s.translation.progress)
  const tokensPerSec = useAppStore(s => s.translation.tokensPerSec)
  const elapsed = useAppStore(s => s.translation.elapsed)
  const firstTokenMs = useAppStore(s => s.translation.firstTokenMs)
  const statusMsg = useAppStore(s => s.translation.statusMsg)
  const activeSentenceId = useAppStore(s => s.translation.activeSentenceId)
  const lastScope = useAppStore(s => s.translation.lastScope)
  const lastPageNum = useAppStore(s => s.translation.lastPageNum)
  const selectedRects = useAppStore(s => s.translation.selectedRects)
  const lastStyle = useAppStore(s => s.translation.lastStyle)
  const [scope, setScopeState] = useState<'full' | 'page' | 'selection'>(lastScope)
  const [pageNum, setPageNum] = useState(lastPageNum)
  const [localError, setLocalError] = useState<string | null>(null)
  const autoFilledRef = useRef(false)

  // Persist scope + pageNum changes so they survive component remount
  const setScope = (s: typeof scope) => {
    setScopeState(s)
    useAppStore.getState().setTranslationScope(s, pageNum)
  }
  const updatePageNum = (val: string) => {
    setPageNum(val)
    useAppStore.getState().setTranslationScope(scope, val)
  }

  // Debug: trace phase transitions
  if (typeof window !== 'undefined' && (window as any).__debugTransPhase !== phase) {
    (window as any).__debugTransPhase = phase
    console.log(`[TRANS-TOOLBAR] render phase=${phase} isTranslating=${isTranslating} firstTokenMs=${firstTokenMs} blocksLen=${useAppStore.getState().translation.blocks.length}`)
  }
  const showTranslateBtn = phase === 'idle' || phase === 'done' || phase === 'error'
  const showSpinner = isTranslating && !firstTokenMs

  const phaseLabel: Record<string, string> = {
    idle: '',
    extracting: '提取文档结构...',
    translating: `翻译中 (已响应 ${progress.current}/${progress.total} 句)`,
    done: '翻译完成',
    error: '翻译失败',
  }

  // Auto-detect current page when switching to "某页翻译"
  useEffect(() => {
    if (scope !== 'page') { autoFilledRef.current = false; return }
    if (autoFilledRef.current && pageNum) return
    const store = useAppStore.getState()
    const activeId = activeSentenceId || store.translation.activeSentenceId
    if (activeId) {
      for (const b of store.translation.blocks) {
        for (const s of b.sentences) {
          if (s.id === activeId) {
            updatePageNum(String(b.pageNum + 1))
            autoFilledRef.current = true
            return
          }
        }
      }
    }
    // Fallback: if no active sentence, default to page 1
    if (!pageNum) updatePageNum('1')
  }, [scope])

  // Scroll to page on valid input (debounced to avoid flicker on multi-digit typing)
  const pageScrollTimer = useRef<ReturnType<typeof setTimeout>>()
  const handlePageInput = (val: string) => {
    updatePageNum(val)
    autoFilledRef.current = false
    const p = parseInt(val, 10)
    if (isNaN(p) || p < 1) return
    clearTimeout(pageScrollTimer.current)
    pageScrollTimer.current = setTimeout(() => {
      useAppStore.getState().setScrollPageIndex(p - 1)
    }, 350)
  }

  const handleTranslate = (style?: 'academic' | 'popular') => {
    const effectiveStyle = style || lastStyle
    if (!filePath) return
    const store = useAppStore.getState()
    const send = store.sendMessage || (window as any).__zhiban_wsSend
    if (!send) { setLocalError('WebSocket 未连接'); return }
    const baseUrl = store.settings.llmBaseUrl || ''
    const isLocal = baseUrl === '__local__' || baseUrl.includes('localhost') || baseUrl.includes('127.0.0.1')
    if (!isLocal && !store.settings.llmApiKey) { setLocalError('请先配置 API Key'); return }

    let selectionRange: { startPage: number; endPage: number } | undefined
    let page: number | undefined
    let sendRects: Array<{ pageIndex: number; x: number; y: number; w: number; h: number }> | undefined

    if (scope === 'page') {
      const p = parseInt(pageNum, 10)
      if (isNaN(p) || p < 1) { setLocalError('请输入有效的页码'); return }
      page = p - 1  // 1-indexed (user) → 0-indexed (server)
    } else if (scope === 'selection') {
      sendRects = [...store.translation.selectedRects]  // snapshot before startTranslation clears
      if (sendRects.length === 0) {
        setLocalError('请先在左侧 PDF 页面中拖拽框选要翻译的段落（按住 Shift 可追加框选）')
        return
      }
    }

    setLocalError(null)
    store.setTranslationStyle(effectiveStyle)
    store.startTranslation(filePath)
    send({
      type: 'translation_request', filePath, scope, page,
      selectionRange,
      selectionRects: sendRects,
      apiKey: store.settings.llmApiKey || undefined,
      baseUrl: store.settings.llmBaseUrl || undefined,
      model: store.settings.llmModel || undefined,
      thinking: store.settings.thinkingMode,
      useLocal: isLocal || undefined,
      style: effectiveStyle,
    })
  }

  const handleCancel = () => {
    if (isTranslating) {
      const detail = progress.total > 0
        ? `已响应 ${progress.current}/${progress.total} 句`
        : '正在提取文档结构'
      if (!confirm(`翻译进行中（${detail}），确定取消？`)) return
    }
    const store = useAppStore.getState()
    const send = store.sendMessage || (window as any).__zhiban_wsSend
    if (send && isTranslating) send({ type: 'cancel_translation' })
    if (isTranslating) store.cancelTranslation()
    onCancel()
  }

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 8, padding: '6px 16px',
      borderBottom: '1px solid var(--border)', background: 'var(--bg-elevated)', flexShrink: 0,
    }}>
      {showTranslateBtn ? (
        <>
          <select value={scope} onChange={e => { setScope(e.target.value as typeof scope); setLocalError(null) }}
            style={{ fontSize: 11, padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--btn-bg)', color: 'var(--text-primary)', fontFamily: 'inherit' }}>
            <option value="full">全文翻译</option>
            <option value="page">某页翻译</option>
            <option value="selection">选中段落</option>
          </select>
          {scope === 'page' && (
            <input
              type="number" min="1" placeholder="页码"
              value={pageNum}
              onChange={e => handlePageInput(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') handleTranslate() }}
              style={{ fontSize: 11, padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--btn-bg)', color: 'var(--text-primary)', fontFamily: 'inherit', width: 52 }}
            />
          )}
          {scope === 'selection' && selectedRects.length > 0 && (
            <>
              <span style={{ fontSize: 11, color: 'var(--text-accent)' }}>
                已框选 {selectedRects.length} 块
              </span>
              <button onClick={() => useAppStore.getState().clearSelectionRects()} style={{
                fontSize: 10, padding: '2px 8px', borderRadius: 4,
                border: '1px solid var(--border)', background: 'transparent',
                color: 'var(--text-muted)', cursor: 'pointer', fontFamily: 'inherit',
              }}>
                清除
              </button>
            </>
          )}
          <button
            onClick={() => handleTranslate('academic')}
            style={{
              fontSize: 12, padding: '5px 14px', borderRadius: 6,
              border: '1px solid var(--border)',
              background: lastStyle === 'academic' ? 'var(--accent-bg)' : 'var(--btn-bg)',
              color: lastStyle === 'academic' ? 'var(--text-accent)' : 'var(--text-primary)',
              cursor: 'pointer', fontFamily: 'inherit', fontWeight: 600,
            }}
          >学术翻译</button>
          <button
            onClick={() => handleTranslate('popular')}
            style={{
              fontSize: 12, padding: '5px 14px', borderRadius: 6,
              border: '1px solid var(--accent)',
              background: lastStyle === 'popular' ? 'var(--accent-bg)' : 'var(--btn-bg)',
              color: lastStyle === 'popular' ? 'var(--text-accent)' : 'var(--text-primary)',
              cursor: 'pointer', fontFamily: 'inherit', fontWeight: 600,
            }}
          >通俗翻译</button>
        </>
      ) : (
        <>
          <span style={{ fontSize: 11, padding: '2px 8px', borderRadius: 4, background: 'var(--accent-bg)', color: 'var(--text-accent)', fontFamily: 'inherit', fontWeight: 500 }}>
            {lastStyle === 'popular' ? '通俗' : '学术'}
          </span>
          <span style={{ fontSize: 12, color: 'var(--text-accent)' }}>{statusMsg || phaseLabel[phase] || phase}</span>
          {showSpinner && <Spinner />}
          <button onClick={handleCancel} style={{ fontSize: 11, padding: '4px 10px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--btn-bg)', color: 'var(--text-muted)', cursor: 'pointer', fontFamily: 'inherit' }}>取消</button>
          {firstTokenMs > 0 && (<span style={{ fontSize: 10, color: 'var(--text-muted)' }}>首token {firstTokenMs < 1000 ? `${firstTokenMs}ms` : `${(firstTokenMs / 1000).toFixed(1)}s`}</span>)}
          {tokensPerSec > 0 && (<span style={{ fontSize: 10, color: 'var(--text-accent)', fontWeight: 500 }}>{tokensPerSec} tok/s</span>)}
          {elapsed > 0 && (<span style={{ fontSize: 10, color: 'var(--text-muted)' }}>{elapsed.toFixed(0)}s</span>)}
        </>
      )}
      {localError && (<span style={{ fontSize: 11, color: '#e9a845', marginLeft: 8 }}>{localError}</span>)}
    </div>
  )
}

function Spinner() {
  return <span style={{ display: 'inline-block', width: 12, height: 12, border: '2px solid var(--border)', borderTopColor: 'var(--accent)', borderRadius: '50%', animation: 'spin 0.7s linear infinite' }} />
}

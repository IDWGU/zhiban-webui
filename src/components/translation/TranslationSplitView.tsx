import { useState, useRef, useCallback, useEffect } from 'react'
import { useAppStore } from '@/stores/appStore'
import OriginalPane from './OriginalPane'
import TranslationPane from './TranslationPane'

interface Props {
  filePath?: string
  paperTitle?: string
  paperId?: string
}

export default function TranslationSplitView({ filePath, paperTitle, paperId }: Props) {
  const [ratio, setRatio] = useState(0.5)
  const dragging = useRef(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const selectedIds = useAppStore(s => s.translation.selectedSentenceIds)
  const selectedRects = useAppStore(s => s.translation.selectedRects)
  const clearSelection = useAppStore(s => s.clearSentenceSelection)
  const clearRects = useAppStore(s => s.clearSelectionRects)
  const blocks = useAppStore(s => s.translation.blocks)
  const phase = useAppStore(s => s.translation.phase)

  // On mount or file change: load cached translation from path-based key.
  // Same guard as original code — only fires when idle with no blocks.
  useEffect(() => {
    if (filePath && phase === 'idle' && blocks.length === 0) {
      useAppStore.getState().loadCachedTranslation(filePath)
    }
  }, [filePath])

  // Request SHA256 identity for content-based cache migration (non-blocking bonus)
  useEffect(() => {
    if (!filePath) return
    const s = useAppStore.getState().translation
    if (!s.fileSha256 && !s.fileIdentityPending) {
      const send = (window as any).__zhiban_wsSend
      if (send) send({ type: 'compute_file_identity', filePath })
    }
  }, [filePath, phase, blocks.length])

  const handleAskAI = useCallback(() => {
    const context = selectedIds.map(id => {
      for (const b of blocks) {
        for (const s of b.sentences) {
          if (s.id === id) return `原文: ${s.text}\n译文: ${s.translation || '(等待翻译)'}`
        }
      }
      return ''
    }).filter(Boolean).join('\n\n---\n\n')

    useAppStore.getState().updateOcrResult(
      [{ index: 0, text: context.slice(0, 3000), bbox: { x: 0, y: 0, w: 1000, h: 40 }, confidence: 1.0 }],
      0, `翻译选中: ${paperTitle || '论文'}`, 'translation'
    )
    clearSelection()
  }, [selectedIds, blocks, paperTitle, clearSelection])

  const clamp = (v: number) => Math.min(0.7, Math.max(0.3, v))

  useEffect(() => {
    if (!containerRef.current || blocks.length === 0) return

    const observer = new IntersectionObserver(
      (entries) => {
        let bestId: string | null = null
        let bestRatio = 0
        for (const e of entries) {
          if (e.intersectionRatio > bestRatio) {
            bestRatio = e.intersectionRatio
            bestId = e.target.getAttribute('data-sentence-id')
          }
        }
        if (!bestId) return

        for (const b of blocks) {
          for (const s of b.sentences) {
            if (s.id === bestId) {
              useAppStore.getState().updateOcrResult(
                [{ index: 0, text: s.text.slice(0, 500), bbox: { x: 0, y: 0, w: 1000, h: 40 }, confidence: 1.0 }],
                0, `翻译: ${paperTitle || '论文'}`, 'translation'
              )
              return
            }
          }
        }
      },
      { root: containerRef.current, threshold: [0.3, 0.6] }
    )

    const timer = setTimeout(() => {
      const els = containerRef.current?.querySelectorAll('[data-sentence-id]')
      els?.forEach(el => observer.observe(el))
    }, 200)

    return () => { observer.disconnect(); clearTimeout(timer) }
  }, [blocks, paperTitle])

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    dragging.current = true
  }, [])

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!dragging.current || !containerRef.current) return
      const rect = containerRef.current.getBoundingClientRect()
      const x = e.clientX - rect.left
      setRatio(clamp(x / rect.width))
    }
    const onUp = () => { dragging.current = false }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
  }, [])

  return (
    <div ref={containerRef} style={{ height: '100%', display: 'flex', overflow: 'hidden', userSelect: 'none' }}>
      {/* Left: Original */}
      <div style={{ flex: `0 0 calc(${ratio * 100}% - 4px)`, minWidth: 0, overflow: 'hidden' }}>
        <div style={{
          fontSize: 12, fontWeight: 600, padding: '10px 16px 6px',
          color: 'hsl(var(--muted-foreground))',
          borderBottom: '1px solid hsl(var(--border))',
          background: 'hsl(var(--muted) / 0.40)',
        }}>
          原文
        </div>
        <div style={{ height: 'calc(100% - 37px)', overflow: 'auto' }}>
          <OriginalPane filePath={filePath} />
        </div>
      </div>

      {/* Divider — handle width = shell gap (8px) */}
      <div style={{
        flex: '0 0 8px', position: 'relative',
        cursor: 'col-resize', zIndex: 5,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>
        <div
          onMouseDown={onMouseDown}
          style={{
            width: 8, height: 40, borderRadius: 4,
            background: 'hsl(var(--muted-foreground) / 0.35)',
            cursor: 'col-resize',
            transition: 'background 0.15s ease',
          }}
          onMouseEnter={e => (e.currentTarget.style.background = 'hsl(var(--primary) / 0.55)')}
          onMouseLeave={e => {
            if (!dragging.current) e.currentTarget.style.background = 'hsl(var(--muted-foreground) / 0.35)'
          }}
        />
      </div>

      {/* Right: Translation */}
      <div style={{ flex: `0 0 calc(${(1 - ratio) * 100}% - 4px)`, minWidth: 0, overflow: 'hidden' }}>
        <div style={{
          fontSize: 12, fontWeight: 600, padding: '10px 16px 6px',
          color: 'hsl(var(--muted-foreground))',
          borderBottom: '1px solid hsl(var(--border))',
          background: 'hsl(var(--muted) / 0.40)',
        }}>
          译文
        </div>
        <div style={{ height: 'calc(100% - 37px)', overflow: 'auto' }}>
          <TranslationPane />
        </div>
      </div>

      {/* Floating action bar — sentence click or drag-select */}
      {(selectedIds.length > 0 || selectedRects.length > 0) && (
        <div style={{
          position: 'fixed', bottom: 24, left: '50%', transform: 'translateX(-50%)',
          zIndex: 100,
          display: 'flex', alignItems: 'center', gap: 12,
          padding: '10px 20px',
          background: 'hsl(var(--popover))',
          border: '1px solid hsl(var(--border))',
          borderRadius: 12,
          boxShadow: '0 4px 20px rgba(0,0,0,0.20)',
        }}>
          <span style={{ fontSize: 13, color: 'hsl(var(--muted-foreground))' }}>
            {selectedIds.length > 0
              ? <>已选 <strong style={{ color: 'hsl(var(--primary))' }}>{selectedIds.length}</strong> 句</>
              : <>已框选 <strong style={{ color: 'hsl(var(--primary))' }}>{selectedRects.length}</strong> 块</>
            }
          </span>
          {selectedIds.length > 0 && (
            <button onClick={handleAskAI} style={{
              padding: '6px 16px',
              background: 'hsl(var(--primary))',
              color: 'hsl(var(--primary-foreground))',
              border: 'none', borderRadius: 8, cursor: 'pointer', fontSize: 13,
              fontFamily: 'inherit', fontWeight: 500,
            }}>
              询问 AI
            </button>
          )}
          <button onClick={() => { clearSelection(); clearRects() }} style={{
            padding: '6px 12px', background: 'transparent',
            color: 'hsl(var(--muted-foreground))',
            border: '1px solid hsl(var(--border))', borderRadius: 8,
            cursor: 'pointer', fontSize: 13, fontFamily: 'inherit',
          }}>
            取消选择
          </button>
        </div>
      )}
    </div>
  )
}

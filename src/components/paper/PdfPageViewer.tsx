import { useRef, useLayoutEffect, useEffect, useState, useCallback } from 'react'
import { useAppStore } from '@/stores/appStore'
import { getPdfjsLib, getCachedDoc } from './pdfCache'

interface Props {
  filePath: string
  numPages: number
  fullText: string
  textParts: string[]
  paperTitle?: string
  paperId?: string
}

export default function PdfPageViewer({ filePath, numPages, fullText, textParts, paperTitle, paperId }: Props) {
  const selectedIndices: number[] = useAppStore(s => s.selectedParagraphIndices ?? [])
  const containerRef = useRef<HTMLDivElement>(null)
  const blobUrlsRef = useRef<string[]>([])
  const [visiblePage, setVisiblePage] = useState(0)
  const [zoom, setZoom] = useState(100)
  const [renderedPages, setRenderedPages] = useState<string[]>([])
  const [rendering, setRendering] = useState(false)
  const [renderProgress, setRenderProgress] = useState('')
  const zoomRef = useRef(zoom)
  zoomRef.current = zoom

  const [isPanning, setIsPanning] = useState(false)
  const panRef = useRef({ x: 0, y: 0, sx: 0, sy: 0 })

  const pendingAdjustRef = useRef<{ cx: number; cy: number; ratio: number } | null>(null)

  function getScrollEl(): HTMLElement | null {
    return containerRef.current?.parentElement ?? null
  }

  useLayoutEffect(() => {
    const adj = pendingAdjustRef.current
    if (!adj) return
    pendingAdjustRef.current = null
    const scrollEl = getScrollEl()
    if (!scrollEl) return
    scrollEl.scrollLeft = (scrollEl.scrollLeft + adj.cx) * adj.ratio - adj.cx
    scrollEl.scrollTop = (scrollEl.scrollTop + adj.cy) * adj.ratio - adj.cy
  }, [zoom])

  // Wheel zoom handler
  useEffect(() => {
    const el = getScrollEl()
    if (!el) return
    function onWheel(e: WheelEvent) {
      if (!e.ctrlKey && !e.shiftKey) return
      e.preventDefault()
      e.stopPropagation()
      const scrollEl = getScrollEl()
      if (!scrollEl) return
      const rect = scrollEl.getBoundingClientRect()
      const cx = e.clientX - rect.left
      const cy = e.clientY - rect.top
      const oldZoom = zoomRef.current
      let newZoom: number
      if (e.ctrlKey) {
        newZoom = Math.round(oldZoom - e.deltaY * 1.2)
      } else {
        newZoom = oldZoom + (e.deltaY > 0 ? -15 : 15)
      }
      newZoom = Math.max(25, Math.min(300, newZoom))
      if (newZoom === oldZoom) return
      const ratio = newZoom / oldZoom
      pendingAdjustRef.current = { cx, cy, ratio }
      setZoom(newZoom)
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [])

  // Drag-to-pan
  useEffect(() => {
    if (!isPanning) return
    const onMove = (e: MouseEvent) => {
      const dx = e.clientX - panRef.current.x
      const dy = e.clientY - panRef.current.y
      const scrollEl = getScrollEl()
      if (scrollEl) {
        scrollEl.scrollLeft = panRef.current.sx - dx
        scrollEl.scrollTop = panRef.current.sy - dy
      }
    }
    const onUp = () => setIsPanning(false)
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
  }, [isPanning])

  function handleMouseDown(e: React.MouseEvent) {
    if (zoom <= 100) return
    if (e.button !== 0) return
    e.preventDefault()
    const scrollEl = getScrollEl()
    setIsPanning(true)
    panRef.current = { x: e.clientX, y: e.clientY, sx: scrollEl?.scrollLeft ?? 0, sy: scrollEl?.scrollTop ?? 0 }
  }

  function revokeBlobUrls() {
    for (const url of blobUrlsRef.current) URL.revokeObjectURL(url)
    blobUrlsRef.current = []
  }
  useEffect(() => () => revokeBlobUrls(), [])

  const abortRef = useRef<AbortController | null>(null)

  const doRender = useCallback(async () => {
    abortRef.current?.abort()
    const abort = new AbortController()
    abortRef.current = abort
    const signal = abort.signal

    const currentZoom = zoomRef.current
    const dpr = window.devicePixelRatio || 1
    const scale = (currentZoom / 100) * dpr * 1.5
    const BATCH1 = 4

    revokeBlobUrls()
    setRendering(true)
    setRenderProgress(`0/${numPages}`)
    setRenderedPages([])

    const pdfjsLib = await getPdfjsLib()
    if (signal.aborted) return
    const cached = getCachedDoc(filePath)
    if (!cached) return
    const pdfDoc = cached.doc

    const newUrls: string[] = new Array(numPages).fill('')

    async function renderOne(i: number): Promise<string> {
      if (signal.aborted) return ''
      const page = await pdfDoc.getPage(i + 1)
      if (signal.aborted) return ''
      const viewport = page.getViewport({ scale })
      const canvas = document.createElement('canvas')
      canvas.width = viewport.width
      canvas.height = viewport.height
      const ctx = canvas.getContext('2d')!
      await page.render({ canvasContext: ctx, viewport }).promise
      if (signal.aborted) return ''
      return new Promise<string>(resolve => {
        canvas.toBlob(blob => {
          if (!blob) { resolve(''); return }
          if (signal.aborted) { resolve(''); return }
          const url = URL.createObjectURL(blob)
          blobUrlsRef.current.push(url)
          resolve(url)
        }, 'image/jpeg', 0.85)
      })
    }

    // Phase 1: first batch in parallel
    const batch1 = Array.from({ length: Math.min(BATCH1, numPages) }, (_, i) => i)
    const results1 = await Promise.all(batch1.map(i => renderOne(i)))
    if (signal.aborted) return
    for (let j = 0; j < batch1.length; j++) newUrls[batch1[j]] = results1[j]
    setRenderedPages([...newUrls])
    setRenderProgress(`${batch1.length}/${numPages}`)

    // Phase 2: remaining pages sequentially
    for (let i = BATCH1; i < numPages; i++) {
      if (signal.aborted) break
      newUrls[i] = await renderOne(i)
      if (signal.aborted) break
      setRenderedPages([...newUrls])
      setRenderProgress(`${i + 1}/${numPages}`)
      await new Promise(r => requestAnimationFrame(r))
    }

    if (!signal.aborted) {
      setRendering(false)
      setRenderProgress('')
    }
  }, [filePath, numPages])

  useEffect(() => { doRender() }, [doRender])

  const pageOffsetsRef = useRef<number[]>([])

  // Scroll-based visible page detection
  useEffect(() => {
    const scrollEl = containerRef.current?.parentElement
    if (!scrollEl || renderedPages.length === 0) return

    const measure = () => {
      const imgs = scrollEl.querySelectorAll('img[data-page]')
      const offsets: number[] = [0]
      imgs.forEach(img => {
        const h = (img as HTMLElement).offsetHeight || 0
        offsets.push(offsets[offsets.length - 1] + h)
      })
      if (offsets.length > 1) pageOffsetsRef.current = offsets
    }
    measure()

    const findVisiblePage = () => {
      const offsets = pageOffsetsRef.current
      if (offsets.length < 2) return
      const st = scrollEl.scrollTop
      const viewH = scrollEl.clientHeight
      const midY = st + viewH * 0.3
      for (let i = 0; i < offsets.length - 1; i++) {
        if (midY >= offsets[i] && midY < offsets[i + 1]) {
          setVisiblePage(i)
          return
        }
      }
      setVisiblePage(offsets.length - 2)
    }

    const t1 = setTimeout(() => { measure(); findVisiblePage() }, 200)
    const t2 = setTimeout(() => { measure(); findVisiblePage() }, 600)
    scrollEl.addEventListener('scroll', findVisiblePage, { passive: true })
    return () => {
      scrollEl.removeEventListener('scroll', findVisiblePage)
      clearTimeout(t1); clearTimeout(t2)
    }
  }, [renderedPages, zoom])

  // Active paragraph sync
  useEffect(() => {
    if (!textParts[visiblePage]) return
    useAppStore.getState().setActiveParagraph(visiblePage)
  }, [visiblePage, textParts])

  const canZoomIn = zoom < 300
  const canZoomOut = zoom > 25
  const showGrabCursor = zoom > 100

  return (
    <div ref={containerRef} style={{ position: 'relative', padding: '0', userSelect: 'none' }}>
      {/* Zoom controls */}
      <div style={{
        position: 'sticky', top: 0, zIndex: 10,
        display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
        padding: '6px 12px', background: 'var(--bg-elevated)',
        borderBottom: '1px solid var(--border)', backdropFilter: 'blur(8px)',
      }}>
        <button onClick={() => setZoom(z => Math.max(25, z - 25))} disabled={!canZoomOut}
          style={zoomBtnStyle(canZoomOut)}>−</button>
        <span style={{ fontSize: 12, color: 'var(--text-secondary)', minWidth: 48, textAlign: 'center', userSelect: 'none' }}>
          {zoom}%
        </span>
        <button onClick={() => setZoom(z => Math.min(300, z + 25))} disabled={!canZoomIn}
          style={zoomBtnStyle(canZoomIn)}>+</button>
        <button onClick={() => setZoom(100)} title="Fit width"
          style={{
            ...zoomBtnStyle(true), padding: 0,
            background: zoom === 100 ? 'var(--accent-bg)' : 'var(--btn-bg)',
            color: zoom === 100 ? 'var(--text-accent)' : 'var(--text-muted)',
            borderColor: zoom === 100 ? 'var(--accent)' : 'var(--border)',
          }}>
          <svg viewBox="0 0 16 14" width="15" height="13" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round">
            <line x1="1.5" y1="2" x2="1.5" y2="12" />
            <polyline points="5,5 2,7 5,9" />
            <polyline points="11,5 14,7 11,9" />
            <line x1="14.5" y1="2" x2="14.5" y2="12" />
          </svg>
        </button>
        {rendering && (
          <span style={{ fontSize: 10, color: 'var(--text-muted)', marginLeft: 8 }}>
            {renderProgress || 'Rendering...'}
          </span>
        )}
      </div>

      <div
        style={{ padding: '8px 0', cursor: isPanning ? 'grabbing' : showGrabCursor ? 'grab' : 'default' }}
        onMouseDown={handleMouseDown}
      >
        {renderedPages.map((blobUrl, i) => {
          const isSelected = selectedIndices.includes(i)
          return (
          <div key={i} style={{
            margin: '0 auto 8px', textAlign: 'center', position: 'relative',
            outline: isSelected ? '2px solid var(--accent)' : 'none',
            outlineOffset: 2, borderRadius: 4,
          }}>
            {/* Quote button — top right of each page */}
            <button
              onClick={() => {
                if (isSelected) {
                  useAppStore.getState().toggleParagraphSelection(i)
                } else {
                  useAppStore.getState().addQuote({
                    paperId: paperId || filePath,
                    paperName: paperTitle || `Paper Page`,
                    pageNumber: i + 1,
                    text: textParts[i]?.slice(0, 500) || `第${i + 1}页内容`,
                  })
                }
              }}
              title={'引用本页作为提问上下文'}
              style={{
                position: 'absolute', top: 0, right: 4, zIndex: 5,
                width: 28, height: 28, borderRadius: 6,
                border: '1.5px solid var(--border)',
                background: 'var(--bg-elevated)',
                color: 'var(--text-muted)',
                cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 14, opacity: 0.4,
                transition: 'opacity 0.15s, background 0.15s',
              }}
              onMouseEnter={e => { (e.target as HTMLElement).style.opacity = '0.85' }}
              onMouseLeave={e => { (e.target as HTMLElement).style.opacity = '0.4' }}
            >"</button>
            {blobUrl ? (
              <img src={blobUrl} data-page={i} alt={`Page ${i + 1}`} draggable={false}
                style={{
                  width: `${zoom}%`, maxWidth: zoom < 100 ? '100%' : 'none',
                  height: 'auto', boxShadow: '0 2px 8px rgba(0,0,0,0.3)', pointerEvents: 'none',
                }}
              />
            ) : (
              <div style={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: 12 }}>
                Rendering...
              </div>
            )}
            <div style={{ fontSize: 10, color: 'var(--text-muted)', padding: 4 }}>
              Page {i + 1}/{numPages} {i === visiblePage ? '👁' : ''}
            </div>
          </div>
        )})}
      </div>
    </div>
  )
}

const zoomBtnStyle = (enabled: boolean): React.CSSProperties => ({
  width: 28, height: 28, borderRadius: 6, border: '1px solid var(--border)',
  background: enabled ? 'var(--btn-bg)' : 'var(--bg-panel)',
  color: enabled ? 'var(--text-primary)' : 'var(--text-muted)',
  fontSize: 16, cursor: enabled ? 'pointer' : 'default',
  display: 'flex', alignItems: 'center', justifyContent: 'center',
  fontFamily: 'inherit', lineHeight: 1, padding: 0,
})

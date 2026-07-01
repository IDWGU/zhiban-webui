import { useRef, useEffect, useLayoutEffect, useState, useCallback, useMemo } from 'react'
import { useAppStore } from '@/stores/appStore'
import { getCachedPdfDoc } from '@/components/paper/pdfCache'

interface Props { filePath?: string }

export default function OriginalPane({ filePath }: Props) {
  const blocks = useAppStore(s => s.translation.blocks)
  const activeId = useAppStore(s => s.translation.activeSentenceId)
  const scrollTargetId = useAppStore(s => s.translation.scrollTargetId)
  const scrollPageIndex = useAppStore(s => s.translation.scrollPageIndex)
  const selectedIds = useAppStore(s => s.translation.selectedSentenceIds)
  const selectedRects = useAppStore(s => s.translation.selectedRects)
  const addSelectionRect = useAppStore(s => s.addSelectionRect)
  const clearSelectionRects = useAppStore(s => s.clearSelectionRects)
  const containerRef = useRef<HTMLDivElement>(null)
  const pagesRef = useRef<HTMLDivElement>(null)
  const blobUrlsRef = useRef<string[]>([])

  const [numPages, setNumPages] = useState(0)
  const [zoom, setZoom] = useState(100)
  const zoomRef = useRef(zoom); zoomRef.current = zoom
  const [renderedPages, setRenderedPages] = useState<string[]>([])
  const [rendering, setRendering] = useState(false)
  const [renderProgress, setRenderProgress] = useState('')
  const [ready, setReady] = useState(false)

  // Pan
  const [isPanning, setIsPanning] = useState(false)
  const panRef = useRef({ x: 0, y: 0, sx: 0, sy: 0 })
  const pendingAdjustRef = useRef<{ cx: number; cy: number; ratio: number } | null>(null)

  // Drag-to-select
  const [isSelecting, setIsSelecting] = useState(false)
  const [selRect, setSelRect] = useState<{ left: number; top: number; width: number; height: number } | null>(null)
  const selRef = useRef({ startX: 0, startY: 0 })
  const selLatestRef = useRef<{ left: number; top: number; width: number; height: number } | null>(null)

  function getScrollEl(): HTMLElement | null {
    return containerRef.current?.parentElement ?? null
  }

  // ---- Get numPages from shared PDF cache ----
  useEffect(() => {
    if (!filePath) return
    const cached = getCachedPdfDoc(filePath)
    if (cached) {
      setNumPages(cached.numPages)
    } else {
      // Retry: PaperViewer may still be loading
      const timer = setInterval(() => {
        const c = getCachedPdfDoc(filePath!)
        if (c) { setNumPages(c.numPages); clearInterval(timer) }
      }, 200)
      return () => clearInterval(timer)
    }
  }, [filePath])

  // ---- Render pages (runs when numPages is first set) ----
  useEffect(() => {
    if (numPages === 0 || !filePath || ready) return
    let cancelled = false
    ;(async () => {
      const currentZoom = zoomRef.current
      const dpr = window.devicePixelRatio || 1
      const scale = (currentZoom / 100) * dpr * 1.5

      for (const u of blobUrlsRef.current) URL.revokeObjectURL(u)
      blobUrlsRef.current = []
      setRendering(true)
      setRenderProgress(`0/${numPages}`)

      try {
        const cached = getCachedPdfDoc(filePath!)
        if (!cached || cancelled) return
        const doc = cached.doc

        const newUrls: string[] = new Array(numPages).fill('')

        async function renderOne(i: number): Promise<string> {
          const page = await doc.getPage(i + 1)
          const vp = page.getViewport({ scale })
          const c = document.createElement('canvas')
          c.width = vp.width; c.height = vp.height
          const ctx = c.getContext('2d')!
          await page.render({ canvasContext: ctx, viewport: vp }).promise
          return new Promise<string>(res => {
            c.toBlob(b => {
              if (!b) { res(''); return }
              const url = URL.createObjectURL(b)
              blobUrlsRef.current.push(url)
              res(url)
            }, 'image/jpeg', 0.85)
          })
        }

        // First 4 in parallel
        const B1 = Math.min(4, numPages)
        const batch1 = await Promise.all(Array.from({ length: B1 }, (_, i) => renderOne(i)))
        if (cancelled) return
        for (let j = 0; j < B1; j++) newUrls[j] = batch1[j]
        setRenderedPages([...newUrls])
        setRenderProgress(`${B1}/${numPages}`)

        // Remainder sequentially
        for (let i = B1; i < numPages; i++) {
          if (cancelled) return
          newUrls[i] = await renderOne(i)
          setRenderedPages([...newUrls])
          setRenderProgress(`${i + 1}/${numPages}`)
          await new Promise(r => requestAnimationFrame(r))
        }

        if (!cancelled) {
          setRendering(false)
          setRenderProgress('')
          setReady(true)
        }
      } catch (err) { console.error('Render error:', err); if (!cancelled) setRendering(false) }
    })()
    return () => { cancelled = true }
  }, [numPages, filePath])

  // ---- Zoom-to-point scroll correction ----
  useLayoutEffect(() => {
    const adj = pendingAdjustRef.current
    if (!adj) return
    pendingAdjustRef.current = null
    const el = getScrollEl()
    if (!el) return
    el.scrollLeft = (el.scrollLeft + adj.cx) * adj.ratio - adj.cx
    el.scrollTop = (el.scrollTop + adj.cy) * adj.ratio - adj.cy
  }, [zoom])

  // ---- Wheel handler (re-attaches when ready) ----
  useEffect(() => {
    if (!ready) return
    const el = getScrollEl()
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      if (!e.ctrlKey && !e.shiftKey) return
      e.preventDefault(); e.stopPropagation()
      const sel = getScrollEl()
      if (!sel) return
      const rect = sel.getBoundingClientRect()
      const cx = e.clientX - rect.left
      const cy = e.clientY - rect.top
      const old = zoomRef.current
      let nz = e.ctrlKey ? Math.round(old - e.deltaY * 1.2) : old + (e.deltaY > 0 ? -15 : 15)
      nz = Math.max(25, Math.min(300, nz))
      if (nz === old) return
      pendingAdjustRef.current = { cx, cy, ratio: nz / old }
      setZoom(nz)
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [ready])

  // ---- Drag-to-pan ----
  useEffect(() => {
    if (!isPanning) return
    const onMove = (e: MouseEvent) => {
      const el = getScrollEl()
      if (el) {
        el.scrollLeft = panRef.current.sx - (e.clientX - panRef.current.x)
        el.scrollTop = panRef.current.sy - (e.clientY - panRef.current.y)
      }
    }
    const onUp = () => setIsPanning(false)
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    return () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp) }
  }, [isPanning])

  // ---- Drag-to-select ----
  useEffect(() => {
    if (!isSelecting) return
    const pagesEl = pagesRef.current
    if (!pagesEl) return
    const onMove = (e: MouseEvent) => {
      const rect = pagesEl.getBoundingClientRect()
      const curX = e.clientX - rect.left
      const curY = e.clientY - rect.top
      const { startX, startY } = selRef.current
      const left = Math.min(startX, curX)
      const top = Math.min(startY, curY)
      const width = Math.abs(curX - startX)
      const height = Math.abs(curY - startY)
      selLatestRef.current = { left, top, width, height }
      setSelRect({ left, top, width, height })
    }
    const onUp = (e: MouseEvent) => {
      setIsSelecting(false)
      const sr = selLatestRef.current
      setSelRect(null)
      selLatestRef.current = null
      if (!sr || (sr.width <= 5 && sr.height <= 5)) return
      const { left, top, width, height } = sr
      // Convert pagesRef-relative selection to viewport coords, then to normalized page rects
      const pagesRect = pagesEl.getBoundingClientRect()
      const selScreen = { left: pagesRect.left + left, top: pagesRect.top + top, right: pagesRect.left + left + width, bottom: pagesRect.top + top + height }
      const newRects: Array<{ pageIndex: number; x: number; y: number; w: number; h: number }> = []
      const pageEls = containerRef.current?.querySelectorAll('[data-page-idx]')
      if (pageEls) {
        for (let i = 0; i < pageEls.length; i++) {
          const pageEl = pageEls[i] as HTMLElement
          const imgEl = pageEl.querySelector('img')
          if (!imgEl) continue
          const imgRect = imgEl.getBoundingClientRect()
          if (selScreen.right <= imgRect.left || selScreen.left >= imgRect.right ||
              selScreen.bottom <= imgRect.top || selScreen.top >= imgRect.bottom) continue
          const pageIdx = parseInt(pageEl.getAttribute('data-page-idx') || '0', 10)
          const nx = Math.max(0, (selScreen.left - imgRect.left) / imgRect.width)
          const ny = Math.max(0, (selScreen.top - imgRect.top) / imgRect.height)
          const nw = Math.min(1, (selScreen.right - imgRect.left) / imgRect.width) - nx
          const nh = Math.min(1, (selScreen.bottom - imgRect.top) / imgRect.height) - ny
          if (nw > 0.003 && nh > 0.003) {
            newRects.push({ pageIndex: pageIdx, x: nx, y: ny, w: nw, h: nh })
          }
        }
      }
      if (newRects.length > 0) {
        const store = useAppStore.getState()
        for (const r of newRects) store.addSelectionRect(r)
      }
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    return () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp) }
  }, [isSelecting])

  function handleMouseDown(e: React.MouseEvent) {
    if (e.button !== 0) return
    if (zoom > 100) {
      e.preventDefault()
      const el = getScrollEl()
      setIsPanning(true)
      panRef.current = { x: e.clientX, y: e.clientY, sx: el?.scrollLeft ?? 0, sy: el?.scrollTop ?? 0 }
    } else {
      const pagesEl = pagesRef.current
      if (!pagesEl) return
      const rect = pagesEl.getBoundingClientRect()
      e.preventDefault()
      // Clear existing selection unless shift is held (multi-select)
      if (!e.shiftKey) clearSelectionRects()
      selRef.current = { startX: e.clientX - rect.left, startY: e.clientY - rect.top }
      setIsSelecting(true)
      setSelRect(null)
    }
  }

  // Cleanup blobs
  useEffect(() => () => { for (const u of blobUrlsRef.current) URL.revokeObjectURL(u) }, [])

  // Build sentence rects lookup from blocks (keyed by sentenceId)
  // Strategy: use word-level rects when available; for gaps, distribute
  // missing sentences evenly between known neighbors within each block.
  // For entirely-missing blocks, use block bbox as the region.
  const sentenceRects = useMemo(() => {
    const map = new Map<string, Array<{ x: number; y: number; w: number; h: number }>>()
    for (const b of blocks) {
      const sents = b.sentences
      if (sents.length === 0) continue
      const n = sents.length
      const bx = b.bbox?.x ?? 0.05; const by = b.bbox?.y ?? 0.05
      const bw = b.bbox?.w ?? 0.90; const bh = b.bbox?.h ?? 0.90
      const hasRect = sents.map(s => !!(s.rects && s.rects.length > 0))
      const runs: Array<{ start: number; end: number; known: boolean }> = []
      for (let i = 0; i < n; ) {
        const known = hasRect[i]
        let end = i + 1
        while (end < n && hasRect[end] === known) end++
        runs.push({ start: i, end, known })
        i = end
      }
      for (const run of runs) {
        if (run.known) {
          for (let i = run.start; i < run.end; i++) map.set(sents[i].id, sents[i].rects || [])
        }
      }
      for (let ri = 0; ri < runs.length; ri++) {
        const run = runs[ri]
        if (run.known) continue
        const count = run.end - run.start
        let prevR: { y: number; h: number; x: number; w: number } | null = null
        if (ri > 0 && runs[ri - 1].known && runs[ri - 1].end > 0) {
          const pr = sents[runs[ri - 1].end - 1].rects
          if (pr && pr.length > 0) { const lr = pr[pr.length - 1]; prevR = { y: lr.y + lr.h, h: lr.h, x: lr.x, w: lr.w } }
        }
        let nextR: { y: number; x: number; w: number } | null = null
        if (ri < runs.length - 1 && runs[ri + 1].known && runs[ri + 1].start < n) {
          const nr = sents[runs[ri + 1].start].rects
          if (nr && nr.length > 0) nextR = { y: nr[0].y, x: nr[0].x, w: nr[0].w }
        }
        if (prevR && nextR) {
          const gapTop = prevR.y; const gapBot = nextR.y; const gapH = gapBot - gapTop
          const eachH = Math.max(0.005, gapH / count * 0.85)
          const x = (prevR.x + nextR.x) / 2; const w = (prevR.w + nextR.w) / 2
          for (let j = 0; j < count; j++) {
            const y = gapTop + (gapH / count) * j + (gapH / count - eachH) / 2
            map.set(sents[run.start + j].id, [{ x, y, w, h: eachH }])
          }
        } else if (prevR) {
          const eachH = Math.max(0.005, (by + bh - prevR.y) / count * 0.85)
          for (let j = 0; j < count; j++) {
            const y = prevR.y + (j + 0.5) * (bh / n)
            map.set(sents[run.start + j].id, [{ x: prevR.x, y, w: prevR.w, h: eachH }])
          }
        } else if (nextR) {
          const eachH = Math.max(0.005, (nextR.y - by) / count * 0.85)
          for (let j = 0; j < count; j++) {
            const y = Math.max(by, nextR.y - (count - j) * (nextR.y - by) / count)
            map.set(sents[run.start + j].id, [{ x: nextR.x, y, w: nextR.w, h: eachH }])
          }
        } else {
          const eachH = bh / n * 0.85
          for (let j = 0; j < count; j++) {
            const y = by + (run.start + j + 0.5) / n * bh - eachH / 2
            map.set(sents[run.start + j].id, [{ x: bx, y, w: bw, h: eachH }])
          }
        }
      }
    }
    return map
  }, [blocks])

  // Group sentence IDs by page
  const pageSentenceIds = useMemo(() => {
    const groups: string[][] = []
    for (const b of blocks) {
      const pn = b.pageNum
      if (!groups[pn]) groups[pn] = []
      for (const s of b.sentences) groups[pn].push(s.id)
    }
    return groups
  }, [blocks])

  // Scroll to clicked sentence at 1/5 of viewport
  useEffect(() => {
    if (!scrollTargetId || !ready || !containerRef.current) return
    const scrollEl = getScrollEl()
    if (!scrollEl) return

    // Use the same sentenceRects map (via memo result) to find target position
    let pageIdx = -1
    let targetY = -1
    for (const b of blocks) {
      for (const s of b.sentences) {
        if (s.id === scrollTargetId) {
          pageIdx = b.pageNum
          const rects = sentenceRects.get(s.id)
          if (rects && rects.length > 0) targetY = rects[0].y
          break
        }
      }
      if (pageIdx >= 0) break
    }
    if (pageIdx < 0 || targetY < 0) return

    const pageEls = containerRef.current.querySelectorAll('[data-page-idx]')
    const pageEl = pageEls[pageIdx] as HTMLElement
    if (!pageEl) return

    const img = pageEl.querySelector('img')
    if (!img) return

    const pageTop = pageEl.offsetTop
    const imgHeight = img.clientHeight
    const highlightYInPage = pageTop + targetY * imgHeight
    const targetScroll = highlightYInPage - scrollEl.clientHeight * 0.2

    scrollEl.scrollTo({ top: Math.max(0, targetScroll), behavior: 'smooth' })
  }, [scrollTargetId, ready, blocks, sentenceRects])

  // Scroll to page by index — direct, no blocks dependency
  useEffect(() => {
    if (scrollPageIndex == null || !ready || !containerRef.current) return
    const scrollEl = getScrollEl()
    if (!scrollEl) return
    const pageEls = containerRef.current.querySelectorAll('[data-page-idx]')
    const pageEl = pageEls[scrollPageIndex] as HTMLElement
    if (!pageEl) return
    const targetScroll = pageEl.offsetTop - 4
    scrollEl.scrollTo({ top: Math.max(0, targetScroll), behavior: 'smooth' })
  }, [scrollPageIndex, ready])

  const showGrabCursor = zoom > 100
  const canZoomIn = zoom < 300
  const canZoomOut = zoom > 25

  return (
    <div ref={containerRef} style={{ position: 'relative', padding: 0, userSelect: 'none' }}>
      {/* Zoom bar */}
      <div style={{
        position: 'sticky', top: 0, zIndex: 10,
        display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
        padding: '4px 8px', background: 'var(--bg-elevated)',
        borderBottom: '1px solid var(--border)', backdropFilter: 'blur(8px)',
      }}>
        <Btn onClick={() => setZoom(z => Math.max(25, z - 25))} disabled={!canZoomOut} label="−" />
        <span style={{ fontSize: 11, color: 'var(--text-secondary)', minWidth: 40, textAlign: 'center', userSelect: 'none' }}>{zoom}%</span>
        <Btn onClick={() => setZoom(z => Math.min(300, z + 25))} disabled={!canZoomIn} label="+" />
        <Btn onClick={() => setZoom(100)} active={zoom === 100} label="⊡" />
        {rendering && <span style={{ fontSize: 10, color: 'var(--text-muted)', marginLeft: 4 }}>{renderProgress || '渲染中...'}</span>}
      </div>

      {/* Pages */}
      <div ref={pagesRef} style={{ padding: '4px 0', cursor: isPanning ? 'grabbing' : showGrabCursor ? 'grab' : 'crosshair', position: 'relative' }} onMouseDown={handleMouseDown}>
        {/* Selection rectangle */}
        {selRect && selRect.width > 3 && selRect.height > 3 && (
          <div style={{
            position: 'absolute', left: selRect.left, top: selRect.top,
            width: selRect.width, height: selRect.height,
            border: '2px solid rgba(66, 133, 244, 0.8)',
            background: 'rgba(66, 133, 244, 0.12)',
            zIndex: 20, pointerEvents: 'none', borderRadius: 2,
          }} />
        )}
        {!ready ? (
          <div style={{ padding: 40, textAlign: 'center' }}>
            <span style={{ fontSize: 13, color: 'var(--text-muted)' }}>
              {numPages === 0 ? '加载文档中...' : rendering ? '渲染页面中...' : '准备中...'}
            </span>
          </div>
        ) : (
          renderedPages.map((url, i) =>
            url ? (
              <div key={i} data-page-idx={i} style={{ margin: '0 auto 6px', textAlign: 'center' }}>
                <div style={{ position: 'relative', width: `${zoom}%`, margin: '0 auto', lineHeight: 0 }}>
                  <img
                    src={url}
                    alt={`P${i + 1}`}
                    draggable={false}
                    style={{ width: '100%', height: 'auto', boxShadow: '0 2px 8px rgba(0,0,0,0.3)', pointerEvents: 'none' }}
                  />
                  {/* SVG sentence highlight overlay */}
                  <svg style={{
                    position: 'absolute', top: 0, left: 0,
                    width: '100%', height: '100%',
                    pointerEvents: 'none',
                  }} viewBox="0 0 1 1" preserveAspectRatio="none">
                    {(pageSentenceIds[i] || []).map(sid => {
                      const rects = sentenceRects.get(sid)
                      if (!rects || rects.length === 0) return null
                      const isActive = sid === activeId
                      const isSelected = selectedIds.includes(sid)
                      return (
                        <g
                          key={sid}
                          onMouseEnter={() => useAppStore.getState().setActiveSentenceId(sid)}
                          onMouseLeave={() => useAppStore.getState().setActiveSentenceId(null)}
                          style={{ cursor: 'pointer', pointerEvents: 'auto' }}
                        >
                          {rects.map((r, ri) => (
                            <rect
                              key={`${sid}-${ri}`}
                              data-rect-for={sid}
                              x={r.x} y={r.y} width={r.w} height={r.h}
                              fill="rgba(0,0,0,0)"
                              stroke="rgba(0,0,0,0)"
                              strokeWidth={0.002}
                              style={{ pointerEvents: 'all' }}
                            />
                          ))}
                        </g>
                      )
                    })}
                  </svg>
                  {/* Highlight overlays: CSS divs for reliable visual rendering */}
                  {(pageSentenceIds[i] || []).map(sid => {
                    const rects = sentenceRects.get(sid)
                    if (!rects || rects.length === 0) return null
                    const isActive = sid === activeId
                    const isSelected = selectedIds.includes(sid)
                    if (!isActive && !isSelected) return null
                    return rects.map((r, ri) => (
                      <div
                        key={`hl-${sid}-${ri}`}
                        style={{
                          position: 'absolute',
                          left: `${r.x * 100}%`,
                          top: `${r.y * 100}%`,
                          width: `${r.w * 100}%`,
                          height: `${r.h * 100}%`,
                          background: isActive ? 'rgba(255, 235, 59, 0.35)' : 'rgba(66, 133, 244, 0.15)',
                          border: isActive ? '2px solid rgba(255, 193, 7, 0.6)' : '2px solid rgba(66, 133, 244, 0.5)',
                          borderRadius: 2,
                          zIndex: 12,
                          pointerEvents: 'none',
                          boxSizing: 'border-box',
                        }}
                      />
                    ))
                  })}
                  {/* Persistent selection rectangles (normalized coords) */}
                  {selectedRects.filter(r => r.pageIndex === i).map((r, ri) => (
                    <div
                      key={`sel-${i}-${ri}`}
                      style={{
                        position: 'absolute',
                        left: `${r.x * 100}%`, top: `${r.y * 100}%`,
                        width: `${r.w * 100}%`, height: `${r.h * 100}%`,
                        border: '2px solid rgba(66, 133, 244, 0.85)',
                        background: 'rgba(66, 133, 244, 0.12)',
                        pointerEvents: 'none',
                        zIndex: 15,
                      }}
                    />
                  ))}
                </div>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', padding: 2 }}>第 {i + 1}/{numPages} 页</div>
              </div>
            ) : (
              <div key={i} style={{ height: 120, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: 11 }}>渲染中...</div>
            )
          )
        )}
      </div>
    </div>
  )
}

function Btn({ onClick, disabled, active, label }: { onClick: () => void; disabled?: boolean; active?: boolean; label: string }) {
  return (
    <button onClick={onClick} disabled={disabled} style={{
      width: 26, height: 26, borderRadius: 6,
      border: active ? '1px solid var(--accent)' : '1px solid var(--border)',
      background: active ? 'var(--accent-bg)' : 'var(--btn-bg)',
      color: active ? 'var(--text-accent)' : disabled ? 'var(--text-muted)' : 'var(--text-primary)',
      cursor: disabled ? 'default' : 'pointer',
      fontSize: 14, fontFamily: 'inherit', display: 'flex', alignItems: 'center', justifyContent: 'center',
      padding: 0, lineHeight: 1,
    }}>{label}</button>
  )
}

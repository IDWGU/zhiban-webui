import { useRef, useEffect, useState, useMemo } from 'react'
import { useAppStore } from '@/stores/appStore'

interface Props {
  text: string
  paperTitle?: string
  paperId?: string
}

export default function TextParagraphs({ text }: Props) {
  const selectedIndices: number[] = useAppStore(s => s.selectedParagraphIndices ?? [])
  const containerRef = useRef<HTMLDivElement>(null)
  const [activeIdx, setActiveIdx] = useState<number | null>(null)

  const paragraphs = useMemo(
    () => text.split(/\n\n+/).filter(p => p.trim().length > 20),
    [text]
  )

  useEffect(() => {
    const scrollEl = containerRef.current?.parentElement
    if (!scrollEl || paragraphs.length === 0) return

    const findActiveParagraph = () => {
      const containerRect = scrollEl.getBoundingClientRect()
      const centerY = containerRect.top + containerRect.height * 0.3
      let bestIdx: number | null = null
      let bestDist = Infinity
      scrollEl.querySelectorAll('[data-para-idx]').forEach(el => {
        const rect = el.getBoundingClientRect()
        if (rect.bottom < containerRect.top || rect.top > containerRect.bottom) return
        const dist = Math.abs((rect.top + rect.bottom) / 2 - centerY)
        if (dist < bestDist) {
          bestDist = dist
          bestIdx = Number(el.getAttribute('data-para-idx'))
        }
      })
      if (bestIdx !== null) setActiveIdx(bestIdx)
    }

    scrollEl.addEventListener('scroll', findActiveParagraph, { passive: true })
    setTimeout(findActiveParagraph, 100)
    return () => scrollEl.removeEventListener('scroll', findActiveParagraph)
  }, [paragraphs])

  if (!text) return null

  return (
    <div ref={containerRef} style={{ padding: '8px 0' }}>
      {paragraphs.map((para, i) => {
        const isSelected = selectedIndices.includes(i)
        return (
        <div key={i} data-para-idx={i} style={{
          fontSize: 13, lineHeight: 1.8, color: 'var(--text-primary)',
          margin: '2px 12px', borderRadius: 6, padding: '10px 14px 10px 36px',
          background: isSelected ? 'var(--accent-bg-subtle)' : i === activeIdx ? 'var(--accent-bg-subtle)' : 'transparent',
          borderLeft: isSelected ? '3px solid var(--accent)' : i === activeIdx ? '3px solid var(--accent)' : '3px solid transparent',
          outline: isSelected ? '1px solid var(--accent)' : 'none',
          outlineOffset: -1,
          transition: 'background 0.3s, border-color 0.3s',
          position: 'relative', cursor: 'default', userSelect: 'text',
        }}>
          {/* Quote button */}
          <button
            onClick={() => useAppStore.getState().toggleParagraphSelection(i)}
            title={isSelected ? '取消引用' : '引用本段作为提问上下文'}
            style={{
              position: 'absolute', left: 4, top: 8,
              width: 24, height: 24, borderRadius: 5,
              border: `1.5px solid ${isSelected ? 'var(--accent)' : 'var(--border)'}`,
              background: isSelected ? 'var(--accent-bg)' : 'transparent',
              color: isSelected ? 'var(--text-accent)' : 'var(--text-muted)',
              cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 12, opacity: isSelected ? 1 : 0.25,
              transition: 'opacity 0.15s, background 0.15s',
            }}
            onMouseEnter={e => { if (!isSelected) (e.target as HTMLElement).style.opacity = '0.7' }}
            onMouseLeave={e => { if (!isSelected) (e.target as HTMLElement).style.opacity = '0.25' }}
          >"</button>
          {isSelected && <span style={{
            position: 'absolute', left: 32, top: 10, fontSize: 9,
            color: 'var(--text-accent)', fontWeight: 600,
          }}>已引用</span>}
          <span style={{ fontSize: 9, color: 'var(--text-muted)', position: 'absolute', right: 8, top: 6, opacity: i === activeIdx || isSelected ? 1 : 0.3 }}>
            {isSelected ? '📌' : i === activeIdx ? '👁' : `§${i + 1}`}
          </span>
          <span style={{ fontSize: 11, color: 'var(--text-muted)', display: 'inline-block', marginRight: 6, minWidth: 20 }}>[{i + 1}]</span>
          {para}
        </div>
      )})}
    </div>
  )
}

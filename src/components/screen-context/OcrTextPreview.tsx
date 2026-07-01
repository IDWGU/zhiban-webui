import type { OcrParagraph } from '@/types'

interface Props {
  paragraphs: OcrParagraph[]
  activeIndex: number | null
}

export default function OcrTextPreview({ paragraphs, activeIndex }: Props) {
  const displayParagraphs = paragraphs.slice(
    Math.max(0, (activeIndex ?? 0) - 0),
    Math.min(paragraphs.length, (activeIndex ?? 0) + 2)
  )

  return (
    <div style={{
      maxHeight: 60, overflow: 'hidden',
      fontSize: 11, lineHeight: 1.5,
    }}>
      {displayParagraphs.map(p => (
        <span
          key={p.index}
          style={{
            color: p.index === activeIndex ? 'var(--text-primary)' : 'var(--text-muted)',
            transition: 'color 0.3s',
          }}
        >
          {p.index === activeIndex && (
            <span style={{
              display: 'inline-block', width: 4, height: 4, borderRadius: '50%',
              background: 'var(--accent)', marginRight: 4, marginBottom: 2,
              boxShadow: '0 0 4px var(--accent)',
            }} />
          )}
          {p.text.slice(0, 150)}
          {p.text.length > 150 ? '...' : ''}{' '}
        </span>
      ))}
    </div>
  )
}

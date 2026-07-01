import type { Citation } from '@/types'

interface Props { citations: Citation[] }

export default function CitationList({ citations }: Props) {
  if (citations.length === 0) return null

  return (
    <div style={{ padding: '4px 12px', fontSize: 11 }}>
      <div style={{
        color: 'hsl(var(--foreground) / 0.38)',
        marginBottom: 4, fontSize: 10, fontWeight: 500,
      }}>
        引用来源
      </div>
      {citations.map((c, i) => (
        <div key={i} style={{
          display: 'flex', gap: 6, alignItems: 'flex-start',
          padding: '4px 0',
          borderBottom: '1px solid hsl(var(--border) / 0.60)',
        }}>
          <span style={{
            color: 'hsl(var(--citation-accent))',
            fontWeight: 600, minWidth: 20, flexShrink: 0,
          }}>
            [{i + 1}]
          </span>
          <div>
            <span style={{ color: 'hsl(var(--citation-text))' }}>
              {c.title || `Paper #${c.paperId}`}
            </span>
            <span style={{ color: 'hsl(var(--muted-foreground))' }}>
              {' '}· {c.sectionType}
            </span>
            <div style={{
              color: 'hsl(var(--muted-foreground))',
              fontSize: 10, marginTop: 1,
            }}>
              {c.chunkText.length > 80 ? c.chunkText.slice(0, 80) + '...' : c.chunkText}
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

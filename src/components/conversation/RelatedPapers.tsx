import type { PaperRef } from '@/types'

interface Props { papers: PaperRef[] }

const RELATION_LABELS: Record<string, string> = {
  successor: '后继', precursor: '前驱', featured: '特色', related: '相关',
}

export default function RelatedPapers({ papers }: Props) {
  if (papers.length === 0) return null

  return (
    <div style={{ padding: '4px 12px', fontSize: 11 }}>
      <div style={{
        color: 'hsl(var(--foreground) / 0.38)',
        marginBottom: 4, fontSize: 10, fontWeight: 500,
      }}>
        知识图谱关联
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
        {papers.map(p => (
          <span key={p.paperId}
            onClick={() => {
              const input = document.querySelector('.query-textarea') as HTMLTextAreaElement
              if (input) {
                const ref = `Paper #${p.paperId} `
                const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')?.set
                if (!((input as any).value || '').includes(ref)) {
                  setter?.call(input, ((input as any).value || '') + ref)
                  input.dispatchEvent(new Event('input', { bubbles: true }))
                }
                input.focus()
              }
            }}
            style={{
            padding: '3px 8px', borderRadius: 9999,
            background: 'hsl(var(--tag-bg))',
            border: '1px solid hsl(var(--tag-border))',
            fontSize: 10, color: 'hsl(var(--tag-text))',
            cursor: 'pointer',
            transition: 'background 0.1s ease, border-color 0.1s ease',
            fontWeight: 500,
          }}>
            #{p.paperId} {p.title.slice(0, 20)}...
            <span style={{
              color: 'hsl(var(--muted-foreground))',
              marginLeft: 4, fontWeight: 400,
            }}>
              {RELATION_LABELS[p.relationType] || p.relationType}
            </span>
          </span>
        ))}
      </div>
    </div>
  )
}

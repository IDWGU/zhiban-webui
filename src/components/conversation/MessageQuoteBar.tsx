import { useAppStore } from '@/stores/appStore'
import type { QuoteItem } from '@/stores/slices/paperSlice'

function quoteLabel(q: QuoteItem): string {
  if ('msgIndex' in q) {
    return `M${q.msgIndex + 1}§${q.paraIndex + 1}`
  }
  return `${q.paperName}:P${q.pageNumber}`
}

function quotePreview(q: QuoteItem): string {
  return q.text.slice(0, 80)
}

function quoteKey(q: QuoteItem): string {
  if ('msgIndex' in q) return `msg:${q.msgIndex}:${q.paraIndex}`
  return `paper:${q.paperId}:${q.pageNumber}`
}

export default function MessageQuoteBar() {
  const quotes = useAppStore(s => s.selectedQuotes)
  const removeQuote = useAppStore(s => s.removeQuote)
  const undoLastQuote = useAppStore(s => s.undoLastQuote)
  const clearQuoteSelections = useAppStore(s => s.clearQuoteSelections)

  if (quotes.length === 0) return null

  return (
    <div style={{
      position: 'fixed', bottom: 128, left: '50%', transform: 'translateX(-50%)',
      zIndex: 1000, display: 'flex', alignItems: 'center', gap: 8,
      padding: '8px 16px',
      background: 'hsl(var(--popover))',
      border: '1px solid hsl(var(--accent) / 0.4)',
      borderRadius: 12,
      boxShadow: '0 4px 24px rgba(0,0,0,0.35)',
      backdropFilter: 'blur(12px)',
      fontSize: 13, color: 'hsl(var(--foreground))',
    }}>
      <span style={{ fontWeight: 600 }}>
        已引用 {quotes.length} 段
      </span>

      <div style={{ display: 'flex', gap: 4, maxWidth: 360, overflow: 'auto' }}>
        {quotes.map((q) => (
          <button
            key={quoteKey(q)}
            onClick={() => {
              if ('msgIndex' in q) {
                removeQuote(q.msgIndex, q.paraIndex)
              } else {
                // For paper quotes, just clear by key
                const store = useAppStore.getState()
                store.undoLastQuote()
              }
            }}
            title={quotePreview(q)}
            style={{
              padding: '2px 8px', borderRadius: 6,
              border: '1px solid hsl(var(--accent) / 0.4)',
              background: 'hsl(var(--accent) / 0.08)',
              color: 'hsl(var(--foreground))',
              cursor: 'pointer', fontSize: 11, whiteSpace: 'nowrap',
              fontFamily: 'inherit', maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis',
            }}
          >
            {quoteLabel(q)} ✕
          </button>
        ))}
      </div>

      <div style={{ display: 'flex', gap: 4, borderLeft: '1px solid var(--border)', paddingLeft: 8 }}>
        <button
          onClick={() => undoLastQuote()}
          title="撤销最后一次引用"
          style={{
            padding: '4px 10px', borderRadius: 6,
            border: '1px solid var(--border)',
            background: 'hsl(var(--muted))',
            color: 'hsl(var(--muted-foreground))',
            cursor: 'pointer', fontSize: 12, fontFamily: 'inherit',
          }}
        >
          ↩ 撤销
        </button>
        <button
          onClick={() => clearQuoteSelections()}
          title="清除所有引用"
          style={{
            padding: '4px 10px', borderRadius: 6,
            border: '1px solid var(--border)',
            background: 'hsl(var(--muted))',
            color: 'hsl(var(--muted-foreground))',
            cursor: 'pointer', fontSize: 12, fontFamily: 'inherit',
          }}
        >
          清除
        </button>
      </div>
    </div>
  )
}

import { useAppStore } from '@/stores/appStore'

export default function SelectionBar() {
  const indices: number[] = useAppStore(s => s.selectedParagraphIndices ?? [])
  if (indices.length === 0) return null

  return (
    <div style={{
      position: 'fixed', bottom: 80, left: '50%', transform: 'translateX(-50%)',
      zIndex: 1000, display: 'flex', alignItems: 'center', gap: 8,
      padding: '8px 16px',
      background: 'hsl(var(--popover))',
      border: '1px solid var(--border)',
      borderRadius: 12,
      boxShadow: '0 4px 24px rgba(0,0,0,0.35)',
      backdropFilter: 'blur(12px)',
      fontSize: 13, color: 'var(--text-primary)',
    }}>
      <span style={{ fontWeight: 600 }}>
        已引用 {indices.length} 段
      </span>

      {/* Individual deselect buttons */}
      <div style={{ display: 'flex', gap: 4, maxWidth: 280, overflow: 'auto' }}>
        {indices.map(idx => (
          <button
            key={idx}
            onClick={() => useAppStore.getState().toggleParagraphSelection(idx)}
            title={`取消引用第 ${idx + 1} 段`}
            style={{
              padding: '2px 8px', borderRadius: 6,
              border: '1px solid var(--border)',
              background: 'var(--accent-bg)',
              color: 'var(--text-accent)',
              cursor: 'pointer', fontSize: 12, whiteSpace: 'nowrap',
              fontFamily: 'inherit',
            }}
          >
            §{idx + 1} ✕
          </button>
        ))}
      </div>

      {/* Undo last / Clear all */}
      <div style={{ display: 'flex', gap: 4, borderLeft: '1px solid var(--border)', paddingLeft: 8 }}>
        <button
          onClick={() => {
            const current = [...useAppStore.getState().selectedParagraphIndices]
            current.pop()
            useAppStore.getState().setSelectedParagraphIndices(current)
          }}
          title="撤销最后一次引用"
          style={{
            padding: '4px 10px', borderRadius: 6,
            border: '1px solid var(--border)',
            background: 'var(--btn-bg)',
            color: 'var(--text-secondary)',
            cursor: 'pointer', fontSize: 12, fontFamily: 'inherit',
          }}
        >
          ↩ 撤销
        </button>
        <button
          onClick={() => useAppStore.getState().clearParagraphSelections()}
          title="清除所有引用"
          style={{
            padding: '4px 10px', borderRadius: 6,
            border: '1px solid var(--border)',
            background: 'var(--btn-bg)',
            color: 'var(--text-muted)',
            cursor: 'pointer', fontSize: 12, fontFamily: 'inherit',
          }}
        >
          清除
        </button>
      </div>
    </div>
  )
}

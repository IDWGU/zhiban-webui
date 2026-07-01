import { useCallback, useMemo } from 'react'
import { useAppStore } from '@/stores/appStore'
import type { TranslationBlock } from '@/types'

export default function TranslationPane() {
  const blocks = useAppStore(s => s.translation.blocks)
  const activeId = useAppStore(s => s.translation.activeSentenceId)
  const selectedIds = useAppStore(s => s.translation.selectedSentenceIds)
  const setActiveId = useAppStore(s => s.setActiveSentenceId)
  const setScrollId = useAppStore(s => s.setScrollTargetId)
  const toggleSelection = useAppStore(s => s.toggleSentenceSelection)
  const phase = useAppStore(s => s.translation.phase)

  const handleEnter = useCallback((id: string) => setActiveId(id), [setActiveId])
  const handleLeave = useCallback(() => setActiveId(null), [setActiveId])
  const handleClick = useCallback((id: string, e: React.MouseEvent) => {
    setActiveId(id)
    setScrollId(id)
    if (e.shiftKey || e.metaKey || e.ctrlKey) {
      toggleSelection(id)
    } else {
      // Plain click: clear any existing multi-selection, move highlight only
      if (selectedIds.length > 0) {
        useAppStore.getState().clearSentenceSelection()
      }
    }
  }, [setActiveId, setScrollId, toggleSelection, selectedIds])

  const sentStyle = useCallback((sId: string) => {
    const isActive = sId === activeId
    const isSelected = selectedIds.includes(sId)
    return {
      background: isActive ? 'rgba(255, 235, 59, 0.35)' : 'transparent',
      border: isSelected ? '1px solid rgba(66, 133, 244, 0.6)' : '1px solid transparent',
      borderRadius: 2,
      cursor: 'default' as const,
      transition: 'background 0.2s, border-color 0.2s',
    }
  }, [activeId, selectedIds])

  const pageGroups = useMemo(() => {
    const map = new Map<number, TranslationBlock[]>()
    for (const b of blocks) {
      const list = map.get(b.pageNum) || []
      list.push(b)
      map.set(b.pageNum, list)
    }
    return Array.from(map.entries()).sort(([a], [b]) => a - b)
  }, [blocks])

  if (phase === 'idle' || phase === 'extracting') {
    return (
      <div style={{
        height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>
        <span style={{ fontSize: 13, color: 'var(--text-muted)' }}>
          {phase === 'extracting' ? '正在提取文档...' : '等待翻译开始'}
        </span>
      </div>
    )
  }

  return (
    <div style={{ padding: '4px 0 16px' }}>
      {pageGroups.map(([pageNum, pageBlocks]) => (
        <div key={pageNum}>
          <div style={{
            fontSize: 10, color: 'var(--text-muted)',
            padding: '4px 12px', marginTop: 8,
            borderBottom: '1px solid var(--border)',
          }}>
            第 {pageNum + 1} 页
          </div>
          {pageBlocks.map(block => (
            <div key={block.id} style={{ padding: '2px 12px' }}>
              {block.type === 'heading' ? (
                <div style={{
                  fontSize: headingFontSize(block.level),
                  fontWeight: 600,
                  color: 'var(--text-primary)',
                  margin: '10px 0 4px',
                  lineHeight: 1.4,
                }}>
                  {block.sentences.map(s => (
                    <span
                      key={s.id}
                      data-sentence-id={s.id}
                      onMouseEnter={() => handleEnter(s.id)}
                      onMouseLeave={handleLeave}
                      onClick={(e) => handleClick(s.id, e)}
                      style={sentStyle(s.id)}
                    >
                      {s.translation || (block.type === 'heading' ? s.text : '　')}
                    </span>
                  ))}
                </div>
              ) : block.type === 'table' ? (
                <TableBlock block={block} activeId={activeId} selectedIds={selectedIds} onEnter={handleEnter} onLeave={handleLeave} onClick={handleClick} sentStyle={sentStyle} />
              ) : (
                <p style={{
                  fontSize: 13, lineHeight: 1.8,
                  color: 'var(--text-primary)',
                  margin: '4px 0',
                }}>
                  {block.sentences.map((s, i) => (
                    <span
                      key={s.id}
                      data-sentence-id={s.id}
                      onMouseEnter={() => handleEnter(s.id)}
                      onMouseLeave={handleLeave}
                      onClick={(e) => handleClick(s.id, e)}
                      style={sentStyle(s.id)}
                    >
                      {s.translation || (
                        <span style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>
                          {s.isComplete ? '—' : '...'}
                        </span>
                      )}
                      {i < block.sentences.length - 1 ? ' ' : ''}
                    </span>
                  ))}
                </p>
              )}
            </div>
          ))}
        </div>
      ))}
    </div>
  )
}

function TableBlock({ block, activeId, selectedIds, onEnter, onLeave, onClick, sentStyle }: {
  block: TranslationBlock
  activeId: string | null
  selectedIds: string[]
  onEnter: (id: string) => void
  onLeave: () => void
  onClick: (id: string, e: React.MouseEvent) => void
  sentStyle: (sId: string) => React.CSSProperties
}) {
  // Render sentences as table rows
  if (block.sentences.length === 0) return null

  const text = block.sentences.map(s => s.translation || s.text).join('\n')

  // Simple table detection: try to parse as pipe-separated
  const lines = text.split('\n')
  const isMarkdownTable = lines.some(l => l.includes('|'))

  if (isMarkdownTable) {
    return (
      <table style={{
        fontSize: 12, borderCollapse: 'collapse', margin: '8px 0',
        width: '100%',
      }}>
        <tbody>
          {lines.filter(l => l.includes('|') && !l.match(/^\|?[\s\-:]+\|?$/)).map((line, ri) => (
            <tr key={ri}>
              {line.split('|').filter(c => c.trim()).map((cell, ci) => (
                <td key={ci} style={{
                  border: '1px solid var(--border)',
                  padding: '4px 8px',
                  color: 'var(--text-primary)',
                }}>
                  {cell.trim()}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    )
  }

  return (
    <div style={{
      fontSize: 12, padding: '8px', margin: '8px 0',
      border: '1px solid var(--border)', borderRadius: 6,
      background: 'var(--bg-panel)',
    }}>
      {block.sentences.map(s => (
        <div
          key={s.id}
          data-sentence-id={s.id}
          onMouseEnter={() => onEnter(s.id)}
          onMouseLeave={onLeave}
          onClick={(e) => onClick(s.id, e)}
          style={{
            ...sentStyle(s.id),
            padding: '1px 4px',
          }}
        >
          {s.translation || '...'}
        </div>
      ))}
    </div>
  )
}

function headingFontSize(level?: number): number {
  switch (level) {
    case 1: return 18
    case 2: return 16
    case 3: return 14
    case 4: return 13
    default: return 13
  }
}

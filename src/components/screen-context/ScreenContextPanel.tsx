import { useState, useEffect } from 'react'
import { useAppStore } from '@/stores/appStore'
import OcrStatusDot from './OcrStatusDot'
import ActiveDocBadge from './ActiveDocBadge'

function timeAgo(timestamp: number | null): string {
  if (!timestamp) return ''
  const seconds = Math.floor((Date.now() - timestamp) / 1000)
  if (seconds < 5) return '刚刚'
  if (seconds < 60) return `${seconds}秒前`
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}分钟前`
  return `${Math.floor(minutes / 60)}小时前`
}

export default function ScreenContextPanel() {
  const screenContext = useAppStore(s => s.screenContext)
  const [now, setNow] = useState(Date.now())
  const [expanded, setExpanded] = useState(false)
  const activeText = screenContext.activeParagraphIndex !== null && screenContext.ocrParagraphs.length > 0
    ? screenContext.ocrParagraphs[screenContext.activeParagraphIndex]?.text
    : null

  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 10000)
    return () => clearInterval(timer)
  }, [])

  const sourceLabel = screenContext.source === 'document' ? 'PDF原文'
    : screenContext.source === 'translation' ? '翻译原文'
    : screenContext.source === 'ax' ? 'AX 感知'
    : screenContext.source === 'ocr' ? '屏幕识别'
    : ''

  const sourceColor = screenContext.source === 'document' ? '#4ade80'
    : screenContext.source === 'ax' ? '#60a5fa'
    : screenContext.source === 'ocr' ? '#facc15'
    : 'hsl(var(--muted-foreground))'

  const hasContent = screenContext.ocrParagraphs.length > 0

  return (
    <div style={{
      flexShrink: 0,
      borderBottom: '1px solid hsl(var(--border))',
      background: expanded ? 'hsl(var(--muted) / 0.60)' : 'hsl(var(--muted) / 0.40)',
      transition: 'background 0.15s ease',
    }}>
      {/* Compact bar — click to expand */}
      <div
        onClick={() => hasContent && setExpanded(!expanded)}
        style={{
          display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap',
          padding: '6px 14px',
          cursor: hasContent ? 'pointer' : 'default',
          userSelect: 'none',
        }}
        title={hasContent ? '点击查看 AI 正在"看"的内容' : ''}
      >
        {/* Label */}
        <span style={{
          fontSize: 10, fontWeight: 600,
          color: 'hsl(var(--foreground) / 0.38)',
          letterSpacing: 0.5, textTransform: 'uppercase',
        }}>
          AI 视野
        </span>

        {/* Source badge */}
        {sourceLabel ? (
          <span style={{
            fontSize: 9, fontWeight: 500,
            color: sourceColor,
            background: `${sourceColor}15`,
            padding: '1px 6px', borderRadius: 9999,
            border: `1px solid ${sourceColor}33`,
          }}>
            {sourceLabel}
          </span>
        ) : (
          <span style={{
            fontSize: 9,
            color: 'hsl(var(--muted-foreground))',
            background: 'hsl(var(--muted))',
            padding: '1px 6px', borderRadius: 9999,
          }}>
            暂停
          </span>
        )}

        <OcrStatusDot isActive={screenContext.isActive || activeText !== null} source={screenContext.source} />

        {screenContext.currentDoc && (
          <ActiveDocBadge docName={screenContext.currentDoc} />
        )}

        {screenContext.lastUpdateTime && (
          <span style={{
            fontSize: 9, color: 'hsl(var(--foreground) / 0.38)',
            whiteSpace: 'nowrap',
          }}>
            {timeAgo(screenContext.lastUpdateTime)}
          </span>
        )}

        {/* Active paragraph preview */}
        {activeText ? (
          <span style={{
            fontSize: 10, color: 'hsl(var(--primary))', fontStyle: 'italic',
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
            flex: 1, minWidth: 0,
          }}>
            {activeText.slice(0, 80)}...
          </span>
        ) : (
          <span style={{
            fontSize: 10, color: 'hsl(var(--foreground) / 0.30)',
            fontStyle: 'italic', flex: 1,
          }}>
            {screenContext.source === 'document' ? '已加载全文上下文' : '等待论文加载...'}
          </span>
        )}

        {/* Expand arrow */}
        {hasContent && (
          <span style={{ fontSize: 9, color: 'hsl(var(--muted-foreground))', flexShrink: 0 }}>
            {expanded ? '▲' : '▼'}
          </span>
        )}
      </div>

      {/* Expanded detail */}
      {expanded && hasContent && (
        <div style={{
          padding: '8px 14px 12px',
          borderTop: '1px solid hsl(var(--border) / 0.50)',
          maxHeight: 200, overflowY: 'auto',
        }}>
          <div style={{
            fontSize: 11, fontWeight: 600,
            color: 'hsl(var(--muted-foreground))',
            marginBottom: 8,
          }}>
            AI 正在"看"以下内容并基于此回答你的问题：
          </div>
          {screenContext.ocrParagraphs.map((p, i) => (
            <div key={i} style={{
              padding: '8px 10px', borderRadius: 8, marginBottom: 6,
              background: i === screenContext.activeParagraphIndex
                ? 'hsl(var(--primary) / 0.08)'
                : 'hsl(var(--muted) / 0.50)',
              border: i === screenContext.activeParagraphIndex
                ? '1px solid hsl(var(--primary) / 0.20)'
                : '1px solid hsl(var(--border) / 0.50)',
              fontSize: 12, lineHeight: 1.7,
              color: i === screenContext.activeParagraphIndex
                ? 'hsl(var(--foreground))'
                : 'hsl(var(--foreground) / 0.70)',
            }}>
              {i === screenContext.activeParagraphIndex && (
                <div style={{
                  fontSize: 9, fontWeight: 600,
                  color: 'hsl(var(--primary))',
                  marginBottom: 4,
                }}>
                  ▸ 当前段落
                </div>
              )}
              {p.text}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

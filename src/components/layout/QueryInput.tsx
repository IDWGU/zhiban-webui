import { useState, useEffect, useRef, useCallback } from 'react'
import { useAppStore } from '@/stores/appStore'

interface Props {
  value: string
  onChange: (v: string) => void
  onSend: (text: string) => boolean
  inputRef: React.RefObject<HTMLTextAreaElement>
}

export default function QueryInput({ value, onChange, onSend, inputRef }: Props) {
  const isStreaming = useAppStore(s => s.conversation.isStreaming)
  const isQueryRunning = useAppStore(s => s.isQueryRunning)

  const [collapsed, setCollapsed] = useState(false)
  // ref 替代 render-phase layout 读取
  const [showCollapseBtn, setShowCollapseBtn] = useState(false)

  // Compute 2-line max height from textarea computed styles
  const getTwoLineHeight = useCallback((el: HTMLTextAreaElement): number => {
    const cs = getComputedStyle(el)
    const lh = parseFloat(cs.lineHeight)
    const pt = parseFloat(cs.paddingTop)
    const pb = parseFloat(cs.paddingBottom)
    const bt = parseFloat(cs.borderTopWidth)
    const bb = parseFloat(cs.borderBottomWidth)
    return lh * 2 + pt + pb + bt + bb
  }, [])

  // autoResize — 延迟到 microtask 执行，避免阻塞输入事件处理
  const autoResize = useCallback((el: HTMLTextAreaElement) => {
    el.style.height = 'auto'
    if (collapsed) {
      el.style.height = getTwoLineHeight(el) + 'px'
      el.style.overflowY = 'hidden'
    } else {
      el.style.height = el.scrollHeight + 'px'
      el.style.overflowY = 'hidden'
    }
  }, [collapsed, getTwoLineHeight])

  // 用 rAF 节流 autoResize：避免每次 onChange → setState → render → effect 链路上的 layout thrashing
  const rafIdRef = useRef(0)
  useEffect(() => {
    if (!inputRef.current) return
    cancelAnimationFrame(rafIdRef.current)
    rafIdRef.current = requestAnimationFrame(() => {
      if (inputRef.current) {
        autoResize(inputRef.current)
        // 更新 collapse 按钮可见性（layout 读取放在 rAF 里，不阻塞 render）
        const twoLine = getTwoLineHeight(inputRef.current)
        setShowCollapseBtn(inputRef.current.scrollHeight > twoLine + 1)
      }
    })
  }, [value, autoResize, inputRef, collapsed, getTwoLineHeight])

  // 内容被清除或低于 2 行时自动取消折叠
  useEffect(() => {
    if (!inputRef.current) return
    // 由 rAF 中的 check 来驱动，这里通过 showCollapseBtn 的变化触发
    if (!showCollapseBtn && collapsed) {
      setCollapsed(false)
    }
  }, [showCollapseBtn, collapsed, inputRef])

  const handleSend = () => {
    const text = value.trim()
    if (!text) return
    if (onSend(text)) {
      onChange('')
      setCollapsed(false)
    }
  }

  const hasText = value.trim().length > 0
  const isRunning = isQueryRunning || isStreaming

  return (
    <div style={{
      flexShrink: 0,
      borderTop: '1px solid hsl(var(--border))',
      padding: '10px 10px 6px',
      background: 'hsl(var(--muted) / 0.50)',
    }}>
      <div style={{ display: 'flex', alignItems: 'flex-end' }}>
        {/* Textarea wrapper */}
        <div style={{ position: 'relative', flex: 1 }}>
          <textarea
            ref={inputRef}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend() }
            }}
            placeholder={'Enter 发送，Shift+Enter 换行'}
            rows={1}
            className="query-textarea"
            style={{
              width: '100%', resize: 'none', overflowY: 'hidden',
              background: 'hsl(var(--foreground) / 0.05)',
              border: '1px solid hsl(var(--border))',
              borderRadius: 10,
              padding: '8px 40px 8px 12px',
              color: 'hsl(var(--foreground))',
              fontSize: 13, fontFamily: 'inherit', lineHeight: 1.5, outline: 'none',
              transition: 'border-color 0.15s ease, background 0.15s ease',
            }}
          />

          {/* Collapse / Expand arrow — top-right, visible when >2 lines */}
          {showCollapseBtn && (
            <button
              onClick={(e) => { e.preventDefault(); setCollapsed(!collapsed) }}
              title={collapsed ? '展开全部' : '收起至两行'}
              style={{
                position: 'absolute',
                right: 6, top: 4,
                width: 22, height: 18,
                border: 'none', borderRadius: 4,
                background: 'transparent',
                color: 'hsl(var(--muted-foreground) / 0.50)',
                cursor: 'pointer',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 11, fontWeight: 600,
                lineHeight: 1,
                transition: 'color 0.15s ease',
              }}
            >
              {collapsed ? (
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                  strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="18 15 12 9 6 15" />
                </svg>
              ) : (
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                  strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="6 9 12 15 18 9" />
                </svg>
              )}
            </button>
          )}

          {/* Enter / Stop — bottom-right */}
          <button
            onClick={(e) => {
              e.preventDefault()
              if (isRunning) {
                useAppStore.getState().cancelQuery()
              } else {
                handleSend()
              }
            }}
            disabled={!isRunning && !hasText}
            title={isRunning ? '停止生成' : 'Enter 发送'}
            style={{
              position: 'absolute',
              right: 6, bottom: 6,
              width: 24, height: 24,
              borderRadius: 10,
              border: 'none',
              background: 'transparent',
              color: isRunning
                ? 'hsl(var(--destructive))'
                : hasText
                  ? 'hsl(var(--primary))'
                  : 'hsl(var(--muted-foreground) / 0.30)',
              cursor: (isRunning || hasText) ? 'pointer' : 'default',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              transition: 'color 0.15s ease',
              opacity: (isRunning || hasText) ? 1 : 0.35,
            }}
          >
            {isRunning ? (
              <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor">
                <rect x="4" y="4" width="16" height="16" rx="2" />
              </svg>
            ) : (
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="9 10 4 15 9 20" />
                <path d="M20 4v7a4 4 0 0 1-4 4H4" />
              </svg>
            )}
          </button>
        </div>
      </div>
      <style>{`
        .query-textarea::-webkit-scrollbar { width: 6px; margin-right: 2px; }
        .query-textarea::-webkit-scrollbar-track { background: transparent; border-radius: 8px; margin: 3px 0; }
        .query-textarea::-webkit-scrollbar-thumb { background: transparent; border-radius: 8px; transition: background 0.3s; }
        .query-textarea:hover::-webkit-scrollbar-thumb { background: hsl(var(--foreground) / 0.12); }
        .query-textarea::-webkit-scrollbar-thumb:active { background: hsl(var(--foreground) / 0.22); }
      `}</style>
    </div>
  )
}

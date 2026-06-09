import { useCallback, useState, useEffect, useRef } from 'react'
import { useAppStore } from '@/stores/appStore'
import ConversationPanel from '@/components/conversation/ConversationPanel'
import PaperViewer from '@/components/paper/PaperViewer'
import NotesPanel from '@/components/paper/NotesPanel'
import ScreenContextPanel from '@/components/screen-context/ScreenContextPanel'
import QueryInput from './QueryInput'

interface Props {
  onSendQuery: (text: string) => boolean
  wsSend: (data: unknown) => void
}

export default function MainContent({ onSendQuery, wsSend }: Props) {
  const [inputText, setInputText] = useState('')
  const papers = useAppStore(s => s.papers)
  const activeTabId = useAppStore(s => s.activeTabId)
  const setActiveTab = useAppStore(s => s.setActiveTab)
  const removePaper = useAppStore(s => s.removePaper)

  const handleSend = useCallback((text: string) => {
    if (onSendQuery(text)) setInputText('')
  }, [onSendQuery])

  const [splitRatio, setSplitRatio] = useState(0.7)
  const draggingRef = useRef(false)
  const mainRef = useRef<HTMLDivElement>(null)

  const clamp = (v: number) => Math.min(0.8, Math.max(0.4, v))

  const handleSplitMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    draggingRef.current = true
  }, [])

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!draggingRef.current || !mainRef.current) return
      const rect = mainRef.current.getBoundingClientRect()
      setSplitRatio(clamp((e.clientX - rect.left) / rect.width))
    }
    const onUp = () => { draggingRef.current = false }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
  }, [])

  const inputRef = useRef<HTMLTextAreaElement>(null)

  const activePaper = papers.find(p => p.id === activeTabId)
  const showNotes = activeTabId === 'notes'

  // Auto-select first paper when papers change and no valid tab is active
  useEffect(() => {
    if (papers.length > 0) {
      if (!activeTabId || !papers.find(p => p.id === activeTabId)) {
        setActiveTab(papers[0].id)
      }
    }
  }, [papers, activeTabId, setActiveTab])

  const onTabClose = useCallback((e: React.MouseEvent, id: string) => {
    e.stopPropagation()
    const paper = papers.find(p => p.id === id)
    if (!paper) return
    removePaper(id)
    const store = useAppStore.getState()
    // Notify backend to remove paper from conversation
    store.sendMessage?.({ type: 'unbind_paper', paperId: id })
    store.pushNotification('info', `已关闭「${paper.name}」`, () => {
      store.addPaper(paper)
    })
  }, [removePaper, papers])

  return (
    <div ref={mainRef} style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
      {/* Left: Reading Area — Proma panel */}
      <div className="proma-panel" style={{
        flex: `0 0 calc(${splitRatio * 100}% - 4px)`,
        display: 'flex', flexDirection: 'column',
        minWidth: 0,
      }}>
        {/* Tab bar */}
        <div style={{
          flexShrink: 0, display: 'flex', alignItems: 'stretch', height: 38,
          background: 'hsl(var(--muted) / 0.50)',
          borderBottom: '1px solid hsl(var(--border))',
          overflow: 'hidden', paddingLeft: 4,
        }}>
          <div onClick={() => setActiveTab('notes')} title="Shared Notes" style={{
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            width: 36, minWidth: 36, height: '100%',
            cursor: 'pointer', fontSize: 13, userSelect: 'none',
            background: showNotes ? 'hsl(var(--accent))' : 'transparent',
            borderBottom: showNotes ? '2px solid hsl(var(--primary))' : '2px solid transparent',
            transition: 'background 0.1s ease, border-color 0.1s ease',
          }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
              strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
              style={{ color: showNotes ? 'hsl(var(--primary))' : 'hsl(var(--muted-foreground))' }}>
              <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
              <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
            </svg>
          </div>
          <div style={{ width: 1, height: 20, background: 'hsl(var(--border))', margin: '9px 4px', flexShrink: 0 }} />
          {papers.map(p => (
            <div key={p.id} onClick={() => setActiveTab(p.id)} title={p.name} style={{
              display: 'flex', alignItems: 'center', gap: 4,
              padding: '0 10px', height: '100%',
              cursor: 'pointer', fontSize: 11,
              color: p.id === activeTabId ? 'hsl(var(--foreground))' : 'hsl(var(--muted-foreground))',
              background: p.id === activeTabId ? 'hsl(var(--accent))' : 'transparent',
              borderBottom: p.id === activeTabId ? '2px solid hsl(var(--primary))' : '2px solid transparent',
              borderRight: '1px solid hsl(var(--border))',
              minWidth: 0, maxWidth: 200, flexShrink: 0,
              transition: 'background 0.1s ease, border-color 0.1s ease',
              userSelect: 'none',
            }}>
              <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {p.name}
              </span>
              <span onClick={(e) => onTabClose(e, p.id)} style={{
                fontSize: 14, fontWeight: 300,
                color: 'hsl(var(--muted-foreground))',
                padding: '0 2px', borderRadius: 3, flexShrink: 0, cursor: 'pointer',
                lineHeight: 1,
              }}>×</span>
            </div>
          ))}
          {papers.length === 0 && (
            <div style={{ fontSize: 11, color: 'hsl(var(--muted-foreground) / 0.60)', padding: '0 12px', fontStyle: 'italic', display: 'flex', alignItems: 'center', height: '100%' }}>
              Drag PDF/DOCX/TXT anywhere to open
            </div>
          )}
          <div style={{ flex: 1 }} />
        </div>

        {/* Paper / Notes content */}
        <div style={{ flex: 1, overflow: 'hidden' }}>
          {showNotes && <NotesPanel />}
          {!showNotes && activePaper && (
            <PaperViewer
              key={activePaper.id}
              filePath={activePaper.path}
              paperTitle={activePaper.name}
              paperId={activePaper.id}
              paperContent={activePaper.extractedText || undefined}
              onTextExtracted={(text) => {
                useAppStore.getState().updatePaperText(activePaper.id, text)
              }}
            />
          )}
          {!showNotes && !activePaper && (
            <div style={{ height: '100%', display: 'flex', flexDirection: 'column',
              alignItems: 'center', justifyContent: 'center', color: 'hsl(var(--muted-foreground))' }}>
              <div style={{ fontSize: 48, opacity: 0.12, marginBottom: 16 }}>
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                  strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                  <polyline points="14 2 14 8 20 8" />
                  <line x1="16" y1="13" x2="8" y2="13" />
                  <line x1="16" y1="17" x2="8" y2="17" />
                  <polyline points="10 9 9 9 8 9" />
                </svg>
              </div>
              <div style={{ fontSize: 13, fontWeight: 500 }}>Drag paper files anywhere on the window</div>
              <div style={{ fontSize: 11, opacity: 0.50, marginTop: 6 }}>Supports PDF / DOCX / TXT / Markdown</div>
            </div>
          )}
        </div>
      </div>

      {/* Draggable divider — handle width = shell gap (8px) */}
      <div style={{
        flex: '0 0 8px', position: 'relative',
        cursor: 'col-resize', zIndex: 5,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>
        <div onMouseDown={handleSplitMouseDown} style={{
          width: 8, height: 40, borderRadius: 4,
          background: 'hsl(var(--muted-foreground) / 0.35)',
          cursor: 'col-resize',
          transition: 'background 0.15s ease',
        }}
          onMouseEnter={e => (e.currentTarget.style.background = 'hsl(var(--primary) / 0.55)')}
          onMouseLeave={e => { if (!draggingRef.current) e.currentTarget.style.background = 'hsl(var(--muted-foreground) / 0.35)' }}
        />
      </div>

      {/* Right: AI Conversation Area — Proma panel */}
      <div className="proma-panel" style={{
        flex: `0 0 calc(${(1 - splitRatio) * 100}% - 4px)`,
        display: 'flex', flexDirection: 'column',
        minWidth: 280,
      }}>
        <div style={{
          height: 38, display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 12, fontWeight: 600,
          padding: '0 16px',
          color: 'hsl(var(--muted-foreground))',
          letterSpacing: 0.5,
          borderBottom: '1px solid hsl(var(--border))',
          flexShrink: 0,
        }}>
          AI Companion · ZhiBan
        </div>
        <ScreenContextPanel />
        <ConversationPanel />
        <ContextIndicator />
        <QueryInput
          value={inputText}
          onChange={setInputText}
          onSend={handleSend}
          inputRef={inputRef}
        />
      </div>
    </div>
  )
}

// ── 上下文指示器 ──
// 显示文档上下文 + 段落/消息引用状态，位于对话框上方
function ContextIndicator() {
  const screenCtx = useAppStore(s => s.screenContext)
  const paused = useAppStore(s => (s as any).contextPaused ?? false)
  const resumeContext = useAppStore(s => (s as any).resumeContext as () => void)
  const pauseContext = useAppStore(s => (s as any).pauseContext as () => void)
  const paraSelections: number[] = useAppStore(s => (s as any).selectedParagraphIndices ?? [])
  const allQuotes = useAppStore(s => s.selectedQuotes ?? [])
  const msgQuotes = allQuotes.filter(q => 'msgIndex' in q) as Array<{msgIndex: number; paraIndex: number; text: string}>
  const paperQuotes = allQuotes.filter(q => 'paperId' in q) as Array<{paperId: string; paperName: string; pageNumber: number; text: string}>
  const clearQuotes = useAppStore(s => s.clearQuoteSelections)
  const clearPara = useAppStore(s => (s as any).clearParagraphSelections as () => void)

  const paragCount = screenCtx.ocrParagraphs.length
  const activeIdx = screenCtx.activeParagraphIndex ?? 0
  const source = screenCtx.source === 'document' ? '文档' : screenCtx.source === 'translation' ? '翻译' : '屏幕'
  const hasDocCtx = screenCtx.currentDoc && paragCount > 0
  const hasQuotes = msgQuotes.length > 0 || paperQuotes.length > 0
  const hasParas = paraSelections.length > 0

  if (!hasDocCtx && !hasQuotes && !hasParas) return null

  return (
    <div style={{
      flexShrink: 0,
      margin: '2px 8px 4px',
      padding: '7px 10px',
      borderRadius: 10,
      background: paused
        ? 'hsl(var(--muted) / 0.55)'
        : 'hsl(var(--accent) / 0.05)',
      border: `1px solid ${paused
        ? 'hsl(var(--border) / 0.70)'
        : 'hsl(var(--accent) / 0.12)'}`,
      fontSize: 11,
      color: 'hsl(var(--foreground) / 0.72)',
      display: 'flex', flexDirection: 'column', gap: 5,
      lineHeight: 1.5,
    }}>
      {/* ── 文档上下文行 ── */}
      {hasDocCtx && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
          {/* 状态图标 */}
          <span style={{
            flexShrink: 0, width: 18, height: 18, borderRadius: 6,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 10,
            background: paused
              ? 'hsl(var(--muted-foreground) / 0.10)'
              : 'hsl(142 60% 40% / 0.12)',
            color: paused
              ? 'hsl(var(--muted-foreground) / 0.55)'
              : 'hsl(142 60% 42%)',
          }}>
            {paused ? '⏸' : '●'}
          </span>
          {/* 文档名 + 页码 */}
          <span style={{
            flex: 1, minWidth: 0,
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
            fontWeight: 500,
          }}>
            {screenCtx.currentDoc!.length > 36
              ? screenCtx.currentDoc!.slice(0, 36) + '…'
              : screenCtx.currentDoc}
          </span>
          <span style={{
            flexShrink: 0, fontSize: 10,
            color: 'hsl(var(--foreground) / 0.42)',
            background: 'hsl(var(--muted) / 0.50)',
            padding: '1px 7px', borderRadius: 9999,
          }}>
            第 {activeIdx + 1}/{paragCount} 页
          </span>
          {/* 暂停/恢复 */}
          {paused ? (
            <button onClick={resumeContext} title="恢复自动上下文" style={{
              flexShrink: 0,
              background: 'hsl(142 60% 40% / 0.10)',
              border: '1px solid hsl(142 60% 40% / 0.20)',
              borderRadius: 5, cursor: 'pointer',
              color: 'hsl(142 60% 40%)', fontSize: 10, fontFamily: 'inherit',
              padding: '2px 8px', fontWeight: 500,
              lineHeight: 1.4,
            }}>
              恢复
            </button>
          ) : (
            <button onClick={pauseContext} title="暂停自动上下文" style={{
              flexShrink: 0,
              width: 20, height: 20, borderRadius: 5,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              background: 'transparent', border: 'none', cursor: 'pointer',
              color: 'hsl(var(--muted-foreground) / 0.55)',
              fontSize: 12, padding: 0, lineHeight: 1,
            }}>✕</button>
          )}
        </div>
      )}

      {/* ── 文档段落引用 ── */}
      {hasParas && (
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 6 }}>
          <span style={{
            flexShrink: 0, width: 18, height: 18, borderRadius: 6,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 10, marginTop: 1,
            background: 'hsl(38 92% 50% / 0.10)',
            color: 'hsl(38 92% 48%)',
          }}>§</span>
          <div style={{ flex: 1, minWidth: 0, display: 'flex', flexWrap: 'wrap', gap: 4, alignItems: 'center' }}>
            <span style={{
              fontSize: 10, fontWeight: 600, flexShrink: 0,
              color: 'hsl(var(--foreground) / 0.45)',
            }}>{paraSelections.length} 段引用</span>
            {paraSelections.slice(0, 3).map(idx => {
              const p = screenCtx.ocrParagraphs[idx]
              const preview = p ? p.text.slice(0, 60).replace(/\n/g, ' ') : ''
              return (
                <span key={idx} title={p?.text.slice(0, 200)} style={{
                  padding: '2px 7px', borderRadius: 9999, fontSize: 10,
                  background: 'hsl(38 92% 50% / 0.08)',
                  border: '1px solid hsl(38 92% 50% / 0.15)',
                  color: 'hsl(var(--foreground) / 0.65)',
                  maxWidth: 200, overflow: 'hidden',
                  textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                }}>{preview}</span>
              )
            })}
          </div>
          <button onClick={clearPara} title="清除文档段落引用" style={{
            flexShrink: 0,
            width: 20, height: 20, borderRadius: 5,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: 'transparent', border: 'none', cursor: 'pointer',
            color: 'hsl(var(--muted-foreground) / 0.50)',
            fontSize: 12, padding: 0, lineHeight: 1,
          }}>✕</button>
        </div>
      )}

      {/* ── 对话消息引用 ── */}
      {hasQuotes && (
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 6 }}>
          <span style={{
            flexShrink: 0, width: 18, height: 18, borderRadius: 6,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 10, marginTop: 1,
            background: 'hsl(262 60% 55% / 0.10)',
            color: 'hsl(262 60% 52%)',
          }}>❝</span>
          <div style={{ flex: 1, minWidth: 0, display: 'flex', flexWrap: 'wrap', gap: 4, alignItems: 'center' }}>
            <span style={{
              fontSize: 10, fontWeight: 600, flexShrink: 0,
              color: 'hsl(var(--foreground) / 0.45)',
            }}>{msgQuotes.length} 段引用</span>
            {msgQuotes.slice(0, 3).map((q, i) => {
              const preview = q.text.slice(0, 60).replace(/\n/g, ' ')
              return (
                <span key={i} title={q.text.slice(0, 200)} style={{
                  padding: '2px 7px', borderRadius: 9999, fontSize: 10,
                  background: 'hsl(262 60% 55% / 0.08)',
                  border: '1px solid hsl(262 60% 55% / 0.15)',
                  color: 'hsl(var(--foreground) / 0.65)',
                  maxWidth: 200, overflow: 'hidden',
                  textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                }}>{preview}</span>
              )
            })}
          </div>
          <button onClick={clearQuotes} title="清除消息段落引用" style={{
            flexShrink: 0,
            width: 20, height: 20, borderRadius: 5,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: 'transparent', border: 'none', cursor: 'pointer',
            color: 'hsl(var(--muted-foreground) / 0.50)',
            fontSize: 12, padding: 0, lineHeight: 1,
          }}>✕</button>
        </div>
      )}
    </div>
  )
}

import { useEffect, useRef, useState, useCallback } from 'react'
import { useAppStore } from '@/stores/appStore'
import MessageBubble from './MessageBubble'
import type { AgentStep } from '@/types'

interface DebugStep {
  code: string
  message: string
  timestamp: number
}

const STAGE_ICONS: Record<string, string> = {
  classifying: '▶',
  intent_analysis: '\u{1F50D}',
  query_understanding: '\u{1F9E0}',
  searching: '\u{1F50E}',
  searching_r2: '\u{1F50E}',
  retrieval: '\u{1F4DA}',
  filtering: '\u{1F4CB}',
  ranking: '\u{1F4CA}',
  reasoning: '\u{1F4AD}',
  thinking: '\u{1F4AD}',
  generating: '✏️',
  writing: '✏️',
  summarizing: '\u{1F4DD}',
  translating: '\u{1F310}',
  processing: '⚙️',
  chatting: '\u{1F4AC}',
  done: '✅',
  error: '❌',
  refused: '\u{1F6AB}',
  expanding: '\u{1F4D0}',
  expanded: '✅',
  idle: '',
}

function getStageIcon(code: string): string {
  return STAGE_ICONS[code] || STAGE_ICONS.processing
}

function formatTime(ts: number): string {
  const d = new Date(ts)
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}:${String(d.getSeconds()).padStart(2, '0')}`
}

export default function ConversationPanel() {
  const messages = useAppStore(s => s.conversation.messages)
  const isStreaming = useAppStore(s => s.conversation.isStreaming)
  const streamingText = useAppStore(s => s.conversation.streamingText)
  const streamingThinkingText = useAppStore(s => s.conversation.streamingThinkingText)
  const streamingAgentSteps = useAppStore(s => s.conversation.streamingAgentSteps)
  const streamingCitations = useAppStore(s => s.conversation.streamingCitations)
  const streamingRelatedPapers = useAppStore(s => s.conversation.streamingRelatedPapers)
  const workflowStatus = useAppStore(s => s.workflowStatus)
  const isQueryRunning = useAppStore(s => s.isQueryRunning)
  const papers = useAppStore(s => s.papers)
  const sessionUsage = useAppStore(s => s.sessionUsage)
  const healthHistory = useAppStore(s => s.healthHistory)
  const localSpeed = useAppStore(s => s.localSpeed)
  const bottomRef = useRef<HTMLDivElement>(null)

  const wsStatus = useAppStore(s => s.connection.wsStatus)
  const wsDisconnected = wsStatus === 'disconnected'
  const showDevPanel = useAppStore(s => s.connection.showDevPanel)
  const setDebugMode = useAppStore(s => s.setDebugMode)
  const [debugSteps, setDebugSteps] = useState<DebugStep[]>([])
  const [debugExpanded, setDebugExpanded] = useState(false)
  const [healthExpanded, setHealthExpanded] = useState(false)
  const [demoRunning, setDemoRunning] = useState(false)
  const prevCodeRef = useRef(workflowStatus.code)

  // ── 模拟 Agent 模式演示 (离线演示用) ──
  const simulateAgentDemo = useCallback(() => {
    if (demoRunning || isStreaming) return
    setDemoRunning(true)
    import('./AgentDemo').then(m => {
      m.runAgentDemo().finally(() => setDemoRunning(false))
    }).catch(() => setDemoRunning(false))
  }, [demoRunning, isStreaming])

  useEffect(() => {
    if (showDevPanel && workflowStatus.code !== 'idle' && workflowStatus.code !== prevCodeRef.current) {
      setDebugSteps(prev => {
        const next = [
          ...prev,
          { code: workflowStatus.code, message: workflowStatus.message, timestamp: Date.now() },
        ]
        return next.length > 50 ? next.slice(-50) : next
      })
    }
    prevCodeRef.current = workflowStatus.code
  }, [workflowStatus, showDevPanel])

  const queryStartRef = useRef(false)
  useEffect(() => {
    if (showDevPanel && isQueryRunning && !queryStartRef.current) {
      queryStartRef.current = true
      if (debugSteps.length > 0) {
        setDebugSteps(prev => [
          ...prev,
          { code: '──', message: '新查询 ──', timestamp: Date.now() },
        ])
      }
    }
    if (!isQueryRunning) {
      queryStartRef.current = false
    }
  }, [isQueryRunning, showDevPanel])

  const streamStartedRef = useRef(false)
  useEffect(() => {
    if (isStreaming && streamingText.length === 0) {
      streamStartedRef.current = true
      if (!showDevPanel) {
        setDebugSteps([])
      }
      // Clear health history for new query
      useAppStore.getState().clearHealthHistory()
    }
    if (!isStreaming) {
      streamStartedRef.current = false
    }
  }, [isStreaming, streamingText.length, showDevPanel])

  // Instant scroll-to-bottom on any content change, no animation lag
  useEffect(() => {
    requestAnimationFrame(() => {
      const el = bottomRef.current?.parentElement
      if (el) el.scrollTop = el.scrollHeight
    })
  }, [messages.length, streamingText, streamingThinkingText, streamingAgentSteps, debugSteps.length])

  const showDebugPanel = showDevPanel && debugSteps.length > 0

  // ── Empty states ──

  if (messages.length === 0 && !isStreaming) {
    if (papers.length === 0) {
      return (
        <div style={{
          flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
          padding: 20,
        }}>
          <div style={{
            maxWidth: 400, textAlign: 'center',
            padding: '40px 32px', borderRadius: 16,
            border: '2px dashed hsl(var(--dashed-border))',
            background: 'hsl(var(--background))',
          }}>
            <div style={{ fontSize: 36, marginBottom: 16, opacity: 0.5 }}>
              <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"
                style={{ color: 'hsl(var(--muted-foreground))' }}>
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
            </div>
            <div style={{
              fontSize: 15, fontWeight: 600,
              color: 'hsl(var(--foreground))',
              marginBottom: 8,
            }}>
              拖入论文 PDF 开始阅读
            </div>
            <div style={{
              fontSize: 12, color: 'hsl(var(--muted-foreground))',
              lineHeight: 1.8, marginBottom: 24,
            }}>
              支持 PDF / DOCX / TXT / MD 格式<br />
              AI 自动检索知识库，智能回答你的问题<br />
              按住 Tab 键语音输入，AI 自动读取当前段落
            </div>
            <div style={{ display: 'flex', gap: 8, justifyContent: 'center', flexWrap: 'wrap' }}>
              <button
                onClick={() => {
                  const input = document.querySelector('.query-textarea') as HTMLTextAreaElement
                  if (input) input.focus()
                }}
                style={{
                  padding: '8px 20px', borderRadius: 10,
                  border: '1px solid hsl(var(--primary) / 0.25)',
                  background: 'hsl(var(--primary) / 0.08)',
                  color: 'hsl(var(--primary))',
                  fontSize: 13, cursor: 'pointer',
                  fontFamily: 'inherit',
                  fontWeight: 500,
                }}
              >
                跳过，直接提问
              </button>
              {wsDisconnected && (
                <button
                  onClick={simulateAgentDemo}
                  disabled={demoRunning}
                  style={{
                    padding: '8px 20px', borderRadius: 10,
                    border: '1px solid #c792ea',
                    background: 'hsl(280 50% 30% / 0.10)',
                    color: '#c792ea',
                    fontSize: 13, cursor: demoRunning ? 'not-allowed' : 'pointer',
                    fontFamily: 'inherit',
                    fontWeight: 500,
                    opacity: demoRunning ? 0.5 : 1,
                  }}
                >
                  {'\u{1F9E0} 演示 Agent 思考过程'}
                </button>
              )}
            </div>
          </div>
        </div>
      )
    }

    return (
      <div style={{
        flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
        color: 'hsl(var(--muted-foreground))', fontSize: 13,
        textAlign: 'center', padding: 20,
      }}>
        <div>
          <div style={{ fontSize: 32, marginBottom: 12, opacity: 0.25 }}>{'\u{1F4D6}'}</div>
          <div>{'拖拽论文到窗口，直接开口提问'}</div>
          <div style={{
            fontSize: 11, marginTop: 6,
            color: 'hsl(var(--foreground) / 0.30)',
          }}>
            {'按住 Tab 键语音输入 · AI 自动读取当前段落'}
          </div>
        </div>
      </div>
    )
  }

  // ── Conversation ──

  return (
    <div style={{
      flex: 1, overflowY: 'auto',
      display: 'flex', flexDirection: 'column',
    }}>
      {/* Messages */}
      {messages.map(msg => (
        <MessageBubble key={msg.id} message={msg} />
      ))}

      {/* Workflow debug panel */}
      {showDebugPanel && (
        <div style={{
          margin: '4px 12px',
          border: '1px solid hsl(var(--border))',
          borderRadius: 10,
          background: 'hsl(var(--muted) / 0.40)',
          overflow: 'hidden',
          flexShrink: 0,
        }}>
          <div
            onClick={() => setDebugExpanded(!debugExpanded)}
            style={{
              display: 'flex', alignItems: 'center',
              padding: '6px 12px', gap: 8,
              fontSize: 10, fontWeight: 600,
              color: 'hsl(var(--muted-foreground))',
              cursor: 'pointer', userSelect: 'none',
            }}
          >
            <span>{debugExpanded ? '▼' : '▶'}</span>
            <span>WORKFLOW DEBUG ({debugSteps.length} steps)</span>
            <div style={{ flex: 1 }} />
            <button
              onClick={e => { e.stopPropagation(); setDebugMode(false); setDebugSteps([]) }}
              style={{
                background: 'none', border: 'none',
                color: 'hsl(var(--muted-foreground))',
                cursor: 'pointer', fontSize: 12, padding: '2px 4px',
                borderRadius: 4,
              }}
              title="Close debug mode"
            >
              {'✕'}
            </button>
            <button
              onClick={e => { e.stopPropagation(); setDebugSteps([]) }}
              style={{
                background: 'none', border: 'none',
                color: 'hsl(var(--muted-foreground))',
                cursor: 'pointer', fontSize: 11, padding: '2px 4px',
                borderRadius: 4,
              }}
              title="Clear debug steps"
            >
              {'\u{1F5D1}'}
            </button>
          </div>
          {debugExpanded && (
            <div>
              {debugSteps.map((step, i) => (
                <div key={i} style={{
                  fontSize: 11, padding: '3px 12px 3px 22px',
                  display: 'flex', alignItems: 'center',
                  fontFamily: 'monospace',
                  borderTop: '1px solid hsl(var(--border) / 0.50)',
                }}>
                  <span style={{ marginRight: 6, fontSize: 10 }}>
                    {getStageIcon(step.code)}
                  </span>
                  <span style={{
                    color: 'hsl(var(--muted-foreground))',
                    background: 'hsl(var(--muted))',
                    padding: '1px 5px', borderRadius: 3, fontSize: 10,
                  }}>
                    {step.code}
                  </span>
                  <span style={{ marginLeft: 8, color: 'hsl(var(--foreground))' }}>
                    {step.message}
                  </span>
                  <span style={{ marginLeft: 'auto', color: 'hsl(var(--muted-foreground))', fontSize: 9 }}>
                    {formatTime(step.timestamp)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Phase 3: LLM Health panel */}
      {healthHistory.length > 0 && showDevPanel && (
        <div style={{
          margin: '4px 12px',
          border: '1px solid hsl(var(--border))',
          borderRadius: 10,
          background: 'hsl(var(--muted) / 0.40)',
          overflow: 'hidden',
          flexShrink: 0,
        }}>
          <div
            onClick={() => setHealthExpanded(!healthExpanded)}
            style={{
              display: 'flex', alignItems: 'center',
              padding: '6px 12px', gap: 8,
              fontSize: 10, fontWeight: 600,
              color: 'hsl(var(--muted-foreground))',
              cursor: 'pointer', userSelect: 'none',
            }}
          >
            <span>{healthExpanded ? '▼' : '▶'}</span>
            <span>⚡ LLM 运行状态</span>
            <div style={{ flex: 1 }} />
          </div>
          {healthExpanded && (
            <div style={{ padding: '0 12px 10px' }}>
              {healthHistory.slice(-2).reverse().map((h, i) => {
                const cacheRate = h.tokens.cache_hit_rate
                const prefillMs = h.timing.prefill_ms
                const decodePerTok = h.timing.decode_per_token_ms
                const tokPerSec = decodePerTok && decodePerTok > 0
                  ? Math.round(1000 / decodePerTok)
                  : null
                const vramMb = h.memory?.vram_mb ?? h.memory?.vram_after_mb

                return (
                  <div key={i} style={{
                    fontSize: 11,
                    padding: '6px 0',
                    borderTop: i > 0 ? '1px solid hsl(var(--border) / 0.50)' : 'none',
                  }}>
                    {/* Call label */}
                    <div style={{
                      display: 'flex', alignItems: 'center', gap: 6,
                      marginBottom: 4,
                    }}>
                      <span style={{
                        background: h.call === 'classify'
                          ? 'hsl(var(--primary) / 0.15)'
                          : 'hsl(150 60% 40% / 0.15)',
                        color: h.call === 'classify'
                          ? 'hsl(var(--primary))'
                          : 'hsl(150 60% 40%)',
                        padding: '1px 6px', borderRadius: 4,
                        fontSize: 10, fontWeight: 600,
                      }}>
                        {h.call === 'classify' ? 'Call 1 · 分类' : 'Call 2 · 回答'}
                      </span>
                      {prefillMs != null && (
                        <span style={{ color: 'hsl(var(--muted-foreground))' }}>
                          首 token: {prefillMs < 1000
                            ? `${prefillMs.toFixed(0)}ms`
                            : `${(prefillMs / 1000).toFixed(2)}s`}
                        </span>
                      )}
                      {tokPerSec != null && (
                        <span style={{ color: 'hsl(var(--muted-foreground))' }}>
                          · {tokPerSec} tok/s
                        </span>
                      )}
                      {h.timing.total_ms != null && (
                        <span style={{
                          marginLeft: 'auto',
                          color: 'hsl(var(--foreground) / 0.45)',
                          fontSize: 10,
                        }}>
                          {h.timing.total_ms < 1000
                            ? `${h.timing.total_ms.toFixed(0)}ms`
                            : `${(h.timing.total_ms / 1000).toFixed(1)}s`}
                        </span>
                      )}
                    </div>

                    {/* Cache hit rate bar */}
                    {cacheRate != null && (
                      <div style={{
                        display: 'flex', alignItems: 'center', gap: 6,
                        marginBottom: 2,
                      }}>
                        <span style={{
                          color: 'hsl(var(--muted-foreground))',
                          fontSize: 10, minWidth: 52,
                        }}>
                          Cache 命中
                        </span>
                        <div style={{
                          flex: 1, height: 6,
                          background: 'hsl(var(--muted))',
                          borderRadius: 3, overflow: 'hidden',
                        }}>
                          <div style={{
                            width: `${Math.round(cacheRate * 100)}%`,
                            height: '100%',
                            background: cacheRate > 0.7
                              ? 'hsl(150 60% 40%)'
                              : cacheRate > 0.3
                                ? 'hsl(45 80% 50%)'
                                : 'hsl(0 70% 50%)',
                            borderRadius: 3,
                            transition: 'width 0.3s',
                          }} />
                        </div>
                        <span style={{
                          fontSize: 10, fontWeight: 600,
                          color: cacheRate > 0.7
                            ? 'hsl(150 60% 40%)'
                            : cacheRate > 0.3
                              ? 'hsl(45 80% 50%)'
                              : 'hsl(0 70% 50%)',
                          minWidth: 36, textAlign: 'right',
                        }}>
                          {Math.round(cacheRate * 100)}%
                        </span>
                      </div>
                    )}

                    {/* Token counts */}
                    <div style={{
                      display: 'flex', gap: 10,
                      color: 'hsl(var(--foreground) / 0.50)',
                      fontSize: 10, marginTop: 2,
                    }}>
                      <span>prefill: {h.tokens.prefill_tokens.toLocaleString()} tok</span>
                      <span>output: {h.tokens.output_tokens.toLocaleString()} tok</span>
                      {h.tokens.cache_hit_tokens != null && (
                        <span style={{ color: 'hsl(150 60% 40%)' }}>
                          命中: {h.tokens.cache_hit_tokens.toLocaleString()} tok
                        </span>
                      )}
                    </div>

                    {/* VRAM */}
                    {vramMb != null && (
                      <div style={{
                        fontSize: 10, color: 'hsl(var(--foreground) / 0.40)',
                        marginTop: 2,
                      }}>
                        显存: {vramMb.toFixed(0)} MB
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}

      {/* Streaming message — ThinkingProcess inside MessageBubble handles all status display */}
      {isStreaming && (
        <MessageBubble
          message={{
            id: 'streaming',
            role: 'assistant',
            content: streamingText,
            citations: streamingCitations,
            relatedPapers: streamingRelatedPapers,
            timestamp: Date.now(),
          }}
          isStreaming
        />
      )}

      {/* Session usage footer */}
      {(sessionUsage.input > 0 || sessionUsage.output > 0) && (
        <div style={{
          padding: '6px 14px 10px',
          fontSize: 10,
          color: 'hsl(var(--foreground) / 0.38)',
          textAlign: 'center',
          flexShrink: 0,
        }}>
          <span>本轮: {(sessionUsage.input + sessionUsage.output).toLocaleString()} tokens</span>
          <span style={{ marginLeft: 8, opacity: 0.60 }}>
            (输入 {(sessionUsage.input / 1000).toFixed(1)}K / 输出 {(sessionUsage.output / 1000).toFixed(1)}K)
          </span>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  )
}

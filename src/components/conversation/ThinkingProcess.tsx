import { useState, useMemo, useRef, useEffect, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { AgentStep } from '@/types'
import { useAppStore } from '@/stores/appStore'
import ToolCallBlock from './ToolCallBlock'

interface Props {
  agentSteps?: AgentStep[]
  thinkingContent?: string
  isStreaming?: boolean
  contentStarted?: boolean
  prefillTokens?: number
  prefillMs?: number
  tokPerSec?: number
}

const PHASE_CONFIG: Record<string, { icon: string; label: string; color: string }> = {
  thinking: { icon: '💭', label: '思考', color: '#c792ea' },
  tool_call: { icon: '🔧', label: '调用工具', color: '#ffb86c' },
  tool_result: { icon: '📋', label: '工具结果', color: '#82aaff' },
}

interface Round {
  roundIndex: number
  steps: AgentStep[]
  hasToolCall: boolean
}

function groupStepsIntoRounds(steps: AgentStep[] | undefined | null): Round[] {
  if (!steps || !Array.isArray(steps)) return []
  const rounds: Round[] = []
  let currentRound: Round | null = null

  for (const step of steps) {
    if (!currentRound || currentRound.roundIndex !== step.stepIndex) {
      currentRound = {
        roundIndex: step.stepIndex,
        steps: [step],
        hasToolCall: false,
      }
      rounds.push(currentRound)
    } else {
      currentRound.steps.push(step)
    }
    if (step.phase === 'tool_call' || step.phase === 'tool_result') {
      currentRound.hasToolCall = true
    }
  }

  return rounds
}

export default function ThinkingProcess({
  agentSteps, thinkingContent, isStreaming,
  contentStarted, prefillTokens, prefillMs, tokPerSec,
}: Props) {
  const [expanded, setExpanded] = useState(false)

  const prefillStartTime = useAppStore(s => s.prefillStartTime)
  const prefillTotalTokens = useAppStore(s => s.prefillTotalTokens)

  const hasSteps = !!(agentSteps && agentSteps.length > 0)
  const hasThinking = !!(thinkingContent && thinkingContent.length > 0)
  // agent rounds 中已展示思考阶段时，避免重复显示思考内容
  const hasAgentThinking = useMemo(() => {
    if (!hasSteps) return false
    return (agentSteps!).some(s => s.phase === 'thinking' && s.content)
  }, [agentSteps, hasSteps])
  const chatState = useAppStore(s => s.conversation.chatState)

  // ── Prefill progress: from backend prefill_progress callback ──
  const isPrefill = isStreaming && !hasSteps && !hasThinking && chatState !== 'streaming'
  const prefillCurrentTokens = useAppStore(s => s.prefillCurrentTokens)
  const prefillPct = prefillTotalTokens > 0
    ? Math.round((prefillCurrentTokens / prefillTotalTokens) * 100)
    : 0

  const rounds = useMemo(
    () => groupStepsIntoRounds(hasSteps ? agentSteps : null),
    [agentSteps, hasSteps],
  )

  // 纯手动切换，不自动折叠
  // Streaming 折叠: 滑动窗口流式文本（去换行，逐字流动）。非流/展开: 最后 3 行或 markdown。
  const displayThinking = useMemo(() => {
    if (!thinkingContent) return ''
    if (expanded) return thinkingContent
    if (isStreaming) {
      // 流式滑动窗口：去换行 + 空白合并，取最后 200 字
      const flat = thinkingContent.replace(/\s+/g, ' ')
      if (flat.length <= 200) return flat
      return '…' + flat.slice(-200)
    }
    // 非流式折叠：最后 3 行
    const lines = thinkingContent.split('\n').filter((l, i, arr) => {
      if (i === 0 || i === arr.length - 1) return l.trim()
      return true
    })
    const last3 = lines.slice(-3)
    let result = last3.join('\n').trim()
    if (!result) {
      const t = thinkingContent.replace(/\s+/g, ' ').trim()
      result = t.length > 200 ? '…' + t.slice(-200) : t
    }
    return result
  }, [thinkingContent, expanded, isStreaming])

  // 内容开始后自动折叠（但用户可以重新展开）
  const isAutoCollapsed = useMemo(() => {
    if (isStreaming && contentStarted && expanded) {
      // 用 setTimeout 不可取，直接通过 ref 追踪
      return false  // 已手动展开，不强制折叠
    }
    return false
  }, [isStreaming, contentStarted, expanded])

  // 内容开始后自动折叠 thinking 面板
  const [autoCollapsed, setAutoCollapsed] = useState(false)
  const prevContentStarted = useRef(contentStarted)
  useEffect(() => {
    if (contentStarted && !prevContentStarted.current && isStreaming) {
      // 正文刚开始输出时自动折叠，但保留展开按钮
      setAutoCollapsed(true)
    }
    prevContentStarted.current = contentStarted
  }, [contentStarted, isStreaming])

  // 实际展开状态：手动展开优先，未折叠时在正文前自动展示
  const actuallyExpanded = expanded || (!autoCollapsed && isStreaming && !contentStarted)

  // 展开/折叠处理
  const handleToggle = useCallback(() => {
    if (autoCollapsed) {
      setAutoCollapsed(false)
    }
    setExpanded(!expanded)
  }, [expanded, autoCollapsed])

  const stepNames: string[] = []
  if (hasSteps) {
    for (const s of agentSteps!) {
      if (s.phase === 'tool_call' && s.toolName) {
        stepNames.push(s.toolName)
      }
    }
  }

  // Prefill/speed info for the header
  const speedInfo: string[] = []
  if (prefillTokens && prefillTokens > 0) {
    speedInfo.push(`prefill ${prefillTokens} tok`)
  }
  if (prefillMs && prefillMs > 0) {
    speedInfo.push(prefillMs < 1000 ? `${prefillMs}ms` : `${(prefillMs / 1000).toFixed(1)}s`)
  }
  if (tokPerSec && tokPerSec > 0) {
    speedInfo.push(`${tokPerSec} tok/s`)
  }

  const summaryText = [
    hasSteps ? `${rounds.length} 轮决策` : isPrefill ? 'prefilling…' : isStreaming ? (hasThinking && !contentStarted ? '思考中…' : hasThinking ? `已思考 ${thinkingContent!.length} 字 · 点击展开` : '分析中…') : (hasThinking ? `已思考 ${thinkingContent!.length} 字 · 点击展开` : ''),
    stepNames.length > 0 ? stepNames.join(' → ') : '',
    speedInfo.length > 0 ? speedInfo.join(' ') : '',
  ].filter(Boolean).join(' · ')

  // Show panel while streaming OR if has content (steps/thinking) in committed message
  if (!hasSteps && !hasThinking && !isStreaming) return null
  // For committed messages, still show but auto-collapsed
  const showContent = actuallyExpanded || (isStreaming && !contentStarted && !autoCollapsed)

  const lastRoundIndex = rounds.length > 0 ? rounds[rounds.length - 1].roundIndex : -1

  return (
    <div style={{
      marginBottom: 8,
      border: '1px solid hsl(var(--border) / 0.6)',
      borderRadius: 10,
      background: 'hsl(var(--muted) / 0.30)',
      overflow: 'hidden',
      fontSize: 12,
      width: '100%',
      boxSizing: 'border-box',
    }}>
      {/* Header */}
      <div
        onClick={handleToggle}
        style={{
          display: 'flex', alignItems: 'center',
          padding: '8px 12px', gap: 8,
          cursor: 'pointer', userSelect: 'none',
        }}
      >
        <span style={{ fontSize: 11, color: 'hsl(var(--muted-foreground))' }}>
          {actuallyExpanded ? '▼' : '▶'}
        </span>
        <span style={{ fontWeight: 600, color: '#c792ea', fontSize: 12 }}>
          {'🧠 思考过程'}
        </span>
        <span style={{
          fontSize: 11, color: 'hsl(var(--foreground) / 0.45)',
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          flex: 1, minWidth: 0,
        }}>
          {summaryText}
        </span>
        {isStreaming && !contentStarted && (
          <span style={{
            marginLeft: 'auto', flexShrink: 0,
            width: 6, height: 6,
            borderRadius: '50%',
            background: '#c792ea',
            animation: 'pulse 1s infinite',
          }} />
        )}
      </div>

      {/* Expanded content */}
      {showContent && (
        <div style={{ padding: '0 12px 10px' }}>
          {/* Prefill progress */}
          {isPrefill && (
            <div style={{
              padding: '8px 0', display: 'flex', flexDirection: 'column', gap: 6,
            }}>
              <div style={{
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                fontSize: 11, color: 'hsl(var(--foreground) / 0.55)',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{
                    width: 8, height: 8, borderRadius: '50%',
                    background: '#c792ea', animation: 'pulse 0.8s infinite',
                  }} />
                  <span>预填充中…</span>
                </div>
                {prefillTotalTokens > 0 && (
                  <span style={{
                    fontWeight: 600, color: '#c792ea', fontFamily: 'monospace',
                    fontSize: 11, minWidth: 36, textAlign: 'right',
                  }}>
                    {prefillPct}%
                  </span>
                )}
              </div>
              <div style={{
                height: 4, borderRadius: 2, overflow: 'hidden',
                background: 'hsl(280 50% 30% / 0.12)',
              }}>
                {prefillTotalTokens > 0 ? (
                  <div style={{
                    height: '100%',
                    width: `${prefillPct}%`,
                    background: 'linear-gradient(90deg, #c792ea, #82aaff)',
                    borderRadius: 2,
                    transition: 'width 0.2s linear',
                  }} />
                ) : (
                  <div style={{
                    height: '100%', width: '40%',
                    background: 'linear-gradient(90deg, transparent, #c792ea, transparent)',
                    borderRadius: 2,
                    animation: 'shimmer 2s infinite',
                  }} />
                )}
              </div>
              {prefillTotalTokens > 0 && (
                <div style={{ fontSize: 10, color: 'hsl(var(--foreground) / 0.35)', textAlign: 'right' }}>
                  {prefillCurrentTokens.toLocaleString()} / {prefillTotalTokens.toLocaleString()} tokens
                </div>
              )}
            </div>
          )}

          {/* Agent decision rounds */}
          {rounds.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {rounds.map((round) => {
                const isCurrentRound = isStreaming && round.roundIndex === lastRoundIndex

                return (
                  <div key={round.roundIndex} style={{
                    border: isCurrentRound
                      ? '1px solid hsl(280 50% 30% / 0.25)'
                      : '1px solid hsl(var(--border) / 0.20)',
                    borderRadius: 8,
                    background: isCurrentRound
                      ? 'hsl(280 50% 30% / 0.06)'
                      : 'hsl(var(--muted) / 0.25)',
                    overflow: 'hidden',
                  }}>
                    <div style={{
                      display: 'flex', alignItems: 'center', gap: 6,
                      padding: '5px 10px',
                      background: 'hsl(var(--muted) / 0.30)',
                      borderBottom: '1px solid hsl(var(--border) / 0.15)',
                    }}>
                      <span style={{
                        fontSize: 10, fontWeight: 700,
                        color: 'hsl(var(--muted-foreground))',
                        background: isCurrentRound
                          ? 'hsl(280 50% 30% / 0.15)'
                          : 'hsl(var(--muted))',
                        padding: '1px 6px', borderRadius: 4,
                      }}>
                        第 {round.roundIndex + 1} 轮
                      </span>
                      {isCurrentRound && (
                        <>
                          <span style={{ width: 4, height: 4, borderRadius: '50%', background: '#c792ea', animation: 'pulse 1s infinite' }} />
                          <span style={{ fontSize: 10, color: '#c792ea', fontWeight: 500 }}>进行中</span>
                        </>
                      )}
                      {round.hasToolCall && !isCurrentRound && (
                        <span style={{ fontSize: 10, color: 'hsl(var(--foreground) / 0.35)' }}>完成</span>
                      )}
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 4, padding: '6px 8px' }}>
                      {round.steps.map((step, i) => {
                        const cfg = PHASE_CONFIG[step.phase] || PHASE_CONFIG.thinking

                        return (
                          <div key={i} style={{
                            padding: '5px 8px', borderRadius: 6,
                            background: 'hsl(var(--muted) / 0.50)',
                            border: '1px solid hsl(var(--border) / 0.25)',
                            fontSize: 11, lineHeight: 1.6,
                          }}>
                            <div style={{
                              display: 'flex', alignItems: 'center', gap: 5,
                              marginBottom: (step.content || step.toolName || step.toolResult) ? 3 : 0,
                            }}>
                              <span style={{ fontSize: 13 }}>{cfg.icon}</span>
                              <span style={{ fontWeight: 600, color: cfg.color, fontSize: 11 }}>{cfg.label}</span>
                            </div>
                            {step.phase === 'thinking' && step.content && (
                              <div style={{ color: 'hsl(var(--foreground) / 0.75)', fontSize: 11 }}>
                                {step.content}
                              </div>
                            )}
                            {step.phase === 'tool_call' && (
                              <ToolCallBlock
                                toolName={step.toolName || ''}
                                toolArgs={step.toolArgs}
                                isPending={isStreaming && isCurrentRound}
                              />
                            )}
                            {step.phase === 'tool_result' && step.toolResult && (
                              <ToolCallBlock
                                toolName={step.toolName || ''}
                                toolResult={step.toolResult}
                                isError={step.success === false}
                                durationMs={step.durationMs}
                              />
                            )}
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )
              })}
            </div>
          )}

          {/* Deep thinking content: only when agent rounds don't already show it */}
          {hasThinking && !hasSteps && (!contentStarted || !isStreaming || actuallyExpanded) && (
            <div style={{
              marginTop: rounds.length > 0 ? 8 : 0,
              padding: '6px 10px',
              borderRadius: 8,
              background: 'hsl(280 50% 30% / 0.08)',
              border: '1px solid hsl(280 50% 30% / 0.15)',
              width: '100%',
              boxSizing: 'border-box',
            }}>
              <div style={{
                fontSize: 10, fontWeight: 600,
                color: '#c792ea', marginBottom: 3,
              }}>
                {!isStreaming ? '💭 思考内容' : (contentStarted ? '💭 思考完成' : '💭 思考中')}
              </div>
              {expanded ? (
                <div style={{
                  color: 'hsl(var(--foreground) / 0.65)',
                  fontSize: 11, lineHeight: 1.6,
                  wordBreak: 'break-word',
                  maxHeight: isStreaming && !contentStarted ? '3em' : 'none',
                  overflowY: isStreaming && !contentStarted ? 'hidden' : 'auto',
                }}
                className="markdown-content"
                >
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {displayThinking}
                  </ReactMarkdown>
                </div>
              ) : (
                <div style={{
                  color: 'hsl(var(--foreground) / 0.65)',
                  fontSize: 11, lineHeight: 1.6,
                  fontFamily: 'monospace',
                  ...(isStreaming && !contentStarted ? {
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'clip',
                  } : {
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                  }),
                }}>
                  {displayThinking}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function formatToolArgs(jsonStr: string): string {
  try {
    return JSON.stringify(JSON.parse(jsonStr), null, 2)
  } catch {
    return jsonStr
  }
}

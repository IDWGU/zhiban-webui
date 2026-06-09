import { useEffect, useRef, useCallback } from 'react'
import { useAppStore } from '@/stores/appStore'
import type { WsServerMessage, ImportVectorResultMessage } from '@/types'

// ── 连接参数（从 window.location 动态推导） ──
function getConnectionParams() {
  const port = window.location.port || '18921'
  const host = window.location.hostname || '127.0.0.1'
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const httpProto = window.location.protocol || 'http:'
  return {
    wsUrl: `${proto}//${host}:${port}/ws`,
    readyUrl: `${httpProto}//${host}:${port}/ready`,
    httpBase: `${httpProto}//${host}:${port}`,
    host,
    port,
  }
}

const RECONNECT_DELAYS = [1, 2, 4, 8, 15, 30, 30, 30]
const READY_POLL_INTERVAL_MS = 500
const READY_POLL_TIMEOUT_MS = 90_000
const WS_CONNECT_TIMEOUT_MS = 10_000
const MAX_RECONNECT_ATTEMPTS = 20
const PONG_GRACE_PERIOD_MS = 60_000
const PONG_CHECK_INTERVAL_MS = 10_000

// ── 模块级共享状态（handleServerMessage 和 hook 都需访问） ──
let _lastPongTimestamp = Date.now()
let _initialConvSyncDone = false
function _updatePongTs() { _lastPongTimestamp = Date.now() }

// ── </think> 分流状态（模块级，跨消息保持）──
// 后端将所有 reasoning 包裹为 <think>...</think> 标签统一在 content 流中输出。
// 前端仅通过 </think> 标签分离思考/正文，无需 isThinking flag。
let _thinkBuf = ""               // 思考内容缓冲区（保留最后 30 字符防 </think> 跨 token）
let _thinkClosed = false         // </think> 已检测到
const _THINK_SAFE_MARGIN = 30    // 缓冲区保留字符数

function _resetThinkSplit() {
  _thinkBuf = ""
  _thinkClosed = false
}

function _stripThinkTags(text: string): string {
  return text
    .replace(/<\s*(?:\/?\s*think|\/?\s*opti-q-think)[^>]*>/gi, '')
    .trim()
}

// ── 流式停滞检测：超时无新 token → 强制结束 ──
// Agent 模式有工具调用间隙，超时更长（60s vs 15s）
// 总结类查询可能生成很长回答，token间隔更稀疏，进一步延长
let _streamStallTimer: ReturnType<typeof setTimeout> | null = null
let _lastStreamTokenTime = 0
let _isAgentMode = false
let _isSummaryQuery = false

function _resetStreamingCaches() {
  try {
    const { resetCitationCache, resetSafeCutCache } = require('@/components/conversation/StreamingMarkdown')
    resetCitationCache()
    resetSafeCutCache()
  } catch {}
}

function _getStallTimeout(): number {
  if (_isSummaryQuery) return 90_000  // 总结类查询：90s
  return _isAgentMode ? 60_000 : 30_000  // Agent: 60s, 普通: 30s (was 15s)
}

function _resetStallTimer() {
  _lastStreamTokenTime = Date.now()
  if (_streamStallTimer) clearTimeout(_streamStallTimer)
  const timeout = _getStallTimeout()
  _streamStallTimer = setTimeout(() => {
    const store = useAppStore.getState()
    if (!store.conversation.isStreaming) return
    if (Date.now() - _lastStreamTokenTime < timeout) return
    const conv = store.conversation
    // Agent 模式下有思考内容但无正文 → 可能在执行工具，延长超时，不强制结束
    if (_isAgentMode && !conv.streamingText && (conv.streamingThinkingText || conv.streamingAgentSteps.length > 0)) {
      console.warn('[ws] Agent still working (thinking=%d steps=%d), extending timeout',
        conv.streamingThinkingText.length, conv.streamingAgentSteps.length)
      _resetStallTimer()
      return
    }
    console.warn('[ws] Streaming stalled for %ds, force-finishing', Math.round(timeout / 1000))
    store.finishStreaming(conv.streamingMessageId || 'stalled', {
      mode: '', model: '', usage: null, error: '流式输出超时',
    })
    store.setQueryRunning(false)
  }, timeout)
}

function _clearStallTimer() {
  _isAgentMode = false
  _isSummaryQuery = false
  if (_streamStallTimer) { clearTimeout(_streamStallTimer); _streamStallTimer = null }
}

// 标记当前查询为全文总结类（延长 stall timeout）
export function markSummaryQuery() {
  _isSummaryQuery = true
}

// ── 全局诊断信息（window.__zhiban_diag 暴露给调试） ──
type DiagEntry = { ts: number; event: string; detail?: string }
const _diagLog: DiagEntry[] = []
function _diag(event: string, detail?: string) {
  const entry: DiagEntry = { ts: Date.now(), event, detail }
  _diagLog.push(entry)
  if (_diagLog.length > 200) _diagLog.shift()
}

export function useWebSocket() {
  // ── refs（跨 render 保持，避免闭包陷阱） ──
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>()
  const reconnectAttempt = useRef(0)
  const sendBuffer = useRef<string[]>([])
  const shownDisconnectNotice = useRef(false)
  const connectTimer = useRef<ReturnType<typeof setTimeout>>()

  // 每次调用 getConnectionParams() 确保取到最新值
  const drainBuffer = useCallback(() => {
    while (sendBuffer.current.length > 0 && wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(sendBuffer.current.shift()!)
    }
  }, [])

  // ── 等待后端就绪（轮询 /ready） ──
  const waitForReady = useCallback((): Promise<boolean> => {
    _diag('waitForReady:start')
    return new Promise((resolve) => {
      const { readyUrl, httpBase } = getConnectionParams()
      const start = Date.now()
      const store = useAppStore.getState()

      const poll = () => {
        const elapsed = Date.now() - start
        if (elapsed > READY_POLL_TIMEOUT_MS) {
          _diag('waitForReady:timeout', `${Math.round(elapsed / 1000)}s`)
          resolve(false)
          return
        }

        // 同时轮询启动状态（展示给用户）
        fetch(`${httpBase}/startup-status`)
          .then(r => r.json())
          .then(d => { if (d.message) store.setStartupMessage(d.message) })
          .catch(() => {})

        fetch(readyUrl)
          .then((r) => r.json())
          .then((data) => {
            if (data.ready === true) {
              _diag('waitForReady:ok', `${Math.round(elapsed / 1000)}s`)
              store.setStartupMessage('')
              resolve(true)
            } else {
              setTimeout(poll, READY_POLL_INTERVAL_MS)
            }
          })
          .catch((err) => {
            // 网络错误 — 后端可能还在启动，继续轮询
            setTimeout(poll, READY_POLL_INTERVAL_MS)
          })
      }

      poll()
    })
  }, [])

  // ── 消息路由 ──
  const handleMessage = useCallback((msg: WsServerMessage) => {
    handleServerMessage(msg)
  }, [])

  // ── 建立 WebSocket 连接 ──
  const connect = useCallback(() => {
    const existing = wsRef.current
    if (existing?.readyState === WebSocket.OPEN) return
    if (existing?.readyState === WebSocket.CONNECTING) return

    const { wsUrl } = getConnectionParams()
    const store = useAppStore.getState()
    store.setWsStatus('connecting')
    _diag('connect', wsUrl)

    const ws = new WebSocket(wsUrl)

    // 连接超时
    connectTimer.current = setTimeout(() => {
      if (wsRef.current !== ws) return
      if (ws.readyState !== WebSocket.OPEN) {
        _diag('connect:timeout', '10s')
        ws.close()
      }
    }, WS_CONNECT_TIMEOUT_MS)

    ws.addEventListener('message', (event: MessageEvent) => {
      try { handleMessage(JSON.parse(event.data)) } catch {}
    })

    ws.addEventListener('open', () => {
      if (connectTimer.current) { clearTimeout(connectTimer.current); connectTimer.current = undefined }
      const s = useAppStore.getState()
      s.setWsStatus('connected')
      s.setSendMessage(send)
      s.setReconnectAttempt(0)
      s.setQueryRunning(false)
      reconnectAttempt.current = 0
      _lastPongTimestamp = Date.now()
      _initialConvSyncDone = false
      shownDisconnectNotice.current = false
      if (s.buildIndex.phase === 'error') {
        s.setBuildIndexResult(true, 0, '')
      }
      _diag('ws:open')
      drainBuffer()
      s.sendMessage?.({ type: 'list_conversations' })
    })

    ws.addEventListener('close', (event) => {
      if (wsRef.current !== ws) return
      if (connectTimer.current) { clearTimeout(connectTimer.current); connectTimer.current = undefined }
      _diag('ws:close', `code=${event.code} reason=${event.reason || '(none)'}`)
      const s = useAppStore.getState()
      s.setQueryRunning(false)
      const buildingPhases = new Set(['scanning', 'extracting', 'embedding', 'paused'])
      if (buildingPhases.has(s.buildIndex.phase)) {
        s.setBuildIndexResult(false, 0, '后端进程意外终止，请重启')
      }
      scheduleReconnect()
    })

    ws.addEventListener('error', (event) => {
      if (connectTimer.current) { clearTimeout(connectTimer.current); connectTimer.current = undefined }
      _diag('ws:error', 'WebSocket error event')
    })

    wsRef.current = ws
  }, [])

  // ── 自动重连（阶梯回退） ──
  const scheduleReconnect = useCallback(() => {
    if (reconnectTimer.current) clearTimeout(reconnectTimer.current)

    if (reconnectAttempt.current >= MAX_RECONNECT_ATTEMPTS) {
      const store = useAppStore.getState()
      store.setWsStatus('disconnected')
      _diag('reconnect:giveup', `after ${MAX_RECONNECT_ATTEMPTS} attempts`)
      store.pushNotification('error', '重连失败次数过多，请检查知伴后端是否正在运行')
      return
    }

    const store = useAppStore.getState()
    if (!shownDisconnectNotice.current) {
      shownDisconnectNotice.current = true
      store.pushNotification('warn', '与知伴后端的连接已断开，正在尝试重连...')
    }
    store.setWsStatus('reconnecting')
    store.setReconnectAttempt(reconnectAttempt.current + 1)

    const delayIdx = Math.min(reconnectAttempt.current, RECONNECT_DELAYS.length - 1)
    const delay = RECONNECT_DELAYS[delayIdx]
    _diag('reconnect:schedule', `attempt=${reconnectAttempt.current + 1} delay=${delay}s`)
    reconnectAttempt.current++
    reconnectTimer.current = setTimeout(() => connect(), delay * 1000)
  }, [connect])

  // ── 发送消息（带缓冲） ──
  const send = useCallback((data: unknown) => {
    const json = JSON.stringify(data)
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(json)
    } else {
      if (sendBuffer.current.length < 50) {
        sendBuffer.current.push(json)
      } else {
        const s = useAppStore.getState()
        if (!s.notifications.some(n => n.message.includes('消息发送队列已满'))) {
          s.pushNotification('warn', '消息发送队列已满，请等待连接恢复后重试')
        }
      }
    }
  }, [])

  // ── 手动断开 ──
  const disconnect = useCallback(() => {
    if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
    if (connectTimer.current) clearTimeout(connectTimer.current)
    wsRef.current?.close()
    wsRef.current = null
    useAppStore.getState().setWsStatus('disconnected')
    _diag('disconnect:manual')
  }, [])

  // ── 主 effect：初始化连接 + 心跳 ──
  useEffect(() => {
    let cancelled = false
    let fallbackTimer: ReturnType<typeof setTimeout> | null = null

    const doConnect = () => {
      if (cancelled) return
      if (fallbackTimer) { clearTimeout(fallbackTimer); fallbackTimer = null }
      connect()
    }

    _diag('init:start')
    useAppStore.getState().setWsStatus('waiting')

    // Electron: 主进程通过 IPC 直接通知 sidecar 端口就绪，无需 HTTP poll
    const api = (window as any).electronAPI
    let unsubReady: (() => void) | null = null
    if (api?.onSidecarReady) {
      _diag('init:using_ipc_ready')
      unsubReady = api.onSidecarReady(() => {
        _diag('init:ipc_ready_received')
        doConnect()
      })
    }

    // HTTP poll 作为兜底（Electron 立即启动，非 Electron 也立即启动）
    const pollDelay = 0
    const startPoll = async () => {
      const ready = await waitForReady()
      if (cancelled) return
      if (ready) {
        doConnect()
      } else {
        _diag('init:ready_timeout')
        // 报错前先检查 WebSocket 是否其实已经连上（IPC 通知可能在我们 polling 期间到达）
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          _diag('init:ws_already_connected_on_timeout')
          return
        }
        useAppStore.getState().setWsStatus('disconnected')
        useAppStore.getState().pushNotification(
          'error',
          '知伴后端启动超时，请检查服务是否正常运行。您可以尝试重启应用。'
        )
      }
    }
    fallbackTimer = setTimeout(startPoll, pollDelay)

    return () => {
      cancelled = true
      unsubReady?.()
      if (fallbackTimer) clearTimeout(fallbackTimer)
    }

    // 心跳：每 30s 发 ping
    const pingInterval = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }))
      }
    }, 30000)

    // 心跳检测：60s 无 pong → 断开重连
    const pongCheck = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        if (Date.now() - _lastPongTimestamp > PONG_GRACE_PERIOD_MS) {
          _diag('pong:timeout', `${PONG_GRACE_PERIOD_MS / 1000}s no response`)
          wsRef.current.close()
        }
      }
    }, PONG_CHECK_INTERVAL_MS)

    return () => {
      cancelled = true
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
      if (connectTimer.current) clearTimeout(connectTimer.current)
      clearInterval(pingInterval)
      clearInterval(pongCheck)
    }
  }, [waitForReady, connect])

  return { send, disconnect, reconnect: connect }
}

// ── 服务端消息处理 ──
function handleServerMessage(msg: WsServerMessage) {
  const store = useAppStore.getState()

  switch (msg.type) {
    case 'pong':
      _updatePongTs()
      store.setSidecarStatus('running')
      break

    case 'llm_token':
      if ((msg as any).conversationId && (msg as any).conversationId !== store.activeConversationId) break
      if (msg.isFirst && !store.conversation.isStreaming) {
        store.startStreaming(msg.messageId)
        _resetThinkSplit()
      }
      // 所有 token 统一通过 content 流到达。后端的 reasoning 已包裹
      // 为 <think>...</think> 标签。前端只靠 </think> 标签来分离思考/正文。
      // msg.isThinking 仅用于老版本兼容，新版本完全不依赖它。
      if (msg.isThinking) {
        // 老版本兼容：后端显式标记的思考 token
        _isAgentMode = true
        store.setChatState('streaming')
        store.appendThinkingToken(msg.token)
      } else if (_thinkClosed) {
        // </think> 之后：正文
        store.setChatState('streaming')
        store.appendStreamingToken(msg.token)
      } else {
        // 还未检测到 </think>：累积 + 流式推送思考
        store.setChatState('streaming')
        _thinkBuf += msg.token

        const closeIdx = _thinkBuf.indexOf('</think>')
        if (closeIdx >= 0) {
          // 检测到 </think>：分离
          const thinkingPart = _thinkBuf.slice(0, closeIdx)   // 不含 </think>
          const contentPart = _thinkBuf.slice(closeIdx + 7)    // </think> 之后

          if (thinkingPart) store.appendThinkingToken(thinkingPart)

          _thinkClosed = true
          _thinkBuf = ""
          if (contentPart) store.appendStreamingToken(contentPart)
        } else {
          // 安全 emit 缓冲前缀（留着尾部防 </think> 跨 token）
          const safeLen = Math.max(0, _thinkBuf.length - _THINK_SAFE_MARGIN)
          if (safeLen > 0) {
            const safePart = _thinkBuf.slice(0, safeLen)
            _thinkBuf = _thinkBuf.slice(safeLen)
            store.appendThinkingToken(safePart)
          }
        }
      }
      _resetStallTimer()
      break

    case 'agent_step': {
      if ((msg as any).conversationId && (msg as any).conversationId !== store.activeConversationId) break
      if (!store.conversation.isStreaming) {
        store.startStreaming((msg as any).messageId || 'agent-streaming')
        _resetThinkSplit()
      }
      _isAgentMode = true
      store.setChatState('streaming')

      // 将 agent 步骤格式化为内联文本，直接流入气泡
      let stepText = ''
      if (msg.phase === 'thinking') {
        const content = msg.content || ''
        if (content.trim()) {
          stepText = `\n\n> 💭 ${content.trim()}\n\n`
        }
      } else if (msg.phase === 'tool_call') {
        const toolName = msg.toolName || '工具'
        const args = msg.toolArgs ? (() => {
          try { return JSON.parse(msg.toolArgs) } catch { return {} }
        })() : {}
        const query = args.query || ''
        const label = toolName === 'search_knowledge_base' ? '搜索知识库'
          : toolName === 'get_paper_section' ? '读取论文章节'
          : toolName === 'get_reading_context' ? '获取阅读位置'
          : toolName
        if (query) {
          stepText = `\n> 🔧 ${label}: \`${query.slice(0, 60)}${query.length > 60 ? '...' : ''}\`\n`
        } else {
          stepText = `\n> 🔧 ${label}\n`
        }
      } else if (msg.phase === 'tool_result') {
        const dur = msg.durationMs ? ` ${msg.durationMs}ms` : ''
        const ok = msg.success !== false
        stepText = `\n> ${ok ? '✅' : '❌'} 完成${dur}\n\n`
      }
      if (stepText) {
        store.appendStreamingToken(stepText)
      }
      _resetStallTimer()
      break
    }

    case 'agent_thinking':
      if ((msg as any).conversationId && (msg as any).conversationId !== store.activeConversationId) break
      // 思考 token → 思考面板
      store.appendThinkingToken(msg.token)
      _resetStallTimer()
      break

    case 'agent_thinking_done':
      // thinking stream complete — no action needed, content is already in store
      break

    case 'llm_citation':
      if ((msg as any).conversationId && (msg as any).conversationId !== store.activeConversationId) break
      msg.citations.forEach(c => store.addStreamingCitation(c))
      break

    case 'llm_related_papers':
      if ((msg as any).conversationId && (msg as any).conversationId !== store.activeConversationId) break
      msg.papers.forEach(p => store.addStreamingRelatedPaper(p))
      break

    case 'llm_done':
      if ((msg as any).conversationId && (msg as any).conversationId !== store.activeConversationId) break
      // 回退：输出完成时 </think> 从未出现 → 清空缓冲区到正文
      if (!_thinkClosed && _thinkBuf) {
        const remaining = _stripThinkTags(_thinkBuf)
        if (remaining) store.appendStreamingToken(remaining)
        _thinkBuf = ""
        _thinkClosed = true
      }
      // 如果 streaming 文本为空但有错误信息，注入错误提示
      if ((msg as any).error && !store.conversation.streamingText) {
        store.appendStreamingToken(`（${(msg as any).error}）`)
      }
      if ((msg as any).cancelled && !store.conversation.streamingText) {
        store.appendStreamingToken('（查询已取消）')
      }
      store.finishStreaming(msg.messageId, {
        mode: (msg as any).mode || '',
        model: (msg as any).model || '',
        usage: (msg as any).usage || null,
        loopDetected: (msg as any).loopDetected ?? false,
        error: (msg as any).error || '',
        cancelled: (msg as any).cancelled ?? false,
      })
      store.setQueryRunning(false)
      store.setChatState('idle')
      store.setLocalSpeed({ prefillTokens: 0, prefillMs: 0, tokPerSec: 0, phase: '' })
      if ((msg as any).loopDetected) {
        store.pushNotification('warn', '模型回复出现重复，已自动重试')
      }
      const usage = (msg as any).usage || {}
      if (usage.input || usage.output) {
        store.addSessionUsage(usage.input || 0, usage.output || 0)
      }
      store.setLastResponseMeta({
        refused: (msg as any).refused ?? false,
        expanded: (msg as any).expanded ?? false,
        responseType: (msg as any).responseType ?? '',
      })
      // 重置 citation/safe-cut 缓存
      _resetStreamingCaches()
      _clearStallTimer()
      break

    case 'llm_health':
      if ((msg as any).conversationId && (msg as any).conversationId !== store.activeConversationId) break
      store.addHealthRecord({
        call: msg.call,
        timing: msg.timing,
        tokens: msg.tokens,
        memory: msg.memory,
        timestamp: msg.timestamp,
      })
      const decodeMs = msg.timing.decode_per_token_ms
      const tokPerSec = decodeMs && decodeMs > 0 ? Math.round(1000 / decodeMs) : 0
      store.setLocalSpeed({
        prefillTokens: msg.tokens.prefill_tokens || 0,
        prefillMs: Math.round(msg.timing.prefill_ms || 0),
        tokPerSec,
        phase: msg.call,
      })
      break

    case 'status':
      if (msg.level === 'error') {
        store.pushNotification('error', msg.message || '服务器错误')
        store.setQueryRunning(false)
        store.setLocalSpeed({ prefillTokens: 0, prefillMs: 0, tokPerSec: 0, phase: '' })
        if (msg.code === 'empty_query' || msg.code === 'rag_error' || msg.code === 'llm_error') {
          store.finishStreaming('error')
        }
        if (msg.code === 'translation_error') {
          store.setTranslationError(msg.message || '翻译失败')
        }
      } else if (msg.level === 'warn') {
        store.pushNotification('warn', msg.message || '警告')
      } else if (msg.level === 'info') {
        if (msg.code === 'loading_model') store.setLocalEngineLoading(true)
        if (msg.code === 'translation_extracting' || msg.code === 'translation_prefilling') {
          store.setTranslationStatusMsg(msg.message || '')
        }
        if (!msg.code.startsWith('translation_')) {
          store.setWorkflowStatus(msg.code || 'processing', msg.message || '')
        }
      }
      break

    case 'workflow_status':
      if ((msg as any).conversationId && (msg as any).conversationId !== store.activeConversationId) break
      store.setWorkflowStatus((msg as any).code || 'processing', (msg as any).message || '')
      store.setQueryRunning(true)
      // 首条 workflow_status 触发 streaming 状态，确保 prefill 阶段面板可见
      if (!store.conversation.isStreaming) {
        store.startStreaming((msg as any).conversationId ? `msg_${Date.now()}` : 'prefill-stream')
        _resetThinkSplit()
      }
      // prefill_start: backend sends total token count as plain number string
      if ((msg as any).code === 'prefill_start' && (msg as any).message) {
        const total = parseInt((msg as any).message, 10)
        if (total > 0) store.setPrefillStart(total)
      }
      // prefill_progress: backend sends { current: N, total: M } for real-time updates
      if ((msg as any).code === 'prefill_progress') {
        try {
          const info = typeof (msg as any).message === 'string'
            ? JSON.parse((msg as any).message) : (msg as any).message
          if (info && typeof info.total === 'number' && info.total > 0) {
            store.setPrefillProgress(info.current ?? 0, info.total)
          }
        } catch {}
      }
      break

    case 'conversation_list':
      store.setConversations(msg.conversations)
      if (msg.conversations.length > 0 && !_initialConvSyncDone) {
        _initialConvSyncDone = true
        const s = useAppStore.getState()
        const exists = msg.conversations.some((c: any) => c.id === s.activeConversationId)
        const targetId = exists ? s.activeConversationId : msg.conversations[0].id
        if (!exists) s.setActiveConversationId(targetId)
        const sendFn = (window as any).__zhiban_wsSend
        if (sendFn) sendFn({ type: 'switch_conversation', conversationId: targetId })
      }
      break

    case 'conversation_created': {
      const convId = (msg as any).conversationId || ''
      store.addConversation({
        id: convId, name: (msg as any).name || '新对话',
        messageCount: 0, paperCount: 0, topic: '', isActive: false,
      })
      break
    }

    case 'conversation_switched':
      store.syncConversationMessages(
        (msg as any).conversationId || '',
        ((msg as any).messages || []).filter((m: any) => m.role === 'user' || m.role === 'assistant').map((m: any) => ({
          id: crypto.randomUUID(),
          role: m.role as 'user' | 'assistant',
          content: m.content,
          timestamp: m.timestamp || Date.now(),
          mode: m.mode || '', model: m.model || '',
        })),
        (msg as any).openPapers || [],
        (msg as any).currentTopic || '',
      )
      break

    case 'conversation_renamed':
      store.updateConversation((msg as any).conversationId || '', { name: (msg as any).name || '' })
      break

    case 'conversation_branched':
      store.pushNotification('success', `已创建分支对话: ${(msg as any).name || ''}`)
      store.setActiveConversationId((msg as any).conversationId || '')
      const sendFn2 = (window as any).__zhiban_wsSend
      if (sendFn2) sendFn2({ type: 'switch_conversation', conversationId: (msg as any).conversationId })
      break

    case 'message_deleted':
      // 后端广播了 conversation_switched 来刷新消息列表，
      // 前端 syncConversationMessages 会更新 store
      if ((msg as any).conversationId === store.activeConversationId) {
        const idx = (msg as any).messageIndex
        if (idx >= 0) {
          const msgs = store.conversation.messages.filter((_: any, i: number) => i !== idx)
          store.syncConversationMessages(
            store.activeConversationId, msgs,
            [], store.conversation.messages[0]?.screenContext?.docName || '',
          )
        }
      }
      break

    case 'translation_blocks':
      if (!msg.blocks || !Array.isArray(msg.blocks)) break
      store.setTranslationBlocks(msg.blocks, msg.totalSentences)
      break

    case 'translation_token':
      store.appendTranslationToken(msg.sentenceId, msg.token, msg.isFirst)
      const ftm = (msg as any).firstTokenMs
      const tps = (msg as any).tokensPerSec
      const el = (msg as any).elapsed
      if (ftm !== undefined || tps !== undefined) {
        const ts = useAppStore.getState().translation
        store.setTranslationSpeed(tps ?? ts.tokensPerSec, el ?? ts.elapsed, ftm ?? ts.firstTokenMs)
      }
      break

    case 'translation_done':
      store.finishTranslation(msg.totalBlocks, msg.totalSentences)
      break

    case 'file_identity_result':
      if (msg.sha256) {
        store.setFileIdentity(msg.filePath, msg.sha256, msg.size || 0)
      } else if (msg.error) {
        const t = useAppStore.getState().translation
        if (t.translatedFilePath === msg.filePath) {
          useAppStore.setState((s: any) => ({
            translation: { ...s.translation, fileIdentityPending: false }
          }))
        }
      }
      break

    case 'llm_test_result':
      store.setLlmTestResult(msg.success, msg.error || '', msg.model || '')
      break

    case 'llm_models_result':
      if (msg.success) {
        const entries: any[] = (msg as any).model_entries || msg.models
        if (entries) {
          store.setAvailableModels(entries.map((m: any) =>
            typeof m === 'string' ? { name: m, path: '' } : m
          ))
        }
      }
      store.setLlmTestResult(msg.success, msg.error || '', '')
      break

    case 'import_vector_result':
      window.dispatchEvent(new CustomEvent('import-vector-result', {
        detail: { success: msg.success, chunks: msg.chunks, error: msg.error },
      }))
      break

    case 'import_paper_progress':
      store.setImportPaperProgress?.(msg.phase, msg.message, msg.progress)
      break

    case 'import_paper_exists':
      store.pushNotification('warn', `论文已存在 (${msg.chunks} chunks)，将覆盖重新索引`)
      break

    case 'import_paper_result':
      if (msg.success) {
        const dupInfo = msg.duplicate
          ? `（已存在，跳过向量化，已有 ${msg.chunks} chunks）`
          : `（新增 ${msg.chunks} chunks，doc_id: ${msg.doc_id}）`
        store.pushNotification('success', `论文导入成功 ${dupInfo}`)
      } else {
        store.pushNotification('error', msg.error || '论文导入失败')
      }
      break

    case 'build_index_progress':
      store.setBuildIndexProgress(msg.phase, msg.current, msg.total, msg.message)
      break

    case 'build_index_result': {
      store.setBuildIndexResult(msg.success, msg.chunks, msg.error)
      const { httpBase } = getConnectionParams()
      fetch(`${httpBase}/health`)
        .then(r => r.json())
        .then(d => window.dispatchEvent(new CustomEvent('health-update', {
          detail: { chunks: d.vectors_chunks ?? 0, graphs: d.graphs_papers ?? 0 },
        })))
        .catch(() => {})
      break
    }

    case 'add_papers_result':
      if (msg.success && msg.library) {
        window.dispatchEvent(new CustomEvent('library-update', { detail: { papers: msg.library } }))
      }
      break

    case 'list_library_result':
      window.dispatchEvent(new CustomEvent('library-update', { detail: { papers: msg.papers } }))
      break

    case 'embedding_progress':
      store.setEmbeddingLoadProgress(msg.percent, msg.message)
      break

    case 'clear_vector_result':
      if (msg.success) {
        store.pushNotification('info', '向量库已清空')
        window.dispatchEvent(new CustomEvent('health-update', { detail: { chunks: 0 } }))
      } else {
        store.pushNotification('error', msg.error || '清空失败')
      }
      break

    case 'remove_paper_result':
      if (msg.success) {
        store.pushNotification('info', msg.message || `已删除 ${msg.removed} 条向量`)
        const { httpBase: hb } = getConnectionParams()
        fetch(`${hb}/health`).then(r => r.json()).then(d => {
          window.dispatchEvent(new CustomEvent('health-update', {
            detail: { chunks: d.vectors_chunks ?? 0, graphs: d.graphs_papers ?? 0 },
          }))
        }).catch(() => {})
        window.dispatchEvent(new CustomEvent('refresh-indexed-papers'))
      } else {
        store.pushNotification('error', msg.error || '删除失败')
      }
      break

    case 'indexed_papers_result':
      window.dispatchEvent(new CustomEvent('indexed-papers-update', {
        detail: { papers: msg.papers || [] },
      }))
      break

    case 'delete_library_result':
      if (msg.success) {
        store.pushNotification('info', `已删除 ${msg.deleted} 篇论文`)
        const sendMsg = (window as any).__zhiban_wsSend
        if (sendMsg) sendMsg({ type: 'list_library' })
      } else {
        store.pushNotification('error', msg.error || '删除失败')
      }
      break

    case 'model_config_result': {
      window.dispatchEvent(new CustomEvent('model-config-result', { detail: msg }))
      if (msg.action === 'get' && msg.config) {
        if (msg.config.debug !== undefined) store.setDebugMode(msg.config.debug)
        if (msg.config.local_engine_loading !== undefined) store.setLocalEngineLoading(msg.config.local_engine_loading)
        store.updateSettings({
          modelCacheDir: msg.config.model_cache_dir || '',
          ...(msg.config.llm_model_path ? { llmModelPath: msg.config.llm_model_path } : {}),
          ...(msg.config.translation_model_path ? { translationModelPath: msg.config.translation_model_path } : {}),
          ...(msg.config.embedding_model ? { embeddingModel: msg.config.embedding_model } : {}),
          ...(msg.config.llm_flash_attn !== undefined ? { llmFlashAttn: msg.config.llm_flash_attn } : {}),
          ...(msg.config.llm_use_mmap !== undefined ? { llmUseMmap: msg.config.llm_use_mmap } : {}),
          ...(msg.config.llm_n_batch !== undefined ? { llmNBatch: msg.config.llm_n_batch } : {}),
          ...(msg.config.llm_n_ubatch !== undefined ? { llmNUbatch: msg.config.llm_n_ubatch } : {}),
        })
      }
      if (msg.action === 'set_local_model') {
        store.setLocalEngineLoading(false)
        if (msg.success) {
          store.pushNotification('info', msg.status || '模型路径已更新')
          if (msg.path) store.updateSettings({ llmModelPath: msg.path })
        } else {
          store.pushNotification('error', msg.error || '模型加载失败')
        }
      }
      if (msg.action === 'set_debug' && msg.enabled !== undefined) store.setDebugMode(msg.enabled)
      if (msg.action === 'set_llm_params') {
        store.pushNotification(msg.success ? 'info' : 'error', msg.success ? (msg.status || 'LLM 参数已更新') : (msg.error || 'LLM 参数更新失败'))
      }
      if (msg.action === 'set_embedding_model') {
        if (msg.success) {
          store.pushNotification('info', `嵌入模型已加载，维度: ${msg.dim || '?'}`)
          store.setEmbeddingLoadResult({ success: true, dim: msg.dim })
        } else {
          store.pushNotification('error', msg.error || '嵌入模型加载失败')
          store.setEmbeddingLoadResult({ success: false, error: msg.error })
        }
      }
      break
    }
  }
}

// ── 暴露诊断信息到全局 ──
if (typeof window !== 'undefined') {
  (window as any).__zhiban_diag = {
    get log() { return [..._diagLog] },
    get last() { return _diagLog[_diagLog.length - 1] || null },
    clear() { _diagLog.length = 0 },
  }
}

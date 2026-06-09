import { useEffect, useState, useCallback, useRef } from 'react'
import { useAppStore } from '@/stores/appStore'
import { useWebSocket, markSummaryQuery } from '@/hooks/useWebSocket'
import TitleBar from '@/components/layout/TitleBar'
import MainContent from '@/components/layout/MainContent'
import ConversationSidebar from '@/components/conversation/ConversationSidebar'
import SettingsPanel from '@/components/settings/SettingsPanel'
import DebugPanel from '@/components/DebugPanel'
import type { PaperTab } from '@/types'

console.log('🀄 ZhiBan App mounting...', new Date().toLocaleTimeString())

let tabCounter = 0

export default function App() {
  const theme = useAppStore(s => s.settings.theme)
  const activeConvId = useAppStore(s => s.activeConversationId)
  const { send } = useWebSocket()
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => {
    try {
      const stored = localStorage.getItem('zhiban-sidebar-collapsed')
      return stored !== null ? stored === 'true' : false
    } catch {
      return false
    }
  })
  const addPaper = useAppStore(s => s.addPaper)
  const notifications = useAppStore(s => s.notifications)
  const dismissNotification = useAppStore(s => s.dismissNotification)
  const pushNotification = useAppStore(s => s.pushNotification)
  const [isDragging, setIsDragging] = useState(false)
  const [debugPanelVisible, setDebugPanelVisible] = useState(false)
  const wsStatus = useAppStore(s => s.connection.wsStatus)
  const debugMode = useAppStore(s => s.connection.debugMode)
  const [jsErrors, setJsErrors] = useState<string[]>([])

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
  }, [theme])

  useEffect(() => {
    localStorage.setItem('zhiban-sidebar-collapsed', String(sidebarCollapsed))
  }, [sidebarCollapsed])

  useEffect(() => {
    const handler = (e: ErrorEvent) => {
      setJsErrors(prev => [...prev.slice(-4), `${e.message}`.slice(0, 80)])
    }
    window.addEventListener('error', handler)
    return () => window.removeEventListener('error', handler)
  }, [])

  // Register WebSocket send function in store for components that need it
  useEffect(() => {
    useAppStore.setState({ sendMessage: send })
    return () => { useAppStore.setState({ sendMessage: null }) }
  }, [send])

  // WebSocket 连接上后，自动同步本地存储的设置与后端
  const autoApplied = useRef(false)
  useEffect(() => {
    if (wsStatus !== 'connected' || autoApplied.current) return
    autoApplied.current = true

    const state = useAppStore.getState()
    const { settings } = state
    const msgSend = useAppStore.getState().sendMessage

    if (!msgSend) return

    // 先从后端拉取当前状态
    msgSend({ type: 'model_config', action: 'get' })

    // 如果前端本地有存储的模型路径，也推送给后端加载
    const storedPath = settings.llmModelPath?.trim()
    if (storedPath && storedPath.length > 0) {
      // 延迟推送，等后端 get 响应回来后再读取最新值
      setTimeout(() => {
        const msg = useAppStore.getState().sendMessage
        const path = useAppStore.getState().settings.llmModelPath?.trim()
        if (path && path.length > 0) {
          msg?.({ type: 'model_config', action: 'set_local_model', path })
        }
      }, 2000)
    }
  }, [wsStatus])

  // WS 断连时重置 autoApplied，确保重连后重新同步设置
  useEffect(() => {
    if (wsStatus === 'disconnected' || wsStatus === 'reconnecting') {
      autoApplied.current = false
    }
  }, [wsStatus])

  const toggleSettings = useCallback(() => setSettingsOpen(v => !v), [])

  const _sendGuard = useRef(false)
  function sendQuery(queryText: string): boolean {
    if (!queryText.trim()) return false
    const state = useAppStore.getState()
    // 本地引擎加载中，禁止查询
    if (state.connection.localEngineLoading) {
      state.pushNotification('warn', '本地模型正在加载中，请稍候再提问')
      return false
    }
    // 防止重复发送 (快速连按回车) — 只用 ref guard，不用 isQueryRunning
    // 因为 setIsQueryRunning(true) 也在这个函数里，不能在 guard 里读它
    if (_sendGuard.current) return false
    _sendGuard.current = true
    setTimeout(() => { _sendGuard.current = false }, 500)

    useAppStore.getState().setQueryRunning(true)
    // 用户叉掉阅读区上下文 = 明确选择不用，不再自动恢复
    const settings = state.settings
    const screenCtx = state.screenContext
    const isContextPaused: boolean = (state as any).contextPaused ?? false

    // 组装上下文：三种情况的优先级 —
    //   Case 3（最高）: 用户显式选中文档段落 / 引用对话消息 → 按引用内容回答
    //   Case 1: 无引用 + 上下文未暂停 → 自动注入当前阅读页内容
    //   Case 2: 无引用 + 上下文已暂停（叉掉了阅读区）→ 空上下文，走 RAG 二次搜索
    const hasContext = screenCtx.source !== null && screenCtx.ocrParagraphs.length > 0
    const selections: number[] = (state as any).selectedParagraphIndices ?? []
    const msgQuotes = (state.selectedQuotes ?? []).filter(q => 'msgIndex' in q)
    const paperQuotes = (state.selectedQuotes ?? []).filter(q => 'paperId' in q)
    const hasQuotes = msgQuotes.length > 0 || paperQuotes.length > 0

    // 上下文摘要（消息气泡下方的小字提示）
    let contextSummary = ''
    if (hasQuotes) {
      const parts: string[] = []
      if (msgQuotes.length > 0) parts.push(`${msgQuotes.length} 段对话`)
      if (paperQuotes.length > 0) parts.push(`${paperQuotes.length} 页论文`)
      contextSummary = `已引用 ${parts.join(' + ')}`
    } else if (selections.length > 0) {
      const snippets = selections.slice(0, 2).map(idx => {
        const p = screenCtx.ocrParagraphs[idx]
        return p ? p.text.slice(0, 40).replace(/\n/g, ' ') : ''
      })
      contextSummary = `已引用 ${selections.length} 段文档 · ${snippets.join(' | ')}`
    } else if (!isContextPaused && hasContext && screenCtx.currentDoc) {
      const idx = (screenCtx.activeParagraphIndex ?? 0) + 1
      contextSummary = `AI 阅读了 ${screenCtx.currentDoc} · 第${idx}/${screenCtx.ocrParagraphs.length}页`
    }

    // === Case 3: 用户显式选中 ===
    let activeParagraph = ''
    if (hasContext && selections.length > 0) {
      const parts: string[] = [`[用户选中了以下 ${selections.length} 段文字]`]
      for (const idx of selections) {
        if (screenCtx.ocrParagraphs[idx]) {
          parts.push(`[第${idx + 1}段] ${screenCtx.ocrParagraphs[idx].text}`)
        }
      }
      activeParagraph = parts.join('\n\n')
    }

    // 对话消息引用可叠加在文档引用之上
    if (msgQuotes.length > 0) {
      const quotedParts: string[] = [`[用户引用了以下 ${msgQuotes.length} 段对话内容]`]
      for (const q of msgQuotes) {
        quotedParts.push(`[引自消息 #${(q as any).msgIndex + 1}] ${q.text.slice(0, 2000)}`)
      }
      const quotedBlock = quotedParts.join('\n\n')
      activeParagraph = activeParagraph
        ? `${activeParagraph}\n\n${quotedBlock}`
        : quotedBlock
    }

    // 论文页面引用
    if (paperQuotes.length > 0) {
      const parts: string[] = [`[用户引用了以下 ${paperQuotes.length} 个论文页面]`]
      for (const pq of paperQuotes) {
        parts.push(`[${pq.paperName} 第${pq.pageNumber}页] ${pq.text}`)
      }
      activeParagraph = activeParagraph
        ? `${activeParagraph}\n\n${parts.join('\n')}`
        : parts.join('\n')
    }

    // === Case 1: 无引用 + 上下文未暂停 → 自动注入当前阅读页 ===
    if (!activeParagraph && !isContextPaused && hasContext) {
      const aIdx = screenCtx.activeParagraphIndex ?? 0
      if (screenCtx.ocrParagraphs[aIdx]) {
        const parts: string[] = []
        if (aIdx > 0 && screenCtx.ocrParagraphs[aIdx - 1]) {
          parts.push(`[前页末尾] ${screenCtx.ocrParagraphs[aIdx - 1].text.slice(-600)}`)
        }
        parts.push(`[当前页] ${screenCtx.ocrParagraphs[aIdx].text}`)
        if (screenCtx.ocrParagraphs[aIdx + 1]) {
          parts.push(`[后页开头] ${screenCtx.ocrParagraphs[aIdx + 1].text.slice(0, 600)}`)
        }
        activeParagraph = parts.join('\n\n')
      }
    }
    // Case 2: 无引用 + 上下文已暂停 → activeParagraph 保持空，后端走 RAG 二次搜索

    const activeDoc = hasContext
      ? (screenCtx.currentDoc || state.papers.map(p => p.name).join(', ') || '')
      : ''

    // 添加用户消息到对话（带上下文摘要）
    useAppStore.getState().addMessage({
      id: `user-${Date.now()}`,
      role: 'user',
      content: queryText,
      timestamp: Date.now(),
      screenContext: contextSummary ? { docName: contextSummary, paragraphText: '' } : undefined,
    })

    // 历史只发送用户消息 (不含 AI 长篇回复)，避免重启后上下文臃肿
    const history = state.conversation.messages
      .filter(m => m.role === 'user')
      .slice(-20)
      .map(m => ({
        role: m.role as const,
        content: m.content,
        timestamp: m.timestamp,
        mode: m.mode,
        model: m.model,
      }))

    // Build open papers list for V10 engine
    const openPapers = state.papers.map(p => ({
      paperId: p.id,
      title: p.name,
      filename: p.name,
      filepath: p.path,
    }))

    // 标记总结类查询（延长流式超时）
    const _summaryKw = ["总结全文", "全文总结", "全文核心内容", "概括全文",
      "总结这篇文章", "全文概括", "总结本论文", "全文内容总结", "概括这篇文章"]
    if (_summaryKw.some(kw => queryText.includes(kw))) {
      markSummaryQuery()
    }

    send({
      type: 'user_query',
      queryText,
      context: {
        activeDoc,
        activeParagraph,
        paragraphIndex: screenCtx.activeParagraphIndex ?? 0,
        conversationId: useAppStore.getState().activeConversationId,
      },
      timestamp: Date.now(),
      apiKey: settings.llmApiKey || undefined,
      baseUrl: settings.llmBaseUrl || undefined,
      systemPrompt: settings.systemPrompt || undefined,
      model: settings.llmModel || undefined,
      thinking: settings.thinkingMode,
      topK: settings.topK,
      history,
      openPapers,
    })
    return true
  }

  // Drag-drop
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const files = Array.from(e.dataTransfer.files)
    const skipped: string[] = []
    const sendMsg = useAppStore.getState().sendMessage
    for (const file of files) {
      const ext = file.name.split('.').pop()?.toLowerCase()
      if (!ext || !['pdf', 'docx', 'txt', 'md'].includes(ext)) {
        skipped.push(file.name)
        continue
      }
      const id = `paper-${++tabCounter}`

      if (window.electronAPI?.getPathForFile) {
        // Electron mode: get local file path
        const filePath = window.electronAPI.getPathForFile(file) || ''
        addPaper({ id, name: file.name, type: ext as PaperTab['type'], path: filePath, extractedText: '' })
        if (filePath && sendMsg) {
          try { sendMsg({ type: 'import_paper', filePath }) } catch {}
          try { sendMsg({ type: 'bind_paper', filePath, paperId: id, title: file.name }) } catch {}
        }
      } else {
        // WebUI mode: upload file
        const formData = new FormData()
        formData.append('file', file)
        fetch('/upload', { method: 'POST', body: formData })
          .then(r => r.json())
          .then(data => {
            if (data.filePath) {
              addPaper({ id, name: file.name, type: ext as PaperTab['type'], path: data.filePath, extractedText: '' })
              if (sendMsg) {
                try { sendMsg({ type: 'import_paper', filePath: data.filePath }) } catch {}
                try { sendMsg({ type: 'bind_paper', filePath: data.filePath, paperId: id, title: file.name }) } catch {}
              }
            }
          })
          .catch(err => {
            console.error('Upload failed:', err)
            useAppStore.getState().pushNotification('error', `上传失败: ${file.name}`)
          })
      }
    }
    if (skipped.length > 0) {
      const names = skipped.map(f => f.length > 40 ? f.slice(0, 37) + '...' : f).join(', ')
      useAppStore.getState().pushNotification(
        'warn',
        `跳过 ${skipped.length} 个不支持的文件（仅支持 PDF/DOCX/TXT/MD）: ${names}`
      )
    }
  }, [addPaper])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'copy'
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    if (e.currentTarget === e.target || !e.currentTarget.contains(e.relatedTarget as Node)) {
      setIsDragging(false)
    }
  }, [])

  // Menu File > Open handler
  useEffect(() => {
    const unsub = window.electronAPI?.onMenuOpenFile?.((filePaths: string[]) => {
      const sendMsg = useAppStore.getState().sendMessage
      for (const fp of filePaths) {
        const name = fp.split('/').pop() || fp.split('\\').pop() || 'unknown'
        const ext = name.split('.').pop()?.toLowerCase() || ''
        if (!['pdf', 'docx', 'txt', 'md'].includes(ext)) continue
        const id = `paper-${++tabCounter}`
        addPaper({ id, name, type: ext as PaperTab['type'], path: fp, extractedText: '' })
        // 触发后端向量化 + 绑定到当前会话
        if (sendMsg) {
          try { sendMsg({ type: 'import_paper', filePath: fp }) } catch {}
          try { sendMsg({ type: 'bind_paper', filePath: fp, paperId: id, title: name }) } catch {}
        }
      }
    })
    return () => { unsub?.() }
  }, [addPaper])

  // Expose WS send for components that need to send messages directly
  ;(window as any).__zhiban_wsSend = send

  // Listen for menu-triggered actions via IPC (instead of executeJavaScript)
  useEffect(() => {
    const unsubs: (() => void)[] = []
    if (window.electronAPI?.onMenuAction) {
      unsubs.push(window.electronAPI.onMenuAction('toggle-settings', () => setSettingsOpen(v => !v)))
      unsubs.push(window.electronAPI.onMenuAction('toggle-theme', () => useAppStore.getState().toggleTheme()))
      unsubs.push(window.electronAPI.onMenuAction('clear-conv', () => useAppStore.setState(s => ({
        conversation: { ...s.conversation, messages: [], isStreaming: false, streamingText: '', streamingCitations: [], streamingRelatedPapers: [] }
      }))))
    }
    return () => unsubs.forEach(u => u())
  }, [])

  // Listen for sidecar startup errors from main process → show as notification
  useEffect(() => {
    const unsub = window.electronAPI?.onSidecarError?.((message: string) => {
      pushNotification('error', message)
    })
    return () => unsub?.()
  }, [pushNotification])

  // Listen for Cmd+Shift+D toggle debug panel
  useEffect(() => {
    const unsub = window.electronAPI?.onShortcut?.('shortcut:toggle-debug', () => {
      setDebugPanelVisible(v => !v)
    })
    return () => unsub?.()
  }, [])

  // Forward WebSocket diag entries to debug panel
  useEffect(() => {
    const interval = setInterval(() => {
      const diag = (window as any).__zhiban_diag
      if (diag && debugPanelVisible) {
        const entries = diag.log || []
        const last = entries[entries.length - 1]
        if (last && last._forwarded !== true) {
          last._forwarded = true
          window.dispatchEvent(new CustomEvent('debug:diag', { detail: `[WS] ${last.event} ${last.detail || ''}` }))
        }
      }
    }, 1000)
    return () => clearInterval(interval)
  }, [debugPanelVisible])

  return (
    <div className={`app-shell${isDragging ? ' drag-over' : ''}`}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
    >
      <TitleBar onSettingsClick={toggleSettings} onToggleSidebar={() => setSidebarCollapsed(v => !v)} />

      {/* Main layout with panel gaps exposing shell gradient */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden', padding: 8, gap: sidebarCollapsed ? 0 : 8 }}>
        <ConversationSidebar
          collapsed={sidebarCollapsed}
          onToggleCollapse={() => setSidebarCollapsed(true)}
          wsSend={send}
        />
        <MainContent onSendQuery={sendQuery} wsSend={send} />
      </div>

      <SettingsPanel open={settingsOpen} onClose={() => setSettingsOpen(false)} />

      <DebugPanel
        visible={debugPanelVisible}
        onClose={() => setDebugPanelVisible(false)}
      />

      {/* Toast notifications — Proma style */}
      {notifications.length > 0 && (
        <div style={{
          position: 'fixed', bottom: 24, right: 24, zIndex: 10000,
          display: 'flex', flexDirection: 'column', gap: 8, maxWidth: 420,
        }}>
          {notifications.map(n => (
            <div key={n.id} style={{
              padding: '10px 14px', borderRadius: 10, fontSize: 13,
              display: 'flex', alignItems: 'flex-start', gap: 10,
              background: n.type === 'error' ? 'var(--toast-error-bg)' : n.type === 'warn' ? 'var(--toast-warn-bg)' : n.type === 'success' ? 'var(--toast-success-bg)' : 'var(--toast-info-bg)',
              border: `1px solid ${n.type === 'error' ? 'var(--toast-error-border)' : n.type === 'warn' ? 'var(--toast-warn-border)' : n.type === 'success' ? 'var(--toast-success-border)' : 'var(--toast-info-border)'}`,
              color: 'hsl(var(--foreground))',
              boxShadow: '0 4px 20px rgba(0,0,0,0.25)',
              animation: 'slideInRight 0.25s ease-out',
              backdropFilter: 'blur(12px)',
              WebkitBackdropFilter: 'blur(12px)',
            }}>
              <span style={{ flex: 1, lineHeight: 1.4 }}>{n.message}</span>
              {n.onUndo && (
                <button onClick={() => {
                  n.onUndo?.()
                  dismissNotification(n.id)
                }} style={{
                  background: 'none', border: '1px solid hsl(var(--primary) / 0.50)',
                  color: 'hsl(var(--primary))', cursor: 'pointer',
                  fontSize: 11, padding: '3px 8px', borderRadius: 6,
                  fontFamily: 'inherit', fontWeight: 500,
                }}>撤销</button>
              )}
              <button onClick={() => dismissNotification(n.id)} style={{
                background: 'none', border: 'none', color: 'hsl(var(--muted-foreground))', cursor: 'pointer',
                fontSize: 16, lineHeight: 1, padding: '0 2px', borderRadius: 4, opacity: 0.6,
              }}>x</button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

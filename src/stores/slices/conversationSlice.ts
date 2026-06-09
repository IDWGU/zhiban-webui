import type { Message, Citation, PaperRef, AgentStep, ConversationState, ConversationSummary, LlmHealthMessage } from '@/types'

function loadSavedConversation(): Message[] {
  try {
    const raw = localStorage.getItem('zhiban-conv')
    if (raw) return JSON.parse(raw)
  } catch {}
  return []
}

export interface HealthRecord {
  call: 'classify' | 'answer'
  timing: LlmHealthMessage['timing']
  tokens: LlmHealthMessage['tokens']
  memory?: LlmHealthMessage['memory']
  timestamp: number
}

export interface ConversationSlice {
  conversation: ConversationState & { chatState: 'idle' | 'thinking' | 'tool_executing' | 'streaming' | 'permission_pending' }
  setChatState: (state: 'idle' | 'thinking' | 'tool_executing' | 'streaming' | 'permission_pending') => void
  conversations: ConversationSummary[]
  activeConversationId: string

  // V10 Workflow
  workflowStatus: { code: string; message: string }
  setWorkflowStatus: (code: string, message: string) => void
  lastResponseMeta: { refused: boolean; expanded: boolean; responseType: string }
  setLastResponseMeta: (meta: { refused: boolean; expanded: boolean; responseType: string }) => void
  isQueryRunning: boolean
  setQueryRunning: (running: boolean) => void
  prefillStartTime: number
  prefillTotalTokens: number
  prefillCurrentTokens: number
  setPrefillStart: (totalTokens: number) => void
  setPrefillProgress: (current: number, total: number) => void
  sessionUsage: { input: number; output: number }
  addSessionUsage: (input: number, output: number) => void

  // Phase 3: Health tracking
  healthHistory: HealthRecord[]
  addHealthRecord: (record: HealthRecord) => void
  clearHealthHistory: () => void

  // Real-time speed display
  localSpeed: { prefillTokens: number; prefillMs: number; tokPerSec: number; phase: string }
  setLocalSpeed: (s: { prefillTokens: number; prefillMs: number; tokPerSec: number; phase: string }) => void

  // Conversation actions
  addMessage: (message: Message) => void
  popLastMessage: () => void
  startStreaming: (messageId: string) => void
  appendStreamingToken: (token: string) => void
  addStreamingCitation: (citation: Citation) => void
  addStreamingRelatedPaper: (paper: PaperRef) => void
  appendAgentStep: (step: AgentStep) => void
  appendThinkingToken: (token: string) => void
  finishStreaming: (messageId: string, meta?: { mode?: string; model?: string; usage?: { input: number; output: number } | null; loopDetected?: boolean; error?: string; cancelled?: boolean }) => void
  cancelQuery: () => void

  // Conversation management
  setConversations: (list: ConversationSummary[]) => void
  setActiveConversationId: (id: string) => void
  addConversation: (conv: ConversationSummary) => void
  removeConversation: (id: string) => void
  updateConversation: (id: string, partial: Partial<ConversationSummary>) => void
  syncConversationMessages: (convId: string, messages: Message[], openPapers: Array<{ paper_id: number | string; title: string; filename: string }>, topic: string) => void

  sendMessage: ((data: unknown) => void) | null
  setSendMessage: (fn: ((data: unknown) => void) | null) => void
}

export function createConversationSlice(set: any, get: any): ConversationSlice {
  return {
    conversation: {
      messages: loadSavedConversation(),
      isStreaming: false,
      streamingMessageId: null,
      streamingText: '',
      streamingCitations: [],
      streamingRelatedPapers: [],
      streamingAgentSteps: [],
      streamingThinkingText: '',
      chatState: 'idle',
    },

    setChatState: (chatState) => set((state: any) => ({
      conversation: { ...state.conversation, chatState },
    })),

    workflowStatus: { code: 'idle', message: '' },
    isQueryRunning: false,
    prefillStartTime: 0,
    prefillTotalTokens: 0,
    prefillCurrentTokens: 0,
    sessionUsage: { input: 0, output: 0 },
    lastResponseMeta: { refused: false, expanded: false, responseType: '' },

    // Phase 3: Health tracking
    healthHistory: [],
    addHealthRecord: (record) => set((state: any) => ({
      healthHistory: [...state.healthHistory.slice(-19), record],
    })),
    clearHealthHistory: () => set({ healthHistory: [] }),

    // Real-time speed
    localSpeed: { prefillTokens: 0, prefillMs: 0, tokPerSec: 0, phase: '' },
    setLocalSpeed: (s) => set({ localSpeed: s }),

    conversations: [],
    activeConversationId: localStorage.getItem('zhiban-active-conv-id') || 'default',
    sendMessage: null,

    // Workflow
    setWorkflowStatus: (code, message) => set({ workflowStatus: { code, message } }),
    setQueryRunning: (running) => set({ isQueryRunning: running }),
    setPrefillStart: (totalTokens) => set({
      prefillStartTime: Date.now(),
      prefillTotalTokens: totalTokens,
      prefillCurrentTokens: 0,
    }),
    setPrefillProgress: (current, total) => set({
      prefillCurrentTokens: current,
      prefillTotalTokens: total > 0 ? total : undefined,
    }),
    addSessionUsage: (input, output) => set((state: any) => ({
      sessionUsage: {
        input: state.sessionUsage.input + input,
        output: state.sessionUsage.output + output,
      },
    })),
    setLastResponseMeta: (meta) => set({ lastResponseMeta: meta }),

    // Conversation actions
    addMessage: (message) => set((state: any) => ({
      conversation: {
        ...state.conversation,
        messages: [...state.conversation.messages, message]
      }
    })),

    popLastMessage: () => set((state: any) => ({
      conversation: {
        ...state.conversation,
        messages: state.conversation.messages.slice(0, -1),
      },
    })),

    startStreaming: (messageId) => {
      // 清理模块级缓存，避免跨会话/跨消息的污染
      try {
        const { resetCitationCache, resetSafeCutCache } = require('@/components/conversation/StreamingMarkdown')
        resetCitationCache()
        resetSafeCutCache()
      } catch {}
      return set((state: any) => ({
        conversation: {
          ...state.conversation,
          isStreaming: true,
          streamingMessageId: messageId,
          streamingText: '',
          streamingCitations: [],
          streamingRelatedPapers: [],
          streamingAgentSteps: [],
          streamingThinkingText: '',
        }
      }))
    },

    appendStreamingToken: (token) => set((state: any) => ({
      conversation: {
        ...state.conversation,
        streamingText: state.conversation.streamingText + token,
      }
    })),

    addStreamingCitation: (citation) => set((state: any) => ({
      conversation: {
        ...state.conversation,
        streamingCitations: [...state.conversation.streamingCitations, citation],
      }
    })),

    addStreamingRelatedPaper: (paper) => set((state: any) => ({
      conversation: {
        ...state.conversation,
        streamingRelatedPapers: [...state.conversation.streamingRelatedPapers, paper],
      }
    })),

    appendAgentStep: (step) => set((state: any) => ({
      conversation: {
        ...state.conversation,
        streamingAgentSteps: [...state.conversation.streamingAgentSteps, step],
      }
    })),

    appendThinkingToken: (token) => set((state: any) => {
      const conv = state.conversation
      const newText = conv.streamingThinkingText + token

      // 自动从 thinking text 中检测推理步骤（仅在没有明确 agent_step 时才补位）
      const decisionMatch = token.match(/^\n([🤔🔍])\s+(.+?)\n$/s)
      let steps = conv.streamingAgentSteps
      if (decisionMatch && steps.length === 0) {
        const phaseIcon = decisionMatch[1]
        const content = decisionMatch[2].trim()
        if (content) {
          steps = [{
            stepIndex: 0,
            phase: 'thinking' as const,
            content: phaseIcon === '🤔'
              ? `决策推理: ${content}`
              : `分析评估: ${content}`,
          }]
        }
      }

      return {
        conversation: {
          ...conv,
          streamingThinkingText: newText,
          streamingAgentSteps: steps,
        }
      }
    }),

    finishStreaming: (messageId, meta?) => set((state: any) => {
      if (messageId === 'error') {
        return {
          conversation: {
            ...state.conversation,
            isStreaming: false,
            streamingMessageId: null,
            streamingText: '',
            streamingCitations: [],
            streamingRelatedPapers: [],
            streamingAgentSteps: [],
            streamingThinkingText: '',
          },
          notifications: [
            ...state.notifications,
            { id: crypto.randomUUID(), type: 'error' as const, message: 'AI 回答生成失败，请重试', timestamp: Date.now() },
          ],
        }
      }
      const msg: Message = {
        id: messageId,
        role: 'assistant',
        content: state.conversation.streamingText,
        citations: state.conversation.streamingCitations,
        relatedPapers: state.conversation.streamingRelatedPapers,
        timestamp: Date.now(),
        mode: meta?.mode || '',
        model: meta?.model || '',
        usage: meta?.usage || undefined,
        loopDetected: meta?.loopDetected ?? false,
        agentSteps: state.conversation.streamingAgentSteps.length > 0
          ? state.conversation.streamingAgentSteps
          : undefined,
        thinkingContent: state.conversation.streamingThinkingText || undefined,
      }
      return {
        conversation: {
          ...state.conversation,
          messages: [...state.conversation.messages, msg],
          isStreaming: false,
          streamingMessageId: null,
          streamingText: '',
          streamingCitations: [],
          streamingRelatedPapers: [],
          streamingAgentSteps: [],
          streamingThinkingText: '',
        }
      }
    }),

    cancelQuery: () => {
      get().sendMessage?.({ type: 'control', action: 'cancel_query' })
      set((state: any) => {
        // 保存进行中的流式内容为不完整消息，避免丢失
        const conv = state.conversation
        let messages = conv.messages
        if (conv.isStreaming && conv.streamingText) {
          const partialMsg: Message = {
            id: conv.streamingMessageId || 'msg-' + Date.now(),
            role: 'assistant',
            content: conv.streamingText + '\n\n*(已中断)*',
            timestamp: Date.now(),
            mode: state.lastResponseMeta?.responseType || '',
            model: '',
            agentSteps: conv.streamingAgentSteps.length > 0
              ? conv.streamingAgentSteps
              : undefined,
            thinkingContent: conv.streamingThinkingText || undefined,
          }
          messages = [...conv.messages, partialMsg]
        }
        return {
          conversation: {
            ...conv,
            messages,
            isStreaming: false,
            streamingMessageId: null,
            streamingText: '',
            streamingCitations: [],
            streamingRelatedPapers: [],
            streamingAgentSteps: [],
            streamingThinkingText: '',
          },
          workflowStatus: { code: 'idle', message: '' },
          isQueryRunning: false,
        }
      })
    },

    // Conversation management
    setConversations: (list) => set({ conversations: list }),

    setActiveConversationId: (id) => {
      localStorage.setItem('zhiban-active-conv-id', id)
      set((state: any) => ({
        activeConversationId: id,
        conversations: state.conversations.map((c: ConversationSummary) => ({
          ...c,
          isActive: c.id === id,
        })),
      }))
    },

    addConversation: (conv) => set((state: any) => ({
      conversations: [
        { ...conv, isActive: false },
        ...state.conversations,
      ],
    })),

    removeConversation: (id) => set((state: any) => {
      const remaining = state.conversations.filter((c: ConversationSummary) => c.id !== id)
      const newActive = state.activeConversationId === id
        ? (remaining[0]?.id || 'default')
        : state.activeConversationId
      return { conversations: remaining, activeConversationId: newActive }
    }),

    updateConversation: (id, partial) => set((state: any) => ({
      conversations: state.conversations.map((c: ConversationSummary) =>
        c.id === id ? { ...c, ...partial } : c
      ),
    })),

    syncConversationMessages: (convId, messages, openPapers, topic) => {
      // 保留已有论文的 extractedText，避免切换会话时重新加载 PDF
      const existingByPath = new Map<string, string>()
      const existingById = new Map<string, string>()
      const s = get() as any
      for (const p of (s.papers || [])) {
        if (p.path && p.extractedText) existingByPath.set(p.path, p.extractedText)
        if (p.id && p.extractedText) existingById.set(p.id, p.extractedText)
      }
      // 将后端消息映射为前端 Message 格式（保留 thinkingContent, agentSteps 等扩展字段）
      const mappedMessages: Message[] = messages.map((m: any) => ({
        id: m.id || `msg-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        role: m.role as 'user' | 'assistant' | 'system',
        content: m.content || '',
        timestamp: m.timestamp || Date.now(),
        mode: m.mode,
        model: m.model,
        thinkingContent: m.thinkingContent,
        agentSteps: m.agentSteps,
      }))
      return set((state: any) => ({
        activeConversationId: convId,
        workflowStatus: { code: 'idle', message: '' },
        isQueryRunning: false,
        conversation: {
          ...state.conversation,
          messages: mappedMessages,
          isStreaming: false,
          streamingText: '',
          streamingCitations: [],
          streamingRelatedPapers: [],
          streamingAgentSteps: [],
          streamingThinkingText: '',
        },
        papers: (() => {
          const seen = new Set<string>()
          return openPapers.map((p: any) => {
            const pid = String(p.paper_id).startsWith('paper-') ? p.paper_id : `paper-${p.paper_id}`
            return {
              id: pid,
              name: p.title || p.filename || `Paper #${p.paper_id}`,
              type: (p.filename || '').endsWith('.pdf') ? 'pdf' as const
                   : (p.filename || '').endsWith('.docx') ? 'docx' as const
                   : 'txt' as const,
              path: p.filepath || '',
              extractedText: existingByPath.get(p.filepath || '') || existingById.get(pid) || '',
            }
          }).filter(p => {
            const key = p.path || p.id
            if (seen.has(key)) return false
            seen.add(key)
            return true
          })
        })(),
        activeTabId: openPapers.length > 0 ? `paper-${openPapers[0].paper_id}` : state.activeTabId,
        conversations: state.conversations.map((c: ConversationSummary) =>
          c.id === convId ? { ...c, isActive: true, topic } : { ...c, isActive: false }
        ),
      }))
    },

    setSendMessage: (fn) => set({ sendMessage: fn }),
  }
}

import type { PaperTab, BuildIndexProgressMessage } from '@/types'

// ── 引用选择类型 ──

export interface ParagraphQuote {
  msgIndex: number
  paraIndex: number
  text: string
}

export interface PaperPageQuote {
  paperId: string
  paperName: string
  pageNumber: number
  text: string  // 页面摘要
}

export type QuoteItem = ParagraphQuote | PaperPageQuote

function isParagraphQuote(q: QuoteItem): q is ParagraphQuote {
  return 'msgIndex' in q
}

function quoteKey(q: QuoteItem): string {
  if (isParagraphQuote(q)) return `msg:${q.msgIndex}:${q.paraIndex}`
  return `paper:${q.paperId}:${q.pageNumber}`
}

function loadSavedPapers(): PaperTab[] {
  try {
    const raw = localStorage.getItem('zhiban-papers')
    if (raw) {
      const papers: PaperTab[] = JSON.parse(raw)
      // Deduplicate by path on load (defensive against accumulated duplicates)
      const seen = new Set<string>()
      return papers.filter(p => {
        const key = p.path || p.id
        if (seen.has(key)) return false
        seen.add(key)
        return true
      })
    }
  } catch {}
  return []
}

function loadSavedNotes(): string {
  return localStorage.getItem('zhiban-notes') || ''
}

export interface AppNotification {
  id: string
  type: 'error' | 'warn' | 'info' | 'success'
  message: string
  timestamp: number
  onUndo?: () => void
}

export interface PaperSlice {
  papers: PaperTab[]
  notesContent: string
  activeTabId: string | null
  notifications: AppNotification[]

  addPaper: (paper: PaperTab) => void
  removePaper: (id: string) => void
  clearPapers: () => void
  setActiveTab: (id: string | null) => void
  setNotesContent: (content: string) => void
  updatePaperText: (id: string, text: string) => void

  // Build Index
  buildIndex: { phase: BuildIndexProgressMessage['phase'] | 'idle'; progress: { current: number; total: number }; message: string; result: { success: boolean; chunks?: number; error?: string } | null }
  startBuildIndex: () => void
  setBuildIndexProgress: (phase: BuildIndexProgressMessage['phase'] | 'paused', current: number, total: number, message: string) => void
  setBuildIndexResult: (success: boolean, chunks?: number, error?: string) => void

  // Notifications
  pushNotification: (type: AppNotification['type'], message: string, onUndo?: () => void) => void
  dismissNotification: (id: string) => void

  // Quote selections
  selectedQuotes: QuoteItem[]
  addQuote: (quote: QuoteItem) => void
  removeQuote: (msgIndex: number, paraIndex: number) => void
  undoLastQuote: () => void
  clearQuoteSelections: () => void
}

export function createPaperSlice(set: any, _get: any): PaperSlice {
  return {
    papers: loadSavedPapers(),
    notesContent: loadSavedNotes(),
    activeTabId: loadSavedPapers()[0]?.id ?? null,
    notifications: [],

    addPaper: (paper) => set((state: any) => {
      // Deduplicate by path (same file) or by id
      const exists = state.papers.some((p: PaperTab) =>
        (paper.path && p.path === paper.path) || p.id === paper.id
      )
      if (exists) {
        return { activeTabId: paper.id }
      }
      return {
        papers: [...state.papers, paper],
        activeTabId: paper.id,
      }
    }),

    removePaper: (id) => set((state: any) => {
      const papers = state.papers.filter((p: PaperTab) => p.id !== id)
      const activeTabId = state.activeTabId === id
        ? (papers.length > 0 ? papers[papers.length - 1].id : null)
        : state.activeTabId
      return { papers, activeTabId }
    }),

    clearPapers: () => {
      localStorage.removeItem('zhiban-papers')
      set({ papers: [], activeTabId: null })
    },

    setActiveTab: (id) => set({ activeTabId: id }),
    setNotesContent: (content) => set({ notesContent: content }),

    updatePaperText: (id, text) => set((state: any) => ({
      papers: state.papers.map((p: PaperTab) => p.id === id ? { ...p, extractedText: text } : p),
    })),

    // Build Index
    buildIndex: {
      phase: 'idle',
      progress: { current: 0, total: 0 },
      message: '',
      result: null,
    },

    startBuildIndex: () => set({
      buildIndex: { phase: 'scanning', progress: { current: 0, total: 0 }, message: '正在启动...', result: null },
    }),

    setBuildIndexProgress: (phase, current, total, message) => set({
      buildIndex: { phase, progress: { current, total }, message, result: null },
    }),

    setBuildIndexResult: (success, chunks, error) => set((state: any) => ({
      buildIndex: {
        ...state.buildIndex,
        phase: success ? 'done' : 'idle',
        result: { success, chunks, error },
      },
    })),

    // Notifications
    pushNotification: (type, message, onUndo?) => set((state: any) => ({
      notifications: [
        ...state.notifications,
        { id: crypto.randomUUID(), type, message, timestamp: Date.now(), onUndo },
      ],
    })),

    dismissNotification: (id) => set((state: any) => ({
      notifications: state.notifications.filter((n: AppNotification) => n.id !== id),
    })),

    // Quote selections
    selectedQuotes: [],

    addQuote: (quote) => set((state: any) => {
      const key = quoteKey(quote)
      if (state.selectedQuotes.some((q: QuoteItem) => quoteKey(q) === key)) return state
      return { selectedQuotes: [...state.selectedQuotes, quote] }
    }),

    removeQuote: (msgIndex, paraIndex) => set((state: any) => ({
      selectedQuotes: state.selectedQuotes.filter(
        (q: QuoteItem) => !(isParagraphQuote(q) && q.msgIndex === msgIndex && q.paraIndex === paraIndex)
      ),
    })),

    undoLastQuote: () => set((state: any) => ({
      selectedQuotes: state.selectedQuotes.slice(0, -1),
    })),

    clearQuoteSelections: () => set({ selectedQuotes: [] }),
  }
}

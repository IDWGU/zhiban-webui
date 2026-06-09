import type { TranslationState, TranslationBlock } from '@/types'

function _cacheKey(sha256: string): string {
  return `zhiban-trans-${sha256}`
}

function _migrateKey(oldPath: string, sha256: string): boolean {
  try {
    const oldRaw = localStorage.getItem(`zhiban-trans-${encodeURIComponent(oldPath)}`)
    if (!oldRaw) return false
    localStorage.setItem(_cacheKey(sha256), oldRaw)
    localStorage.removeItem(`zhiban-trans-${encodeURIComponent(oldPath)}`)
    return true
  } catch {
    return false
  }
}

export interface TranslationSlice {
  translation: TranslationState
  prepareForFile: (filePath: string) => void
  setFileIdentity: (filePath: string, sha256: string, size: number) => void
  startTranslation: (filePath?: string, extractOnly?: boolean) => void
  loadCachedTranslation: (filePath: string) => boolean
  setTranslationBlocks: (blocks: TranslationBlock[], total: number) => void
  appendTranslationToken: (sentenceId: string, token: string, isFirst: boolean) => void
  finishTranslation: (totalBlocks: number, totalSentences: number) => void
  setTranslationPhase: (phase: TranslationState['phase']) => void
  setTranslationStatusMsg: (msg: string) => void
  setTranslationSpeed: (tokensPerSec: number, elapsed: number, firstTokenMs?: number) => void
  setTranslationError: (error: string) => void
  setActiveSentenceId: (id: string | null) => void
  setScrollTargetId: (id: string | null) => void
  setScrollPageIndex: (index: number | null) => void
  setTranslationScope: (scope: TranslationState['lastScope'], pageNum: string) => void
  cancelTranslation: () => void
  toggleSentenceSelection: (sentenceId: string) => void
  selectSentenceRange: (sentenceIds: string[]) => void
  clearSentenceSelection: () => void
  addSelectionRect: (rect: import('@/types').SelectionRect) => void
  removeSelectionRect: (pageIndex: number, idx: number) => void
  clearSelectionRects: () => void
}

export function createTranslationSlice(set: any, _get: any): TranslationSlice {
  return {
    translation: {
      isTranslating: false,
      blocks: [],
      activeSentenceId: null,
      selectedSentenceIds: [],
      scrollTargetId: null,
      scrollPageIndex: null,
      progress: { current: 0, total: 0 },
      phase: 'idle',
      error: null,
      translatedFilePath: null,
      fileSha256: null,
      fileIdentityPending: false,
      tokensPerSec: 0,
      elapsed: 0,
      firstTokenMs: 0,
      statusMsg: '',
      lastScope: 'full',
      lastPageNum: '',
      extractOnly: false,
      selectedRects: [],
    },

    prepareForFile: (filePath) => {
      // Immediately try path-based cache (backward compat), then
      // if found → show it; if not → request SHA256 identity from backend
      let loaded = false
      try {
        const raw = localStorage.getItem(`zhiban-trans-${encodeURIComponent(filePath)}`)
        if (raw) {
          const blocks = JSON.parse(raw)
          if (Array.isArray(blocks) && blocks.length > 0) {
            loaded = true
            set((s: any) => ({
              translation: {
                ...s.translation,
                blocks,
                isTranslating: false,
                activeSentenceId: null,
                selectedSentenceIds: [],
                progress: {
                  current: blocks.reduce((acc: number, b: any) =>
                    acc + (b.sentences || []).filter((sb: any) => sb.translation?.length > 0).length, 0),
                  total: blocks.reduce((acc: number, b: any) => acc + (b.sentences || []).length, 0),
                },
                phase: 'done',
                error: null,
                translatedFilePath: filePath,
                fileSha256: null,
                fileIdentityPending: true,
                tokensPerSec: 0,
                elapsed: 0,
                firstTokenMs: 0,
                statusMsg: '',
              }
            }))
          }
        }
      } catch {}

      if (!loaded) {
        set((s: any) => ({
          translation: {
            ...s.translation,
            isTranslating: false,
            blocks: [],
            activeSentenceId: null,
            selectedSentenceIds: [],
            progress: { current: 0, total: 0 },
            phase: 'idle',
            error: null,
            translatedFilePath: filePath,
            fileSha256: null,
            fileIdentityPending: true,
            tokensPerSec: 0,
            elapsed: 0,
            firstTokenMs: 0,
            statusMsg: '',
          }
        }))
      }
    },

    setFileIdentity: (filePath, sha256, size) => {
      const state = _get().translation
      if (state.translatedFilePath !== filePath) return

      // Migrate old path-based cache to SHA256 key
      if (!state.blocks.length || state.phase !== 'done') {
        if (_migrateKey(filePath, sha256)) {
          try {
            const raw = localStorage.getItem(_cacheKey(sha256))
            if (raw) {
              const blocks = JSON.parse(raw)
              if (Array.isArray(blocks) && blocks.length > 0) {
                set((s: any) => ({
                  translation: {
                    ...s.translation,
                    blocks,
                    isTranslating: false,
                    progress: {
                      current: blocks.reduce((acc: number, b: any) =>
                        acc + (b.sentences || []).filter((sb: any) => sb.translation?.length > 0).length, 0),
                      total: blocks.reduce((acc: number, b: any) => acc + (b.sentences || []).length, 0),
                    },
                    phase: 'done',
                    fileSha256: sha256,
                    fileIdentityPending: false,
                  }
                }))
                return
              }
            }
          } catch {}
        }
      }

      // Already loaded from path-based cache — just record the SHA256
      set((s: any) => ({
        translation: {
          ...s.translation,
          fileSha256: sha256,
          fileIdentityPending: false,
        }
      }))
    },

    startTranslation: (filePath?, extractOnly = false) => set((state: any) => ({
      translation: {
        ...state.translation,
        isTranslating: true,
        blocks: extractOnly ? state.translation.blocks : [],
        activeSentenceId: null,
        scrollPageIndex: null,
        progress: { current: 0, total: 0 },
        phase: 'extracting',
        error: null,
        translatedFilePath: filePath || state.translation.translatedFilePath,
        tokensPerSec: 0,
        elapsed: 0,
        firstTokenMs: 0,
        extractOnly,
        selectedRects: [],
      }
    })),

    loadCachedTranslation: (filePath: string) => {
      // Legacy: only used as fallback. Uses sha256 if available, or old path-based key.
      const state = _get().translation
      const sha256 = state.fileSha256
      try {
        let raw: string | null = null
        if (sha256) {
          raw = localStorage.getItem(_cacheKey(sha256))
          if (!raw) { _migrateKey(filePath, sha256); raw = localStorage.getItem(_cacheKey(sha256)) }
        }
        if (!raw) raw = localStorage.getItem(`zhiban-trans-${encodeURIComponent(filePath)}`)
        if (!raw) return false
        const blocks = JSON.parse(raw)
        if (!Array.isArray(blocks) || blocks.length === 0) return false
        set((state2: any) => ({
          translation: {
            ...state2.translation,
            blocks,
            isTranslating: false,
            progress: { current: blocks.reduce((acc: number, b: any) =>
              acc + (b.sentences || []).filter((s: any) => s.translation?.length > 0).length, 0),
              total: blocks.reduce((acc: number, b: any) => acc + (b.sentences || []).length, 0) },
            phase: 'done',
            translatedFilePath: filePath,
            fileSha256: sha256 || state2.translation.fileSha256,
            error: null,
          }
        }))
        return true
      } catch {
        return false
      }
    },

    setTranslationBlocks: (blocks, total) => set((state: any) => ({
      translation: {
        ...state.translation,
        blocks,
        progress: { current: 0, total },
        phase: 'translating',
        statusMsg: '',  // clear extracting message so phaseLabel shows
      }
    })),

    appendTranslationToken: (sentenceId, token, isFirst) => set((state: any) => {
      const blocks = state.translation.blocks.map((block: TranslationBlock) => ({
        ...block,
        sentences: block.sentences.map((s) => {
          if (s.id !== sentenceId) return s
          return {
            ...s,
            translation: isFirst ? token : s.translation + token,
            isComplete: false,
          }
        }),
      }))
      let completed = 0
      for (const b of blocks) {
        for (const s of b.sentences) {
          if (s.translation.length > 0) completed++
        }
      }
      return {
        translation: {
          ...state.translation,
          blocks,
          progress: { ...state.translation.progress, current: completed },
        }
      }
    }),

    finishTranslation: (totalBlocks, totalSentences) => set((state: any) => {
      const blocks = state.translation.blocks.map((block: TranslationBlock) => ({
        ...block,
        sentences: block.sentences.map((s) => ({ ...s, isComplete: true })),
      }))
      // Reset extractOnly — extraction is done, structure is available
      const extractOnly = false
      const fp = state.translation.translatedFilePath
      const sha = state.translation.fileSha256
      if (blocks.length > 0) {
        try {
          // Always save to path-based key (primary, backward compat)
          if (fp) {
            localStorage.setItem(`zhiban-trans-${encodeURIComponent(fp)}`, JSON.stringify(blocks))
          }
          // Also save to SHA256 key for content-based dedup
          if (sha) {
            localStorage.setItem(_cacheKey(sha), JSON.stringify(blocks))
          }
        } catch {}
      }
      return {
        translation: {
          ...state.translation,
          blocks,
          isTranslating: false,
          progress: { current: totalSentences, total: totalSentences },
          phase: 'done',
          extractOnly,
        }
      }
    }),

    setTranslationSpeed: (tokensPerSec, elapsed, firstTokenMs?) => set((state: any) => ({
      translation: { ...state.translation, tokensPerSec, elapsed, ...(firstTokenMs !== undefined ? { firstTokenMs } : {}) }
    })),

    setTranslationStatusMsg: (msg) => set((state: any) => ({
      translation: { ...state.translation, statusMsg: msg }
    })),

    setTranslationPhase: (phase) => set((state: any) => ({
      translation: { ...state.translation, phase }
    })),

    setTranslationError: (error) => set((state: any) => ({
      translation: {
        ...state.translation,
        isTranslating: false,
        phase: 'error',
        error,
      }
    })),

    setActiveSentenceId: (id) => set((state: any) => ({
      translation: { ...state.translation, activeSentenceId: id }
    })),

    setScrollTargetId: (id) => set((state: any) => ({
      translation: { ...state.translation, scrollTargetId: id }
    })),

    setScrollPageIndex: (index) => set((state: any) => ({
      translation: { ...state.translation, scrollPageIndex: index }
    })),

    setTranslationScope: (scope, pageNum) => set((state: any) => ({
      translation: { ...state.translation, lastScope: scope, lastPageNum: pageNum }
    })),

    cancelTranslation: () => set((state: any) => {
      const shouldClearScreen = state.screenContext.source === 'translation'
      const preserveBlocks = state.translation.extractOnly
      const next: any = {
        translation: {
          ...state.translation,
          isTranslating: false,
          blocks: preserveBlocks ? state.translation.blocks : [],
          activeSentenceId: null,
          selectedSentenceIds: [],
          scrollTargetId: null,
          scrollPageIndex: null,
          tokensPerSec: 0,
          elapsed: 0,
          firstTokenMs: 0,
          statusMsg: '',
          progress: preserveBlocks
            ? state.translation.progress
            : { current: 0, total: 0 },
          phase: preserveBlocks ? 'done' : 'idle',
          error: null,
          fileSha256: preserveBlocks ? state.translation.fileSha256 : null,
          fileIdentityPending: false,
          extractOnly: false,
          selectedRects: [],
        },
      }
      if (shouldClearScreen) {
        next.screenContext = {
          isActive: false,
          currentDoc: null,
          ocrParagraphs: [],
          activeParagraphIndex: null,
          source: null,
          lastUpdateTime: null,
        }
      }
      return next
    }),

    toggleSentenceSelection: (sentenceId) => set((state: any) => {
      const selected = state.translation.selectedSentenceIds
      if (selected.includes(sentenceId)) {
        return { translation: { ...state.translation, selectedSentenceIds: selected.filter((id: string) => id !== sentenceId) } }
      }
      return { translation: { ...state.translation, selectedSentenceIds: [...selected, sentenceId] } }
    }),

    selectSentenceRange: (sentenceIds) => set((state: any) => ({
      translation: { ...state.translation, selectedSentenceIds: sentenceIds }
    })),

    clearSentenceSelection: () => set((state: any) => ({
      translation: { ...state.translation, selectedSentenceIds: [] }
    })),

    addSelectionRect: (rect) => set((state: any) => ({
      translation: {
        ...state.translation,
        selectedRects: [...state.translation.selectedRects, rect],
      }
    })),

    removeSelectionRect: (pageIndex, idx) => set((state: any) => {
      const rects = state.translation.selectedRects.filter((_: any, i: number) => i !== idx)
      return { translation: { ...state.translation, selectedRects: rects } }
    }),

    clearSelectionRects: () => set((state: any) => ({
      translation: { ...state.translation, selectedRects: [] }
    })),
  }
}

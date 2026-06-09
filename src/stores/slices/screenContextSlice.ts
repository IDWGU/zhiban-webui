import type { ScreenContextState, OcrParagraph } from '@/types'

export interface ScreenContextSlice {
  screenContext: ScreenContextState
  contextPaused: boolean
  updateOcrResult: (paragraphs: OcrParagraph[], activeIndex: number | null, doc: string, source?: ScreenContextState['source']) => void
  setActiveParagraph: (index: number) => void
  clearScreenContext: () => void
  pauseContext: () => void
  resumeContext: () => void
  setScreenActive: (active: boolean) => void
  toggleParagraphSelection: (index: number) => void
  clearParagraphSelections: () => void
  selectedParagraphIndices: number[]
  setSelectedParagraphIndices: (indices: number[]) => void
}

export function createScreenContextSlice(set: any, _get: any): ScreenContextSlice {
  return {
    selectedParagraphIndices: [],

    screenContext: {
      isActive: false,
      currentDoc: null,
      ocrParagraphs: [],
      activeParagraphIndex: null,
      source: null,
      lastUpdateTime: null,
    },

    updateOcrResult: (paragraphs, activeIndex, doc, source) => set((state: any) => ({
      screenContext: {
        ...state.screenContext,
        isActive: true,
        ocrParagraphs: paragraphs,
        activeParagraphIndex: activeIndex,
        currentDoc: doc ?? state.screenContext.currentDoc,
        source: source ?? state.screenContext.source,
        lastUpdateTime: Date.now(),
      }
    })),

    setActiveParagraph: (index) => set((state: any) => ({
      screenContext: {
        ...state.screenContext,
        activeParagraphIndex: index,
        lastUpdateTime: Date.now(),
      }
    })),

    contextPaused: false,

    clearScreenContext: () => set((state: any) => ({
      selectedParagraphIndices: [],
      contextPaused: true,
      screenContext: { ...state.screenContext, isActive: false }
    })),

    pauseContext: () => set({ contextPaused: true }),

    resumeContext: () => set((state: any) => ({
      contextPaused: false,
      screenContext: { ...state.screenContext, isActive: true }
    })),

    setScreenActive: (active) => set((state: any) => ({
      screenContext: { ...state.screenContext, isActive: active }
    })),

    toggleParagraphSelection: (index) => set((state: any) => {
      const current: number[] = state.selectedParagraphIndices ?? []
      const exists = current.includes(index)
      return {
        selectedParagraphIndices: exists
          ? current.filter((i: number) => i !== index).sort((a: number, b: number) => a - b)
          : [...current, index].sort((a: number, b: number) => a - b)
      }
    }),

    clearParagraphSelections: () => set({ selectedParagraphIndices: [] }),

    setSelectedParagraphIndices: (indices) => set({ selectedParagraphIndices: indices }),
  }
}

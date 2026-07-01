import type { SettingsState } from '@/types'

function loadSettings(): Partial<SettingsState> {
  try {
    const raw = localStorage.getItem('zhiban-settings')
    if (raw) {
      const parsed = JSON.parse(raw)
      if (!parsed.rememberApiKey) {
        delete parsed.llmApiKey
      }
      return parsed
    }
  } catch {}
  return {}
}

export function saveSettings(s: SettingsState) {
  try {
    if (s.rememberApiKey) {
      localStorage.setItem('zhiban-settings', JSON.stringify(s))
    } else {
      const { llmApiKey: _, ...safe } = s
      localStorage.setItem('zhiban-settings', JSON.stringify(safe))
    }
  } catch {}
}

const saved = loadSettings()

export interface SettingsSlice {
  settings: SettingsState
  toggleTheme: () => void
  updateSettings: (partial: Partial<SettingsState>) => void
  llmTestResult: { success: boolean; error: string; model: string } | null
  setLlmTestResult: (success: boolean, error: string, model: string) => void
  availableModels: { name: string; path: string }[]
  setAvailableModels: (models: { name: string; path: string }[]) => void
  embeddingLoadResult: { success: boolean; dim?: number; error?: string } | null
  setEmbeddingLoadResult: (r: { success: boolean; dim?: number; error?: string } | null) => void
  embeddingProgress: { percent: number; message: string }
  setEmbeddingLoadProgress: (percent: number, message: string) => void
}

export function createSettingsSlice(set: any, get: any): SettingsSlice {
  return {
    settings: {
      theme: saved.theme ?? 'dark',
      llmApiKey: saved.llmApiKey ?? '',
      llmBaseUrl: saved.llmBaseUrl ?? '__local__',
      systemPrompt: saved.systemPrompt ?? '',
      llmModel: saved.llmModel ?? 'deepseek-v4-pro',
      thinkingMode: saved.thinkingMode ?? false,  // 默认关：小模型 thinking 会吃掉全部 token
      topK: saved.topK ?? 5,
      llmTopP: saved.llmTopP ?? 0.8,
      llmTopK: saved.llmTopK ?? 40,
      llmTemperature: saved.llmTemperature ?? 0.05,
      llmRepeatPenalty: saved.llmRepeatPenalty ?? 1.2,
      llmFrequencyPenalty: saved.llmFrequencyPenalty ?? 0.3,
      llmPresencePenalty: saved.llmPresencePenalty ?? 0.4,
      llmStopTokens: saved.llmStopTokens ?? '<|im_end|>,<|endoftext|>',
      llmExtraHeaders: saved.llmExtraHeaders ?? '',
      llmExtraBody: saved.llmExtraBody ?? '',
      rememberApiKey: saved.rememberApiKey ?? false,
      llmModelPath: saved.llmModelPath ?? '',
      translationModelPath: saved.translationModelPath ?? '',
      embeddingModel: saved.embeddingModel ?? 'jinaai/jina-embeddings-v5-text-nano',
      modelCacheDir: saved.modelCacheDir ?? '',
      llmFlashAttn: saved.llmFlashAttn ?? true,
      llmUseMmap: saved.llmUseMmap ?? true,
      llmNBatch: saved.llmNBatch ?? 2048,
      llmNUbatch: saved.llmNUbatch ?? 1024,
    },
    llmTestResult: null,
    availableModels: [],

    toggleTheme: () => set((state: any) => {
      const next = state.settings.theme === 'dark' ? 'light' : 'dark'
      document.documentElement.setAttribute('data-theme', next)
      return { settings: { ...state.settings, theme: next } }
    }),

    setLlmTestResult: (success, error, model) => set({
      llmTestResult: { success, error, model },
    }),

    setAvailableModels: (models) => set({ availableModels: models }),

    embeddingLoadResult: null,
    setEmbeddingLoadResult: (r) => set({ embeddingLoadResult: r }),
    embeddingProgress: { percent: 0, message: '' },
    setEmbeddingLoadProgress: (percent, message) => set({ embeddingProgress: { percent, message } }),

    updateSettings: (partial) => set((state: any) => ({
      settings: { ...state.settings, ...partial }
    })),
  }
}

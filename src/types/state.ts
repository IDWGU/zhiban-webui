// ===== 全局 State 类型 =====

import type { OcrParagraph, Message, Citation, PaperRef, PaperTab, AgentStep } from './models'
import type { BuildIndexProgressMessage } from './websocket'

export interface ScreenContextState {
  isActive: boolean
  currentDoc: string | null
  ocrParagraphs: OcrParagraph[]
  activeParagraphIndex: number | null
  source: 'document' | 'translation' | 'ocr' | 'ax' | null
  lastUpdateTime: number | null
}

export interface ConversationState {
  messages: Message[]
  isStreaming: boolean
  streamingMessageId: string | null
  streamingText: string
  streamingCitations: Citation[]
  streamingRelatedPapers: PaperRef[]
  streamingAgentSteps: AgentStep[]
  streamingThinkingText: string
}

export interface ConnectionState {
  wsStatus: 'waiting' | 'connecting' | 'connected' | 'disconnected' | 'reconnecting'
  sidecarStatus: 'running' | 'stopped' | 'error'
  reconnectAttempt: number
  debugMode: boolean
  localEngineLoading: boolean
  startupMessage: string
  showDevPanel: boolean
}

export interface SettingsState {
  theme: 'dark' | 'light'
  llmApiKey: string
  llmBaseUrl: string
  systemPrompt: string
  llmModel: string
  thinkingMode: boolean
  topK: number
  llmTopP: number
  llmTopK: number
  llmTemperature: number
  llmRepeatPenalty: number
  llmFrequencyPenalty: number
  llmPresencePenalty: number
  llmStopTokens: string
  llmExtraHeaders: string
  llmExtraBody: string
  rememberApiKey: boolean
  // Local model settings
  llmModelPath: string
  translationModelPath: string
  embeddingModel: string
  modelCacheDir: string
  // llama.cpp loading parameters
  llmFlashAttn: boolean
  llmUseMmap: boolean
  llmNBatch: number
  llmNUbatch: number
}

export interface TranslationSentence {
  id: string
  text: string
  translation: string
  isComplete: boolean
  rects: Array<{ x: number; y: number; w: number; h: number }>
}

export interface TranslationBlock {
  id: string
  type: 'heading' | 'paragraph' | 'table' | 'formula'
  level?: number
  sentences: TranslationSentence[]
  pageNum: number
}

export interface TranslationState {
  isTranslating: boolean
  blocks: TranslationBlock[]
  activeSentenceId: string | null
  selectedSentenceIds: string[]
  scrollTargetId: string | null
  scrollPageIndex: number | null
  progress: { current: number; total: number }
  phase: 'idle' | 'extracting' | 'translating' | 'done' | 'error'
  error: string | null
  translatedFilePath: string | null
  fileSha256: string | null
  fileIdentityPending: boolean
  tokensPerSec: number
  elapsed: number
  firstTokenMs: number
  statusMsg: string
  lastScope: 'full' | 'page' | 'selection'
  lastPageNum: string
  extractOnly: boolean
  selectedRects: SelectionRect[]
}

export interface SelectionRect {
  pageIndex: number
  x: number
  y: number
  w: number
  h: number
}

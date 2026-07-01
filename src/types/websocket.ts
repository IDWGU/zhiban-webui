// ===== WebSocket 消息协议 — 客户端 → 服务端 =====

import type { OcrParagraph } from './models'
import type { TranslationBlock } from './state'

export type WsClientMessage =
  | ScreenUpdateMessage
  | UserQueryMessage
  | ControlMessage
  | TranslationRequestMessage
  | ComputeFileIdentityMessage
  | BindPaperMessage
  | LlmTestMessage
  | LlmListModelsMessage
  | ImportVectorStoreMessage
  | BuildIndexMessage
  | BuildControlMessage
  | AddPapersMessage
  | ListLibraryMessage
  | OpenLibraryPaperMessage
  | OcrControlMessage
  | NewConversationMessage
  | SwitchConversationMessage
  | ListConversationsMessage
  | DeleteConversationMessage
  | RenameConversationMessage
  | ModelConfigMessage
  | RegenerateLastMessage

export interface ScreenUpdateMessage {
  type: 'screen_update'
  screenshot: ArrayBuffer
  mouseX: number
  mouseY: number
  activeWindowTitle: string
  timestamp: number
}

export interface UserQueryMessage {
  type: 'user_query'
  queryText: string
  context: {
    activeDoc: string | null
    activeParagraph: string
    paragraphIndex: number | null
    conversationId: string
  }
  timestamp: number
  apiKey?: string
  baseUrl?: string
  systemPrompt?: string
  model?: string
  thinking?: boolean
  topK?: number
  topP?: number
  history?: Array<{ role: string; content: string }>
  openPapers?: Array<{ paperId: number | string; title: string; filename: string }>
}

export interface ControlMessage {
  type: 'control'
  action: 'cancel_query'
  payload?: Record<string, unknown>
}

export interface NewConversationMessage {
  type: 'new_conversation'
  name?: string
}

export interface SwitchConversationMessage {
  type: 'switch_conversation'
  conversationId: string
}

export interface ListConversationsMessage {
  type: 'list_conversations'
}

export interface DeleteConversationMessage {
  type: 'delete_conversation'
  conversationId: string
}

export interface RenameConversationMessage {
  type: 'rename_conversation'
  conversationId: string
  name: string
}

export interface RegenerateLastMessage {
  type: 'regenerate_last'
  conversationId?: string
  apiKey?: string
  model?: string
  baseUrl?: string
  thinking?: boolean
}

export interface LlmTestMessage {
  type: 'llm_test'
  apiKey: string
  model: string
  baseUrl?: string
}

export interface LlmListModelsMessage {
  type: 'llm_list_models'
  apiKey: string
  baseUrl?: string
}

export interface TranslationRequestMessage {
  type: 'translation_request'
  filePath: string
  scope: 'full' | 'page' | 'selection'
  page?: number
  selectionRange?: { startPage: number; endPage: number }
  selectionRects?: Array<{ pageIndex: number; x: number; y: number; w: number; h: number }>
  apiKey?: string
  baseUrl?: string
  model?: string
  thinking?: boolean
  useLocal?: boolean
}

export interface ComputeFileIdentityMessage {
  type: 'compute_file_identity'
  filePath: string
}

export interface FileIdentityResultMessage {
  type: 'file_identity_result'
  filePath: string
  sha256?: string
  size?: number
  error?: string
}

export interface BindPaperMessage {
  type: 'bind_paper'
  filePath: string
  paperId?: string
  title?: string
}

export interface ImportVectorStoreMessage {
  type: 'import_vector_store'
  sourcePath: string
}

export interface ImportVectorResultMessage {
  type: 'import_vector_result'
  success: boolean
  chunks?: number
  error?: string
}

export interface BuildIndexMessage {
  type: 'build_index'
  sourcePath?: string
  force?: boolean
}

export interface BuildIndexProgressMessage {
  type: 'build_index_progress'
  phase: 'scanning' | 'extracting' | 'embedding' | 'paused' | 'done'
  current: number
  total: number
  message: string
}

export interface BuildIndexResultMessage {
  type: 'build_index_result'
  success: boolean
  chunks?: number
  documents?: number
  duration?: number
  error?: string
}

export interface AddPapersMessage {
  type: 'add_papers'
  files: string[]
}

export interface BuildControlMessage {
  type: 'build_control'
  action: 'pause' | 'resume' | 'cancel'
}

export interface AddPapersResultMessage {
  type: 'add_papers_result'
  success: boolean
  added?: number
  library?: Array<{ name: string; path: string; size: number; mtime: number }>
  error?: string
}

export interface ListLibraryMessage {
  type: 'list_library'
}

export interface ListLibraryResultMessage {
  type: 'list_library_result'
  papers: Array<{ name: string; path: string; size: number; mtime: number }>
}

export interface ClearVectorStoreMessage {
  type: 'clear_vector_store'
}

export interface ClearVectorResultMessage {
  type: 'clear_vector_result'
  success: boolean
  error?: string
  message?: string
}

export interface RemovePaperVectorsMessage {
  type: 'remove_paper_vectors'
  doc_ids: string[]
}

export interface RemovePaperResultMessage {
  type: 'remove_paper_result'
  success: boolean
  removed?: number
  error?: string
  message?: string
}

export interface ListIndexedPapersMessage {
  type: 'list_indexed_papers'
}

export interface IndexedPaper {
  doc_id: string
  filename: string
  source: string
  chunks: number
}

export interface IndexedPapersResultMessage {
  type: 'indexed_papers_result'
  papers: IndexedPaper[]
  success?: boolean
  error?: string
}

export interface OpenLibraryPaperMessage {
  type: 'open_library_paper'
  path: string
  name: string
}

export interface OcrControlMessage {
  type: 'ocr_control'
  action: 'pause' | 'resume'
}

export interface PongMessage {
  type: 'pong'
  timestamp: number
}

export interface ImportPaperProgressMessage {
  type: 'import_paper_progress'
  phase: string
  message: string
  progress: number | null
  doc_id: string
  filename: string
}

export interface ImportPaperExistsMessage {
  type: 'import_paper_exists'
  doc_id: string
  filename: string
  chunks: number
}

export interface ImportPaperResultMessage {
  type: 'import_paper_result'
  success: boolean
  doc_id?: string
  chunks?: number
  source?: string
  filename?: string
  error?: string
}

// ===== WebSocket 消息协议 — 服务端 → 客户端 =====

export type WsServerMessage =
  | OcrResultMessage
  | LlmTokenMessage | LlmCitationMessage | LlmRelatedPapersMessage | LlmDoneMessage
  | LlmHealthMessage
  | StatusMessage
  | TranslationBlocksMessage | TranslationTokenMessage | TranslationDoneMessage
  | LlmTestResultMessage
  | LlmModelsResultMessage
  | ImportVectorResultMessage
  | BuildIndexProgressMessage
  | BuildIndexResultMessage
  | AddPapersResultMessage
  | ListLibraryResultMessage
  | WorkflowStatusMessage
  | ConversationCreatedMessage | ConversationSwitchedMessage | ConversationListMessage
  | ConversationRenamedMessage
  | PongMessage
  | ImportPaperProgressMessage | ImportPaperExistsMessage | ImportPaperResultMessage
  | FileIdentityResultMessage
  | ModelConfigResultMessage
  | AgentStepMessage | AgentThinkingMessage | AgentThinkingDoneMessage

export interface OcrResultMessage {
  type: 'ocr_result'
  source?: 'ocr' | 'ax'
  paragraphs: OcrParagraph[]
  activeParagraphIndex: number | null
  activeDoc: string
  timestamp: number
}

export interface LlmTokenMessage {
  type: 'llm_token'
  token: string
  messageId: string
  isFirst: boolean
  isThinking?: boolean
  timestamp: number
}

export interface LlmCitationMessage {
  type: 'llm_citation'
  messageId: string
  citations: Citation[]
}

export interface LlmRelatedPapersMessage {
  type: 'llm_related_papers'
  messageId: string
  papers: PaperRef[]
}

export interface LlmDoneMessage {
  type: 'llm_done'
  messageId: string
  totalTokens: number
  duration: number
  refused?: boolean
  expanded?: boolean
  responseType?: string
  mode?: string
  model?: string
  usage?: { input: number; output: number }
  loopDetected?: boolean
  regenerated?: boolean
}

export interface LlmHealthMessage {
  type: 'llm_health'
  messageId: string
  call: 'classify' | 'answer'
  timing: {
    prefill_ms: number | null
    decode_ms: number | null
    decode_per_token_ms: number | null
    total_ms: number | null
  }
  tokens: {
    prefill_tokens: number
    output_tokens: number
    cache_hit_tokens?: number
    cache_miss_tokens?: number
    cache_hit_rate: number | null
  }
  memory?: {
    vram_before_mb?: number
    vram_after_mb?: number
    vram_mb?: number
  }
  timestamp: number
}

export interface LlmTestResultMessage {
  type: 'llm_test_result'
  success: boolean
  error?: string
  model?: string
}

export interface LlmModelEntry {
  name: string
  path: string
}

export interface LlmModelsResultMessage {
  type: 'llm_models_result'
  success: boolean
  error?: string
  models?: string[]                         // 旧格式（兼容）
  model_entries?: LlmModelEntry[]           // 新格式 {name, path}
}

export interface StatusMessage {
  type: 'status'
  level: 'info' | 'warn' | 'error'
  code: string
  message: string
}

export interface WorkflowStatusMessage {
  type: 'workflow_status'
  code: string
  message: string
  timestamp: number
}

export interface ConversationCreatedMessage {
  type: 'conversation_created'
  conversationId: string
  name: string
}

export interface ConversationSwitchedMessage {
  type: 'conversation_switched'
  conversationId: string
  messages: Array<{
    role: string
    content: string
    id?: string
    thinkingContent?: string
    agentSteps?: import('./models').AgentStep[]
    mode?: string
    model?: string
    timestamp?: number
  }>
  openPapers: Array<{ paper_id: number; title: string; filename: string; filepath?: string }>
  currentTopic: string
}

export interface ConversationListMessage {
  type: 'conversation_list'
  conversations: ConversationSummary[]
}

export interface ConversationRenamedMessage {
  type: 'conversation_renamed'
  conversationId: string
  name: string
}

export interface ConversationBranchedMessage {
  type: 'conversation_branched'
  conversationId: string
  sourceConversationId: string
  name: string
}

export interface BranchConversationMessage {
  type: 'branch_conversation'
  conversationId: string
  messageIndex: number
  name?: string
}

export interface ConversationSummary {
  id: string
  name: string
  messageCount: number
  paperCount: number
  topic: string
  isActive: boolean
  createdAt?: string
  updatedAt?: string
}

function formatRelativeTime(isoString: string): string {
  if (!isoString) return ''
  const diff = Date.now() - new Date(isoString).getTime()
  const sec = Math.floor(diff / 1000)
  if (sec < 60) return '刚刚'
  const min = Math.floor(sec / 60)
  if (min < 60) return `${min}分钟前`
  const hr = Math.floor(min / 60)
  if (hr < 24) return `${hr}小时前`
  return `${Math.floor(hr / 24)}天前`
}

export interface TranslationBlocksMessage {
  type: 'translation_blocks'
  blocks: TranslationBlock[]
  totalSentences: number
}

export interface TranslationTokenMessage {
  type: 'translation_token'
  sentenceId: string
  token: string
  isFirst: boolean
}

export interface TranslationDoneMessage {
  type: 'translation_done'
  totalBlocks: number
  totalSentences: number
  duration: number
}

// ===== Model Config =====

export interface ModelConfigMessage {
  type: 'model_config'
  action: 'get' | 'set_local_model' | 'set_embedding_model' | 'set_llm_params'
  path?: string
  model?: string
  params?: {
    flash_attn?: boolean
    use_mmap?: boolean
    n_batch?: number
    n_ubatch?: number
  }
}

export interface ModelConfigResultMessage {
  type: 'model_config_result'
  action: string
  success?: boolean
  error?: string
  status?: string
  enabled?: boolean
  config?: {
    llm_model_path: string
    translation_model_path: string
    embedding_model: string
    model_cache_dir: string
    chroma_dir: string
    embedding_available: boolean
    debug: boolean
    local_engine_loaded: boolean
    local_engine_loading: boolean
    local_engine_backend: string
    llm_flash_attn: boolean
    llm_use_mmap: boolean
    llm_n_batch: number
    llm_n_ubatch: number
  }
  path?: string
  model?: string
  dim?: number
}

// ===== Agent 模式消息 — 服务端 → 客户端 =====

export interface AgentStepMessage {
  type: 'agent_step'
  messageId: string
  stepIndex: number
  phase: 'thinking' | 'tool_call' | 'tool_result'
  content?: string
  toolName?: string
  toolArgs?: string
  toolResult?: string
  isFinal?: boolean
  success?: boolean
  error?: string
  durationMs?: number
}

export interface AgentThinkingMessage {
  type: 'agent_thinking'
  messageId: string
  token: string
  isFirst: boolean
}

export interface AgentThinkingDoneMessage {
  type: 'agent_thinking_done'
  messageId: string
}

// Forward references from models
import type { Citation, PaperRef } from './models'

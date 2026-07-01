// 知伴 TypeScript 类型定义 — 按领域拆分，统一重导出

// WebSocket 消息协议
export type {
  WsClientMessage, WsServerMessage,
  ScreenUpdateMessage, UserQueryMessage, ControlMessage,
  LlmTestMessage, LlmListModelsMessage,
  TranslationRequestMessage,
  ImportVectorStoreMessage, ImportVectorResultMessage,
  BuildIndexMessage, BuildIndexProgressMessage, BuildIndexResultMessage,
  AddPapersMessage, AddPapersResultMessage,
  OcrControlMessage, PongMessage,
  ImportPaperProgressMessage, ImportPaperExistsMessage, ImportPaperResultMessage,
  NewConversationMessage, SwitchConversationMessage,
  ListConversationsMessage, DeleteConversationMessage, RenameConversationMessage,
  OcrResultMessage,
  LlmTokenMessage, LlmCitationMessage, LlmRelatedPapersMessage, LlmDoneMessage,
  LlmHealthMessage,
  LlmTestResultMessage, LlmModelsResultMessage,
  StatusMessage, WorkflowStatusMessage,
  ConversationCreatedMessage, ConversationSwitchedMessage,
  ConversationListMessage, ConversationRenamedMessage,
  ConversationSummary,
  TranslationBlocksMessage, TranslationTokenMessage, TranslationDoneMessage,
  AgentStepMessage, AgentThinkingMessage, AgentThinkingDoneMessage,
} from './websocket'

// 数据模型
export type {
  OcrParagraph, Message, Citation, PaperRef, PaperTab,
  AgentStep,
} from './models'

// State 类型
export type {
  ScreenContextState, ConversationState, ConnectionState, SettingsState,
  TranslationSentence, TranslationBlock, TranslationState,
} from './state'

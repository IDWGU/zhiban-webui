// ===== 数据模型 =====

export interface OcrParagraph {
  index: number
  text: string
  bbox: { x: number; y: number; w: number; h: number }
  confidence: number
}

export interface AgentStep {
  stepIndex: number
  phase: 'thinking' | 'tool_call' | 'tool_result'
  content?: string
  toolName?: string
  toolArgs?: string
  toolResult?: string
  success?: boolean
  error?: string
  durationMs?: number
}

export interface Message {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  citations?: Citation[]
  relatedPapers?: PaperRef[]
  timestamp: number
  screenContext?: {
    docName: string
    paragraphText: string
  }
  usage?: { input: number; output: number; elapsed?: number }
  mode?: string
  model?: string
  loopDetected?: boolean
  agentSteps?: AgentStep[]
  thinkingContent?: string
}

export interface Citation {
  index: number
  paperId: number
  title: string
  chunkText: string
  sectionType: string
}

export interface PaperRef {
  paperId: number
  title: string
  relationType: 'successor' | 'precursor' | 'featured' | 'related'
  year: number
  relevance: number
}

export interface PaperTab {
  id: string
  name: string
  path?: string
  type: 'pdf' | 'docx' | 'txt' | 'md'
  extractedText: string
}

import { useCallback, useMemo } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkBreaks from 'remark-breaks'
import type { Message } from '@/types'
import CodeBlock from './CodeBlock'
import { useAppStore } from '@/stores/appStore'
import StreamingMarkdown from './StreamingMarkdown'
import CitationList from './CitationList'
import RelatedPapers from './RelatedPapers'
import ThinkingProcess from './ThinkingProcess'

interface Props {
  message: Message
  isStreaming?: boolean
}

export default function MessageBubble({ message, isStreaming }: Props) {
  const isUser = message.role === 'user'

  const msgCount = useAppStore(s => s.conversation.messages.length)

  // 流式消息：从 store 读取思考状态
  const streamingThinkingText = useAppStore(s => s.conversation.streamingThinkingText)
  const streamingAgentSteps = useAppStore(s => s.conversation.streamingAgentSteps)
  const contentStarted = isStreaming ? message.content.length > 0 : false

  // 思考内容来源：流式时从 store，完成时从 message
  const thinkingContent = isStreaming ? streamingThinkingText : message.thinkingContent
  const agentSteps = isStreaming ? streamingAgentSteps : message.agentSteps

  // 用 useMemo 缓存 markdownComponents，避免每次渲染重新创建
  const msgIndex = msgCount - 1  // 当前正在渲染的消息（通常是最后一条 assistant）
  const mdComponents = useMemo(() => makeMarkdownComponents(msgIndex), [msgIndex])

  const handleRegenerate = useCallback(() => {
    const store = useAppStore.getState()
    // 立即删除最后一条 assistant 消息（UI 即时响应）
    const msgs = store.conversation.messages
    if (msgs.length > 0 && msgs[msgs.length - 1].role === 'assistant') {
      store.popLastMessage()
    }
    // 发送重新生成请求
    const send = (window as any).__zhiban_wsSend
    if (send) send({ type: 'regenerate_last' })
  }, [])

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', gap: 4,
      padding: '6px 14px', marginBottom: 4,
      alignItems: isUser ? 'flex-end' : 'flex-start',
    }}>
      {/* Screen context indicator */}
      {message.screenContext && (
        <div style={{
          fontSize: 10, color: 'hsl(var(--muted-foreground))',
          padding: '3px 8px', borderRadius: 6,
          background: 'hsl(var(--muted))',
          borderLeft: '2px solid hsl(var(--border))',
          maxWidth: '88%',
        }}>
          正在阅读: {message.screenContext.docName}
        </div>
      )}

      {/* 思考过程面板 — 仅 assistant 消息 */}
      {!isUser && (
        <ThinkingProcess
          agentSteps={agentSteps}
          thinkingContent={thinkingContent}
          isStreaming={isStreaming}
          contentStarted={contentStarted}
        />
      )}

      {/* Message bubble — Proma style */}
      <div style={{
        display: 'inline-block',
        maxWidth: '88%',
        padding: '10px 14px',
        borderRadius: 10,
        background: isUser
          ? 'hsl(var(--primary) / 0.12)'
          : 'hsl(var(--accent))',
        fontSize: 13, lineHeight: 1.65,
        color: isUser
          ? 'hsl(var(--primary) / 0.90)'
          : 'hsl(var(--foreground))',
        overflowWrap: 'break-word', wordBreak: 'break-word',
      }}>
        {/* 思考和 agent 行为已直接内联到气泡正文流中 */}

        {isStreaming ? (
          <StreamingMarkdown />
        ) : (
          <div
            className="markdown-content"
            onClick={(e) => {
              const target = e.target as HTMLElement
              const badge = target.closest?.('.paper-ref-badge') as HTMLElement | null
              if (badge && badge.dataset.paperId) {
                // Focus query input with paper reference
                const input = document.querySelector('.query-textarea') as HTMLTextAreaElement
                if (input) {
                  const current = (input as any).value || ''
                  const ref = `Paper #${badge.dataset.paperId} `
                  if (!current.includes(ref)) {
                    const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
                      window.HTMLTextAreaElement.prototype, 'value'
                    )?.set
                    nativeInputValueSetter?.call(input, current + ref)
                    input.dispatchEvent(new Event('input', { bubbles: true }))
                  }
                  input.focus()
                }
              }
            }}
          >
            <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]} components={mdComponents}>
              {preprocessCitations(message.content)}
            </ReactMarkdown>
          </div>
        )}
      </div>

      {/* Loop detected — regenerate button */}
      {!isUser && message.loopDetected && !isStreaming && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8,
          padding: '6px 10px', borderRadius: 8,
          background: 'hsl(30, 80%, 15%)', border: '1px solid hsl(30, 60%, 30%)',
          fontSize: 12,
        }}>
          <span style={{ color: 'hsl(30, 70%, 70%)' }}>检测到回复出现重复</span>
          <button onClick={handleRegenerate} style={{
            padding: '4px 12px', borderRadius: 6,
            background: 'hsl(30, 60%, 35%)', color: '#fff',
            border: 'none', cursor: 'pointer', fontSize: 12, fontWeight: 600,
          }}>
            重新生成
          </button>
        </div>
      )}

      {/* Citations */}
      {message.citations && message.citations.length > 0 && (
        <CitationList citations={message.citations} />
      )}

      {/* Related papers */}
      {message.relatedPapers && message.relatedPapers.length > 0 && (
        <RelatedPapers papers={message.relatedPapers} />
      )}

      {/* Meta row: actions + mode badge + model name + timestamp + word count */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 6,
        padding: '0 4px', flexWrap: 'wrap',
      }}>
        {/* Fork + Edit + Delete buttons */}
        {!isStreaming && (
          <>
            <button
              onClick={() => {
                const store = useAppStore.getState()
                const msgIdx = store.conversation.messages.findIndex(m => m.id === message.id)
                if (msgIdx >= 0) {
                  const send = (window as any).__zhiban_wsSend
                  if (send) send({
                    type: 'branch_conversation',
                    conversationId: store.activeConversationId,
                    messageIndex: msgIdx,
                  })
                }
              }}
              title="从此处分叉新对话"
              style={actionBtnStyle}
            >⑂</button>
            {isUser && (
              <button
                onClick={() => {
                  const newText = prompt('编辑消息:', message.content)
                  if (newText && newText.trim() && newText !== message.content) {
                    const store = useAppStore.getState()
                    const msgIdx = store.conversation.messages.findIndex(m => m.id === message.id)
                    if (msgIdx >= 0) {
                      const send = (window as any).__zhiban_wsSend
                      if (send) send({
                        type: 'edit_message',
                        conversationId: store.activeConversationId,
                        messageIndex: msgIdx,
                        content: newText.trim(),
                      })
                    }
                  }
                }}
                title="编辑消息"
                style={actionBtnStyle}
              >✎</button>
            )}
            <button
              onClick={() => {
                if (!confirm('确定删除此消息？')) return
                const store = useAppStore.getState()
                const msgIdx = store.conversation.messages.findIndex(m => m.id === message.id)
                if (msgIdx >= 0) {
                  const send = (window as any).__zhiban_wsSend
                  if (send) send({
                    type: 'delete_message',
                    conversationId: store.activeConversationId,
                    messageIndex: msgIdx,
                  })
                }
              }}
              title="删除消息"
              style={{...actionBtnStyle, color: 'hsl(0 60% 50%)'}}
            >✕</button>
          </>
        )}

        {message.mode && (
          <span style={{
            fontSize: 9, fontWeight: 600,
            color: message.mode === 'deep' ? '#56b6c2' : message.mode === 'quick' ? '#e5c07b' : '#98c379',
            background: 'hsl(var(--muted))',
            padding: '1px 6px', borderRadius: 9999,
          }}>
            {message.mode}
          </span>
        )}
        {message.model && (
          <span style={{
            fontSize: 10,
            color: 'hsl(var(--foreground) / 0.45)',
          }}>
            {message.model}
          </span>
        )}
        <span style={{
          fontSize: 10,
          color: 'hsl(var(--foreground) / 0.38)',
        }}>
          {formatTime(message.timestamp)}
        </span>
        {!isUser && message.content && (
          <span style={{
            fontSize: 10,
            color: 'hsl(var(--foreground) / 0.35)',
          }}>
            {message.content.length} 字
          </span>
        )}
      </div>
    </div>
  )
}

function preprocessCitations(text: string): string {
  return text
    // 新格式 (3段): 【来源: Paper #1, filename.pdf, abstract】
    .replace(
      /【来源:\s*Paper\s*#(\d+)[，,]\s*([^，,]+)[，,]\s*([^】]+)】/g,
      (_, n, fname, section) => `[\u{1F4C4} ${fname.trim()}](${n} "${section.trim()}")`
    )
    // 新格式 (2段, 仅文件名): 【来源: Paper #1, filename.pdf】
    .replace(
      /【来源:\s*Paper\s*#(\d+)[，,]\s*([^】]+?\.[a-zA-Z0-9]{2,5})】/g,
      (_, n, fname) => `[\u{1F4C4} ${fname.trim()}](${n})`
    )
    // 旧格式 (2段, 无文件名): 【来源: Paper #1, abstract】
    .replace(
      /【来源:\s*Paper\s*#(\d+)[，,]\s*([^】]+)】/g,
      (_, n, section) => `[\u{1F4C4} #${n}](${n} "${section.trim()}")`
    )
    // 旧格式 (1段): 【来源: Paper #1】
    .replace(
      /【来源:\s*Paper\s*#(\d+)】/g,
      (_, n) => `[\u{1F4C4} #${n}](${n})`
    )
}

function makeMarkdownComponents(msgIndex: number) {
  let paraCounter = 0
  return {
    a: ({ href, title, children, node, ...props }: any) => {
      if (href && /^\d+$/.test(href)) {
        const paperId = href
        return (
          <span
            className="paper-ref-badge"
            data-paper-id={paperId}
            title={title ? `Paper #${paperId} — ${title}` : `Paper #${paperId}`}
            style={{
              display: 'inline-flex', alignItems: 'center', gap: 4,
              padding: '1px 8px', borderRadius: 9999,
              fontSize: '0.85em',
              background: 'hsl(var(--citation-accent) / 0.12)',
              border: '1px solid hsl(var(--citation-accent) / 0.25)',
              color: 'hsl(var(--citation-accent))',
              cursor: 'pointer', whiteSpace: 'nowrap',
              verticalAlign: 'middle', margin: '0 2px',
            }}
          >
            {children}
          </span>
        )
      }
      return <a href={href} target="_blank" rel="noopener noreferrer" {...props}>{children}</a>
    },
    p: ({ children, ...props }: any) => {
      const text = extractTextContent(children)
      const idx = paraCounter++
      return (
        <ParagraphWithQuote text={text} isEmpty={!text?.trim()} msgIndex={msgIndex} paraIndex={idx}>
          <p {...props}>{children}</p>
        </ParagraphWithQuote>
      )
    },
    li: ({ children, ...props }: any) => {
      // 列表项不显示引用按钮（太细粒度），直接渲染
      return <li {...props}>{children}</li>
    },
    code: ({ node, className, children, inline, ...props }: any) => {
      const match = /language-(\w+)/.exec(className || '')
      if (!inline && match) {
        return <CodeBlock code={String(children).replace(/\n$/, '')} language={match[1]} maxLines={30} />
      }
      if (!inline && !match) {
        return <CodeBlock code={String(children).replace(/\n$/, '')} language="plaintext" maxLines={30} />
      }
      return (
        <code className={className} style={{
          background: 'hsl(var(--muted) / 0.60)',
          padding: '2px 5px', borderRadius: 4,
          fontFamily: 'monospace', fontSize: '0.9em',
        }} {...props}>{children}</code>
      )
    },
  }
}

function extractTextContent(node: any): string {
  if (typeof node === 'string') return node
  if (typeof node === 'number') return String(node)
  if (Array.isArray(node)) return node.map(extractTextContent).join('')
  if (node && typeof node === 'object' && 'props' in node) {
    return extractTextContent(node.props.children)
  }
  return ''
}

function ParagraphWithQuote({ text, isEmpty, msgIndex, paraIndex, children }: {
  text: string; isEmpty: boolean; msgIndex: number; paraIndex: number; children: React.ReactNode
}) {
  const addQuote = useAppStore(s => s.addQuote)

  // 无实质内容（空、纯数字编号、纯标点、＜5 个有效字符）→ 不显示引用按钮
  const meaningful = text.replace(/[\s\d\.\,\;\:\!\?\-\–\#\*\(\)\[\]【】「」『』""'']/g, '').trim()
  if (isEmpty || meaningful.length < 3) return <>{children}</>

  const handleQuote = (e: React.MouseEvent) => {
    e.stopPropagation()
    addQuote({
      msgIndex,
      paraIndex,
      text: text.slice(0, 300),
    })
  }

  return (
    <div style={{ display: 'flex', gap: 4, alignItems: 'flex-start' }}>
      <div style={{ flex: 1, minWidth: 0 }}>{children}</div>
      <button
        onClick={handleQuote}
        title="引用此段落"
        style={{
          flexShrink: 0,
          width: 20, height: 20, marginTop: 1,
          borderRadius: 4,
          border: '1px solid hsl(var(--border) / 0.40)',
          background: 'hsl(var(--muted))',
          color: 'hsl(var(--muted-foreground))',
          cursor: 'pointer', fontSize: 10,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          padding: 0, lineHeight: 1,
          opacity: 0.5,
        }}
      >
        {'"'}
      </button>
    </div>
  )
}

const actionBtnStyle: React.CSSProperties = {
  padding: '1px 6px', borderRadius: 4,
  border: '1px solid hsl(var(--border))',
  background: 'hsl(var(--muted))',
  color: 'hsl(var(--muted-foreground))',
  cursor: 'pointer', fontSize: 10, fontFamily: 'inherit',
  opacity: 0.55, lineHeight: 1.5,
}

function formatTime(ts: number): string {
  return new Date(ts).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
}

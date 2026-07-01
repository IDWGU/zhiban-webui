import { useState } from 'react'
import CodeBlock from './CodeBlock'

// ── Tool icon mapping ──
const TOOL_ICONS: Record<string, string> = {
  search_knowledge_base: '🔍',
  get_paper_section: '📄',
  get_reading_context: '📖',
  web_search: '🌐',
  web_fetch: '📥',
}

const TOOL_LABELS: Record<string, string> = {
  search_knowledge_base: '搜索知识库',
  get_paper_section: '获取论文章节',
  get_reading_context: '获取阅读上下文',
  web_search: '网络搜索',
  web_fetch: '获取网页',
}

interface ToolCallBlockProps {
  toolName: string
  toolArgs?: string
  toolResult?: string
  isPending?: boolean
  isError?: boolean
  durationMs?: number
}

export default function ToolCallBlock({
  toolName, toolArgs, toolResult, isPending = false, isError = false, durationMs,
}: ToolCallBlockProps) {
  const [expandedArgs, setExpandedArgs] = useState(false)
  const [expandedResult, setExpandedResult] = useState(false)

  const icon = TOOL_ICONS[toolName] || '🔧'
  const label = TOOL_LABELS[toolName] || toolName || '工具结果'

  const argsPreview = toolArgs
    ? formatArgPreview(toolArgs)
    : ''
  const resultPreview = toolResult
    ? toolResult.replace(/\s+/g, ' ').slice(0, 80)
    : ''
  const resultLineCount = toolResult ? toolResult.split('\n').length : 0
  const hasResult = !!toolResult && !isPending

  return (
    <div style={{
      overflow: 'hidden', borderRadius: 8,
      border: `1px solid ${isError
        ? 'hsl(0 60% 40% / 0.30)'
        : 'hsl(var(--border))'}`,
      background: isError
        ? 'hsl(0 60% 40% / 0.05)'
        : 'hsl(var(--muted) / 0.25)',
      marginBottom: 4,
    }}>
      {/* ── Summary bar ── */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 6,
        padding: '5px 10px',
        fontSize: 11, fontWeight: 600,
        color: 'hsl(var(--muted-foreground))',
        backgroundColor: 'hsl(var(--muted) / 0.30)',
        borderBottom: `1px solid hsl(var(--border) / 0.25)`,
      }}>
        <span style={{ fontSize: 13 }}>{icon}</span>
        <span style={{ color: '#ffb86c', fontFamily: 'monospace', fontSize: 11 }}>
          {label}
        </span>

        {argsPreview && (
          <span style={{
            flex: 1, minWidth: 0,
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
            fontSize: 10, fontWeight: 400,
            color: 'hsl(var(--foreground) / 0.45)',
            fontFamily: 'monospace',
          }}>
            {argsPreview}
          </span>
        )}

        {isPending && (
          <span style={{
            display: 'flex', alignItems: 'center', gap: 4,
            fontSize: 10, color: 'hsl(var(--muted-foreground))',
          }}>
            <span style={{
              width: 10, height: 10, borderRadius: '50%',
              border: '2px solid #ffb86c',
              borderTopColor: 'transparent',
              animation: 'spin 0.8s linear infinite',
            }} />
            执行中…
          </span>
        )}

        {hasResult && (
          <span style={{
            fontSize: 10, fontWeight: 500,
            color: isError ? 'hsl(0 60% 50%)' : 'hsl(var(--muted-foreground))',
          }}>
            {isError ? '失败' : resultLineCount > 1
              ? `${resultLineCount} 行结果`
              : resultPreview
                ? `${resultPreview.length} 字符`
                : '完成'}
          </span>
        )}

        {durationMs != null && (
          <span style={{
            fontSize: 10, color: 'hsl(var(--foreground) / 0.35)',
          }}>
            {durationMs < 1000 ? `${durationMs}ms` : `${(durationMs / 1000).toFixed(1)}s`}
          </span>
        )}

        <div style={{ flex: 1, minWidth: 0 }} />

        {/* Expand/collapse buttons */}
        {toolArgs && (
          <button
            onClick={() => setExpandedArgs(v => !v)}
            title="查看参数"
            style={{
              background: 'none', border: 'none', cursor: 'pointer',
              color: 'hsl(var(--muted-foreground))',
              fontSize: 10, padding: '1px 4px',
              fontFamily: 'inherit',
            }}
          >
            {expandedArgs ? '收起参数' : '参数'}
          </button>
        )}
        {hasResult && (
          <button
            onClick={() => setExpandedResult(v => !v)}
            title="查看结果"
            style={{
              background: 'none', border: 'none', cursor: 'pointer',
              color: 'hsl(var(--muted-foreground))',
              fontSize: 10, padding: '1px 4px',
              fontFamily: 'inherit',
            }}
          >
            {expandedResult ? '收起结果' : '结果'}
          </button>
        )}
      </div>

      {/* ── Expanded args ── */}
      {expandedArgs && toolArgs && (
        <div style={{
          borderBottom: hasResult || expandedResult ? '1px solid hsl(var(--border) / 0.15)' : 'none',
        }}>
          <CodeBlock code={formatToolArgsJson(toolArgs)} language="json" maxLines={12} />
        </div>
      )}

      {/* ── Expanded result ── */}
      {expandedResult && toolResult && (
        <div style={{
          maxHeight: 200, overflow: 'auto',
        }}>
          <CodeBlock
            code={toolResult}
            language={isError ? 'plaintext' : 'plaintext'}
            maxLines={15}
          />
        </div>
      )}
    </div>
  )
}

function formatArgPreview(args: string): string {
  try {
    const obj = JSON.parse(args)
    const entries = Object.entries(obj).slice(0, 2)
    return entries.map(([k, v]) => `${k}: ${String(v).slice(0, 40)}`).join(', ')
  } catch {
    return args.slice(0, 100)
  }
}

function formatToolArgsJson(args: string): string {
  try {
    return JSON.stringify(JSON.parse(args), null, 2)
  } catch {
    return args
  }
}

import { useEffect, useRef, useState, useCallback } from 'react'

interface DebugPanelProps {
  visible: boolean
  onClose: () => void
}

const MAX_LINES = 1000

export default function DebugPanel({ visible, onClose }: DebugPanelProps) {
  const [logs, setLogs] = useState<string[]>([])
  const [copied, setCopied] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const scrollRef = useRef(true)

  // Track scroll position to auto-scroll
  const handleScroll = useCallback(() => {
    const el = containerRef.current
    if (!el) return
    scrollRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 30
  }, [])

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (scrollRef.current && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight
    }
  }, [logs])

  // Listen for debug logs from Electron via IPC
  useEffect(() => {
    const api = (window as any).electronAPI
    if (!api?.onDebugLog) {
      setLogs(prev => {
        const entry = `[--:--:--] ⚠ electronAPI.onDebugLog 不可用 (非 Electron 环境?)`
        return [...prev, entry].slice(-MAX_LINES)
      })
      return
    }

    // Pull buffered logs first (emitted before renderer was ready)
    if (api.getDebugLogs) {
      api.getDebugLogs().then((buffered: string[]) => {
        if (buffered && buffered.length > 0) {
          setLogs(prev => [...buffered, ...prev].slice(-MAX_LINES))
        }
      }).catch(() => {})
    }

    const unsub = api.onDebugLog((entry: string) => {
      setLogs(prev => [...prev, entry].slice(-MAX_LINES))
    })

    return unsub
  }, [])

  // Listen for WebSocket diag events from App.tsx
  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail as string
      if (detail) {
        const ts = new Date().toISOString().split('T')[1]?.slice(0, 12) ?? ''
        setLogs(prev => [...prev, `[${ts}] ${detail}`].slice(-MAX_LINES))
      }
    }
    window.addEventListener('debug:diag', handler)
    return () => window.removeEventListener('debug:diag', handler)
  }, [])

  // Dump browser/env snapshot + pull buffered logs when panel opens
  useEffect(() => {
    if (!visible) return
    const api = (window as any).electronAPI
    const ts = () => new Date().toISOString().split('T')[1]?.slice(0, 12) ?? ''
    const snapshot = [
      `[${ts()}] === 浏览器环境 ===`,
      `[${ts()}] userAgent: ${navigator.userAgent.slice(0, 80)}`,
      `[${ts()}] protocol: ${window.location.protocol}`,
      `[${ts()}] hostname: ${window.location.hostname}`,
      `[${ts()}] port: ${window.location.port || '(default)'}`,
      `[${ts()}] electronAPI: ${api ? 'available' : 'MISSING'}`,
    ]
    // Pull buffered Electron-side logs that may have been missed
    if (api?.getDebugLogs) {
      api.getDebugLogs().then((buffered: string[]) => {
        if (buffered && buffered.length > 0) {
          setLogs(prev => [...buffered, ...prev, ...snapshot].slice(-MAX_LINES))
        } else {
          setLogs(prev => [...prev, ...snapshot].slice(-MAX_LINES))
        }
      }).catch(() => {
        setLogs(prev => [...prev, ...snapshot].slice(-MAX_LINES))
      })
    } else {
      setLogs(prev => [...prev, ...snapshot].slice(-MAX_LINES))
    }
  }, [visible])

  // Listen for WebSocket status changes
  useEffect(() => {
    const originalWarn = console.warn
    const originalError = console.error

    console.warn = (...args: any[]) => {
      const msg = args.map(a => typeof a === 'string' ? a : JSON.stringify(a)).join(' ')
      setLogs(prev => [...prev, `[--:--:--] [warn] ${msg}`].slice(-MAX_LINES))
      originalWarn.apply(console, args)
    }

    console.error = (...args: any[]) => {
      const msg = args.map(a => typeof a === 'string' ? a : JSON.stringify(a)).join(' ')
      setLogs(prev => [...prev, `[--:--:--] [error] ${msg}`].slice(-MAX_LINES))
      originalError.apply(console, args)
    }

    return () => {
      console.warn = originalWarn
      console.error = originalError
    }
  }, [])

  const copyAll = () => {
    const text = logs.join('\n')
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    }).catch(() => {
      // Fallback for older browsers
      const ta = document.createElement('textarea')
      ta.value = text
      document.body.appendChild(ta)
      ta.select()
      document.execCommand('copy')
      document.body.removeChild(ta)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    })
  }

  const clear = () => setLogs([])

  if (!visible) return null

  return (
    <div style={{
      position: 'fixed', bottom: 0, left: 0, right: 0, height: '40vh',
      background: '#0d1117', borderTop: '2px solid #30363d',
      zIndex: 99999, display: 'flex', flexDirection: 'column',
      fontFamily: "'SF Mono', 'Menlo', 'Monaco', monospace",
      fontSize: 11, color: '#c9d1d9',
    }}>
      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '4px 12px', background: '#161b22', borderBottom: '1px solid #30363d',
        flexShrink: 0,
      }}>
        <span style={{ fontWeight: 600, color: '#58a6ff', fontSize: 12 }}>
          🔍 调试面板 ({logs.length} 条)
        </span>
        <div style={{ display: 'flex', gap: 6 }}>
          <button onClick={copyAll} style={btnStyle}>
            {copied ? '已复制!' : '复制全部'}
          </button>
          <button onClick={clear} style={btnStyle}>清空</button>
          <button onClick={onClose} style={{ ...btnStyle, color: '#f85149' }}>关闭</button>
        </div>
      </div>

      {/* Log content */}
      <div
        ref={containerRef}
        onScroll={handleScroll}
        style={{
          flex: 1, overflow: 'auto', padding: '6px 12px',
          whiteSpace: 'pre-wrap', wordBreak: 'break-all',
          lineHeight: 1.45,
        }}
      >
        {logs.length === 0 ? (
          <div style={{ color: '#484f58', padding: 20, textAlign: 'center' }}>
            等待日志... (按 Cmd+Shift+D 切换面板)
          </div>
        ) : (
          logs.map((line, i) => (
            <div
              key={i}
              style={{
                color: line.includes('[stderr]') ? '#f85149'
                     : line.includes('ERROR') ? '#f85149'
                     : line.includes('[stdout]') ? '#c9d1d9'
                     : line.includes('✅') || line.includes('ready') ? '#3fb950'
                     : line.includes('⏳') || line.includes('Health') ? '#d2991d'
                     : line.includes('=== ') ? '#58a6ff'
                     : '#8b949e',
                padding: '1px 0',
              }}
            >
              {line}
            </div>
          ))
        )}
      </div>
    </div>
  )
}

const btnStyle: React.CSSProperties = {
  padding: '2px 10px', borderRadius: 4, border: '1px solid #30363d',
  background: '#21262d', color: '#c9d1d9', cursor: 'pointer',
  fontSize: 11, fontFamily: 'inherit',
}

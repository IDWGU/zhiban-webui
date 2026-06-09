import type { CSSProperties } from 'react'
import { useAppStore } from '@/stores/appStore'

interface Props {
  onSettingsClick: () => void
  onToggleSidebar?: () => void
}

function getStatusText(status: string, attempt: number, msg: string): string {
  if (msg) return msg
  switch (status) {
    case 'waiting':      return '引擎启动中...'
    case 'connecting':   return '连接中...'
    case 'connected':    return '就绪'
    case 'reconnecting': return `重连中 (${attempt})`
    case 'disconnected': return '后端已断开连接'
    default:             return status
  }
}

function getDotColor(status: string): string {
  switch (status) {
    case 'waiting':      return 'var(--dot-blue)'
    case 'connecting':   return 'var(--dot-yellow)'
    case 'connected':    return 'var(--dot-green)'
    case 'reconnecting': return 'var(--dot-yellow)'
    case 'disconnected': return 'var(--dot-red)'
    default:             return 'var(--dot-red)'
  }
}

export default function TitleBar({ onSettingsClick, onToggleSidebar }: Props) {
  const wsStatus = useAppStore(s => s.connection.wsStatus)
  const reconnectAttempt = useAppStore(s => s.connection.reconnectAttempt)
  const startupMessage = useAppStore(s => s.connection.startupMessage)

  return (
    <div className="titlebar-drag" style={{
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      height: 40, padding: '0 16px',
      background: 'hsl(var(--background) / 0.85)',
      backdropFilter: 'blur(12px)',
      WebkitBackdropFilter: 'blur(12px)',
      borderBottom: '1px solid hsl(var(--border))',
      flexShrink: 0,
      zIndex: 50,
    } as CSSProperties}>
      {/* Left: brand + sidebar toggle */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        {onToggleSidebar && (
          <button
            onClick={onToggleSidebar}
            className="titlebar-no-drag"
            style={iconBtnStyle}
            title="会话列表"
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor"
              strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="3" y1="6" x2="21" y2="6" />
              <line x1="3" y1="12" x2="21" y2="12" />
              <line x1="3" y1="18" x2="21" y2="18" />
            </svg>
          </button>
        )}
        <span style={{
          fontSize: 13, fontWeight: 600, color: 'hsl(var(--foreground))',
          letterSpacing: 0.5, userSelect: 'none',
        }}>
          知伴 ZhiBan
        </span>
      </div>

      {/* Right: status + settings */}
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <StatusDot color={getDotColor(wsStatus)} />
          <span style={{ fontSize: 11, color: 'hsl(var(--muted-foreground))' }}>
            {getStatusText(wsStatus, reconnectAttempt, startupMessage)}
          </span>
        </div>
        <button
          onClick={onSettingsClick}
          className="titlebar-no-drag"
          style={iconBtnStyle}
          title="设置"
        >
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor"
            strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="3" />
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
          </svg>
        </button>
      </div>
    </div>
  )
}

function StatusDot({ color }: { color: string }) {
  return (
    <span style={{
      display: 'inline-block', width: 7, height: 7, borderRadius: '50%', background: color,
      boxShadow: `0 0 6px ${color}`,
    }} />
  )
}

const iconBtnStyle: React.CSSProperties = {
  width: 30, height: 28,
  display: 'flex', alignItems: 'center', justifyContent: 'center',
  background: 'transparent',
  border: 'none',
  borderRadius: 6,
  color: 'hsl(var(--muted-foreground))',
  cursor: 'pointer',
}

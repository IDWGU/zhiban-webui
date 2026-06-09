import { useState, useEffect, useCallback, useRef } from 'react'
import { useAppStore } from '@/stores/appStore'

interface Props {
  collapsed: boolean
  onToggleCollapse: () => void
  wsSend: ((data: unknown) => void) | null
}

function relativeTime(iso: string): string {
  if (!iso) return ''
  const diff = Date.now() - new Date(iso).getTime()
  const sec = Math.floor(diff / 1000)
  if (sec < 60) return '刚刚'
  const min = Math.floor(sec / 60)
  if (min < 60) return `${min}分钟前`
  const hr = Math.floor(min / 60)
  if (hr < 24) return `${hr}小时前`
  return `${Math.floor(hr / 24)}天前`
}

export default function ConversationSidebar({ collapsed, onToggleCollapse, wsSend }: Props) {
  const conversations = useAppStore(s => s.conversations)
  const activeConvId = useAppStore(s => s.activeConversationId)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editName, setEditName] = useState('')
  const [contextMenu, setContextMenu] = useState<{ id: string; x: number; y: number } | null>(null)
  const [search, setSearch] = useState('')
  const editInputRef = useRef<HTMLInputElement>(null)

  const filtered = search.trim()
    ? conversations.filter(c =>
        c.name.toLowerCase().includes(search.toLowerCase()) ||
        (c.topic || '').toLowerCase().includes(search.toLowerCase()))
    : conversations

  useEffect(() => {
    if (wsSend) {
      wsSend({ type: 'list_conversations' })
    }
  }, [wsSend])

  useEffect(() => {
    if (editingId) {
      editInputRef.current?.focus()
      editInputRef.current?.select()
    }
  }, [editingId])

  useEffect(() => {
    if (!contextMenu) return
    const close = () => setContextMenu(null)
    window.addEventListener('click', close)
    return () => window.removeEventListener('click', close)
  }, [contextMenu])

  const handleNew = useCallback(() => {
    wsSend?.({ type: 'new_conversation', name: '新对话' })
  }, [wsSend])

  const handleSwitch = useCallback((convId: string) => {
    wsSend?.({ type: 'switch_conversation', conversationId: convId })
  }, [wsSend])

  const handleDelete = useCallback((convId: string) => {
    const conv = conversations.find(c => c.id === convId)
    if (!conv) return
    setContextMenu(null)
    useAppStore.getState().removeConversation(convId)
    wsSend?.({ type: 'delete_conversation', conversationId: convId })
  }, [wsSend, conversations])

  const handleRenameStart = useCallback((convId: string, currentName: string) => {
    setEditingId(convId)
    setEditName(currentName)
    setContextMenu(null)
  }, [])

  const handleRenameSubmit = useCallback((convId: string) => {
    const name = editName.trim()
    if (name && name.length > 0) {
      wsSend?.({ type: 'rename_conversation', conversationId: convId, name })
    }
    setEditingId(null)
  }, [editName, wsSend])

  const handleContextMenu = useCallback((e: React.MouseEvent, convId: string) => {
    e.preventDefault()
    setContextMenu({ id: convId, x: e.clientX, y: e.clientY })
  }, [])

  return (
    <div className={collapsed ? undefined : 'proma-panel'} style={{
      width: collapsed ? 0 : 260,
      minWidth: 0,
      flexShrink: 0,
      overflow: 'hidden',
      transition: 'width 300ms ease-in-out',
      display: 'flex', flexDirection: 'column',
    }}>
      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '10px 14px',
        borderBottom: '1px solid hsl(var(--border))',
        flexShrink: 0,
      }}>
        <span style={{ fontSize: 13, fontWeight: 600, color: 'hsl(var(--foreground))' }}>
          会话
        </span>
        <button
          onClick={handleNew}
          className="proma-dashed-entry"
          style={{
            fontSize: 11, padding: '3px 10px', borderRadius: 8,
            color: 'hsl(var(--muted-foreground))',
            cursor: 'pointer', fontFamily: 'inherit',
          }}
        >
          + 新建
        </button>
      </div>

      {/* Search */}
      {conversations.length > 0 && (
        <div style={{ padding: '6px 10px' }}>
          <input
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="搜索会话..."
            style={{
              width: '100%', padding: '5px 10px', borderRadius: 8,
              border: '1px solid hsl(var(--border))',
              background: 'hsl(var(--foreground) / 0.04)',
              color: 'hsl(var(--foreground))',
              fontSize: 12, outline: 'none',
              fontFamily: 'inherit',
            }}
          />
        </div>
      )}

      {/* List */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '4px 0' }}>
        {conversations.length === 0 && (
          <div style={{
            padding: '24px 14px', fontSize: 12,
            color: 'hsl(var(--muted-foreground) / 0.60)',
            textAlign: 'center',
          }}>
            暂无会话，点击"新建"开始
          </div>
        )}
        {filtered.map(conv => {
          const isActive = conv.id === activeConvId
          const isEditing = editingId === conv.id
          const topicName = conv.topic || '无话题'

          return (
            <div
              key={conv.id}
              onClick={() => handleSwitch(conv.id)}
              onContextMenu={e => handleContextMenu(e, conv.id)}
              onDoubleClick={() => handleRenameStart(conv.id, conv.name)}
              className={isActive ? 'proma-selected' : ''}
              style={sidebarItemStyle}
            >
              {/* Left accent status bar */}
              {isActive && (
                <div className="proma-status-bar" style={{
                  background: 'hsl(var(--primary))',
                }} />
              )}

              <div style={{ flex: 1, minWidth: 0 }}>
                {isEditing ? (
                  <input
                    ref={editInputRef}
                    value={editName}
                    onChange={e => setEditName(e.target.value)}
                    onBlur={() => handleRenameSubmit(conv.id)}
                    onKeyDown={e => {
                      if (e.key === 'Enter') handleRenameSubmit(conv.id)
                      if (e.key === 'Escape') setEditingId(null)
                    }}
                    onClick={e => e.stopPropagation()}
                    style={{
                      background: 'hsl(var(--foreground) / 0.06)',
                      border: 'none',
                      borderBottom: '1px solid hsl(var(--primary) / 0.50)',
                      borderRadius: 0,
                      color: 'hsl(var(--foreground))',
                      fontSize: 13, padding: '2px 0',
                      width: '100%', outline: 'none',
                      fontFamily: 'inherit',
                    }}
                    maxLength={50}
                  />
                ) : (
                  <>
                    <div style={{
                      fontSize: 13, fontWeight: 500, lineHeight: 1.4,
                      color: isActive ? 'hsl(var(--foreground))' : 'hsl(var(--foreground) / 0.80)',
                      overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                    }}>
                      {conv.name}
                    </div>
                    <div style={{
                      fontSize: 10, color: 'hsl(var(--muted-foreground))',
                      marginTop: 1,
                      display: 'flex', gap: 8,
                    }}>
                      {conv.messageCount > 0 && (
                        <span>{Math.floor(conv.messageCount / 2)} 轮</span>
                      )}
                      {conv.paperCount > 0 && (
                        <span>{conv.paperCount} 篇</span>
                      )}
                      {conv.messageCount === 0 && conv.paperCount === 0 && (
                        <span>新会话</span>
                      )}
                    </div>
                    {topicName && (
                      <div style={{
                        fontSize: 10, color: 'hsl(var(--primary))',
                        overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                        marginTop: 1, opacity: 0.85,
                      }}>
                        {topicName.length > 18 ? topicName.slice(0, 18) + '...' : topicName}
                      </div>
                    )}
                    {conv.updatedAt && (
                      <div style={{
                        fontSize: 9, color: 'hsl(var(--foreground) / 0.30)',
                        marginTop: 1,
                      }}>
                        {relativeTime(conv.updatedAt)}
                      </div>
                    )}
                  </>
                )}
              </div>

              {/* 3-dot menu button — 常驻可见 */}
              <button
                onClick={e => {
                  e.stopPropagation()
                  const rect = (e.target as HTMLElement).getBoundingClientRect()
                  setContextMenu({ id: conv.id, x: rect.right, y: rect.bottom })
                }}
                style={{
                  width: 24, height: 24, flexShrink: 0,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  background: 'transparent', border: 'none', borderRadius: 6,
                  color: 'hsl(var(--muted-foreground))', cursor: 'pointer',
                  fontSize: 14, opacity: 0.50,
                }}
              >
                ···
              </button>
            </div>
          )
        })}
      </div>

      {/* Footer */}
      <div style={{
        borderTop: '1px solid hsl(var(--border))',
        padding: '6px 14px', flexShrink: 0,
      }}>
        <button onClick={onToggleCollapse} style={{
          background: 'transparent', border: 'none',
          color: 'hsl(var(--muted-foreground) / 0.60)',
          fontSize: 11, cursor: 'pointer', padding: '2px 0',
          fontFamily: 'inherit',
        }}>
          收起
        </button>
      </div>

      {/* Context Menu — Proma dropdown style */}
      {contextMenu && (
        <div style={{
          position: 'fixed',
          left: contextMenu.x,
          top: contextMenu.y,
          zIndex: 3000,
          background: 'hsl(var(--popover))',
          border: '1px solid hsl(var(--border))',
          borderRadius: 10,
          padding: '4px 0',
          minWidth: 120,
          boxShadow: '0 4px 20px rgba(0,0,0,0.3)',
          backdropFilter: 'blur(12px)',
          WebkitBackdropFilter: 'blur(12px)',
        }}>
          <div
            onClick={() => handleRenameStart(contextMenu.id,
              conversations.find(c => c.id === contextMenu.id)?.name || '')}
            style={menuItemStyle}
          >
            重命名
          </div>
          <div
            onClick={() => handleDelete(contextMenu.id)}
            style={{ ...menuItemStyle, color: 'hsl(var(--destructive))' }}
          >
            删除
          </div>
        </div>
      )}
    </div>
  )
}

const sidebarItemStyle: React.CSSProperties = {
  position: 'relative',
  display: 'flex', alignItems: 'center', gap: 8,
  padding: '7px 14px 7px 14px',
  borderRadius: 10,
  cursor: 'pointer',
  transition: 'background 0.1s ease, color 0.1s ease',
  userSelect: 'none',
  margin: '1px 6px',
}

const menuItemStyle: React.CSSProperties = {
  padding: '6px 16px',
  fontSize: 12,
  color: 'hsl(var(--foreground))',
  cursor: 'pointer',
  whiteSpace: 'nowrap',
  transition: 'background 0.1s ease',
}

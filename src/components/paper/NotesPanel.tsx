import { useAppStore } from '@/stores/appStore'

export default function NotesPanel() {
  const notesContent = useAppStore(s => s.notesContent)
  const setNotesContent = useAppStore(s => s.setNotesContent)

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{
        padding: '10px 16px',
        borderBottom: '1px solid hsl(var(--border))',
        fontSize: 12, fontWeight: 600,
        color: 'hsl(var(--foreground))',
        display: 'flex', alignItems: 'center', gap: 8,
      }}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
          style={{ color: 'hsl(var(--primary))' }}>
          <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
          <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
        </svg>
        共享笔记
        <span style={{
          fontSize: 10, color: 'hsl(var(--foreground) / 0.38)',
          fontWeight: 400,
        }}>
          — 你和 AI 都可以在这里书写
        </span>
      </div>
      <textarea
        value={notesContent}
        onChange={(e) => setNotesContent(e.target.value)}
        placeholder={'在这里写笔记... AI 回答时也可以引用这里的内容。\n按 Space 键提问，AI 可以看到所有打开的论文和这份笔记。'}
        style={{
          flex: 1, width: '100%', border: 'none', outline: 'none',
          padding: '16px 20px', fontSize: 13, lineHeight: 1.8,
          fontFamily: 'inherit', resize: 'none',
          background: 'hsl(var(--background))',
          color: 'hsl(var(--foreground))',
        }}
      />
    </div>
  )
}

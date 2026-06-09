interface Props { docName: string | null }

export default function ActiveDocBadge({ docName }: Props) {
  if (!docName) {
    return (
      <span style={{
        fontSize: 10, color: 'hsl(var(--muted-foreground))', fontStyle: 'italic',
        padding: '2px 8px', borderRadius: 9999,
        background: 'hsl(var(--muted))',
      }}>
        未检测到文档
      </span>
    )
  }

  return (
    <span style={{
      fontSize: 10, fontWeight: 500,
      color: 'hsl(var(--tag-text))',
      padding: '2px 8px', borderRadius: 9999,
      background: 'hsl(var(--tag-bg))',
      border: '1px solid hsl(var(--tag-border))',
      maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis',
      whiteSpace: 'nowrap',
      transition: 'background 0.1s ease, border-color 0.1s ease',
    }}>
      {docName}
    </span>
  )
}

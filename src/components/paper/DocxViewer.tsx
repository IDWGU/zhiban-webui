interface Props {
  html: string
  fullText?: string
  paperTitle?: string
  paperId?: string
}

export default function DocxViewer({ html }: Props) {
  return (
    <div
      style={{ padding: '16px 24px', fontSize: 13, lineHeight: 1.8, color: 'var(--text-primary)' }}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  )
}

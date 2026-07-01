interface Props { isActive: boolean; source: string | null }

export default function OcrStatusDot({ isActive, source }: Props) {
  // document = green (accurate), ax = blue (accurate, no OCR), ocr = yellow (unreliable), null = gray
  const color = !isActive ? 'var(--text-muted)'
    : source === 'document' ? '#4ade80'
    : source === 'ax' ? '#60a5fa'
    : source === 'ocr' ? '#facc15'
    : '#facc15'

  return (
    <span style={{
      display: 'inline-block', width: 6, height: 6, borderRadius: '50%',
      background: color,
      boxShadow: isActive ? `0 0 6px ${color}` : 'none',
      transition: 'all 0.3s',
      animation: isActive ? 'pulse 2s infinite' : 'none',
    }} />
  )
}

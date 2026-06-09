import { useAppStore } from '@/stores/appStore'

const IS_LOCAL = (url: string) =>
  url === '__local__' || url.includes('localhost') || url.includes('127.0.0.1') || url.includes('ollama')

export default function TranslationSection() {
  const settings = useAppStore(s => s.settings)
  const isLocal = IS_LOCAL(settings.llmBaseUrl)

  return (
    <div style={{ padding: '12px 0' }}>
      <h3 style={{ fontSize: 15, fontWeight: 600, margin: '0 0 8px', color: 'hsl(var(--foreground))' }}>
        翻译模型
      </h3>
      <p style={{ fontSize: 13, color: 'hsl(var(--muted-foreground))', lineHeight: 1.5 }}>
        {isLocal
          ? 'Hy-MT2-1.8B (~1.1GB)'
          : `复用伴读模型 (${settings.llmBaseUrl || 'API'})`}
      </p>
    </div>
  )
}

import { useAppStore } from '@/stores/appStore'
import { SectionBox, SettingRow, inputStyle } from './SharedComponents'

export default function ModelsSection() {
  const settings = useAppStore(s => s.settings)
  const embedResult = useAppStore(s => s.embeddingLoadResult)

  return (
    <SectionBox title="模型管理">
      {/* 嵌入模型 — 固定 jina，启动自动加载，不可切换 */}
      <SettingRow label="嵌入模型" desc="启动时自动加载 jina-embeddings-v5-text-nano (768维, ~0.5GB)">
        <span style={{ fontSize: 12, color: embedResult?.success ? '#4caf50' : 'hsl(var(--muted-foreground))' }}>
          {embedResult?.success
            ? `✅ 已加载 · ${embedResult.dim ?? '?'}维`
            : '启动时自动加载'}
        </span>
      </SettingRow>

      {/* 模型缓存目录 */}
      <SettingRow label="模型缓存目录" desc="所有模型文件下载到此目录（通过 MODEL_CACHE 环境变量设置）">
        <input value={settings.modelCacheDir || 'models/bundled/（项目内置）'}
          readOnly
          style={{ ...inputStyle, color: 'hsl(var(--muted-foreground))', cursor: 'not-allowed' }} />
      </SettingRow>

    </SectionBox>
  )
}

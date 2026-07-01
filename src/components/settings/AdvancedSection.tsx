import { useState, useEffect } from 'react'
import { useAppStore } from '@/stores/appStore'
import { SectionBox, SettingRow, ToggleCheckbox, inputStyle, actionBtnStyle } from './SharedComponents'

function doSend() {
  return useAppStore.getState().sendMessage || (window as any).__zhiban_wsSend
}

export default function AdvancedSection() {
  const settings = useAppStore(s => s.settings)
  const updateSettings = useAppStore(s => s.updateSettings)
  const wsStatus = useAppStore(s => s.connection.wsStatus)
  const [applying, setApplying] = useState(false)
  const [flashAttn, setFlashAttn] = useState(settings.llmFlashAttn)
  const [useMmap, setUseMmap] = useState(settings.llmUseMmap)
  const [nBatch, setNBatch] = useState(settings.llmNBatch)
  const [nUbatch, setNUbatch] = useState(settings.llmNUbatch)

  // Sync from store on first load
  useEffect(() => {
    if (wsStatus === 'connected' && applying) setApplying(false)
  }, [wsStatus])

  const handleApply = () => {
    const send = doSend()
    if (!send) return
    setApplying(true)
    // Persist to localStorage via Zustand
    updateSettings({ llmFlashAttn: flashAttn, llmUseMmap: useMmap, llmNBatch: nBatch, llmNUbatch: nUbatch })
    // Send to backend to update config + reload model
    send({
      type: 'model_config',
      action: 'set_llm_params',
      params: {
        flash_attn: flashAttn,
        use_mmap: useMmap,
        n_batch: nBatch,
        n_ubatch: nUbatch,
      },
    })
  }

  return (
    <SectionBox title="llama.cpp 加载参数">
      <SettingRow
        label="Flash Attention"
        desc="Metal 优化注意力计算，显著加速长上下文 Prefill 和 Decode。关闭后大幅降速。"
      >
        <ToggleCheckbox checked={flashAttn} onChange={setFlashAttn} disabled={applying} />
      </SettingRow>

      <SettingRow
        label="内存映射 (mmap)"
        desc="零拷贝加载模型权重，减少启动内存峰值。关闭后使用完整读入模式。"
      >
        <ToggleCheckbox checked={useMmap} onChange={setUseMmap} disabled={applying} />
      </SettingRow>

      <SettingRow
        label="Batch 大小 (n_batch)"
        desc="Prompt 预填充的物理批次大小。Metal 推荐 2048-4096，越大 prefilling 越快但占用更多 GPU 内存。"
      >
        <input
          type="number"
          value={nBatch}
          onChange={e => setNBatch(parseInt(e.target.value) || 512)}
          min={512} max={8192} step={512}
          style={{ ...inputStyle, width: 120 }}
          disabled={applying}
        />
      </SettingRow>

      <SettingRow
        label="微批次大小 (n_ubatch)"
        desc="Flash Attention 逻辑分块大小。一般 ≤ n_batch 的一半，推荐 512-2048。"
      >
        <input
          type="number"
          value={nUbatch}
          onChange={e => setNUbatch(parseInt(e.target.value) || 256)}
          min={256} max={4096} step={256}
          style={{ ...inputStyle, width: 120 }}
          disabled={applying}
        />
      </SettingRow>

      <SettingRow
        label=""
        desc={applying ? '参数已更新，后端重启中，WebSocket 重连后自动生效...' : '修改后点击"应用"生效（需重启模型进程）'}
      >
        <button
          onClick={handleApply}
          disabled={applying}
          style={{
            ...actionBtnStyle,
            background: applying ? 'hsl(var(--border))' : 'hsl(var(--accent))',
            color: applying ? 'hsl(var(--muted-foreground))' : '#fff',
            cursor: applying ? 'not-allowed' : 'pointer',
            opacity: applying ? 0.7 : 1,
          }}
        >
          {applying ? '重启中...' : '应用（重启模型）'}
        </button>
      </SettingRow>
    </SectionBox>
  )
}

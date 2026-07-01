import { useState, useEffect, useCallback } from 'react'
import { useAppStore } from '@/stores/appStore'
import { SectionBox, SettingRow, ToggleCheckbox, toggleBtnStyle } from './SharedComponents'

function doSend() {
  return useAppStore.getState().sendMessage || (window as any).__zhiban_wsSend
}

export default function GeneralSection() {
  const theme = useAppStore(s => s.settings.theme)
  const toggleTheme = useAppStore(s => s.toggleTheme)
  const rememberApiKey = useAppStore(s => s.settings.rememberApiKey)
  const updateSettings = useAppStore(s => s.updateSettings)
  const wsStatus = useAppStore(s => s.connection.wsStatus)
  const showDevPanel = useAppStore(s => s.connection.showDevPanel)
  const setShowDevPanel = useAppStore(s => s.setShowDevPanel)
  const [debug, setDebug] = useState(false)
  const [restarting, setRestarting] = useState(false)

  // WebSocket 重连后清除 restarting 状态
  useEffect(() => {
    if (wsStatus === 'connected') setRestarting(false)
  }, [wsStatus])

  // debug 状态由 App.tsx 在 WS 连接后统一拉取，通过 model-config-result CustomEvent 广播

  // 监听 model_config_result 中的 debug 状态
  useEffect(() => {
    const handler = (e: Event) => {
      const msg = (e as CustomEvent).detail
      if (msg?.action === 'get' && msg.config?.debug !== undefined) {
        setDebug(msg.config.debug)
      }
      if (msg?.action === 'set_debug' && msg.enabled !== undefined) {
        setDebug(msg.enabled)
        setRestarting(true)
        // 后端会自行重启，页面在 WS 重连后自动恢复
      }
    }
    window.addEventListener('model-config-result', handler as EventListener)
    return () => window.removeEventListener('model-config-result', handler as EventListener)
  }, [])

  const toggleDebug = useCallback(() => {
    const send = doSend()
    if (send) {
      send({ type: 'model_config', action: 'set_debug', enabled: !debug })
    }
  }, [debug])

  return (
    <SectionBox title="通用设置">
      <SettingRow label="主题" desc="切换深色/浅色界面">
        <button onClick={toggleTheme} style={toggleBtnStyle}>
          {theme === 'dark' ? '🌙 深色' : '☀️ 浅色'}
        </button>
      </SettingRow>
      <SettingRow label="记住 API 密钥" desc="密钥以明文保存在本地，关闭后每次启动需重新输入">
        <ToggleCheckbox checked={rememberApiKey} onChange={(v) => updateSettings({ rememberApiKey: v })} />
      </SettingRow>
      <SettingRow
        label="调试模式"
        desc={restarting
          ? '后端重启中，WebSocket 重连后自动恢复...'
          : '输出更详细的日志用于排查问题，开关后自动重启后端'}
      >
        <ToggleCheckbox checked={debug} onChange={toggleDebug} disabled={restarting} />
      </SettingRow>
      <SettingRow label="开发者面板" desc="在对话区显示工作流步骤和 LLM 运行状态的调试面板，用于排查问题和性能分析">
        <ToggleCheckbox checked={showDevPanel} onChange={(v) => setShowDevPanel(v)} />
      </SettingRow>
    </SectionBox>
  )
}

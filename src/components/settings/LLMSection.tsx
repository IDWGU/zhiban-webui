import { useState, useEffect } from 'react'
import { useAppStore } from '@/stores/appStore'
import { SectionBox, SettingRow, ToggleCheckbox, inputStyle } from './SharedComponents'
import type { SettingsState } from '@/types'

function doSend() {
  return useAppStore.getState().sendMessage || (window as any).__zhiban_wsSend
}

function ModelStatus() {
  const conn = useAppStore(s => s.connection)
  const loading = conn.localEngineLoading
  const [loaded, setLoaded] = useState(false)
  const [backend, setBackend] = useState('')

  useEffect(() => {
    const h = (e: Event) => {
      const cfg = (e as CustomEvent).detail?.config
      if (cfg?.local_engine_loaded) {
        setLoaded(true)
        setBackend(cfg.local_engine_backend || '')
      }
    }
    window.addEventListener('model-config-result', h)
    return () => window.removeEventListener('model-config-result', h)
  }, [])

  if (loaded) {
    return (
      <div style={{ fontSize: 11, color: '#4caf50', marginTop: 6 }}>
        已加载 · {backend}
      </div>
    )
  }
  if (loading) {
    return (
      <div style={{ fontSize: 11, color: 'hsl(var(--muted-foreground))', marginTop: 6 }}>
        <span style={{
          display: 'inline-block', width: 8, height: 8, borderRadius: '50%',
          background: 'hsl(var(--primary))', marginRight: 6,
          animation: 'pulse 1.5s infinite',
        }} />
        加载中...
      </div>
    )
  }
  return null
}

const BASE_URL_PRESETS: { label: string; url: string }[] = [
  { label: 'DeepSeek', url: 'https://api.deepseek.com' },
  { label: 'Ollama', url: 'http://localhost:11434/v1' },
  { label: 'vLLM', url: 'http://localhost:8000/v1' },
  { label: 'LM Studio', url: 'http://localhost:1234/v1' },
  { label: 'Local', url: '__local__' },
  { label: '自定义', url: '' },
]

const IS_LOCAL = (url: string) => url === '__local__'

interface SamplingPreset {
  label: string
  desc: string
  params: Partial<SettingsState>
}
const SAMPLING_PRESETS: SamplingPreset[] = [
  {
    label: '精确',
    desc: 'RAG 事实问答，低随机性',
    params: { llmTemperature: 0.05, llmTopP: 0.8, llmTopK: 40, llmRepeatPenalty: 1.2, llmFrequencyPenalty: 0.3, llmPresencePenalty: 0.4, llmStopTokens: '<|im_end|>,<|endoftext|>' },
  },
  {
    label: '均衡',
    desc: '通用对话，推荐默认',
    params: { llmTemperature: 0.1, llmTopP: 0.85, llmTopK: 40, llmRepeatPenalty: 1.15, llmFrequencyPenalty: 0.2, llmPresencePenalty: 0.3, llmStopTokens: '<|im_end|>,<|endoftext|>' },
  },
  {
    label: '稳定',
    desc: '小模型防重复',
    params: { llmTemperature: 0.15, llmTopP: 0.8, llmTopK: 30, llmRepeatPenalty: 1.2, llmFrequencyPenalty: 0.3, llmPresencePenalty: 0.4, llmStopTokens: '<|im_end|>,<|endoftext|>' },
  },
  {
    label: '创意',
    desc: '头脑风暴，多样输出',
    params: { llmTemperature: 0.6, llmTopP: 0.9, llmTopK: 50, llmRepeatPenalty: 1.05, llmFrequencyPenalty: 0.0, llmPresencePenalty: 0.0, llmStopTokens: '' },
  },
]

export default function LLMSection() {
  const settings = useAppStore(s => s.settings)
  const updateSettings = useAppStore(s => s.updateSettings)
  const testResult = useAppStore(s => s.llmTestResult)
  const availableModels = useAppStore(s => s.availableModels)
  const [testing, setTesting] = useState(false)
  const [fetchingModels, setFetchingModels] = useState(false)
  const [showAdvanced, setShowAdvanced] = useState(false)

  const { llmApiKey, llmBaseUrl, llmModel, thinkingMode, systemPrompt,
    llmTopP, llmTemperature, llmExtraHeaders, llmExtraBody } = settings

  const handleTest = () => {
    setTesting(true)
    useAppStore.setState({ llmTestResult: null })
    const send = doSend()
    if (!send) {
      useAppStore.getState().setLlmTestResult(false, 'WebSocket 未连接', '')
      setTesting(false)
      return
    }
    if (IS_LOCAL(llmBaseUrl)) {
      send({ type: 'model_config', action: 'test_local_model', path: settings.llmModelPath || '', model: llmModel })
    } else {
      send({ type: 'llm_test', apiKey: llmApiKey, model: llmModel || 'deepseek-chat', baseUrl: llmBaseUrl })
    }
    setTimeout(() => setTesting(false), 15000)
  }

  const handleFetchModels = () => {
    setFetchingModels(true)
    useAppStore.setState({ llmTestResult: null })
    const send = doSend()
    if (!send) {
      useAppStore.getState().setLlmTestResult(false, 'WebSocket 未连接', '')
      setFetchingModels(false)
      return
    }
    if (IS_LOCAL(llmBaseUrl)) {
      // 本地模式：扫描本地模型文件
      send({ type: 'model_config', action: 'scan_local_models' })
    } else {
      send({ type: 'llm_list_models', apiKey: llmApiKey, baseUrl: llmBaseUrl })
    }
    setTimeout(() => setFetchingModels(false), 15000)
  }

  useEffect(() => {
    if (testResult) { setTesting(false); setFetchingModels(false) }
  }, [testResult])

  useEffect(() => {
    if (availableModels.length > 0 && llmModel && !availableModels.some(m => m.name === llmModel)) {
      useAppStore.setState({ availableModels: [...availableModels, { name: llmModel, path: '' }] })
    }
  }, [availableModels])

  // 模型配置由 App.tsx 在 WS 连接后统一拉取，通过 model_config_result 广播到 store

  const handlePreset = (preset: typeof BASE_URL_PRESETS[number]) => {
    if (preset.url) {
      updateSettings({ llmBaseUrl: preset.url })
      if (IS_LOCAL(preset.url) !== isLocal) {
        // 切换模式时清空模型列表
        useAppStore.setState({ availableModels: [] })
      }
    }
  }

  const isLocal = IS_LOCAL(llmBaseUrl)
  const hasModels = availableModels.length > 0

  return (
    <SectionBox title="LLM 模型">
      {/* API Key */}
      <SettingRow label="API 密钥" desc={isLocal ? '本地推理无需 API Key' : '云端 API 需填 Key，本地模型（Ollama 等）可留空'}>
        <input type="password" value={llmApiKey}
          onChange={(e) => updateSettings({ llmApiKey: e.target.value })}
          placeholder={isLocal ? '本地模式无需 API Key' : 'sk-... 或留空（本地模型）'}
          disabled={isLocal}
          style={{ ...inputStyle, opacity: isLocal ? 0.4 : 1 }} />
      </SettingRow>

      {/* Base URL + Presets */}
      <SettingRow label="Base URL" desc="API 端点地址，支持任何 OpenAI 兼容服务">
        <div style={{ display: 'flex', gap: 6, marginBottom: 8, flexWrap: 'wrap' }}>
          {BASE_URL_PRESETS.map(p => (
            <button key={p.label} onClick={() => handlePreset(p)}
              style={{
                padding: '3px 10px', borderRadius: 12, border: '1px solid hsl(var(--border))',
                background: llmBaseUrl === p.url ? 'hsl(var(--primary))' : 'transparent',
                color: llmBaseUrl === p.url ? '#fff' : 'hsl(var(--muted-foreground))',
                fontSize: 11, cursor: 'pointer', fontFamily: 'inherit',
                fontWeight: llmBaseUrl === p.url ? 600 : 400,
              }}>
              {p.label}
            </button>
          ))}
        </div>
        <input value={llmBaseUrl}
          onChange={(e) => updateSettings({ llmBaseUrl: e.target.value })}
          placeholder="https://api.deepseek.com"
          style={inputStyle} />
        {isLocal && (
          <ModelStatus />
        )}
      </SettingRow>

      {/* Model */}
      <SettingRow label="模型" desc={
        isLocal ? '选择本地扫描到的模型'
        : hasModels ? `${availableModels.length} 个可用模型`
        : '模型名称或 ID'
      }>
        {hasModels ? (
          <select value={llmModel}
            onChange={(e) => {
              const name = e.target.value
              updateSettings({ llmModel: name })
              // 本地模式：选中模型后自动加载对应路径
              if (isLocal) {
                const entry = availableModels.find(m => m.name === name)
                if (entry?.path) {
                  const send = doSend()
                  if (send) send({ type: 'model_config', action: 'set_local_model', path: entry.path })
                }
              }
            }}
            style={{ ...inputStyle, cursor: 'pointer' }}>
            {availableModels.map(m => <option key={m.name} value={m.name}>{m.name}</option>)}
          </select>
        ) : (
          <div style={{ display: 'flex', gap: 8 }}>
            <input value={llmModel}
              onChange={(e) => updateSettings({ llmModel: e.target.value })}
              placeholder={isLocal ? '点击"扫描本地"查找模型' : 'deepseek-chat'}
              style={{ ...inputStyle, flex: 1 }} />
            <button onClick={handleFetchModels} disabled={fetchingModels}
              style={{
                padding: '5px 12px', borderRadius: 6, border: '1px solid var(--btn-border)',
                background: fetchingModels ? 'var(--border)' : 'var(--btn-bg)',
                color: fetchingModels ? 'var(--text-muted)' : 'var(--text-secondary)',
                fontSize: 11, cursor: fetchingModels ? 'not-allowed' : 'pointer',
                fontFamily: 'inherit', whiteSpace: 'nowrap',
              }}>
              {fetchingModels ? '扫描中...' : isLocal ? '扫描本地' : '获取列表'}
            </button>
          </div>
        )}
      </SettingRow>
      {hasModels && (
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: -10 }}>
          <button onClick={handleFetchModels} disabled={fetchingModels} style={{
            padding: '3px 10px', borderRadius: 4, border: 'none',
            background: 'transparent', color: 'var(--text-muted)',
            fontSize: 10, cursor: fetchingModels ? 'not-allowed' : 'pointer',
            fontFamily: 'inherit',
          }}>
            {fetchingModels ? '扫描中...' : '刷新'}
          </button>
        </div>
      )}

      {/* Thinking Toggle */}
      <SettingRow label="思考/推理模式" desc="启用深度思考模式（Provider 自行翻译为 API 参数）">
        <ToggleCheckbox checked={thinkingMode}
          onChange={(v) => updateSettings({ thinkingMode: v })} />
      </SettingRow>

      {/* System Prompt */}
      <SettingRow label="系统提示词" desc="作用于所有对话，留空使用默认值">
        <textarea value={systemPrompt}
          onChange={(e) => updateSettings({ systemPrompt: e.target.value })}
          rows={6}
          style={{ ...inputStyle, resize: 'vertical', fontFamily: 'monospace', fontSize: 11, minHeight: 80 }} />
      </SettingRow>

      {/* Sampling Presets */}
      <div style={{ marginTop: 12 }}>
        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6, color: 'hsl(var(--muted-foreground))' }}>
          采样预设
        </div>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {SAMPLING_PRESETS.map(p => (
            <button key={p.label}
              onClick={() => {
                updateSettings(p.params)
                // 同步到后端
                const send = doSend()
                if (send) {
                  send({ type: 'model_config', action: 'set_sampling_params', params: {
                    LLM_TEMPERATURE: p.params.llmTemperature,
                    LLM_TOP_P: p.params.llmTopP,
                    LLM_TOP_K: p.params.llmTopK,
                    LLM_REPEAT_PENALTY: p.params.llmRepeatPenalty,
                    LLM_FREQUENCY_PENALTY: p.params.llmFrequencyPenalty,
                    LLM_PRESENCE_PENALTY: p.params.llmPresencePenalty,
                    LLM_STOP_TOKENS: p.params.llmStopTokens || '',
                  }})
                }
              }}
              title={p.desc}
              style={{
                padding: '4px 12px', borderRadius: 12, border: '1px solid hsl(var(--border))',
                background: llmTemperature === p.params.llmTemperature ? 'hsl(var(--primary))' : 'transparent',
                color: llmTemperature === p.params.llmTemperature ? '#fff' : 'hsl(var(--muted-foreground))',
                fontSize: 11, cursor: 'pointer', fontFamily: 'inherit',
                fontWeight: llmTemperature === p.params.llmTemperature ? 600 : 400,
              }}>
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {/* Advanced Toggle */}
      <div style={{ marginTop: -4 }}>
        <button onClick={() => setShowAdvanced(!showAdvanced)} style={{
          padding: '4px 0', border: 'none', background: 'transparent',
          color: 'hsl(var(--muted-foreground))', fontSize: 12, cursor: 'pointer',
          fontFamily: 'inherit',
        }}>
          {showAdvanced ? '▾ 高级设置' : '▸ 高级设置'}
        </button>
      </div>

      {/* Advanced Settings */}
      {showAdvanced && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16, paddingLeft: 8, borderLeft: '2px solid hsl(var(--border))' }}>
          <SettingRow label="温度" desc={`低=确定性，高=随机性 (${llmTemperature})`}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <input type="range" min="0" max="2" step="0.05" value={llmTemperature}
                onChange={(e) => updateSettings({ llmTemperature: parseFloat(e.target.value) })}
                style={{ flex: 1 }} />
              <span style={{ fontSize: 12, color: 'hsl(var(--muted-foreground))', minWidth: 28 }}>{llmTemperature}</span>
            </div>
          </SettingRow>

          <SettingRow label="Top P" desc={`核采样阈值 (${llmTopP})`}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <input type="range" min="0" max="1" step="0.05" value={llmTopP}
                onChange={(e) => updateSettings({ llmTopP: parseFloat(e.target.value) })}
                style={{ flex: 1 }} />
              <span style={{ fontSize: 12, color: 'hsl(var(--muted-foreground))', minWidth: 28 }}>{llmTopP}</span>
            </div>
          </SettingRow>

          <SettingRow label="Top K" desc={`限制候选词数，0=禁用 (${settings.llmTopK})`}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <input type="range" min="0" max="100" step="5" value={settings.llmTopK}
                onChange={(e) => updateSettings({ llmTopK: parseInt(e.target.value) })}
                style={{ flex: 1 }} />
              <span style={{ fontSize: 12, color: 'hsl(var(--muted-foreground))', minWidth: 28 }}>{settings.llmTopK}</span>
            </div>
          </SettingRow>

          <SettingRow label="重复惩罚" desc={`越高越避免重复 (${settings.llmRepeatPenalty})`}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <input type="range" min="1" max="2" step="0.05" value={settings.llmRepeatPenalty}
                onChange={(e) => updateSettings({ llmRepeatPenalty: parseFloat(e.target.value) })}
                style={{ flex: 1 }} />
              <span style={{ fontSize: 12, color: 'hsl(var(--muted-foreground))', minWidth: 28 }}>{settings.llmRepeatPenalty}</span>
            </div>
          </SettingRow>

          <SettingRow label="频率惩罚" desc={`惩罚高频词 (${settings.llmFrequencyPenalty})`}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <input type="range" min="0" max="2" step="0.1" value={settings.llmFrequencyPenalty}
                onChange={(e) => updateSettings({ llmFrequencyPenalty: parseFloat(e.target.value) })}
                style={{ flex: 1 }} />
              <span style={{ fontSize: 12, color: 'hsl(var(--muted-foreground))', minWidth: 28 }}>{settings.llmFrequencyPenalty}</span>
            </div>
          </SettingRow>

          <SettingRow label="存在惩罚" desc={`惩罚已出现概念 (${settings.llmPresencePenalty})`}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <input type="range" min="0" max="2" step="0.1" value={settings.llmPresencePenalty}
                onChange={(e) => updateSettings({ llmPresencePenalty: parseFloat(e.target.value) })}
                style={{ flex: 1 }} />
              <span style={{ fontSize: 12, color: 'hsl(var(--muted-foreground))', minWidth: 28 }}>{settings.llmPresencePenalty}</span>
            </div>
          </SettingRow>

          <SettingRow label="停止词" desc="逗号分隔，模型遇到即停止">
            <input value={settings.llmStopTokens}
              onChange={(e) => updateSettings({ llmStopTokens: e.target.value })}
              placeholder="<|im_end|>,<|endoftext|>"
              style={inputStyle} />
          </SettingRow>

          <SettingRow label="额外请求头" desc={'自定义 HTTP 头，JSON 格式，如 {"X-Custom": "value"}'}>
            <textarea value={llmExtraHeaders}
              onChange={(e) => updateSettings({ llmExtraHeaders: e.target.value })}
              rows={2}
              placeholder='{}'
              style={{ ...inputStyle, resize: 'vertical', fontFamily: 'monospace', fontSize: 11, minHeight: 40 }} />
          </SettingRow>

          <SettingRow label="额外请求体" desc={'自定义请求体参数，JSON 格式，如 {"thinking": {"type": "enabled"}}'}>
            <textarea value={llmExtraBody}
              onChange={(e) => updateSettings({ llmExtraBody: e.target.value })}
              rows={2}
              placeholder='{}'
              style={{ ...inputStyle, resize: 'vertical', fontFamily: 'monospace', fontSize: 11, minHeight: 40 }} />
          </SettingRow>
        </div>
      )}

      {/* Local Model Section */}
      <div style={{ marginTop: 12, paddingTop: 12, borderTop: '1px solid hsl(var(--border))' }}>
        <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12, color: 'hsl(var(--foreground))' }}>
          本地 LLM 路径
        </div>

        {/* Local Model Path */}
        <SettingRow label="本地模型路径" desc="GGUF (.gguf) 文件或 MLX (model.safetensors 目录) 的路径">
          <div style={{ display: 'flex', gap: 8 }}>
            <input value={settings.llmModelPath}
              onChange={(e) => updateSettings({ llmModelPath: e.target.value })}
              placeholder="/path/to/model.gguf 或 /mlx-model-dir/"
              style={{ ...inputStyle, flex: 1 }} />
            <button onClick={() => {
              const send = doSend()
              if (send) send({ type: 'model_config', action: 'set_local_model', path: settings.llmModelPath })
            }} style={{
              padding: '5px 12px', borderRadius: 6, border: '1px solid var(--btn-border)',
              background: 'var(--btn-bg)', color: 'var(--text-secondary)',
              fontSize: 11, cursor: 'pointer', fontFamily: 'inherit', whiteSpace: 'nowrap',
            }}>
              应用
            </button>
          </div>
        </SettingRow>
      </div>

      {/* Test Connection */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 12 }}>
        <button onClick={handleTest} disabled={testing} style={{
          padding: '6px 16px', borderRadius: 6, border: '1px solid var(--accent)',
          background: testing ? 'var(--border)' : 'var(--accent-bg)',
          color: testing ? 'var(--text-muted)' : 'var(--text-accent)',
          fontSize: 12, cursor: testing ? 'not-allowed' : 'pointer',
          fontFamily: 'inherit', fontWeight: 600,
        }}>
          {testing ? '测试中...' : '测试连接'}
        </button>
        {testResult && (
          <span style={{ fontSize: 12, color: testResult.success ? '#4caf50' : '#e0556a' }}>
            {testResult.success
              ? (testResult.model ? `已连接 (${testResult.model})` : '成功')
              : `失败: ${testResult.error}`}
          </span>
        )}
      </div>
    </SectionBox>
  )
}

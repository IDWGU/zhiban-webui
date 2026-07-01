import { useState, useEffect, useRef, useCallback } from 'react'
import { useAppStore } from '@/stores/appStore'
import { SectionBox, SettingRow, toggleBtnStyle, actionBtnStyle, inputStyle } from './SharedComponents'
import type { PaperTab } from '@/types'

const EMBEDDING_PRESETS = [
  { label: 'jina-v5-nano', model: 'jinaai/jina-embeddings-v5-text-nano', desc: '768维 / ~0.5GB / 轻量' },
  { label: 'BGE-M3', model: 'BAAI/bge-m3', desc: '1024维 / ~2.2GB / 中文强' },
  { label: 'KaLM-V2.5', model: 'KaLM-Embedding-V2.5', desc: '768维 / ~0.8GB' },
]

export default function DatabaseSection() {
  const storePapers = useAppStore(s => s.papers)
  const addPaper = useAppStore(s => s.addPaper)
  const clearPapers = useAppStore(s => s.clearPapers)
  const topK = useAppStore(s => s.settings.topK)
  const updateSettings = useAppStore(s => s.updateSettings)
  const buildIndex = useAppStore(s => s.buildIndex)
  const startBuildIndex = useAppStore(s => s.startBuildIndex)
  const buildingPhases = new Set(['scanning', 'extracting', 'embedding', 'paused'])
  const isBuilding = buildingPhases.has(buildIndex.phase)
  const isPaused = buildIndex.phase === 'paused'
  const embeddingModel = useAppStore(s => s.settings.embeddingModel)
  const embedResult = useAppStore(s => s.embeddingLoadResult)
  const embedProgress = useAppStore(s => s.embeddingProgress)
  const clearEmbedResult = useAppStore(s => s.setEmbeddingLoadResult)
  const [buildWarning, setBuildWarning] = useState(false)
  const [sysCheckResult, setSysCheckResult] = useState<string | null>(null)
  const [healthInfo, setHealthInfo] = useState<{ chunks: number; graphs: number } | null>(null)
  const [libPapers, setLibPapers] = useState<Array<{ name: string; path: string; size: number; mtime: number }>>([])
  const [selectedPaths, setSelectedPaths] = useState<Set<string>>(new Set())
  const lastClickIdx = useRef(-1)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const tabCounter = useRef(0)
  const [dialogModel, setDialogModel] = useState(embeddingModel)
  const [isModelLoading, setIsModelLoading] = useState(false)
  const [modelLoadStatus, setModelLoadStatus] = useState('')
  const [embeddingAvailable, setEmbeddingAvailable] = useState(false)
  const [forceRebuild, setForceRebuild] = useState(false)
  const buildAfterModelRef = useRef(false)

  // 挂载时拉取健康信息 + 论文库列表 + 嵌入模型状态（合并为一次 fetch）
  useEffect(() => {
    fetch('http://127.0.0.1:18921/health')
      .then(r => r.json())
      .then(d => {
        setHealthInfo({ chunks: d.vectors_chunks ?? 0, graphs: d.graphs_papers ?? 0 })
        setEmbeddingAvailable(!!d.embedding_available)
      })
      .catch(() => {})
    const send = (window as any).__zhiban_wsSend
    if (send) send({ type: 'list_library' })
  }, [])

  // 监听论文库更新
  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail
      if (detail.papers) setLibPapers(detail.papers)
    }
    window.addEventListener('library-update', handler)
    return () => window.removeEventListener('library-update', handler)
  }, [])

  useEffect(() => {
    if (buildIndex.result) {
      if (buildIndex.result.success) {
        fetch('http://127.0.0.1:18921/health')
          .then(r => r.json())
          .then(d => setHealthInfo({ chunks: d.vectors_chunks ?? 0, graphs: d.graphs_papers ?? 0 }))
          .catch(() => {})
      }
    }
  }, [buildIndex.result])

  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail
      setHealthInfo({ chunks: detail.chunks ?? 0, graphs: detail.graphs ?? 0 })
    }
    window.addEventListener('health-update', handler)
    return () => window.removeEventListener('health-update', handler)
  }, [])

  const runSystemCheck = useCallback(() => {
    fetch('http://127.0.0.1:18921/system-info')
      .then(r => r.json())
      .then(info => {
        const lines: string[] = [
          `平台: ${info.platform} (${info.machine})`,
          `CPU: ${info.cpu.physical_cores} 物理核 / ${info.cpu.logical_cores} 逻辑核`,
          `${info.memory.type}: ${info.memory.total_gb} GB (可用 ${info.memory.available_gb} GB)${
            info.memory.note ? `\n  → ${info.memory.note}` : ''
          }`,
        ]
        if (info.gpu.available) {
          lines.push(`GPU: ${info.gpu.name || 'Apple GPU'} (${info.gpu.backend})`)
        }
        if (info.disk.total_gb > 0) {
          lines.push(`磁盘: 共 ${info.disk.total_gb} GB / 剩余 ${info.disk.free_gb} GB (知识库路径)`)
        }
        lines.push('')
        // 检查是否满足最低要求
        const minMem = 4  // GB
        const minDisk = 5  // GB
        const memOk = info.memory.total_gb >= minMem
        const diskOk = info.disk.free_gb >= minDisk
        if (memOk && diskOk) {
          lines.push('✓ 配置满足要求')
        } else {
          if (!memOk) lines.push(`✗ 内存不足 (需要 ≥${minMem} GB)`)
          if (!diskOk) lines.push(`✗ 磁盘空间不足 (需要 ≥${minDisk} GB)`)
        }
        setSysCheckResult(lines.join('\n'))
      })
      .catch(() => {
        setSysCheckResult('✗ 无法连接后端，请确认 sidecar 正在运行')
      })
  }, [])

  const handleAddPaper = useCallback(() => { fileInputRef.current?.click() }, [])

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    if (files.length === 0) { e.target.value = ''; return }
    const send = (window as any).__zhiban_wsSend

    if (window.electronAPI?.getPathForFile) {
      // Electron: 直接拿本地路径
      const realPaths = files.map(f => window.electronAPI!.getPathForFile!(f as any) || f.name)
      if (send && realPaths.length > 0) send({ type: 'add_papers', files: realPaths })
    } else {
      // WebUI: 先上传再发路径
      Promise.all(files.map(async (file) => {
        const formData = new FormData()
        formData.append('file', file)
        const res = await fetch('/upload', { method: 'POST', body: formData })
        const data = await res.json()
        return data.filePath || ''
      })).then((paths) => {
        const valid = paths.filter(Boolean)
        if (send && valid.length > 0) send({ type: 'add_papers', files: valid })
        // add_papers 完成后会自动更新论文库
      }).catch(err => {
        console.error('Upload failed:', err)
        useAppStore.getState().pushNotification('error', '论文上传失败')
      })
    }
    e.target.value = ''
  }, [])

  const handleBuildVector = useCallback(() => {
    setDialogModel(embeddingModel)
    runSystemCheck()
    setBuildWarning(true)
  }, [runSystemCheck, embeddingModel])

  const confirmBuild = useCallback(() => {
    setBuildWarning(false)

    const send = (window as any).__zhiban_wsSend
    if (!send) {
      alert('WebSocket 未连接，请确保后端正在运行')
      return
    }

    if (!embeddingAvailable) {
      // 模型未加载，先发模型加载请求
      buildAfterModelRef.current = true
      setIsModelLoading(true)
      setModelLoadStatus('正在加载嵌入模型...')
      send({ type: 'model_config', action: 'set_embedding_model', model: dialogModel })
    } else {
      startBuildIndex()
      send({ type: 'build_index', force: forceRebuild })
    }
  }, [embeddingAvailable, dialogModel, startBuildIndex, forceRebuild])

  const sendBuildControl = useCallback((action: 'pause' | 'resume' | 'cancel') => {
    const send = (window as any).__zhiban_wsSend
    if (send) {
      send({ type: 'build_control', action })
    }
  }, [])

  const [importing, setImporting] = useState(false)
  const [importResult, setImportResult] = useState<string | null>(null)
  const importTimerRef = useRef<ReturnType<typeof setTimeout>>()
  const importCleanupRef = useRef<() => void>()

  useEffect(() => {
    return () => {
      clearTimeout(importTimerRef.current)
      importCleanupRef.current?.()
    }
  }, [])

  const handleImportVector = useCallback(async () => {
    const dirPath = await window.electronAPI?.selectDirectory?.()
    if (!dirPath) return

    setImporting(true)
    setImportResult(null)

    const send = (window as any).__zhiban_wsSend
    if (!send) {
      setImportResult('WebSocket 未连接')
      setImporting(false)
      return
    }

    const onResult = (e: Event) => {
      const detail = (e as CustomEvent).detail
      clearTimeout(importTimerRef.current!)
      setImporting(false)
      if (detail.success) {
        setImportResult(`导入成功！已加载 ${detail.chunks} 个向量片段`)
      } else {
        setImportResult(`导入失败: ${detail.error}`)
      }
      window.removeEventListener('import-vector-result', onResult)
    }
    window.addEventListener('import-vector-result', onResult)
    importCleanupRef.current = () => window.removeEventListener('import-vector-result', onResult)

    importTimerRef.current = setTimeout(() => {
      window.removeEventListener('import-vector-result', onResult)
      setImporting(false)
      setImportResult('导入超时，请检查后端是否运行')
    }, 30000)

    send({ type: 'import_vector_store', sourcePath: dirPath })
  }, [])

  // 监听模型加载完成 → 自动开始构建
  useEffect(() => {
    if (embedResult && buildAfterModelRef.current) {
      buildAfterModelRef.current = false
      setIsModelLoading(false)
      if (embedResult.success) {
        setModelLoadStatus('模型加载完成，正在构建向量库...')
        setEmbeddingAvailable(true)
        const send = (window as any).__zhiban_wsSend
        if (send) {
          startBuildIndex()
          send({ type: 'build_index', force: forceRebuild })
        }
      } else {
        setModelLoadStatus(`模型加载失败: ${embedResult.error}`)
      }
      clearEmbedResult(null)
    }
  }, [embedResult, clearEmbedResult])

  // 监听嵌入模型下载进度
  useEffect(() => {
    if (isModelLoading && embedProgress.message) {
      setModelLoadStatus(embedProgress.message)
    }
  }, [embedProgress, isModelLoading])

  const openLibraryPaper = useCallback((file: { name: string; path: string }) => {
    const ext = file.name.split('.').pop()?.toLowerCase() || 'pdf'
    const id = `paper-${++tabCounter.current}`
    addPaper({ id, name: file.name, type: ext as PaperTab['type'], path: file.path, extractedText: '' })
  }, [addPaper])
  // 论文标签过多时提示清理（不静默清除）
  useEffect(() => {
    if (storePapers.length > 100) {
      const ok = confirm(`当前打开了 ${storePapers.length} 个论文标签，可能导致性能下降。是否关闭全部标签？（可从论文库重新打开）`)
      if (ok) { clearPapers(); localStorage.removeItem('zhiban-papers') }
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <SectionBox title="知识库管理">
      {/* 论文库（已存储的论文） */}
      <SettingRow label={
        <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span>论文库 ({libPapers.length})</span>
        </span>
      } desc="存储在知识库中的论文，点击可直接打开阅读">
        <div style={{ ...inputStyle, maxHeight: 160, overflowY: 'auto', padding: 4 }}>
          {libPapers.length === 0 ? (
            <div style={{ fontSize: 11, color: 'var(--text-muted)', padding: '4px 8px' }}>
              暂无论文，点击上方「添加论文」存入
            </div>
          ) : libPapers.map((f, i) => {
            const isOpen = storePapers.some(p => p.path === f.path)
            const isSelected = selectedPaths.has(f.path)
            const cbSize = 14
            return (
              <div key={i} onClick={(e) => {
                if (e.shiftKey && lastClickIdx.current >= 0) {
                  const [lo, hi] = [Math.min(lastClickIdx.current, i), Math.max(lastClickIdx.current, i)]
                  const paths = new Set(libPapers.slice(lo, hi + 1).map(p => p.path))
                  setSelectedPaths(paths)
                } else {
                  setSelectedPaths(new Set([f.path]))
                  lastClickIdx.current = i
                }
              }} onDoubleClick={() => openLibraryPaper(f)} style={{
                fontSize: 11, padding: '5px 6px', cursor: 'pointer',
                color: isOpen ? 'var(--text-accent)' : 'var(--text-primary)',
                background: isSelected ? 'var(--accent-bg)' : isOpen ? 'var(--accent-bg)' : 'transparent',
                borderRadius: 4, marginBottom: 1,
                display: 'flex', alignItems: 'center', gap: 6,
                transition: 'background 0.1s',
                outline: isSelected ? '1px solid var(--accent)' : 'none',
              }}
                onMouseEnter={e => { if (!isOpen && !isSelected) (e.currentTarget as HTMLElement).style.background = 'var(--bg-elevated)' }}
                onMouseLeave={e => { if (!isOpen && !isSelected) (e.currentTarget as HTMLElement).style.background = 'transparent' }}
              >
                <span style={{
                  display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                  width: cbSize, height: cbSize, minWidth: cbSize, borderRadius: 3,
                  border: `1.5px solid ${isSelected ? 'var(--accent)' : 'var(--text-muted)'}`,
                  background: isSelected ? 'var(--accent)' : 'transparent',
                  color: '#fff', fontSize: 10, fontWeight: 700,
                  transition: 'all 0.15s',
                }}>
                  {isSelected ? '✓' : ''}
                </span>
                <span>{f.name.endsWith('.pdf') ? '📄' : '📝'}</span>
                <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{f.name}</span>
                {isOpen && <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>已打开</span>}
              </div>
            )
          })}
        </div>
      </SettingRow>

      <div style={{ display: 'flex', gap: 8 }}>
        <button onClick={handleAddPaper} style={{ ...actionBtnStyle, flex: 1 }}>➕ 添加论文到知识库</button>
        <button onClick={() => {
          if (selectedPaths.size === 0) return
          const names = [...selectedPaths].map(p => {
            const n = p.split('/').pop() || ''
            return n.length > 40 ? n.slice(0, 37) + '...' : n
          })
          const preview = names.slice(0, 5).join('\n')
          const more = names.length > 5 ? `\n... 等共 ${selectedPaths.size} 篇` : ''
          if (!confirm(`确定删除以下论文吗？\n\n${preview}${more}\n\n文件将被永久删除。`)) return
          const send = (window as any).__zhiban_wsSend
          if (send) send({ type: 'delete_library_papers', paths: [...selectedPaths] })
          setSelectedPaths(new Set())
        }} disabled={selectedPaths.size === 0} style={{
          ...actionBtnStyle, flex: 1,
          color: selectedPaths.size > 0 ? '#e0556a' : 'var(--text-muted)',
          border: `1px solid ${selectedPaths.size > 0 ? '#e0556a' : 'var(--border)'}`,
          opacity: selectedPaths.size > 0 ? 1 : 0.4,
          cursor: selectedPaths.size > 0 ? 'pointer' : 'default',
        }}>
          🗑️ 删除所选{selectedPaths.size > 0 ? ` (${selectedPaths.size})` : ''}
        </button>
        <input ref={fileInputRef} type="file" accept=".pdf,.docx,.txt,.md" multiple
          onChange={handleFileChange} style={{ display: 'none' }} />
      </div>

      {/* 已打开的阅读标签页 */}
      {storePapers.length > 0 && (
        <SettingRow label={
          <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span>已打开的阅读标签 ({storePapers.length})</span>
            <button onClick={() => { clearPapers(); localStorage.removeItem('zhiban-papers') }} style={{
              ...toggleBtnStyle, fontSize: 10, padding: '2px 8px', color: '#e0556a',
            }}>
              ✕ 关闭全部
            </button>
          </span>
        } desc="拖入或从论文库点击打开的论文">
          <div style={{ ...inputStyle, maxHeight: 100, overflowY: 'auto', padding: 8 }}>
            {storePapers.map((p) => (
              <div key={p.id} style={{ fontSize: 11, padding: '2px 0', color: 'var(--text-primary)' }}>📄 {p.name}</div>
            ))}
          </div>
        </SettingRow>
      )}

      <SettingRow label="搜索结果数量" desc="每次检索返回的论文片段数 (5-50)，越多越丰富但消耗更多 Token">
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <input type="range" min={5} max={50} step={5} value={topK}
            onChange={(e) => updateSettings({ topK: Number(e.target.value) })}
            style={{ flex: 1, accentColor: 'var(--accent)' }} />
          <span style={{ fontSize: 12, color: 'var(--text-primary)', minWidth: 24, textAlign: 'right' }}>{topK}</span>
        </div>
      </SettingRow>

      <div style={{ borderTop: '1px solid var(--border)', margin: '8px 0' }} />

      <SettingRow label="系统兼容性检查" desc="构建前检测内存和 CPU 配置">
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <button onClick={runSystemCheck} style={{ ...actionBtnStyle, fontSize: 11 }}>🔍 检查配置</button>
          {sysCheckResult && (
            <pre style={{ fontSize: 10, color: 'var(--text-secondary)', margin: 0, whiteSpace: 'pre-wrap' }}>{sysCheckResult}</pre>
          )}
        </div>
      </SettingRow>

      <div style={{ marginTop: 8 }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6, cursor: 'pointer', fontSize: 11, color: 'var(--text-secondary)' }}>
          <input type="checkbox" checked={forceRebuild} onChange={e => setForceRebuild(e.target.checked)}
            style={{ width: 14, height: 14, cursor: 'pointer', accentColor: 'var(--accent)' }} />
          全量重建（清空后重新向量化全部论文）
        </label>
        <button onClick={handleBuildVector} disabled={isBuilding && !isPaused} style={{
          ...actionBtnStyle, background: 'var(--accent-bg)', color: 'var(--text-accent)',
          border: '1px solid var(--border-accent)', opacity: (isBuilding && !isPaused) ? 0.5 : 1,
          width: '100%', padding: '10px',
        }}>
          {isPaused ? '⏸️ 构建已暂停' : isBuilding ? '⏳ 构建中...' : forceRebuild ? '🔨 全量重建向量库' : '🔨 增量构建向量库'}
        </button>
        <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>
          {forceRebuild ? '清空全部向量后重新构建' : '仅向量化新增论文，已索引的跳过'}
        </div>

        {isBuilding && buildIndex.phase !== 'idle' && (
          <div style={{ marginTop: 8 }}>
            <div style={{ width: '100%', height: 6, borderRadius: 3, background: 'var(--bg-elevated)', overflow: 'hidden' }}>
              <div style={{
                width: isPaused ? '100%' : `${buildIndex.progress.total > 0 ? (buildIndex.progress.current / buildIndex.progress.total) * 100 : 100}%`,
                height: '100%', borderRadius: 3,
                background: isPaused ? 'var(--text-muted)' : 'var(--accent)',
                transition: 'width 0.3s',
              }} />
            </div>
            <div style={{ fontSize: 10, color: isPaused ? 'var(--text-muted)' : 'var(--text-secondary)', marginTop: 4 }}>
              {buildIndex.message || (isPaused ? '构建已暂停' : '构建中...')}
            </div>
            {/* 暂停/继续 + 取消按钮 */}
            <div style={{ display: 'flex', gap: 6, marginTop: 8 }}>
              {isPaused ? (
                <button onClick={() => sendBuildControl('resume')} style={{
                  ...actionBtnStyle, flex: 1, background: '#2a9d8f', color: '#fff',
                  border: 'none', fontSize: 11,
                }}>
                  ▶ 继续构建
                </button>
              ) : (
                <button onClick={() => sendBuildControl('pause')} style={{
                  ...actionBtnStyle, flex: 1, background: '#e6a817', color: '#fff',
                  border: 'none', fontSize: 11,
                }}>
                  ⏸️ 暂停
                </button>
              )}
              <button onClick={() => sendBuildControl('cancel')} style={{
                ...actionBtnStyle, background: '#d32f2f', color: '#fff',
                border: 'none', fontSize: 11,
              }}>
                ✕ 取消
              </button>
            </div>
          </div>
        )}

        {buildIndex.result && !isBuilding && (
          <div style={{ fontSize: 11, marginTop: 6, padding: '6px 10px', borderRadius: 6,
            background: 'var(--bg-elevated)', color: buildIndex.result.success ? '#4caf50' : '#e0556a' }}>
            {buildIndex.result.success
              ? `向量库构建完成！共 ${buildIndex.result.chunks ?? 0} 个片段`
              : `构建失败: ${buildIndex.result.error}`}
          </div>
        )}
      </div>

      <div style={{ marginTop: 8 }}>
        <button onClick={handleImportVector} disabled={importing} style={{ ...actionBtnStyle, width: '100%', opacity: importing ? 0.5 : 1 }}>
          {importing ? '⏳ 导入中...' : '📂 导入向量库'}
        </button>
        <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>
          导入共享或之前导出的 ChromaDB 目录
        </div>
        {importResult && (
          <div style={{ fontSize: 11, marginTop: 6, padding: '6px 10px', borderRadius: 6,
            background: 'var(--bg-elevated)', color: importResult.startsWith('导入成功') ? '#4caf50' : '#e0556a' }}>
            {importResult}
          </div>
        )}
      </div>

      {/* 清空向量库 */}
      <div style={{ marginTop: 8 }}>
        <button onClick={() => {
          if (confirm('确定清空整个向量库吗？此操作不可恢复。')) {
            const send = (window as any).__zhiban_wsSend
            if (send) send({ type: 'clear_vector_store' })
          }
        }} style={{
          ...actionBtnStyle, width: '100%',
          background: 'transparent', color: '#e0556a',
          border: '1px solid #e0556a',
        }}>
          🗑️ 清空向量库
        </button>
      </div>

      <IndexedPapersList />

      {buildWarning && (
        <div onClick={(e) => e.stopPropagation()} style={warningOverlayStyle}>
          <div style={warningBoxStyle}>
            <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 12, color: 'var(--text-primary)' }}>
              构建向量库
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: 16 }}>
              <p>将使用嵌入模型对所有论文文本进行向量化。</p>
              <p style={{ color: '#FF9800', margin: '12px 0' }}>
                CPU 使用率将达到 100% — 属于正常现象。<br />
                每篇论文约需 15-30 秒。
              </p>
              <p>系统会自动检测配置以确保内存安全。<br />
              模型加载完成后会释放占用的内存。</p>
            </div>

            {/* 模型选择 */}
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 6 }}>选择嵌入模型：</div>
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                {EMBEDDING_PRESETS.map(p => (
                  <button key={p.label} onClick={() => { setDialogModel(p.model); updateSettings({ embeddingModel: p.model }) }}
                    style={{
                      padding: '4px 10px', borderRadius: 10, border: '1px solid var(--border)',
                      background: dialogModel === p.model ? 'var(--accent-bg)' : 'transparent',
                      color: dialogModel === p.model ? 'var(--text-accent)' : 'var(--text-secondary)',
                      fontSize: 11, cursor: 'pointer', fontFamily: 'inherit',
                      fontWeight: dialogModel === p.model ? 600 : 400,
                    }}>
                    {p.label}
                    <span style={{ fontSize: 10, opacity: 0.7, display: 'block' }}>{p.desc}</span>
                  </button>
                ))}
              </div>
            </div>

            {sysCheckResult && (
              <pre style={{ fontSize: 10, color: 'var(--text-secondary)', background: 'var(--bg-elevated)',
                padding: 8, borderRadius: 6, marginBottom: 12, whiteSpace: 'pre-wrap' }}>{sysCheckResult}</pre>
            )}

            {/* 模型加载进度 */}
            {isModelLoading && (
              <div style={{ marginBottom: 12 }}>
                <div style={{ width: '100%', height: 6, borderRadius: 3, background: 'var(--bg-elevated)', overflow: 'hidden' }}>
                  <div style={{
                    width: `${Math.max(2, embedProgress.percent || 5)}%`, height: '100%', borderRadius: 3,
                    background: 'var(--accent)', transition: 'width 1s ease',
                  }} />
                </div>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>{modelLoadStatus}</div>
              </div>
            )}

            <label style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 12, cursor: 'pointer', fontSize: 12, color: 'var(--text-secondary)' }}>
              <input type="checkbox" checked={forceRebuild} onChange={e => setForceRebuild(e.target.checked)}
                style={{ width: 14, height: 14, cursor: 'pointer', accentColor: 'var(--accent)' }} />
              全量重建（清空后重新向量化全部论文）
            </label>
            <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
              <button onClick={() => { setBuildWarning(false); setIsModelLoading(false); buildAfterModelRef.current = false }}
                style={{ ...toggleBtnStyle }}>取消</button>
              <button onClick={confirmBuild} disabled={isModelLoading}
                style={{
                  ...toggleBtnStyle, background: isModelLoading ? 'var(--border)' : 'var(--accent-bg)',
                  color: isModelLoading ? 'var(--text-muted)' : 'var(--text-accent)',
                  opacity: isModelLoading ? 0.6 : 1,
                }}>
                {isModelLoading ? '加载中...' : '确认构建'}
              </button>
            </div>
          </div>
        </div>
      )}
    </SectionBox>
  )
}

function IndexedPapersList() {
  const [papers, setPapers] = useState<Array<{doc_id:string;filename:string;source:string;chunks:number}>>([])
  const [expanded, setExpanded] = useState(false)

  useEffect(() => {
    const onUpdate = (e: Event) => setPapers((e as CustomEvent).detail.papers)
    window.addEventListener('indexed-papers-update', onUpdate)
    // 加载时请求一次
    const send = (window as any).__zhiban_wsSend
    if (send) send({ type: 'list_indexed_papers' })
    return () => window.removeEventListener('indexed-papers-update', onUpdate)
  }, [])

  if (papers.length === 0) return null

  const removePaper = (docId: string) => {
    if (!confirm(`确定删除 "${docId}" 的所有向量吗？`)) return
    const send = (window as any).__zhiban_wsSend
    if (send) send({ type: 'remove_paper_vectors', doc_ids: [docId] })
  }

  const listStyle: React.CSSProperties = {
    maxHeight: expanded ? 400 : 0,
    overflow: 'hidden',
    transition: 'max-height 0.3s',
    fontSize: 11,
  }

  return (
    <div style={{ marginTop: 8 }}>
      <button onClick={() => {
        setExpanded(!expanded)
        if (!expanded) {
          const send = (window as any).__zhiban_wsSend
          if (send) send({ type: 'list_indexed_papers' })
        }
      }} style={{
        ...actionBtnStyle, width: '100%', fontSize: 11,
        background: 'var(--bg-elevated)', color: 'var(--text-secondary)',
        border: '1px solid var(--border)',
      }}>
        📋 已索引文章 ({papers.length} 篇) {expanded ? '▲' : '▼'}
      </button>
      <div style={listStyle}>
        <div style={{ maxHeight: 400, overflow: 'auto', marginTop: 4 }}>
          {papers.map(p => (
            <div key={p.doc_id} style={{
              display: 'flex', alignItems: 'center', gap: 6,
              padding: '4px 8px', borderBottom: '1px solid var(--border)',
              fontSize: 11, color: 'var(--text-secondary)',
            }}>
              <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {p.filename}
              </span>
              <span style={{ color: 'var(--text-muted)', fontSize: 10, minWidth: 40, textAlign: 'right' }}>
                {p.chunks} chunks
              </span>
              <button onClick={() => removePaper(p.doc_id)} style={{
                background: 'transparent', color: '#e0556a', border: 'none',
                cursor: 'pointer', fontSize: 12, padding: '0 4px',
              }} title="删除此文向量">
                ✕
              </button>
            </div>
          ))}
        </div>
        <button onClick={() => {
          if (confirm(`确定删除全部 ${papers.length} 篇文章的向量吗？`)) {
            const send = (window as any).__zhiban_wsSend
            if (send) send({ type: 'clear_vector_store' })
          }
        }} style={{
          ...actionBtnStyle, width: '100%', marginTop: 4, fontSize: 10,
          background: 'transparent', color: '#e0556a', border: 'none',
        }}>
          全部删除
        </button>
      </div>
    </div>
  )
}

const warningOverlayStyle: React.CSSProperties = {
  position: 'fixed', inset: 0, zIndex: 3000,
  background: 'hsl(var(--background) / 0.60)',
  backdropFilter: 'blur(4px)',
  WebkitBackdropFilter: 'blur(4px)',
  display: 'flex', alignItems: 'center', justifyContent: 'center',
}
const warningBoxStyle: React.CSSProperties = {
  width: 420, padding: 28, borderRadius: 12,
  background: 'hsl(var(--dialog))',
  border: '1px solid hsl(var(--border))',
  boxShadow: '0 4px 24px rgba(0,0,0,0.25)',
  animation: 'fadeInZoom 0.2s ease-out',
}

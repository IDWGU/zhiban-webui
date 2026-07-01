/**
 * Python Sidecar 进程管理
 * Dev: spawn Python directly
 * Production: use bundled start-sidecar.sh launcher
 *
 * Logging: 双写策略
 *   - 内存缓冲区 (_debugLogBuffer) → IPC 推送给渲染进程 (调试面板)
 *   - 文件日志 (~/Library/Application Support/ZhiBan/logs/) → 用于事后 Bug 分析
 *   - Crash 状态持久化 → crash.json (记录最后一次崩溃原因和进程退出码)
 */
import { app } from 'electron'
import { spawn, exec, ChildProcess } from 'child_process'
import path from 'path'
import fs from 'fs'
import os from 'os'

let sidecarProcess: ChildProcess | null = null
let healthCheckTimer: ReturnType<typeof setInterval> | null = null
let restartCount = 0
let sidecarStartTime = 0
const MAX_RESTART_COUNT = 5
const HEALTH_CHECK_INTERVAL_MS = 5000
const STARTUP_GRACE_PERIOD_MS = 120000  // 120s grace for model loading (embedding + LLM)
const RESTART_DELAY_MS = 3000

// Error callback for surfacing sidecar startup errors to the renderer
let _onErrorCallback: ((message: string) => void) | null = null

let _onReadyCallback: (() => void) | null = null

export function setSidecarErrorCallback(cb: (message: string) => void) {
  _onErrorCallback = cb
}

export function setSidecarReadyCallback(cb: () => void) {
  _onReadyCallback = cb
}

// ── 文件日志（持久化，用于事后 Bug 分析） ──

const MAX_LOG_FILES = 10
const MAX_LOG_SIZE_BYTES = 5 * 1024 * 1024  // 5MB per log file

let _logDir: string | null = null
let _logStream: fs.WriteStream | null = null
let _logFileSize = 0

function getLogDir(): string {
  if (_logDir) return _logDir
  _logDir = path.join(app.getPath('userData'), 'logs')
  fs.mkdirSync(_logDir, { recursive: true })
  return _logDir
}

function rotateLogFile(): void {
  if (_logStream) {
    _logStream.end()
    _logStream = null
    _logFileSize = 0
  }
  const dir = getLogDir()

  // Shift old logs: zhiban.9.log → zhiban.10.log (删除), ... zhiban.0.log → zhiban.1.log
  const oldestFile = path.join(dir, `zhiban.${MAX_LOG_FILES - 1}.log`)
  if (fs.existsSync(oldestFile)) fs.unlinkSync(oldestFile)
  for (let i = MAX_LOG_FILES - 1; i > 0; i--) {
    const oldPath = path.join(dir, `zhiban.${i - 1}.log`)
    const newPath = path.join(dir, `zhiban.${i}.log`)
    if (fs.existsSync(oldPath)) {
      try { fs.renameSync(oldPath, newPath) } catch {}
    }
  }
  // Rename current to zhiban.0.log
  const currentPath = path.join(dir, 'zhiban.log')
  if (fs.existsSync(currentPath)) {
    try { fs.renameSync(currentPath, path.join(dir, 'zhiban.0.log')) } catch {}
  }
}

function ensureLogStream(): fs.WriteStream {
  if (_logStream && _logFileSize < MAX_LOG_SIZE_BYTES) return _logStream
  rotateLogFile()
  const logPath = path.join(getLogDir(), 'zhiban.log')
  _logStream = fs.createWriteStream(logPath, { flags: 'a' })
  _logFileSize = (() => { try { return fs.statSync(logPath).size } catch { return 0 } })()
  return _logStream
}

function writeToLogFile(line: string): void {
  try {
    const stream = ensureLogStream()
    stream.write(line + '\n')
    _logFileSize += Buffer.byteLength(line) + 1
  } catch {
    // 日志写入失败不应影响主流程
  }
}

/** 持久化 crash 状态 (用于启动后分析上次崩溃原因) */
function persistCrashState(reason: string, exitCode: number | null): void {
  try {
    const crashPath = path.join(getLogDir(), 'crash.json')
    const data = {
      timestamp: new Date().toISOString(),
      reason,
      exitCode,
      restartCount,
      uptime: Date.now() - sidecarStartTime,
      version: app.getVersion(),
      platform: process.platform,
      arch: process.arch,
    }
    fs.writeFileSync(crashPath, JSON.stringify(data, null, 2))
  } catch {
    // 失败不阻塞主流程
  }
}

/** 读取上次 crash 状态 (用于启动诊断) */
export function readCrashState(): Record<string, any> | null {
  try {
    const crashPath = path.join(getLogDir(), 'crash.json')
    if (!fs.existsSync(crashPath)) return null
    return JSON.parse(fs.readFileSync(crashPath, 'utf-8'))
  } catch {
    return null
  }
}

// ── 调试日志回调（发送到前端调试面板 + 持久化到文件） ──
const _debugLogBuffer: string[] = []
const MAX_DEBUG_BUFFER = 1000
let _debugLogCallback: ((entry: string) => void) | null = null
export function setDebugLogCallback(cb: (entry: string) => void) {
  _debugLogCallback = cb
}
export function getDebugLogBuffer(): string[] {
  const buf = [..._debugLogBuffer]
  return buf
}
function emitDebug(entry: string) {
  const ts = new Date().toISOString().split('T')[1]?.slice(0, 12) ?? ''
  const line = `[${ts}] ${entry}`
  console.log(`[debug] ${line}`)
  _debugLogBuffer.push(line)
  if (_debugLogBuffer.length > MAX_DEBUG_BUFFER) _debugLogBuffer.shift()
  // IPC 推送（渲染进程可能未就绪，失败不影响）
  _debugLogCallback?.(line)
  // 持久化到文件（异步写入，不阻塞主流程）
  writeToLogFile(line)
}

// ── 项目根目录检测 ──

let _projectDirCache: string | null = null

function findProjectDir(): string | null {
  if (_projectDirCache) return _projectDirCache

  const candidates: string[] = []

  // 1. Dev mode: __dirname is dist-electron/ → project is one level up
  if (!app.isPackaged) {
    candidates.push(path.join(__dirname, '..'))
  }

  // 2. Production mode: search upward from Resources/
  if (process.resourcesPath) {
    let dir = process.resourcesPath
    for (let i = 0; i < 5; i++) {
      candidates.push(dir)
      const parent = path.dirname(dir)
      if (parent === dir) break
      dir = parent
    }
  }

  // 3. Common download locations
  candidates.push(
    path.join(os.homedir(), 'Downloads', 'zhiban-standalone'),
    path.join(os.homedir(), 'Desktop', 'zhiban-standalone'),
    path.join(os.homedir(), 'Documents', 'zhiban-standalone'),
  )

  for (const p of candidates) {
    const marker = path.join(p, 'sidecar', '.venv', 'bin', 'python3')
    if (fs.existsSync(marker)) {
      _projectDirCache = p
      return p
    }
  }

  return null
}

function isVenvAvailable(): boolean {
  return findProjectDir() !== null
}

export function getSidecarPath(): string {
  const isDev = !app.isPackaged

  if (isDev) {
    const venvPython = path.join(__dirname, '..', 'sidecar', '.venv', 'bin', 'python3')
    if (fs.existsSync(venvPython)) return venvPython
    return 'python3'
  }

  // Production: always use bundled start-sidecar.sh.
  // Never fall back to a venv Python from a project directory on the user's machine —
  // that would load the wrong sidecar sources and bypass the -P isolation flag.
  const resourcesPath = process.resourcesPath || path.join(app.getAppPath(), '..', 'Resources')
  const launcher = path.join(resourcesPath, 'sidecar-dist', 'start-sidecar.sh')

  if (fs.existsSync(launcher)) {
    return launcher
  }

  throw new Error(`Sidecar launcher not found at: ${launcher}\n请重新安装知伴或检查应用完整性。`)
}

export function getSidecarArgs(): string[] {
  const isDev = !app.isPackaged

  if (isDev) {
    return ['-m', 'sidecar.server']
  }

  // Production: start-sidecar.sh handles its own args
  const sidecarPath = getSidecarPath()
  if (sidecarPath.endsWith('.sh')) {
    return []
  }

  // Production PyInstaller binary needs no extra args
  return []
}

function getProjectDir(): string {
  // 打包模式下，app.asar 是文件不是目录，不能作为 spawn cwd
  // 改用 process.resourcesPath（真实目录，如 /Applications/知伴.app/Contents/Resources）
  if (app.isPackaged) {
    return process.resourcesPath || path.dirname(app.getAppPath())
  }
  return findProjectDir() || path.join(__dirname, '..')
}

export function getSidecarEnv(): Record<string, string> {
  const isDev = !app.isPackaged

  // Only pass through environment variables that the sidecar actually needs.
  // Do NOT use ...process.env — that would leak Proma's ANTHROPIC_API_KEY,
  // ANTHROPIC_BASE_URL, PROMADB_HOST, and any other parent-process secrets
  // into the Python child process.
  //
  // ANTHROPIC_API_KEY is intentionally captured for LLM_API_KEY as a convenience
  // fallback (DeepSeek accepts Anthropic-format keys on its main API), but we
  // then blank it so the raw key name is not visible in the child.
  const parentKey = process.env.ANTHROPIC_API_KEY || ''
  const passThrough = [
    'PATH', 'HOME', 'USER', 'SHELL', 'LANG', 'TMPDIR',
    'PYTHONUNBUFFERED', 'SIDECAR_DEBUG', 'DEBUG_WORKFLOW',
  ]
  const parentEnv: Record<string, string> = {}
  for (const k of passThrough) {
    if (process.env[k]) parentEnv[k] = process.env[k]!
  }

  const common = {
    ...parentEnv,
    SIDECAR_NO_RELOAD: '1',  // Electron 管理生命周期，无需 uvicorn 内部 reload
    HF_ENDPOINT: 'https://hf-mirror.com',
    DEEPSEEK_API_KEY: process.env.DEEPSEEK_API_KEY || '',
    DEEPSEEK_BASE_URL: process.env.DEEPSEEK_BASE_URL || '__local__',
    LLM_API_KEY: process.env.LLM_API_KEY || process.env.DEEPSEEK_API_KEY || parentKey,
    LLM_BASE_URL: process.env.LLM_BASE_URL || process.env.DEEPSEEK_BASE_URL || '__local__',
    // Explicitly blank Proma's Anthropic vars so they are NOT visible in the child process
    ANTHROPIC_API_KEY: '',
    ANTHROPIC_BASE_URL: '',
  }

  // Production: inject bundled model paths so PyInstaller sidecar can find them
  const resourcesPath = process.resourcesPath || path.join(app.getAppPath(), '..', 'Resources')
  const sdDir = path.join(resourcesPath, 'sidecar-dist')
  const modelsDir = path.join(sdDir, 'models')

  // Detect which LLM model is bundled (2B or 4B)
  const llmDir = path.join(modelsDir, 'llm')
  let llmModelPath = ''
  if (fs.existsSync(llmDir)) {
    const ggufFiles = fs.readdirSync(llmDir).filter(f => f.endsWith('.gguf'))
    if (ggufFiles.length > 0) {
      llmModelPath = path.join(llmDir, ggufFiles[0])
    }
  }

  // Detect translation model
  const transDir = path.join(modelsDir, 'translation')
  let transModelPath = ''
  if (fs.existsSync(transDir)) {
    const transFiles = fs.readdirSync(transDir).filter(f => f.endsWith('.gguf'))
    if (transFiles.length > 0) {
      transModelPath = path.join(transDir, transFiles[0])
    }
  }

  const prodEnv: Record<string, string> = {
    ...common,
    ZHIBAN_RESOURCES: resourcesPath,
    ZHIBAN_MODELS_DIR: modelsDir,
  }
  if (llmModelPath) prodEnv.LLM_MODEL_PATH = llmModelPath
  if (transModelPath) prodEnv.TRANSLATION_MODEL_PATH = transModelPath

  if (isDev) {
    const projectDir = getProjectDir()
    return {
      ...prodEnv,
      PYTHONPATH: path.resolve(projectDir),
    }
  }

  return prodEnv
}

function isSidecarRunning(): Promise<boolean> {
  return new Promise((resolve) => {
    try {
      const net = require('net')
      const sock = new net.Socket()
      sock.once('connect', () => {
        sock.destroy()
        resolve(true)
      })
      sock.once('error', () => {
        sock.destroy()
        resolve(false)
      })
      sock.setTimeout(500, () => {
        sock.destroy()
        resolve(false)
      })
      sock.connect(18921, '127.0.0.1')
    } catch {
      resolve(false)
    }
  })
}

let _starting = false

export async function startSidecar(): Promise<boolean> {
  if (sidecarProcess) return true
  if (_starting) { console.log('[sidecar] Already starting, skipping duplicate call'); return false }
  _starting = true

  // 检查上次启动是否有 crash 记录
  const prevCrash = readCrashState()

  emitDebug(`=== 知伴 Sidecar 启动 ===`)
  emitDebug(`isPackaged: ${app.isPackaged}`)
  emitDebug(`version: ${app.getVersion()}`)
  emitDebug(`resourcesPath: ${process.resourcesPath || '(not set)'}`)
  emitDebug(`app.getAppPath: ${app.getAppPath()}`)
  if (prevCrash) {
    emitDebug(`上次崩溃记录: ${prevCrash.reason} (exitCode=${prevCrash.exitCode}, uptime=${Math.round((prevCrash.uptime || 0)/1000)}s)`)
  }

  // Check if sidecar is already running (e.g., launched manually in dev mode)
  if (await isSidecarRunning()) {
    emitDebug('端口 18921 已被占用 (已运行的 sidecar)')
    sidecarStartTime = Date.now()
    startHealthCheck()
    _starting = false
    return true
  }

  const cmd = getSidecarPath()
  const args = getSidecarArgs()
  const env = getSidecarEnv()

  emitDebug(`启动命令: ${cmd} ${args.join(' ')}`)
  emitDebug(`CWD: ${getProjectDir()}`)
  emitDebug(`ZHIBAN_RESOURCES: ${env.ZHIBAN_RESOURCES || '(not set)'}`)
  emitDebug(`LLM_MODEL_PATH: ${env.LLM_MODEL_PATH || '(not set)'}`)
  emitDebug(`LLM_BASE_URL: ${env.LLM_BASE_URL || '(not set)'}`)
  emitDebug(`MODEL_CACHE: ${(env as any).MODEL_CACHE || '(not set, from start-sidecar.sh)'}`)

  // macOS Gatekeeper: 清除 sidecar 目录的隔离标记，否则捆绑 Python 被 SIGKILL
  try {
    const { execSync } = require('child_process')
    const sdDir = env.ZHIBAN_RESOURCES ? `${env.ZHIBAN_RESOURCES}/sidecar-dist` : ''
    if (sdDir) {
      execSync(`xattr -cr "${sdDir}" 2>/dev/null; xattr -cr "${sdDir}/python" 2>/dev/null; true`, { timeout: 3000 })
      emitDebug('Quarantine cleared for sidecar-dist')
    }
  } catch {}

  try {
    sidecarProcess = spawn(cmd, args, {
      cwd: getProjectDir(),
      env,
      stdio: ['pipe', 'pipe', 'pipe'],
    })

    emitDebug(`Sidecar PID: ${sidecarProcess.pid}`)
    sidecarStartTime = Date.now()
    restartCount = 0  // reset on successful spawn

    // 启动成功时清除上一次的 crash 状态
    try {
      const crashPath = path.join(getLogDir(), 'crash.json')
      if (fs.existsSync(crashPath)) fs.unlinkSync(crashPath)
    } catch {}

    sidecarProcess.stdout?.on('data', (data) => {
      const text = data.toString().trim()
      if (text) emitDebug(`[stdout] ${text}`)
    })

    sidecarProcess.stderr?.on('data', (data) => {
      const text = data.toString().trim()
      if (text) emitDebug(`[stderr] ${text}`)
    })

    sidecarProcess.on('close', (code) => {
      const reason = code === 0 ? 'normal_exit' : `exit_code_${code}`
      emitDebug(`Sidecar 进程退出, code=${code} (${reason})`)
      if (code !== 0 && code !== null) {
        persistCrashState(reason, code)
      }
      sidecarProcess = null
      _starting = false
    })

    sidecarProcess.on('error', (err) => {
      const msg = `无法启动知伴后端: ${err.message}`
      emitDebug(`ERROR: ${msg}`)
      persistCrashState(err.message, null)
      if (_onErrorCallback) _onErrorCallback(msg)
      sidecarProcess = null
      _starting = false
    })

    // Start health checker (will respect grace period)
    startHealthCheck()
    // 立即执行首次健康检查，不等 5 秒间隔
    setTimeout(() => {
      if (sidecarProcess || healthCheckTimer) {
        // 触发一次即时端口检测
        isSidecarRunning().then(alive => {
          if (alive && !_readyNotified) {
            _readyNotified = true
            const elapsed = Date.now() - sidecarStartTime
            emitDebug(`端口 18921 即时检测通过 (${Math.round(elapsed/1000)}s), IPC 通知渲染进程`)
            _onReadyCallback?.()
          }
        })
      }
    }, 1000)

    return true
  } catch (err) {
    const msg = `启动知伴后端失败: ${err instanceof Error ? err.message : String(err)}`
    emitDebug(`ERROR: ${msg}`)
    persistCrashState(msg, null)
    if (_onErrorCallback) _onErrorCallback(msg)
    _starting = false
    return false
  }
}

let _portAliveLogged = false

let _healthLogCount = 0
let _readyNotified = false
function startHealthCheck() {
  stopHealthCheck()
  _portAliveLogged = false
  _healthLogCount = 0
  _readyNotified = false
  healthCheckTimer = setInterval(async () => {
    _healthLogCount++
    const elapsed = Date.now() - sidecarStartTime

    // 快速检测：端口一开就通知渲染进程，不等宽限期
    if (!_readyNotified) {
      const portAlive = await isSidecarRunning()
      if (portAlive) {
        _readyNotified = true
        emitDebug(`端口 18921 已就绪 (${Math.round(elapsed/1000)}s), IPC 通知渲染进程`)
        _onReadyCallback?.()
      }
    }

    // During startup grace period, only check if process is still alive.
    // Don't penalize slow model loading.
    if (elapsed < STARTUP_GRACE_PERIOD_MS) {
      if (!sidecarProcess) {
        const portAlive = await isSidecarRunning()
        if (portAlive) {
          if (!_portAliveLogged) {
            emitDebug(`Health#${_healthLogCount}: 进程失联但端口存活 — 其他实例运行中`)
            _portAliveLogged = true
          }
          sidecarStartTime = 0
          return
        }
        emitDebug(`Health#${_healthLogCount}: 进程在宽限期内终止 (${Math.round(elapsed/1000)}s), 重启 #${restartCount}`)
        restartCount++
        if (restartCount <= MAX_RESTART_COUNT) {
          setTimeout(() => startSidecar(), RESTART_DELAY_MS)
        } else {
          emitDebug(`Health#${_healthLogCount}: 达到最大重启次数 (${MAX_RESTART_COUNT}), 放弃`)
          stopHealthCheck()
        }
      }
      return
    }

    const alive = await isSidecarRunning()

    if (alive) {
      if (restartCount > 0) {
        emitDebug(`Health#${_healthLogCount}: 端口恢复, 重置重启计数`)
        restartCount = 0
      }
      return
    }

    // Port is down but process reference still exists — likely still in startup/lifespan
    if (sidecarProcess) return

    // Process is dead and port is down — restart if within limits
    if (restartCount < MAX_RESTART_COUNT) {
      restartCount++
      emitDebug(`Health#${_healthLogCount}: 端口不通 (${restartCount}/${MAX_RESTART_COUNT}), ${RESTART_DELAY_MS}ms 后重启`)
      setTimeout(async () => {
        const success = await startSidecar()
        if (success) {
          emitDebug(`Health: 自动重启成功 (#${restartCount})`)
        }
      }, RESTART_DELAY_MS)
    } else {
      emitDebug(`Health#${_healthLogCount}: 达到最大重启次数 (${MAX_RESTART_COUNT}), 放弃`)
      stopHealthCheck()
    }
  }, HEALTH_CHECK_INTERVAL_MS)
}

function stopHealthCheck() {
  if (healthCheckTimer) {
    clearInterval(healthCheckTimer)
    healthCheckTimer = null
  }
}

export function resetHealthCheck() {
  restartCount = 0
}

export function stopSidecar() {
  stopHealthCheck()

  emitDebug('Sidecar 停止: 开始清理')

  if (sidecarProcess) {
    // 1. SIGTERM → Python 的 lifespan shutdown 会卸载模型
    sidecarProcess.kill('SIGTERM')
    sidecarProcess = null

    // 2. 异步等待 3s 后强制清理端口
    const forceCleanup = () => {
      exec('lsof -ti :18921 2>/dev/null | xargs kill -9 2>/dev/null; true', (err) => {
        emitDebug(`强制端口清理完成${err ? ' (或端口已空闲)' : ''}`)
      })
    }
    setTimeout(forceCleanup, 3000)
  } else {
    // 无进程引用，立即清理端口
    exec('lsof -ti :18921 2>/dev/null | xargs kill -9 2>/dev/null; true', (err) => {
      emitDebug(`端口清理完成${err ? ' (或端口已空闲)' : ''}`)
    })
  }

  // 关闭文件日志流
  if (_logStream) {
    try { _logStream.end() } catch {}
    _logStream = null
  }
}

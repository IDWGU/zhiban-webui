import { app, BrowserWindow, Tray, Menu, globalShortcut, screen, ipcMain, dialog } from 'electron'
import path from 'path'
import fs from 'fs'
import { startSidecar, stopSidecar, setSidecarErrorCallback, setSidecarReadyCallback, setDebugLogCallback, getDebugLogBuffer } from './sidecar'

// 防止 stdout/stderr 断开导致 EIO 崩溃（父进程退出时）
process.stdout.on('error', (err: any) => {
  if (err.code === 'EIO' || err.code === 'EPIPE') return
})
process.stderr.on('error', (err: any) => {
  if (err.code === 'EIO' || err.code === 'EPIPE') return
})

// 单实例锁：防止 Electron 被启动两次导致两个窗口
const gotTheLock = app.requestSingleInstanceLock()
if (!gotTheLock) {
  app.quit()
}

let mainWindow: BrowserWindow | null = null
let tray: Tray | null = null
let _sidecarReady = false

const isDev = !app.isPackaged

function createMainWindow() {
  // Close any leaked previous window before creating a new one
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.close()
  }

  const { width, height } = screen.getPrimaryDisplay().workAreaSize

  mainWindow = new BrowserWindow({
    width,
    height,
    minWidth: 900,
    minHeight: 600,
    frame: true,
    title: '知伴 ZhiBan',
    backgroundColor: '#14161c',
    resizable: true,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false
    }
  })

  if (isDev) {
    mainWindow.loadURL('http://localhost:5173')
  } else {
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'))
  }

  mainWindow.on('closed', () => {
    mainWindow = null
  })
}

function createTray() {
  try {
    const fs = require('fs')
    // Apple Silicon Mac: 使用 Template 图标 (系统自动适配深色/浅色菜单栏)
    const iconCandidates = [
      path.join(__dirname, '../build/tray-iconTemplate.png'),
      path.join(app.getAppPath(), 'build/tray-iconTemplate.png'),
    ]
    let iconPath = ''
    for (const p of iconCandidates) {
      if (fs.existsSync(p)) { iconPath = p; break }
    }
    if (!iconPath) return

    tray = new Tray(iconPath)
    const contextMenu = Menu.buildFromTemplate([
      { label: '显示/隐藏', click: toggleMainWindow },
      { type: 'separator' },
      { label: '退出知伴', click: () => { app.quit() } }
    ])
    tray.setContextMenu(contextMenu)
    tray.on('click', toggleMainWindow)
  } catch (e) {
    console.warn('Failed to create tray:', e)
  }
}

function toggleMainWindow() {
  if (!mainWindow) return
  if (mainWindow.isVisible()) {
    mainWindow.hide()
  } else {
    mainWindow.show()
    mainWindow.focus()
  }
}

function registerShortcuts() {
  globalShortcut.register('CmdOrCtrl+Shift+Z', toggleMainWindow)
  globalShortcut.register('CmdOrCtrl+Shift+X', () => {
    mainWindow?.webContents.send('shortcut:toggle-listening')
  })
  globalShortcut.register('CmdOrCtrl+Shift+D', () => {
    mainWindow?.webContents.send('shortcut:toggle-debug')
  })
}

// IPC handlers
ipcMain.handle('get-screen-size', () => {
  return screen.getPrimaryDisplay().workAreaSize
})

const ALLOWED_DOC_EXTS = new Set(['pdf', 'docx', 'txt', 'md'])

ipcMain.handle('read-file', async (_event, filePath: string) => {
  // Security: reject non-absolute paths and path traversal
  if (typeof filePath !== 'string' || !filePath.startsWith('/') || filePath.includes('..')) {
    throw new Error(`读取文件失败: 路径不合法`)
  }
  const ext = filePath.toLowerCase().split('.').pop() || ''
  if (!ALLOWED_DOC_EXTS.has(ext)) {
    throw new Error(`读取文件失败: 不支持的文件类型 .${ext}`)
  }
  try {
    const buffer = await fs.promises.readFile(filePath)
    return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength)
  } catch (err: any) {
    throw new Error(`读取文件失败: ${err.message}`)
  }
})

ipcMain.handle('select-directory', async (_event) => {
  const result = await dialog.showOpenDialog(mainWindow!, {
    properties: ['openDirectory'],
    message: '选择 ChromaDB 向量库目录',
  })
  if (result.canceled || result.filePaths.length === 0) return null
  return result.filePaths[0]
})

ipcMain.handle('get-debug-logs', () => {
  return getDebugLogBuffer()
})

ipcMain.handle('is-sidecar-ready', () => {
  return _sidecarReady
})

function setupMenu() {
  const { Menu } = require('electron')
  const sendToRenderer = (channel: string) =>
    mainWindow?.webContents.send(`menu-action:${channel}`)

  const template: any[] = [
    {
      label: '知伴',
      submenu: [
        { label: '关于知伴', role: 'about' as const },
        { type: 'separator' as const },
        { label: '设置...', accelerator: 'Cmd+,', click: () => sendToRenderer('toggle-settings') },
        { type: 'separator' as const },
        { label: '隐藏知伴', accelerator: 'Cmd+H', role: 'hide' as const },
        { label: '隐藏其他', accelerator: 'Cmd+Shift+H', role: 'hideOthers' as const },
        { type: 'separator' as const },
        { label: '退出知伴', accelerator: 'Cmd+Q', role: 'quit' as const },
      ]
    },
    {
      label: '文件',
      submenu: [
        { label: '打开论文...', accelerator: 'Cmd+O', click: () => {
          const { dialog } = require('electron')
          dialog.showOpenDialog(mainWindow!, {
            filters: [{ name: '文档', extensions: ['pdf', 'docx', 'txt', 'md'] }],
            properties: ['openFile', 'multiSelections'],
          }).then((result: any) => {
            if (!result.canceled && result.filePaths.length > 0) {
              mainWindow?.webContents.send('menu:open-files', result.filePaths)
            }
          })
        }},
        { type: 'separator' as const },
        { label: '关闭标签', accelerator: 'Cmd+W', click: () => {} },
        { label: '清空对话', accelerator: 'Cmd+K', click: () => sendToRenderer('clear-conv') },
      ]
    },
    {
      label: '编辑',
      submenu: [
        { label: '撤销', accelerator: 'Cmd+Z', role: 'undo' as const },
        { label: '重做', accelerator: 'Cmd+Shift+Z', role: 'redo' as const },
        { type: 'separator' as const },
        { label: '剪切', accelerator: 'Cmd+X', role: 'cut' as const },
        { label: '复制', accelerator: 'Cmd+C', role: 'copy' as const },
        { label: '粘贴', accelerator: 'Cmd+V', role: 'paste' as const },
      ]
    },
    {
      label: '视图',
      submenu: [
        { label: '重新加载', accelerator: 'CmdOrCtrl+R', role: 'reload' as const },
        { label: '强制重新加载', accelerator: 'CmdOrCtrl+Shift+R', role: 'forceReload' as const },
        { type: 'separator' as const },
        { label: '切换主题', accelerator: 'Cmd+Shift+T', click: () => sendToRenderer('toggle-theme') },
        { label: '开发者工具', accelerator: 'F12', role: 'toggleDevTools' as const },
        { type: 'separator' as const },
        { label: '放大', accelerator: 'Cmd+=', role: 'zoomIn' as const },
        { label: '缩小', accelerator: 'Cmd+-', role: 'zoomOut' as const },
        { label: '重置缩放', accelerator: 'Cmd+0', role: 'resetZoom' as const },
      ]
    },
    {
      label: '帮助',
      submenu: [
        { label: '检查更新', click: () => {} },
        { label: '反馈问题', click: () => {} },
      ]
    },
  ]
  const menu = Menu.buildFromTemplate(template)
  Menu.setApplicationMenu(menu)
}

app.on('second-instance', () => {
  // 已有实例运行时，聚焦到已有窗口
  if (mainWindow) {
    if (mainWindow.isMinimized()) mainWindow.restore()
    mainWindow.show()
    mainWindow.focus()
  }
})

app.whenReady().then(async () => {
  setupMenu()

  // Wire up sidecar error callback → forward to renderer for user-visible notification
  // mainWindow 在 createMainWindow() 中赋值，回调内部用可选链安全访问
  setSidecarErrorCallback((msg: string) => {
    mainWindow?.webContents.send('sidecar:error', msg)
  })

  // Wire up sidecar ready callback — 端口就绪时通知渲染进程，无需 HTTP poll
  setSidecarReadyCallback(() => {
    _sidecarReady = true
    mainWindow?.webContents.send('sidecar:ready')
  })

  // Wire up debug log callback → send to renderer for debug panel
  setDebugLogCallback((entry: string) => {
    mainWindow?.webContents.send('debug:log', entry)
  })

  // 先创建窗口，再启动后端 — 确保 mainWindow 已就绪后 sidecar 错误才能被推送到前端
  createMainWindow()
  createTray()
  registerShortcuts()

  app.on('activate', () => {
    if (!mainWindow) createMainWindow()
  })

  await startSidecar() // Always try to start sidecar for real backend
})

app.on('window-all-closed', () => {
  // macOS: 不退出，保留托盘图标
})

app.on('before-quit', () => {
  stopSidecar()
})

app.on('will-quit', () => {
  globalShortcut.unregisterAll()
  stopSidecar()  // 双保险：确保进程一定被杀
})

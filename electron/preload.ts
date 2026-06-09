import { contextBridge, ipcRenderer, webUtils } from 'electron'

contextBridge.exposeInMainWorld('electronAPI', {
  getScreenSize: () => ipcRenderer.invoke('get-screen-size'),
  onShortcut: (channel: string, callback: () => void) => {
    ipcRenderer.on(channel, callback)
    return () => ipcRenderer.removeListener(channel, callback)
  },
  readFile: (filePath: string) => ipcRenderer.invoke('read-file', filePath),
  selectDirectory: () => ipcRenderer.invoke('select-directory'),
  getPathForFile: (file: File) => webUtils.getPathForFile(file),
  onMenuOpenFile: (callback: (filePaths: string[]) => void) => {
    const handler = (_event: any, filePaths: string[]) => callback(filePaths)
    ipcRenderer.on('menu:open-files', handler)
    return () => ipcRenderer.removeListener('menu:open-files', handler)
  },
  /** Listen for menu-triggered actions from main process */
  onMenuAction: (action: string, callback: () => void) => {
    const handler = () => callback()
    ipcRenderer.on(`menu-action:${action}`, handler)
    return () => ipcRenderer.removeListener(`menu-action:${action}`, handler)
  },
  /** Listen for sidecar startup errors from main process */
  onSidecarError: (callback: (message: string) => void) => {
    const handler = (_event: any, message: string) => callback(message)
    ipcRenderer.on('sidecar:error', handler)
    return () => ipcRenderer.removeListener('sidecar:error', handler)
  },
  /** Listen for sidecar ready notification from main process (port open, no HTTP poll needed) */
  onSidecarReady: (callback: () => void) => {
    const handler = () => callback()
    ipcRenderer.on('sidecar:ready', handler)
    // 注册时立即检查：如果 sidecar 在渲染进程挂载前已就绪，IPC 事件已丢失，
    // 通过 invoke 检查标志位，已就绪则直接回调
    ipcRenderer.invoke('is-sidecar-ready').then((alreadyReady: boolean) => {
      if (alreadyReady) callback()
    })
    return () => ipcRenderer.removeListener('sidecar:ready', handler)
  },
  /** Listen for debug logs from main process */
  onDebugLog: (callback: (entry: string) => void) => {
    const handler = (_event: any, entry: string) => callback(entry)
    ipcRenderer.on('debug:log', handler)
    return () => ipcRenderer.removeListener('debug:log', handler)
  },
  /** Pull buffered debug logs from main process (for logs emitted before renderer was ready) */
  getDebugLogs: () => ipcRenderer.invoke('get-debug-logs')
})

// ===== 全局类型声明 =====

declare global {
  interface Window {
    electronAPI?: {
      getScreenSize: () => Promise<{ width: number; height: number }>
      onShortcut: (channel: string, callback: () => void) => () => void
      readFile: (filePath: string) => Promise<ArrayBuffer>
      getPathForFile: (file: File) => string
      onMenuOpenFile: (callback: (filePaths: string[]) => void) => () => void
      onMenuAction: (action: string, callback: () => void) => () => void
      selectDirectory: () => Promise<string | null>
      onSidecarError: (callback: (message: string) => void) => () => void
    }
  }
}

export {}

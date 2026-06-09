import type { ConnectionState } from '@/types'

export interface ConnectionSlice {
  connection: ConnectionState
  setWsStatus: (status: ConnectionState['wsStatus']) => void
  setReconnectAttempt: (n: number) => void
  setSidecarStatus: (status: ConnectionState['sidecarStatus']) => void
  setDebugMode: (enabled: boolean) => void
  setLocalEngineLoading: (loading: boolean) => void
  setStartupMessage: (msg: string) => void
  showDevPanel: boolean
  setShowDevPanel: (show: boolean) => void
}

export function createConnectionSlice(set: any, _get: any): ConnectionSlice {
  return {
    connection: {
      wsStatus: 'waiting',
      sidecarStatus: 'stopped',
      reconnectAttempt: 0,
      debugMode: false,
      localEngineLoading: false,
      startupMessage: '',
      showDevPanel: false,
    },

    setWsStatus: (status) => set((state: any) => ({
      connection: { ...state.connection, wsStatus: status }
    })),

    setReconnectAttempt: (n) => set((state: any) => ({
      connection: { ...state.connection, reconnectAttempt: n }
    })),

    setSidecarStatus: (status) => set((state: any) => ({
      connection: { ...state.connection, sidecarStatus: status }
    })),

    setDebugMode: (enabled) => set((state: any) => ({
      connection: { ...state.connection, debugMode: enabled }
    })),

    setLocalEngineLoading: (loading) => {
      set((state: any) => ({
        connection: { ...state.connection, localEngineLoading: loading }
      }))
      if (loading) {
        // 60s 超时兜底：防止后端崩溃导致加载状态永久卡住
        setTimeout(() => {
          set((state: any) => {
            if (state.connection.localEngineLoading) {
              return { connection: { ...state.connection, localEngineLoading: false } }
            }
            return {}
          })
        }, 60_000)
      }
    },

    setStartupMessage: (msg) => set((state: any) => ({
      connection: { ...state.connection, startupMessage: msg }
    })),

    showDevPanel: false,
    setShowDevPanel: (show) => set((state: any) => ({
      connection: { ...state.connection, showDevPanel: show }
    })),
  }
}

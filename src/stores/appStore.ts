import { create } from 'zustand'
import type { PaperTab } from '@/types'

import { createSettingsSlice, saveSettings, type SettingsSlice } from './slices/settingsSlice'
import { createScreenContextSlice, type ScreenContextSlice } from './slices/screenContextSlice'
import { createConnectionSlice, type ConnectionSlice } from './slices/connectionSlice'

import { createConversationSlice, type ConversationSlice } from './slices/conversationSlice'
import { createTranslationSlice, type TranslationSlice } from './slices/translationSlice'
import { createPaperSlice, type PaperSlice } from './slices/paperSlice'

export type { AppNotification } from './slices/paperSlice'

export interface AppStore extends
  SettingsSlice,
  ScreenContextSlice,
  ConnectionSlice,
  ConversationSlice,
  TranslationSlice,
  PaperSlice {}

export const useAppStore = create<AppStore>((set, get) => ({
  ...createSettingsSlice(set, get),
  ...createScreenContextSlice(set, get),
  ...createConnectionSlice(set, get),
  ...createConversationSlice(set, get),
  ...createTranslationSlice(set, get),
  ...createPaperSlice(set, get),
}))

// Auto-persist settings — 仅在变化时写入，避免流式输出期间每帧无意义 disk IO
let _lastSettingsJson = ''
useAppStore.subscribe((state) => {
  const json = JSON.stringify(state.settings)
  if (json === _lastSettingsJson) return
  _lastSettingsJson = json
  saveSettings(state.settings)
})

// Persist active conversation ID across sessions
let _lastActiveConvId = useAppStore.getState().activeConversationId
useAppStore.subscribe((state) => {
  if (state.activeConversationId !== _lastActiveConvId) {
    _lastActiveConvId = state.activeConversationId
    localStorage.setItem('zhiban-active-conv-id', state.activeConversationId)
  }
})

// Persist papers, notes, and recent conversation messages — debounced
// to avoid writing to disk on every streaming token (hundreds/sec).
let _persistTimer: ReturnType<typeof setTimeout> | null = null
let _persistDirty = false
function _flushPersist(state: any) {
  try {
    localStorage.setItem('zhiban-papers', JSON.stringify(state.papers.map((p: PaperTab) => ({
      id: p.id, name: p.name, type: p.type, path: p.path, extractedText: p.extractedText,
    }))))
    localStorage.setItem('zhiban-notes', state.notesContent)
    localStorage.setItem('zhiban-conv', JSON.stringify(state.conversation.messages))
  } catch {}
  _persistDirty = false
}
useAppStore.subscribe((state) => {
  if (state.conversation.isStreaming) {
    // During streaming, mark dirty and defer — flush on finish or after 2s max
    _persistDirty = true
    if (!_persistTimer) {
      _persistTimer = setTimeout(() => {
        _persistTimer = null
        if (_persistDirty) _flushPersist(useAppStore.getState())
      }, 2000)
    }
  } else {
    if (_persistTimer) { clearTimeout(_persistTimer); _persistTimer = null }
    _flushPersist(state)
  }
})

import { useState, useEffect } from 'react'
import GeneralSection from './GeneralSection'
import LLMSection from './LLMSection'
import TranslationSection from './TranslationSection'
import DatabaseSection from './DatabaseSection'
import ModelsSection from './ModelsSection'
import AdvancedSection from './AdvancedSection'

type Section = 'general' | 'llm' | 'translation' | 'models' | 'database' | 'advanced'

interface Props { open: boolean; onClose: () => void }

export default function SettingsPanel({ open, onClose }: Props) {
  const [section, setSection] = useState<Section>('general')

  useEffect(() => {
    if (!open) return
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [open, onClose])

  if (!open) return null

  return (
    <div
      onClick={(e) => { if (e.target === e.currentTarget) onClose() }}
      style={{
        position: 'fixed', inset: 0, zIndex: 2000,
        background: 'hsl(var(--background) / 0.60)',
        backdropFilter: 'blur(4px)',
        WebkitBackdropFilter: 'blur(4px)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}
    >
      <div className="proma-panel-elevated" style={{
        width: 600, height: 500,
        display: 'flex', overflow: 'hidden',
        animation: 'fadeInZoom 0.2s ease-out',
      }}>
        {/* Left nav */}
        <div style={{
          width: 150, flexShrink: 0,
          borderRight: '1px solid hsl(var(--border))',
          display: 'flex', flexDirection: 'column',
          background: 'hsl(var(--muted) / 0.30)',
        }}>
          <div style={{
            fontSize: 13, fontWeight: 600,
            color: 'hsl(var(--foreground))',
            padding: '16px', letterSpacing: 0.5,
          }}>
            设置
          </div>
          <NavItem active={section === 'general'} onClick={() => setSection('general')} label="通用" />
          <NavItem active={section === 'llm'} onClick={() => setSection('llm')} label="伴读 LLM" />
          <NavItem active={section === 'translation'} onClick={() => setSection('translation')} label="翻译" />
          <NavItem active={section === 'models'} onClick={() => setSection('models')} label="向量模型" />
          <NavItem active={section === 'database'} onClick={() => setSection('database')} label="知识库" />
          <NavItem active={section === 'advanced'} onClick={() => setSection('advanced')} label="高级" />
          <div style={{ flex: 1 }} />
          <div style={{
            fontSize: 10, color: 'hsl(var(--foreground) / 0.30)',
            padding: '12px 16px', textAlign: 'center',
          }}>
            知伴 v0.1.0
          </div>
        </div>

        {/* Right content */}
        <div style={{
          flex: 1, padding: '24px', overflowY: 'auto',
          background: 'hsl(var(--dialog))',
        }}>
          {section === 'general' && <GeneralSection />}
          {section === 'llm' && <LLMSection />}
          {section === 'translation' && <TranslationSection />}
          {section === 'models' && <ModelsSection />}
          {section === 'database' && <DatabaseSection />}
          {section === 'advanced' && <AdvancedSection />}
        </div>
      </div>
    </div>
  )
}

function NavItem({ active, onClick, label }: { active: boolean; onClick: () => void; icon?: string; label: string }) {
  return (
    <div onClick={onClick} style={{
      padding: '9px 16px', cursor: 'pointer', fontSize: 13,
      background: active ? 'hsl(var(--accent))' : 'transparent',
      color: active ? 'hsl(var(--foreground))' : 'hsl(var(--muted-foreground))',
      borderLeft: active ? '2px solid hsl(var(--primary))' : '2px solid transparent',
      transition: 'background 0.1s ease, color 0.1s ease, border-color 0.1s ease',
      fontWeight: active ? 500 : 400,
    }}>
      {label}
    </div>
  )
}

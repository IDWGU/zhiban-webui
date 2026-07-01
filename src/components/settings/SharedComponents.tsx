import type { ReactNode } from 'react'

export function SectionBox({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div>
      <div style={{
        fontSize: 14, fontWeight: 600,
        color: 'hsl(var(--foreground))',
        marginBottom: 16,
      }}>
        {title}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        {children}
      </div>
    </div>
  )
}

export function SettingRow({ label, desc, children }: { label: ReactNode; desc?: string; children: ReactNode }) {
  return (
    <div>
      <div style={{
        fontSize: 13, fontWeight: 500,
        color: 'hsl(var(--foreground))',
        marginBottom: 2,
      }}>
        {label}
      </div>
      {desc && (
        <div style={{
          fontSize: 11,
          color: 'hsl(var(--muted-foreground))',
          marginBottom: 6,
        }}>
          {desc}
        </div>
      )}
      <div>{children}</div>
    </div>
  )
}

export function ToggleCheckbox({ checked, onChange, disabled }: { checked: boolean; onChange: (v: boolean) => void; disabled?: boolean }) {
  return (
    <div onClick={() => { if (!disabled) onChange(!checked) }} style={{
      width: 36, height: 20, borderRadius: 10, cursor: disabled ? 'not-allowed' : 'pointer',
      background: checked ? 'hsl(var(--primary))' : 'hsl(var(--border))',
      position: 'relative',
      opacity: disabled ? 0.5 : 1,
      transition: 'background 0.2s ease',
    }}>
      <div style={{
        position: 'absolute', top: 2,
        left: checked ? 18 : 2,
        width: 16, height: 16,
        borderRadius: '50%',
        background: '#fff',
        transition: 'left 0.2s ease',
        boxShadow: '0 1px 3px rgba(0,0,0,0.2)',
      }} />
    </div>
  )
}

// Proma-style shared input/button styles
export const inputStyle: React.CSSProperties = {
  width: '100%', padding: '7px 10px', borderRadius: 8,
  border: '1px solid hsl(var(--border))',
  background: 'hsl(var(--foreground) / 0.05)',
  color: 'hsl(var(--foreground))',
  fontSize: 12, fontFamily: 'inherit', outline: 'none',
  transition: 'border-color 0.15s ease',
}

export const toggleBtnStyle: React.CSSProperties = {
  padding: '6px 14px', borderRadius: 8,
  border: '1px solid hsl(var(--border))',
  background: 'hsl(var(--foreground) / 0.05)',
  color: 'hsl(var(--muted-foreground))',
  fontSize: 12, cursor: 'pointer', fontFamily: 'inherit',
  transition: 'background 0.1s ease',
}

export const actionBtnStyle: React.CSSProperties = {
  padding: '8px 16px', borderRadius: 8,
  border: '1px solid hsl(var(--border))',
  background: 'hsl(var(--foreground) / 0.05)',
  color: 'hsl(var(--foreground))',
  fontSize: 12, cursor: 'pointer', fontFamily: 'inherit',
  transition: 'background 0.1s ease',
}

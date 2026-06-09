import { Highlight, themes } from 'prism-react-renderer'

const CODE_FONT = '"JetBrains Mono", "Fira Code", "Cascadia Code", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace'

const darkTheme = {
  ...themes.vsDark,
  plain: { color: '#d4d4d4', backgroundColor: '#1a1a2e' },
}

interface Props {
  code: string
  language?: string
  maxLines?: number
}

export default function CodeBlock({ code, language = 'plaintext', maxLines }: Props) {
  const lines = code.split('\n')
  const truncated = maxLines && lines.length > maxLines
  const visibleLines = truncated ? lines.slice(0, maxLines) : lines
  const visibleCode = visibleLines.join('\n')

  return (
    <div style={{
      borderRadius: 8,
      overflow: 'hidden',
      border: '1px solid hsl(var(--border))',
      margin: '8px 0',
      fontSize: 12,
    }}>
      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '4px 12px',
        background: 'hsl(var(--muted) / 0.50)',
        borderBottom: '1px solid hsl(var(--border) / 0.60)',
      }}>
        <span style={{
          fontSize: 10, fontWeight: 600,
          color: 'hsl(var(--muted-foreground))',
          textTransform: 'uppercase', letterSpacing: '0.05em',
        }}>
          {language === 'plaintext' ? 'text' : language}
        </span>
        {truncated && (
          <span style={{
            fontSize: 10,
            color: 'hsl(var(--muted-foreground))',
          }}>
            {lines.length - (maxLines || 0)} more lines
          </span>
        )}
      </div>

      {/* Code */}
      <div style={{
        maxHeight: maxLines ? `${maxLines * 20 + 16}px` : undefined,
        overflow: 'auto',
        background: '#1a1a2e',
      }}>
        <Highlight theme={darkTheme} code={visibleCode} language={language}>
          {({ tokens, getLineProps, getTokenProps }) => (
            <pre style={{
              margin: 0,
              padding: '10px 12px',
              fontFamily: CODE_FONT,
              fontSize: 12,
              lineHeight: 1.55,
              overflowX: 'auto',
              whiteSpace: 'pre',
            }}>
              {tokens.map((line, i) => (
                <div key={i} {...getLineProps({ line })} style={{ display: 'table-row' }}>
                  <span style={{
                    display: 'table-cell',
                    textAlign: 'right',
                    paddingRight: 12,
                    userSelect: 'none',
                    color: 'hsl(var(--muted-foreground) / 0.40)',
                    fontSize: 10,
                    minWidth: 28,
                    fontFamily: CODE_FONT,
                  }}>
                    {i + 1}
                  </span>
                  <span style={{ display: 'table-cell' }}>
                    {line.map((token, key) => (
                      <span key={key} {...getTokenProps({ token })} />
                    ))}
                  </span>
                </div>
              ))}
            </pre>
          )}
        </Highlight>
      </div>
    </div>
  )
}

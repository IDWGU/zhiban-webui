import { useRef, useEffect, useState, useCallback, memo } from 'react'
import { useAppStore } from '@/stores/appStore'
import TranslationSplitView from '@/components/translation/TranslationSplitView'
import TranslationToolbar from '@/components/translation/TranslationToolbar'
import PdfPageViewer from './PdfPageViewer'
import DocxViewer from './DocxViewer'
import TextParagraphs from './TextParagraphs'
import { getCachedDoc, cacheDoc } from './pdfCache'
import { loadPdf, loadDocx, readFileBuffer } from './loaders'

interface PdfMeta { filePath: string; numPages: number }

interface Props {
  paperContent?: string
  paperTitle?: string
  paperId?: string
  filePath?: string
  onTextExtracted?: (text: string) => void
}

function categorizeError(err: any, fileType: string): { title: string; detail: string; hint: string } {
  const msg = (err?.message || String(err || '')).toLowerCase()
  if (msg.includes('enoent') || msg.includes('not found') || msg.includes('不存在'))
    return { title: '文件未找到', detail: err?.message || '', hint: '文件可能已被移动或删除，请尝试重新拖入' }
  if (msg.includes('eacces') || msg.includes('permission') || msg.includes('权限'))
    return { title: '没有读取权限', detail: err?.message || '', hint: '请在系统设置中授予知伴文件访问权限' }
  if (msg.includes('invalid') || msg.includes('corrupt') || msg.includes('damaged') || msg.includes('格式'))
    return { title: `${fileType.toUpperCase()} 文件异常`, detail: err?.message || '', hint: '文件可能已损坏或格式不完整' }
  if (msg.includes('不支持'))
    return { title: '不支持的文件格式', detail: err?.message || '', hint: '仅支持 PDF / DOCX / TXT / MD 格式' }
  if (msg.includes('路径不合法'))
    return { title: '文件路径无效', detail: err?.message || '', hint: '请通过拖拽或菜单打开文件' }
  return { title: '文件加载失败', detail: err?.message || '', hint: '请确认文件完整且未损坏' }
}

const PaperViewer = memo(function PaperViewer({ paperContent, paperTitle, paperId, filePath, onTextExtracted }: Props) {
  const [fileType, setFileType] = useState<'text' | 'pdf' | 'docx' | null>(null)
  const [extractedText, setExtractedText] = useState('')
  const [pdfTextParts, setPdfTextParts] = useState<string[]>([])
  const [pdfMeta, setPdfMeta] = useState<PdfMeta | null>(null)
  const [docxHtml, setDocxHtml] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [translationMode, setTranslationMode] = useState(false)
  const [selectionPopover, setSelectionPopover] = useState<{ text: string; x: number; y: number } | null>(null)
  const translationPhase = useAppStore(s => s.translation.phase)

  const handleContentMouseUp = useCallback(() => {
    setTimeout(() => {
      const sel = window.getSelection()
      if (!sel || sel.isCollapsed || !sel.toString().trim()) {
        setSelectionPopover(null)
        return
      }
      const text = sel.toString().trim()
      if (text.length < 5) { setSelectionPopover(null); return }
      const range = sel.getRangeAt(0)
      const rect = range.getBoundingClientRect()
      setSelectionPopover({
        text,
        x: rect.left + rect.width / 2,
        y: rect.top - 8,
      })
    }, 50)
  }, [])

  const handleAskAbout = useCallback((text: string) => {
    const input = document.querySelector('.query-textarea') as HTMLTextAreaElement
    if (input) {
      const prompt = `关于这段话："${text.slice(0, 200)}"\n\n请解释一下这段话的含义。`
      const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
        window.HTMLTextAreaElement.prototype, 'value'
      )?.set
      nativeInputValueSetter?.call(input, prompt)
      input.dispatchEvent(new Event('input', { bubbles: true }))
      input.focus()
    }
    setSelectionPopover(null)
  }, [])

  // Detect file type
  useEffect(() => {
    if (filePath) {
      const ext = filePath.split('.').pop()?.toLowerCase()
      if (ext === 'pdf') setFileType('pdf')
      else if (ext === 'docx') setFileType('docx')
      else setFileType('text')
    } else if (paperContent) {
      setFileType('text')
    }
  }, [filePath, paperContent])

  // Load PDF
  useEffect(() => {
    if (fileType !== 'pdf' || !filePath) return
    if (paperContent) setExtractedText(paperContent)
    const cached = getCachedDoc(filePath)
    if (cached) {
      setPdfMeta({ filePath, numPages: cached.numPages })
      setExtractedText(cached.fullText)
      setPdfTextParts(cached.textParts)
      onTextExtracted?.(cached.fullText)
      return
    }
    setLoading(true)
    setError(null)
    loadPdf(filePath).then(({ doc, fullText, numPages, textParts }) => {
      cacheDoc(filePath, { doc, fullText, numPages, textParts })
      setPdfMeta({ filePath, numPages })
      setPdfTextParts(textParts)
      setExtractedText(fullText)
      onTextExtracted?.(fullText)
      setLoading(false)
    }).catch(err => {
      console.error('PDF load error:', err)
      setError(err?.message || String(err))
      setLoading(false)
    })
  }, [fileType, filePath])

  // Load DOCX
  useEffect(() => {
    if (fileType !== 'docx' || !filePath) return
    if (paperContent) setExtractedText(paperContent)
    setLoading(true)
    loadDocx(filePath).then(({ html, fullText }) => {
      setDocxHtml(html)
      setExtractedText(fullText)
      onTextExtracted?.(fullText)
      setLoading(false)
    }).catch(err => {
      console.error('DOCX load error:', err)
      setError(`DOCX load failed: ${err instanceof Error ? err.message : String(err)}`)
      setLoading(false)
    })
  }, [fileType, filePath])

  // Load plain text
  useEffect(() => {
    if (fileType !== 'text' || !filePath || paperContent) return
    setLoading(true)
    readFileBuffer(filePath).then(buf => {
      const decoder = new TextDecoder('utf-8')
      const text = decoder.decode(buf)
      setExtractedText(text)
      onTextExtracted?.(text)
      setLoading(false)
    }).catch(err => {
      console.error('Text file load error:', err)
      setError(`File load failed: ${err instanceof Error ? err.message : String(err)}`)
      setLoading(false)
    })
  }, [fileType, filePath, paperContent])

  const displayText = paperContent || extractedText

  // Build paragraphs for screen context + send full text to L2 cache
  useEffect(() => {
    const title = paperTitle || filePath?.split('/').pop() || 'Paper'
    if (pdfTextParts.length === 0 && !displayText) {
      useAppStore.getState().updateOcrResult([], null, title, 'document')
      return
    }
    const parts = pdfTextParts.length > 0 ? pdfTextParts : [displayText]
    const paragraphs = parts.map((text, i) => ({
      index: i,
      text: text.slice(0, 3000),
      bbox: { x: 0, y: i * 40, w: 1000, h: 40 },
      confidence: 1.0,
    }))
    useAppStore.getState().updateOcrResult(paragraphs, 0, title, 'document')

    // Send full paper text to backend L2 context (for 总结全文 etc.)
    const MAX_CHARS = 50000
    const fullText = parts.join('\n\n')
    const truncated = fullText.length > MAX_CHARS
    const l2Text = truncated
      ? fullText.slice(0, MAX_CHARS) + `\n\n[全文共 ${fullText.length} 字，此处截取前 ${MAX_CHARS} 字。如需后续章节，请指明页码范围。]`
      : fullText
    const convId = useAppStore.getState().activeConversationId
    const sendMsg = useAppStore.getState().sendMessage
    if (fullText && convId && sendMsg) {
      sendMsg({
        type: 'set_paper_context',
        conversationId: convId,
        text: l2Text,
        title: title,
      })
    }
  }, [pdfTextParts, displayText, paperTitle, filePath])

  // Cleanup on unmount
  useEffect(() => {
    return () => { useAppStore.getState().clearScreenContext() }
  }, [!!displayText])

  const paperViewerRef = useRef<HTMLDivElement>(null)

  return (
    <div ref={paperViewerRef} style={{ height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* Header */}
      <div style={headerStyle}>
        {paperId && !paperId.startsWith('paper-') && (
          <span style={{ fontSize: 11, color: 'var(--text-muted)', marginRight: 8 }}>
            #{paperId}
          </span>
        )}
        <span style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)', flex: 1 }}>
          {paperTitle || filePath?.split('/').pop() || 'No paper loaded'}
        </span>
        {loading && <span style={{ fontSize: 11, color: 'var(--text-accent)' }}>Loading...</span>}
        {fileType === 'pdf' && pdfMeta && (
          <button
            onClick={() => {
              if (translationMode) {
                const _s = useAppStore.getState()
                if (_s.translation.isTranslating) {
                  const _send = _s.sendMessage || (window as any).__zhiban_wsSend
                  if (_send) _send({ type: 'cancel_translation' })
                  _s.cancelTranslation()
                }
              }
              setTranslationMode(v => !v)
            }}
            style={{
              fontSize: 11, padding: '4px 10px', borderRadius: 4,
              border: translationMode ? '1px solid var(--accent)' : '1px solid var(--border)',
              background: translationMode ? 'var(--accent-bg)' : 'var(--btn-bg)',
              color: translationMode ? 'var(--text-accent)' : 'var(--text-muted)',
              cursor: 'pointer', fontFamily: 'inherit',
            }}
          >
            {translationMode ? 'Exit Translation' : 'Translate'}
          </button>
        )}
      </div>

      {/* Translation mode */}
      {translationMode && filePath && (
        <>
          <TranslationToolbar
            filePath={filePath}
            onCancel={() => setTranslationMode(false)}
          />
          <div style={{ flex: 1, overflow: 'hidden' }}>
            <TranslationSplitView filePath={filePath} paperTitle={paperTitle} paperId={paperId} />
          </div>
        </>
      )}

      {/* Normal content area */}
      {!translationMode && (
        <div style={{ flex: 1, overflow: 'auto', position: 'relative' }} onMouseUp={handleContentMouseUp}>
          {loading && !pdfMeta && !docxHtml && <div style={emptyStyle}>Loading...</div>}

          {error && (() => {
            const cat = categorizeError(error, fileType || '')
            return (
              <div style={{ padding: 24, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
                <div style={{ fontSize: 40, opacity: 0.3, marginBottom: 16 }}>⚠️</div>
                <div style={{ color: '#e0556a', fontSize: 15, fontWeight: 600, marginBottom: 8 }}>{cat.title}</div>
                <div style={{ color: 'var(--text-muted)', fontSize: 12, textAlign: 'center', maxWidth: 320, lineHeight: 1.6 }}>
                  {cat.hint}
                </div>
                <div style={{ marginTop: 12, fontSize: 10, color: 'var(--text-muted)', opacity: 0.5, fontFamily: 'monospace', maxWidth: 320, wordBreak: 'break-all' }}>
                  {cat.detail}
                </div>
              </div>
            )
          })()}

          {!loading && !error && !displayText && !pdfMeta && !docxHtml && (
            <div style={emptyStyle}>
              <div style={{ fontSize: 40, opacity: 0.2, marginBottom: 16 }}>📄</div>
              <div style={{ fontSize: 14, color: 'var(--text-muted)', marginBottom: 8 }}>No paper loaded</div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', opacity: 0.7 }}>
                Drag PDF/DOCX/TXT onto window or add via Settings → Database
              </div>
            </div>
          )}

          {pdfMeta && (
            <PdfPageViewer
              filePath={pdfMeta.filePath}
              numPages={pdfMeta.numPages}
              fullText={extractedText}
              textParts={pdfTextParts}
              paperTitle={paperTitle}
              paperId={paperId}
            />
          )}

          {docxHtml && <DocxViewer html={docxHtml} fullText={extractedText} paperTitle={paperTitle} paperId={paperId} />}

          {displayText && fileType !== 'pdf' && fileType !== 'docx' && (
            <TextParagraphs text={displayText} paperTitle={paperTitle} paperId={paperId} />
          )}
        </div>
      )}

      {/* Floating "Ask ZhiBan" popover — shown when text is selected in paper */}
      {selectionPopover && (
        <div style={{
          position: 'fixed',
          left: selectionPopover.x,
          top: selectionPopover.y,
          transform: 'translate(-50%, -100%)',
          zIndex: 5000,
          animation: 'fadeIn 0.15s ease-out',
        }}>
          <div style={{
            display: 'flex', alignItems: 'center', gap: 6,
            padding: '6px 12px',
            background: 'hsl(var(--popover))',
            border: '1px solid hsl(var(--border))',
            borderRadius: 10,
            boxShadow: '0 4px 20px rgba(0,0,0,0.3)',
            backdropFilter: 'blur(12px)',
            WebkitBackdropFilter: 'blur(12px)',
          }}>
            <button
              onClick={() => handleAskAbout(selectionPopover.text)}
              style={{
                padding: '4px 12px', borderRadius: 6,
                border: 'none',
                background: 'hsl(var(--primary))',
                color: 'hsl(var(--primary-foreground))',
                fontSize: 12, fontWeight: 600,
                cursor: 'pointer', fontFamily: 'inherit',
                whiteSpace: 'nowrap',
              }}
            >
              问知伴
            </button>
            <button
              onClick={() => setSelectionPopover(null)}
              style={{
                padding: '4px 8px', borderRadius: 6,
                border: 'none', background: 'transparent',
                color: 'hsl(var(--muted-foreground))',
                fontSize: 12, cursor: 'pointer',
              }}
            >
              ✕
            </button>
          </div>
          {/* Arrow */}
          <div style={{
            width: 0, height: 0,
            borderLeft: '6px solid transparent',
            borderRight: '6px solid transparent',
            borderTop: '6px solid hsl(var(--popover))',
            margin: '0 auto',
          }} />
        </div>
      )}
    </div>
  )
})

export default PaperViewer

const headerStyle: React.CSSProperties = {
  flexShrink: 0, padding: '14px 20px', borderBottom: '1px solid var(--border)',
  display: 'flex', alignItems: 'center', gap: 8,
}
const emptyStyle: React.CSSProperties = {
  height: '100%', display: 'flex', flexDirection: 'column',
  alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)',
}

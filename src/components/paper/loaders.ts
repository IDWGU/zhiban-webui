import { getPdfjsLib } from './pdfCache'

export async function loadPdf(filePath: string): Promise<{ doc: any; fullText: string; numPages: number; textParts: string[] }> {
  const pdfjsLib = await getPdfjsLib()
  const arrayBuffer = await readFileBuffer(filePath)
  const doc = await pdfjsLib.getDocument({ data: arrayBuffer }).promise

  const textParts: string[] = []
  for (let i = 1; i <= doc.numPages; i++) {
    const page = await doc.getPage(i)
    const textContent = await page.getTextContent()
    textParts.push(textContent.items.map((item: any) => item.str).join(' '))
  }

  return { doc, fullText: textParts.join('\n\n'), numPages: doc.numPages, textParts }
}

export async function loadDocx(filePath: string): Promise<{ html: string; fullText: string }> {
  const mammoth = await import('mammoth')
  const arrayBuffer = await readFileBuffer(filePath)
  const result = await mammoth.convertToHtml({ arrayBuffer })
  return { html: result.value, fullText: result.value.replace(/<[^>]+>/g, '') }
}

export async function readFileBuffer(filePath: string): Promise<ArrayBuffer> {
  if (window.electronAPI?.readFile) {
    try {
      return await window.electronAPI.readFile(filePath) as ArrayBuffer
    } catch (err) {
      console.error('IPC readFile failed:', err)
    }
  }
  // WebUI 模式：通过后端 HTTP 端点获取文件内容
  const encoded = encodeURIComponent(filePath)
  const response = await fetch(`/file-content?path=${encoded}`)
  if (!response.ok) throw new Error(`文件加载失败: HTTP ${response.status}`)
  return response.arrayBuffer()
}

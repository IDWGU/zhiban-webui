let _pdfjsLib: any = null
let _workerReady = false

async function ensureWorker(pdfjsLib: any) {
  if (_workerReady) return
  try {
    const workerUrl = (await import('pdfjs-dist/build/pdf.worker.mjs?url')).default
    pdfjsLib.GlobalWorkerOptions.workerSrc = workerUrl
  } catch {
    // Vite dev 模式下可能加载失败，用 CDN 兜底
    const version = pdfjsLib.version || '4.10.38'
    pdfjsLib.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${version}/pdf.worker.min.mjs`
  }
  _workerReady = true
}

export async function getPdfjsLib() {
  if (!_pdfjsLib) {
    _pdfjsLib = await import('pdfjs-dist/build/pdf.mjs')
    await ensureWorker(_pdfjsLib)
  }
  return _pdfjsLib
}

const MAX_DOC_CACHE = 5
const _docCache = new Map<string, { doc: any; fullText: string; numPages: number; textParts: string[] }>()
const _docCacheOrder: string[] = []

export function getCachedPdfDoc(filePath: string) {
  return _docCache.get(filePath) ?? null
}

export function getCachedDoc(filePath: string) {
  const idx = _docCacheOrder.indexOf(filePath)
  if (idx >= 0) {
    _docCacheOrder.splice(idx, 1)
    _docCacheOrder.push(filePath)
  }
  return _docCache.get(filePath) ?? null
}

export function cacheDoc(filePath: string, entry: { doc: any; fullText: string; numPages: number; textParts: string[] }) {
  if (_docCache.size >= MAX_DOC_CACHE) {
    const oldest = _docCacheOrder.shift()
    if (oldest) {
      _docCache.get(oldest)?.doc.destroy?.()
      _docCache.delete(oldest)
    }
  }
  const idx = _docCacheOrder.indexOf(filePath)
  if (idx >= 0) _docCacheOrder.splice(idx, 1)
  _docCache.set(filePath, entry)
  _docCacheOrder.push(filePath)
}

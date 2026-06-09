declare module 'pdfjs-dist' {
  export const GlobalWorkerOptions: { workerSrc: string }
  export function getDocument(config: { data: ArrayBuffer }): {
    promise: Promise<{
      numPages: number
      getPage(num: number): Promise<{
        getViewport(opts: { scale: number }): { width: number; height: number }
        render(opts: { canvasContext: CanvasRenderingContext2D; viewport: any }): { promise: Promise<void> }
        getTextContent(): Promise<{ items: Array<{ str: string }> }>
      }>
    }>
  }
}

declare module 'pdfjs-dist/build/pdf.worker.mjs?url' {
  const url: string
  export default url
}

declare module 'pdfjs-dist/build/pdf.mjs' {
  export * from 'pdfjs-dist'
}

declare module 'mammoth' {
  export function convertToHtml(opts: { arrayBuffer: ArrayBuffer }): Promise<{
    value: string
    messages: any[]
  }>
}

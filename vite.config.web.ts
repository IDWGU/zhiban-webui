import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [
    react(),
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src')
    }
  },
  build: {
    outDir: 'dist-web',
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      '/ws': {
        target: 'ws://127.0.0.1:18921',
        ws: true,
      },
      '/health': 'http://127.0.0.1:18921',
      '/ready': 'http://127.0.0.1:18921',
      '/startup-status': 'http://127.0.0.1:18921',
      '/system-info': 'http://127.0.0.1:18921',
      '/upload': 'http://127.0.0.1:18921',
    }
  }
})

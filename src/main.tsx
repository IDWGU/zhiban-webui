import React, { Component } from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './styles/global.css'

// Error boundary - shows the actual error on screen instead of white screen
class ErrorBoundary extends Component<{ children: React.ReactNode }, { error: Error | null }> {
  constructor(props: { children: React.ReactNode }) {
    super(props)
    this.state = { error: null }
  }
  static getDerivedStateFromError(error: Error) {
    return { error }
  }
  handleRetry = () => {
    this.setState({ error: null })
  }
  render() {
    if (this.state.error) {
      return (
        <div style={{ padding: 30, color: '#f44336', background: '#14161c', fontFamily: 'monospace', fontSize: 13 }}>
          <h2 style={{ color: '#fff' }}>React Error</h2>
          <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
            {this.state.error.message}
          </pre>
          <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all', marginTop: 10, color: '#999', fontSize: 11 }}>
            {this.state.error.stack}
          </pre>
          <button onClick={this.handleRetry} style={{
            marginTop: 16, padding: '8px 20px',
            background: '#f44336', color: '#fff',
            border: 'none', borderRadius: 6, cursor: 'pointer',
            fontSize: 13, fontFamily: 'system-ui, sans-serif',
          }}>
            重试
          </button>
        </div>
      )
    }
    return this.props.children
  }
}

const root = document.getElementById('root')
if (root) {
  ReactDOM.createRoot(root).render(
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  )
} else {
  document.body.innerHTML = '<div style="padding:30px;color:red;background:#14161c">Root element not found</div>'
}

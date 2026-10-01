import { useState } from 'react'
import { CircleAlert, Languages } from 'lucide-react'
import { api, type Translation } from './api'
import { TranslationPanel, TranslationResult } from './components'
import FeedbackPanel from './FeedbackPanel'

export default function App() {
  const [text, setText] = useState('')
  const [submittedText, setSubmittedText] = useState('')
  const [result, setResult] = useState<Translation | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [copied, setCopied] = useState(false)

  async function translate() {
    const source = text.trim()
    if (!source || loading) return
    setLoading(true)
    setError('')
    setResult(null)
    setCopied(false)
    try {
      const answer = await api.translate(source)
      setSubmittedText(source)
      setResult(answer)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Translation failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  async function copy() {
    if (!result?.translation) return
    try {
      await navigator.clipboard.writeText(result.translation)
      setCopied(true)
    } catch {
      setError('Could not copy to clipboard.')
    }
  }

  return (
    <div className="app-shell">
      <header className="site-header">
        <div className="header-inner">
          <a className="brand" href="/" aria-label="HRE-TRANSLATE home">
            <span className="brand-symbol"><Languages size={21} strokeWidth={2.2} /></span>
            <span>HRE<span className="brand-accent">·</span>TRANSLATE</span>
          </a>
          <span className="header-caption">H’rê → Vietnamese</span>
        </div>
      </header>

      <main className="page-container">
        <section className="intro">
          <p className="intro-kicker">H’RÊ → VIETNAMESE</p>
          <h1>Translation made <span>simple.</span></h1>
          <p>Enter H’rê text and get a Vietnamese translation in one step.</p>
        </section>

        <section className="workspace" aria-label="H’rê to Vietnamese translator">
          <div className="workspace-heading">
            <div>
              <p className="eyebrow">TRANSLATOR</p>
              <h2>H’rê <span aria-hidden="true">→</span> Vietnamese</h2>
            </div>
            <span className="language-pair">Fixed language pair</span>
          </div>
          {error && (
            <div className="error-banner" role="alert">
              <CircleAlert size={18} />
              <span>{error}</span>
            </div>
          )}
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <TranslationPanel
              text={text}
              onTextChange={setText}
              onTranslate={() => void translate()}
              loading={loading}
            />
            <TranslationResult
              result={result}
              loading={loading}
              copied={copied}
              onCopy={() => void copy()}
            />
          </div>
          {result && (
            <div className="result-meta flex flex-wrap gap-x-6 gap-y-2" aria-label="Translation details">
              <span>Model: <strong>{result.model.replaceAll('_', ' ')}</strong></span>
              <span>Response: <strong>{Math.round(result.latency_ms).toLocaleString()} ms</strong></span>
            </div>
          )}
        </section>

        {result && (
          <FeedbackPanel
            key={submittedText + result.translation}
            source={submittedText}
            result={result}
          />
        )}
      </main>

      <footer className="site-footer">
        <span>HRE-TRANSLATE</span>
        <span>Research demo · Please review translations before use.</span>
      </footer>
    </div>
  )
}

import { useEffect, useState } from 'react'
import { ArrowUpRight, CircleAlert, Languages, Layers3 } from 'lucide-react'
import { api, type ModelOption, type Translation } from './api'
import { ModelSelector, RetrievedExamples, RetrievedTerms, ResultMetrics, TranslationPanel, TranslationResult } from './components'
import FeedbackPanel from './FeedbackPanel'

export default function App() {
  const [models, setModels] = useState<ModelOption[]>([])
  const [model, setModel] = useState('auto')
  const [text, setText] = useState('')
  const [submittedText, setSubmittedText] = useState('')
  const [result, setResult] = useState<Translation | null>(null)
  const [loading, setLoading] = useState(false)
  const [modelsLoading, setModelsLoading] = useState(true)
  const [error, setError] = useState('')
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    api.models().then(options => {
      setModels(options)
      if (!options.some(option => option.id === 'auto' && option.available)) {
        setModel(options.find(option => option.available)?.id || '')
      }
    }).catch(() => setError('Cannot reach the translation API. Start FastAPI on port 8000, then refresh.'))
      .finally(() => setModelsLoading(false))
  }, [])

  async function translate() {
    if (!text.trim() || !model || loading) return
    setLoading(true)
    setError('')
    setResult(null)
    setCopied(false)
    try {
      const answer = await api.translate(text.trim(), model)
      setSubmittedText(text.trim())
      setResult(answer)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Translation failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  async function copy() {
    if (!result?.translation) return
    try { await navigator.clipboard.writeText(result.translation); setCopied(true) }
    catch { setError('Could not copy to clipboard.') }
  }

  return <div className="app-shell">
    <header className="site-header"><div className="header-inner"><a className="brand" href="/" aria-label="HRE Translate home"><span className="brand-symbol"><Languages size={21} strokeWidth={2.3} /></span><span>hre<span className="brand-accent">.</span>translate</span></a><div className="header-right"><span className="header-badge"><span className="live-dot" /> RESEARCH DEMO</span><a href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer">API docs <ArrowUpRight size={14} /></a></div></div></header>
    <main className="page-container">
      <div className="hero"><div className="hero-kicker"><Layers3 size={15} /> LANGUAGE TECHNOLOGY / H’RÊ → VIETNAMESE</div><h1>Translation, with <em>context.</em></h1><p>Explore H’rê to Vietnamese translation through interpretable models, retrieval, and a carefully evaluated default.</p></div>
      <section className="workspace"><div className="workspace-top"><div><p className="eyebrow">TRANSLATION WORKSPACE</p><h2>Make every word count.</h2></div><ModelSelector models={models} value={model} onChange={setModel} loading={modelsLoading} /></div>
        {error && <div className="error-banner" role="alert"><CircleAlert size={18} /><span>{error}</span></div>}
        <div className="translation-grid"><TranslationPanel text={text} onTextChange={setText} onTranslate={() => void translate()} loading={loading} disabled={modelsLoading || !model} /><TranslationResult result={result} loading={loading} copied={copied} onCopy={() => void copy()} /></div>
      </section>
      {result ? <div className="results-section"><div className="section-intro"><div><p className="eyebrow">UNDER THE HOOD</p><h2>Inside this translation</h2></div><span>Transparent by design</span></div><ResultMetrics result={result} /><div className="context-grid"><RetrievedTerms terms={result.retrieved_terms || []} /><RetrievedExamples examples={result.retrieved_examples || []} /></div><FeedbackPanel key={`${submittedText}-${result.translation}`} source={submittedText} result={result} /></div> : <div className="pre-result"><span className="pre-result-icon"><Layers3 size={20} /></span><div><strong>Built for transparent translation</strong><p>After translating, explore the model route, response time, dictionary matches, and similar examples.</p></div></div>}
    </main>
    <footer className="site-footer"><span>HRE TRANSLATE <span className="footer-separator">/</span> LOW-RESOURCE LANGUAGE RESEARCH</span><span>Research demo · Outputs may contain errors</span></footer>
  </div>
}

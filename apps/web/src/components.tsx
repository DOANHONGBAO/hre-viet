import { ArrowRight, Check, Copy, Sparkles } from 'lucide-react'
import type { Translation } from './api'

export function TranslationPanel({ text, onTextChange, onTranslate, loading, language }: {
  text: string
  onTextChange: (value: string) => void
  onTranslate: () => void
  loading: boolean
  language: 'hre' | 'vi'
}) {
  const label = language === 'hre' ? 'H’rê' : 'Vietnamese'
  return (
    <section className="editor-card">
      <div className="editor-heading">
        <div className="flex items-center gap-3">
          <span className="language-mark">{language === 'hre' ? 'Hr' : 'Vi'}</span>
          <div><p className="eyebrow">SOURCE</p><h3>{label}</h3></div>
        </div>
        <span className="mini-tag">INPUT</span>
      </div>
      <textarea
        aria-label={`${label} input`}
        className="editor-input"
        placeholder={`Type or paste your ${label} text here…`}
        value={text}
        onChange={event => onTextChange(event.target.value)}
        onKeyDown={event => {
          if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') onTranslate()
        }}
        maxLength={2000}
      />
      <div className="editor-footer">
        <span className="character-count">{text.length} / 2,000</span>
        <button
          className="translate-button"
          type="button"
          onClick={onTranslate}
          disabled={loading || !text.trim()}
        >
          {loading ? <><span className="spinner" /> Translating…</> : <>Translate <ArrowRight size={18} /></>}
        </button>
      </div>
    </section>
  )
}

export function TranslationResult({ result, loading, copied, onCopy, language }: {
  result: Translation | null
  loading: boolean
  copied: boolean
  onCopy: () => void
  language: 'hre' | 'vi'
}) {
  const label = language === 'hre' ? 'H’rê' : 'Vietnamese'
  return (
    <section className="editor-card result-card">
      <div className="editor-heading">
        <div className="flex items-center gap-3">
          <span className="language-mark output-mark">{language === 'hre' ? 'Hr' : 'Vi'}</span>
          <div><p className="eyebrow">TARGET</p><h3>{label}</h3></div>
        </div>
        <span className="mini-tag output-tag">OUTPUT</span>
      </div>
      <div className={'output-area' + (!result ? ' empty-output' : '')} aria-live="polite">
        {loading ? (
          <div className="status-content"><span className="spinner dark" /> Translating your text…</div>
        ) : result ? (
          <p className="translated-text">{result.translation || 'No translation returned.'}</p>
        ) : (
          <div className="empty-illustration">
            <Sparkles size={26} strokeWidth={1.5} />
            <p>Your {label} translation will appear here.</p>
          </div>
        )}
      </div>
      <div className="editor-footer result-footer">
        <span className="character-count">{result ? 'Translation ready' : 'Ready when you are'}</span>
        <button
          type="button"
          className="copy-button"
          onClick={onCopy}
          disabled={!result?.translation}
          aria-label="Copy translation"
        >
          {copied ? <Check size={17} /> : <Copy size={17} />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
    </section>
  )
}

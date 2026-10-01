import { useState } from 'react'
import { Check, ThumbsDown, ThumbsUp } from 'lucide-react'
import { api, type Translation } from './api'

export default function FeedbackPanel({ source, result }: { source: string; result: Translation }) {
  const [correction, setCorrection] = useState(result.translation)
  const [rating, setRating] = useState<number | undefined>()
  const [sending, setSending] = useState(false)
  const [message, setMessage] = useState('')

  async function submit(nextRating = rating) {
    if (!correction.trim() || sending) return
    setSending(true)
    setMessage('')
    try {
      await api.feedback({ source, prediction: result.translation, correction: correction.trim(), model: result.model, rating: nextRating })
      setRating(nextRating)
      setMessage('Thanks — your feedback was saved locally.')
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Could not save feedback.')
    } finally {
      setSending(false)
    }
  }

  return <section className="feedback-card">
    <div><p className="eyebrow">HELP US IMPROVE</p><h3>Was this translation useful?</h3><p className="feedback-subtitle">Rate the result or suggest a better Vietnamese translation.</p></div>
    <div className="feedback-actions"><button type="button" className={`rating-button ${rating === 5 ? 'selected' : ''}`} onClick={() => void submit(5)} disabled={sending}><ThumbsUp size={17} /> Helpful</button><button type="button" className={`rating-button ${rating === 1 ? 'selected' : ''}`} onClick={() => void submit(1)} disabled={sending}><ThumbsDown size={17} /> Needs work</button></div>
    <label className="correction-label" htmlFor="correction">SUGGEST A CORRECTION</label>
    <div className="correction-row"><textarea id="correction" value={correction} onChange={event => setCorrection(event.target.value)} maxLength={4000} /><button type="button" onClick={() => void submit()} disabled={sending || !correction.trim() || correction.trim() === result.translation}><Check size={17} /> Submit correction</button></div>
    {message && <p className="feedback-message" role="status">{message}</p>}
  </section>
}

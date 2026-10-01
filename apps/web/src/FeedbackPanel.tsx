import { useState } from 'react'
import { Check, Pencil, ThumbsUp } from 'lucide-react'
import { api, type Translation } from './api'

export default function FeedbackPanel({ source, result }: { source: string; result: Translation }) {
  const [editing, setEditing] = useState(false)
  const [correction, setCorrection] = useState(result.translation)
  const [helpful, setHelpful] = useState(false)
  const [sending, setSending] = useState(false)
  const [message, setMessage] = useState('')

  async function submitFeedback(updatedText: string, rating?: number) {
    const cleaned = updatedText.trim()
    if (!cleaned || sending) return
    setSending(true)
    setMessage('')
    try {
      await api.feedback({
        source,
        prediction: result.translation,
        correction: cleaned,
        model: result.model,
        rating,
      })
      if (rating === 5) setHelpful(true)
      if (editing) setEditing(false)
      setMessage('Thank you — your feedback was saved.')
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Could not save feedback.')
    } finally {
      setSending(false)
    }
  }

  return (
    <section className="feedback-card">
      <div className="flex flex-col items-start gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h3>Was this translation helpful?</h3>
          <p>Help improve this H’rê → Vietnamese translator.</p>
        </div>
        <div className="flex gap-2">
          <button
            type="button"
            className={'rating-button' + (helpful ? ' selected' : '')}
            onClick={() => void submitFeedback(result.translation, 5)}
            disabled={sending || helpful}
          >
            <ThumbsUp size={16} /> Good
          </button>
          <button
            type="button"
            className="rating-button"
            onClick={() => setEditing(value => !value)}
            disabled={sending}
          >
            <Pencil size={16} /> Edit translation
          </button>
        </div>
      </div>
      {editing && (
        <div className="correction-form">
          <label htmlFor="correction">Suggest a better Vietnamese translation</label>
          <textarea
            id="correction"
            value={correction}
            onChange={event => setCorrection(event.target.value)}
            maxLength={4000}
          />
          <button
            type="button"
            onClick={() => void submitFeedback(correction)}
            disabled={sending || !correction.trim() || correction.trim() === result.translation}
          >
            <Check size={16} /> Send correction
          </button>
        </div>
      )}
      {message && <p className="feedback-message" role="status">{message}</p>}
    </section>
  )
}

export type ModelOption = { id: string; label: string; available: boolean }
export type Term = { source?: string; target?: string; match_type?: string }
export type Example = { source?: string; target?: string; similarity?: number }
export type Translation = {
  translation: string
  model: string
  latency_ms: number
  retrieved_terms?: Term[]
  retrieved_examples?: Example[]
  similarity?: number | null
  confidence?: number | null
}

const base = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')

async function request<T>(path: string, init?: RequestInit, timeoutMs = 120000): Promise<T> {
  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs)
  try {
    const response = await fetch(`${base}${path}`, { ...init, signal: controller.signal })
    if (!response.ok) {
      const body = await response.json().catch(() => ({}))
      const detail = typeof body.detail === 'string' ? body.detail : `HTTP ${response.status}`
      throw new Error(detail)
    }
    return response.json() as Promise<T>
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new Error('Request timed out. The selected model may need more time to load.')
    }
    throw error
  } finally {
    window.clearTimeout(timeout)
  }
}

export const api = {
  models: () => request<ModelOption[]>('/models', undefined, 15000),
  translate: (text: string, model: string) => request<Translation>('/translate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, source: 'hre', target: 'vi', model }),
  }),
  feedback: (payload: { source: string; prediction: string; correction: string; model: string; rating?: number }) =>
    request<{ id: number; timestamp: string }>('/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }, 15000),
}

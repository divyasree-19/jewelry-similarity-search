// Backend API client: every call has a timeout and failures become an ApiError with a readable message.
import { API_URL, HEALTH_TIMEOUT_MS, SEARCH_TIMEOUT_MS } from '../config.js'

export class ApiError extends Error {
  constructor(message, status, body = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status // HTTP status, or 'timeout' | 'network' | 'aborted'
    this.body = body
  }
}

const STATUS_MESSAGES = {
  400: 'Invalid image or request',
  404: 'Nothing to search',
  413: 'Image is too large',
  415: 'Unsupported file type',
  502: 'The embedding service (SigLIP) is down',
  503: 'The search service is unavailable',
}

function detailText(body) {
  const detail = body && typeof body === 'object' ? body.detail : body
  if (!detail) return ''
  return typeof detail === 'string' ? detail : JSON.stringify(detail)
}

export function errorMessage(status, body) {
  const detail = detailText(body)
  const summary = STATUS_MESSAGES[status] || (status >= 500 ? `Backend error (${status})` : `Request failed (${status})`)
  return detail && !summary.includes(detail) ? `${summary}: ${detail}` : summary
}

async function readBody(response) {
  const text = await response.text()
  try {
    return text ? JSON.parse(text) : null
  } catch {
    return text
  }
}

async function request(path, { timeoutMs = SEARCH_TIMEOUT_MS, signal, ...init } = {}) {
  const controller = new AbortController()
  let timedOut = false
  const timer = setTimeout(() => {
    timedOut = true
    controller.abort()
  }, timeoutMs)
  const cancel = () => controller.abort()
  signal?.addEventListener('abort', cancel)

  let response, body
  try {
    response = await fetch(`${API_URL}${path}`, { ...init, signal: controller.signal })
    body = await readBody(response)
  } catch {
    if (timedOut) throw new ApiError(`The backend did not answer within ${timeoutMs / 1000} s.`, 'timeout')
    if (signal?.aborted) throw new ApiError('Request cancelled.', 'aborted')
    const origin = typeof window === 'undefined' ? 'this app' : window.location.origin
    throw new ApiError(
      `Cannot reach the backend at ${API_URL}. Check that it is running and that its CORS_ORIGINS includes ${origin}.`,
      'network',
    )
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', cancel)
  }
  if (!response.ok) throw new ApiError(errorMessage(response.status, body), response.status, body)
  return body
}

/** POST /search/image. An empty category means auto-detect. */
export function searchImage({ file, category, text, topK = 10, debug = false }, signal) {
  const form = new FormData()
  form.append('file', file)
  if (category) form.append('category', category)
  if (text?.trim()) form.append('text', text.trim())
  form.append('top_k', String(topK))
  form.append('debug', String(debug))
  return request('/search/image', { method: 'POST', body: form, signal })
}

/** POST /search/text. An empty category lets the backend take it from the text (or search all). */
export function searchText({ text, category, topK = 10, debug = false }, signal) {
  return request('/search/text', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: text.trim(), category: category || null, top_k: topK, debug }),
    signal,
  })
}

/** GET /health. A degraded backend answers 503 with the same body, so that is returned rather than thrown. */
export async function health(signal) {
  try {
    return await request('/health', { timeoutMs: HEALTH_TIMEOUT_MS, signal })
  } catch (err) {
    if (err.status === 503 && err.body?.dependencies) return err.body
    throw err
  }
}

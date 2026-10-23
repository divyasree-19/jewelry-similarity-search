import { useCallback, useEffect, useState } from 'react'

import { health } from '../api/search.js'
import { API_URL } from '../config.js'

const POLL_MS = 15_000

function describe(state) {
  if (state.status === 'checking') return { ok: null, text: 'Checking backend…', title: '' }
  if (state.status === 'unreachable') return { ok: false, text: 'Backend unreachable', title: state.error }
  const failing = Object.entries(state.body.dependencies || {}).filter(([, dep]) => dep.status !== 'ok')
  const title = Object.entries(state.body.dependencies || {})
    .map(([name, dep]) => `${name}: ${dep.status}${dep.error ? ` (${dep.error})` : ''}`)
    .join('\n')
  if (!failing.length) return { ok: true, text: `Backend OK · ${state.body.embed_backend}`, title }
  return { ok: false, text: `Backend degraded: ${failing.map(([name]) => name).join(', ')} down`, title }
}

export default function Header() {
  const [state, setState] = useState({ status: 'checking' })

  const check = useCallback(async (signal) => {
    try {
      setState({ status: 'ok', body: await health(signal) })
    } catch (err) {
      if (!signal?.aborted) setState({ status: 'unreachable', error: err.message })
    }
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    check(controller.signal)
    const timer = setInterval(() => check(controller.signal), POLL_MS)
    return () => {
      clearInterval(timer)
      controller.abort()
    }
  }, [check])

  const { ok, text, title } = describe(state)
  const dot = ok === null ? 'bg-secondary' : ok ? 'bg-success' : 'bg-danger'

  return (
    <header className="navbar navbar-dark bg-dark">
      <div className="container flex-wrap gap-2">
        <span className="navbar-brand mb-0 h1">
          Jewellery Search <small className="text-white-50 fs-6">test harness</small>
        </span>
        <button
          type="button"
          className="btn btn-sm btn-outline-light d-flex align-items-center gap-2"
          title={`${API_URL}\n${title}\n(click to re-check)`}
          onClick={() => check()}
        >
          <span className={`health-dot rounded-circle ${dot}`} />
          {text}
        </button>
      </div>
    </header>
  )
}

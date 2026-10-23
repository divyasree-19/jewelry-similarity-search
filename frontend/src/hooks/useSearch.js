import { useCallback, useEffect, useRef, useState } from 'react'

const IDLE = { status: 'idle', response: null, error: null, latencyMs: null }

/**
 * Runs one search at a time: status is idle | loading | done | error.
 * A new search or reset() aborts the request in flight, and its late answer is ignored.
 */
export function useSearch() {
  const [state, setState] = useState(IDLE)
  const current = useRef(null)

  const run = useCallback(async (call) => {
    current.current?.abort()
    const controller = new AbortController()
    current.current = controller
    setState({ ...IDLE, status: 'loading' })
    const started = performance.now()
    try {
      const response = await call(controller.signal)
      if (current.current !== controller) return
      setState({ status: 'done', response, error: null, latencyMs: Math.round(performance.now() - started) })
    } catch (err) {
      if (current.current !== controller) return
      setState({ ...IDLE, status: 'error', error: err.message })
    }
  }, [])

  const reset = useCallback(() => {
    current.current?.abort()
    current.current = null
    setState(IDLE)
  }, [])

  useEffect(() => () => current.current?.abort(), [])

  return { ...state, run, reset }
}

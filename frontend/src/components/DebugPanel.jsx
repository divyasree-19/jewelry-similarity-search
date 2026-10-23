// Debug view (debug=true): per-result signals, weights and labels, plus weights version and latency.
const SIGNALS = ['image', 'caption', 'gray', 'region']

const fixed = (value, digits = 3) => (value == null ? '–' : Number(value).toFixed(digits))

export function DebugToggle({ checked, onChange }) {
  return (
    <div className="form-check form-switch mb-0">
      <input
        id="debug-toggle"
        className="form-check-input"
        type="checkbox"
        role="switch"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      <label className="form-check-label" htmlFor="debug-toggle">Debug</label>
    </div>
  )
}

/** Stage 9 raw cosine per signal, used when the reranker did not run for this mode (no weights). */
function rawSignal(debug, name) {
  if (name === 'region') return debug.rerank?.region?.score
  return debug[`${name}_score`]
}

function SignalCell({ debug, name }) {
  const detail = debug.signals?.[name]
  if (!detail) {
    const raw = rawSignal(debug, name)
    return <td className={raw == null ? 'text-body-tertiary' : ''}>{fixed(raw)}</td>
  }
  if (detail.raw == null) return <td className="text-body-tertiary">–</td>
  return (
    <td className={detail.used ? '' : 'text-body-tertiary'} title={`raw cosine ${fixed(detail.raw, 4)}`}>
      {fixed(detail.value)}
      <span className="text-body-secondary"> × {fixed(detail.weight, 2)}</span>
    </td>
  )
}

function BoostCell({ debug }) {
  const points = debug.signals?.boost?.points ?? debug.boost_total
  const fields = Object.entries(debug.boosts || {})
  return (
    <td title={fields.map(([field, { matched }]) => `${field}: ${matched.join(', ')}`).join('\n')}>
      {points ? `+${fixed(points, 4)}` : '–'}
      {fields.length > 0 && <div className="small text-body-secondary">{fields.map(([field]) => field).join(', ')}</div>}
    </td>
  )
}

export default function DebugPanel({ response, latencyMs }) {
  const timings = response.timings_ms || {}
  const reranked = response.results.some((r) => r.debug?.signals)
  const steps = Object.entries(timings).filter(([step]) => step !== 'total')

  return (
    <div className="card mb-3 debug-panel">
      <div className="card-header d-flex flex-wrap gap-3 small">
        <span>
          mode <strong>{response.mode}</strong>
        </span>
        <span>
          weights{' '}
          <strong>{response.reranker_weights_version || 'not used (Stage 9 order for this mode)'}</strong>
        </span>
        <span>
          backend <strong>{fixed(timings.total, 0)} ms</strong>
        </span>
        <span>
          round trip <strong>{latencyMs} ms</strong>
        </span>
      </div>
      <div className="card-body p-0">
        {steps.length > 0 && (
          <div className="small text-body-secondary px-3 py-2 border-bottom">
            {steps.map(([step, ms]) => `${step} ${fixed(ms, 0)} ms`).join(' · ')}
          </div>
        )}
        <div className="table-responsive">
          <table className="table table-sm mb-0 align-middle small">
            <thead>
              <tr>
                <th>#</th>
                <th>product</th>
                {SIGNALS.map((name) => (
                  <th key={name}>{name}</th>
                ))}
                <th>boosts</th>
                <th>final</th>
                <th>labels</th>
              </tr>
            </thead>
            <tbody>
              {response.results.map((result, index) => {
                const debug = result.debug || {}
                return (
                  <tr key={`${result.product_id}-${result.image_id}`}>
                    <td>{index + 1}</td>
                    <td className="text-nowrap">
                      {result.product_id}
                      {debug.possible_duplicate_of && (
                        <span className="badge text-bg-warning ms-1" title="possible duplicate">dup</span>
                      )}
                    </td>
                    {SIGNALS.map((name) => (
                      <SignalCell key={name} debug={debug} name={name} />
                    ))}
                    <BoostCell debug={debug} />
                    <td>{fixed(result.score, 4)}</td>
                    <td>
                      {Object.entries(debug.labels || {}).map(([key, value]) => (
                        <span key={key} className="badge text-bg-light border me-1">
                          {key}: {value}
                        </span>
                      ))}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        <div className="small text-body-secondary px-3 py-2 border-top">
          {reranked
            ? 'Signal cells: 0–1 value × normalised weight (hover for the raw cosine). Grey = not used by these weights.'
            : 'Reranker not applied to this mode: cells are raw cosines, final is the Stage 9 fused score.'}
        </div>
      </div>
    </div>
  )
}

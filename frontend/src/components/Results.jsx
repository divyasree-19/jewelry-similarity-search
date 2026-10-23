import DebugPanel from './DebugPanel.jsx'
import ResultCard from './ResultCard.jsx'

const percent = (value) => `${Math.round(value * 100)}%`
const label = (category) => category.replace('_', ' ')

function CategoryChoice({ response, onChoose }) {
  return (
    <div className="alert alert-warning">
      <div className="mb-2">{response.message || 'Not sure which category this is. Choose one to search:'}</div>
      <div className="d-flex flex-wrap gap-2">
        {response.category_candidates.map(({ category, confidence }) => (
          <button key={category} type="button" className="btn btn-warning" onClick={() => onChoose(category)}>
            {label(category)} <span className="badge text-bg-light ms-1">{percent(confidence)}</span>
          </button>
        ))}
      </div>
    </div>
  )
}

function CategorySummary({ response }) {
  const { category, category_source: source, category_confidence: confidence } = response
  if (!category) {
    const searched = response.searched_categories
    return <span>Searched {searched?.length ? searched.map(label).join(', ') : 'all categories'}</span>
  }
  const how = {
    user: 'chosen',
    text: 'from the text',
    detected: confidence != null ? `detected, ${percent(confidence)} confidence` : 'detected',
  }[source]
  return (
    <span>
      Category <strong>{label(category)}</strong> {how && <span className="text-body-secondary">({how})</span>}
    </span>
  )
}

function ParsedText({ parsed }) {
  const parts = Object.entries(parsed || {})
    .filter(([, value]) => value != null && value !== '' && !(Array.isArray(value) && !value.length))
    .map(([key, value]) => `${key}: ${Array.isArray(value) ? value.join(', ') : value}`)
  if (!parts.length) return null
  return <div className="small text-body-secondary">Text understood as: {parts.join(' · ')}</div>
}

export default function Results({ search, onChooseCategory }) {
  const { status, response, error, latencyMs } = search

  if (status === 'loading') {
    return (
      <div className="text-center text-body-secondary py-5">
        <div className="spinner-border mb-2" role="status" />
        <div>Searching…</div>
      </div>
    )
  }
  if (status === 'error') {
    return (
      <div className="alert alert-danger" role="alert">
        <strong>Search failed.</strong> {error}
      </div>
    )
  }
  if (status !== 'done') return null

  if (response.needs_category) return <CategoryChoice response={response} onChoose={onChooseCategory} />

  const results = response.results || []
  return (
    <section>
      <div className="d-flex flex-wrap justify-content-between gap-2 mb-2">
        <CategorySummary response={response} />
        <span className="text-body-secondary small">
          {results.length} result{results.length === 1 ? '' : 's'} · {latencyMs} ms
        </span>
      </div>
      <ParsedText parsed={response.parsed_text} />
      {response.similarity_score && (
        <div className="small text-body-secondary mb-2">Similarity = {response.similarity_score}</div>
      )}
      {response.warnings?.map((warning) => (
        <div key={warning} className="alert alert-warning py-2 small">{warning}</div>
      ))}

      {results.length === 0 ? (
        <div className="text-center text-body-secondary border rounded-3 py-5">
          <div className="fs-5">No results</div>
          <div className="small">Try another category, remove the refinement text, or use a clearer photo.</div>
        </div>
      ) : (
        <>
          {results[0].debug && <DebugPanel response={response} latencyMs={latencyMs} />}
          <div className="row row-cols-1 row-cols-sm-2 row-cols-lg-3 row-cols-xl-4 g-3">
            {results.map((result, index) => (
              <div className="col" key={`${result.product_id}-${result.image_id}`}>
                <ResultCard result={result} rank={index + 1} />
              </div>
            ))}
          </div>
        </>
      )}
    </section>
  )
}

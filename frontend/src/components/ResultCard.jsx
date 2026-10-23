import { imageSrc } from '../config.js'

// Product fields shown as attribute badges, in this order
const ATTRIBUTE_FIELDS = ['metal', 'gemstones', 'stone_colours', 'motifs', 'style']

const values = (value) => (Array.isArray(value) ? value : value ? [value] : [])

/** Attribute values that earned a boost for this query; only known when debug is on. */
function matchedValues(debug) {
  return Object.entries(debug?.boosts || {}).flatMap(([field, { matched }]) =>
    matched.map((value) => ({ field, value })),
  )
}

export default function ResultCard({ result, rank }) {
  const { attributes = {}, debug } = result
  const score = result.similarity_score
  const width = score == null ? 0 : Math.round(Math.min(1, Math.max(0, score)) * 100)
  const duplicate = debug?.possible_duplicate_of
  const src = imageSrc(result.image_url)

  return (
    <div className="card h-100 result-card">
      <div className="ratio ratio-1x1 result-image rounded-top">
        {src ? (
          <img src={src} alt={attributes.name || result.product_id} loading="lazy" />
        ) : (
          <div className="d-flex align-items-center justify-content-center text-body-secondary small">no image</div>
        )}
      </div>
      <div className="card-body d-flex flex-column gap-2 p-3">
        <div>
          <div className="d-flex justify-content-between align-items-start gap-2">
            <span className="fw-semibold">
              #{rank} {result.product_id}
            </span>
            <span className="badge text-bg-secondary">{result.category.replace('_', ' ')}</span>
          </div>
          {attributes.name && <div className="small text-body-secondary">{attributes.name}</div>}
          {duplicate && (
            <span
              className="badge text-bg-warning mt-1"
              title={`Looks almost identical to ${duplicate.product_id} (image similarity ${duplicate.image_similarity})`}
            >
              possible duplicate of {duplicate.product_id}
            </span>
          )}
        </div>

        <div>
          <div className="d-flex justify-content-between small">
            <span>similarity</span>
            <span>{score == null ? 'n/a' : `${(score * 100).toFixed(1)}%`}</span>
          </div>
          <div className="progress" role="progressbar" aria-label="similarity" aria-valuenow={width}>
            <div className="progress-bar" style={{ width: `${width}%` }} />
          </div>
        </div>

        <div className="d-flex flex-wrap gap-1 mt-auto">
          {matchedValues(debug).map(({ field, value }) => (
            <span key={`m-${field}-${value}`} className="badge text-bg-success" title={`matched ${field}`}>
              ✓ {value}
            </span>
          ))}
          {ATTRIBUTE_FIELDS.flatMap((field) =>
            values(attributes[field]).map((value) => (
              <span key={`${field}-${value}`} className="badge attribute-badge" title={field.replace('_', ' ')}>
                {value}
              </span>
            )),
          )}
        </div>
      </div>
    </div>
  )
}

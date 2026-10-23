/** Search (with spinner while loading) and Clear, shared by both tabs. */
export default function SearchButtons({ loading, canSearch, onClear }) {
  return (
    <div className="d-flex gap-2">
      <button type="submit" className="btn btn-primary flex-grow-1" disabled={loading || !canSearch}>
        {loading && <span className="spinner-border spinner-border-sm me-2" aria-hidden="true" />}
        {loading ? 'Searching…' : 'Search'}
      </button>
      <button type="button" className="btn btn-outline-secondary" onClick={onClear}>
        Clear
      </button>
    </div>
  )
}

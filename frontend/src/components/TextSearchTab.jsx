import { useState } from 'react'

import { searchText } from '../api/search.js'
import { useSearch } from '../hooks/useSearch.js'
import CategorySelect from './CategorySelect.jsx'
import Results from './Results.jsx'
import SearchButtons from './SearchButtons.jsx'
import TopKInput from './TopKInput.jsx'

export default function TextSearchTab({ debug }) {
  const [text, setText] = useState('')
  const [category, setCategory] = useState('')
  const [topK, setTopK] = useState(10)
  const search = useSearch()
  const loading = search.status === 'loading'

  const runSearch = (chosenCategory = category) => {
    if (!text.trim()) return
    search.run((signal) => searchText({ text, category: chosenCategory, topK, debug }, signal))
  }

  const chooseCategory = (chosen) => {
    setCategory(chosen)
    runSearch(chosen)
  }

  const clear = () => {
    setText('')
    setCategory('')
    setTopK(10)
    search.reset()
  }

  return (
    <>
      <form
        className="card card-body mb-4"
        onSubmit={(e) => {
          e.preventDefault()
          runSearch()
        }}
      >
        <div className="row g-3 align-items-end">
          <div className="col-12 col-lg-6">
            <label htmlFor="text-query" className="form-label">Describe the piece</label>
            <input
              id="text-query"
              type="text"
              className="form-control"
              maxLength={500}
              placeholder="e.g. red ruby ring, silver peacock earrings"
              value={text}
              disabled={loading}
              onChange={(e) => setText(e.target.value)}
            />
          </div>
          <div className="col-8 col-lg-3">
            <label htmlFor="text-category" className="form-label">Category</label>
            <CategorySelect
              id="text-category"
              value={category}
              onChange={setCategory}
              autoLabel="Any (from the text)"
              disabled={loading}
            />
          </div>
          <div className="col-4 col-lg-1">
            <label htmlFor="text-top-k" className="form-label">Results</label>
            <TopKInput id="text-top-k" value={topK} onChange={setTopK} disabled={loading} />
          </div>
          <div className="col-12 col-lg-2">
            <SearchButtons loading={loading} canSearch={Boolean(text.trim())} onClear={clear} />
          </div>
        </div>
      </form>
      <Results search={search} onChooseCategory={chooseCategory} />
    </>
  )
}

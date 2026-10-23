import { useState } from 'react'

import { searchImage } from '../api/search.js'
import { useSearch } from '../hooks/useSearch.js'
import CategorySelect from './CategorySelect.jsx'
import Results from './Results.jsx'
import SearchButtons from './SearchButtons.jsx'
import TopKInput from './TopKInput.jsx'
import UploadBox from './UploadBox.jsx'

export default function ImageSearchTab({ debug }) {
  const [file, setFile] = useState(null)
  const [category, setCategory] = useState('')
  const [text, setText] = useState('')
  const [topK, setTopK] = useState(10)
  const search = useSearch()
  const loading = search.status === 'loading'

  const runSearch = (chosenCategory = category) => {
    if (!file) return
    search.run((signal) => searchImage({ file, category: chosenCategory, text, topK, debug }, signal))
  }

  // Low-confidence answer: the user picks one of the suggested categories and the same query runs again
  const chooseCategory = (chosen) => {
    setCategory(chosen)
    runSearch(chosen)
  }

  const clear = () => {
    setFile(null)
    setCategory('')
    setText('')
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
        <div className="row g-3">
          <div className="col-md-5">
            <UploadBox file={file} onChange={setFile} disabled={loading} />
          </div>
          <div className="col-md-7 d-flex flex-column gap-3">
            <div className="row g-3">
              <div className="col-8">
                <label htmlFor="image-category" className="form-label">Category</label>
                <CategorySelect id="image-category" value={category} onChange={setCategory} disabled={loading} />
              </div>
              <div className="col-4">
                <label htmlFor="image-top-k" className="form-label">Results</label>
                <TopKInput id="image-top-k" value={topK} onChange={setTopK} disabled={loading} />
              </div>
            </div>
            <div>
              <label htmlFor="image-text" className="form-label">
                Refinement <span className="text-body-secondary">(optional)</span>
              </label>
              <input
                id="image-text"
                type="text"
                className="form-control"
                maxLength={500}
                placeholder="e.g. similar but in silver"
                value={text}
                disabled={loading}
                onChange={(e) => setText(e.target.value)}
              />
            </div>
            <div className="mt-auto">
              <SearchButtons loading={loading} canSearch={Boolean(file)} onClear={clear} />
            </div>
          </div>
        </div>
      </form>
      <Results search={search} onChooseCategory={chooseCategory} />
    </>
  )
}

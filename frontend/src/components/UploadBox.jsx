import { useEffect, useRef, useState } from 'react'

import { ALLOWED_TYPES, MAX_UPLOAD_BYTES, MAX_UPLOAD_MB } from '../config.js'

/** Same checks as the backend (type and size), so obvious mistakes never leave the browser. */
export function checkFile(file) {
  if (!ALLOWED_TYPES.includes(file.type)) return `"${file.name}" is not a JPEG, PNG or WebP image.`
  if (file.size === 0) return `"${file.name}" is empty.`
  if (file.size > MAX_UPLOAD_BYTES) {
    return `"${file.name}" is ${(file.size / 1_000_000).toFixed(1)} MB; the limit is ${MAX_UPLOAD_MB} MB.`
  }
  return null
}

export default function UploadBox({ file, onChange, disabled }) {
  const input = useRef(null)
  const [dragging, setDragging] = useState(false)
  const [error, setError] = useState(null)
  const [preview, setPreview] = useState(null)

  useEffect(() => {
    if (!file) {
      setPreview(null)
      return undefined
    }
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  useEffect(() => {
    if (!file) {
      setError(null)
      if (input.current) input.current.value = ''
    }
  }, [file])

  const pick = (picked) => {
    if (!picked) return
    const problem = checkFile(picked)
    setError(problem)
    if (!problem) onChange(picked)
  }

  const onDrop = (event) => {
    event.preventDefault()
    setDragging(false)
    if (!disabled) pick(event.dataTransfer.files[0])
  }

  return (
    <div>
      <div
        role="button"
        tabIndex={0}
        className={`upload-box rounded-3 d-flex align-items-center justify-content-center text-center p-3 ${
          dragging ? 'dragging' : ''
        } ${disabled ? 'disabled' : ''}`}
        onClick={() => !disabled && input.current.click()}
        onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && !disabled && input.current.click()}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        {preview ? (
          <img src={preview} alt="Query preview" className="upload-preview" />
        ) : (
          <div className="text-body-secondary">
            <div className="fs-5">Drop a jewellery photo here</div>
            <div className="small">or click to choose · JPEG, PNG or WebP · up to {MAX_UPLOAD_MB} MB</div>
          </div>
        )}
      </div>
      <input
        ref={input}
        type="file"
        accept={ALLOWED_TYPES.join(',')}
        className="d-none"
        onChange={(e) => pick(e.target.files[0])}
      />
      {file && (
        <div className="small text-body-secondary mt-1 text-truncate">
          {file.name} · {(file.size / 1_000_000).toFixed(2)} MB
        </div>
      )}
      {error && <div className="small text-danger mt-1">{error}</div>}
    </div>
  )
}

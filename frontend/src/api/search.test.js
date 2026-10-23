import { afterEach, describe, expect, it, vi } from 'vitest'

import { ApiError, errorMessage, health, searchImage, searchText } from './search.js'

const reply = (status, body) =>
  vi.fn().mockResolvedValue(new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } }))

afterEach(() => {
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

describe('errorMessage', () => {
  it.each([
    [400, { detail: 'Image is too small' }, 'Invalid image or request: Image is too small'],
    [413, { detail: 'Image is larger than 10 MB' }, 'Image is too large: Image is larger than 10 MB'],
    [415, { detail: 'upload a JPEG, PNG or WebP image' }, 'Unsupported file type: upload a JPEG, PNG or WebP image'],
    [502, { detail: 'Embedding service is unavailable' }, 'The embedding service (SigLIP) is down: Embedding service is unavailable'],
    [503, { detail: 'Catalogue database is unavailable' }, 'The search service is unavailable: Catalogue database is unavailable'],
    [500, 'Internal Server Error', 'Backend error (500): Internal Server Error'],
  ])('%i', (status, body, expected) => {
    expect(errorMessage(status, body)).toBe(expected)
  })
})

describe('request', () => {
  it('sends the image form with only the fields that are set', async () => {
    const fetch = reply(200, { results: [] })
    vi.stubGlobal('fetch', fetch)
    await searchImage({ file: new Blob(['x'], { type: 'image/png' }), category: '', text: '  ', topK: 5, debug: true })
    const form = fetch.mock.calls[0][1].body
    expect(form.has('category')).toBe(false)
    expect(form.has('text')).toBe(false)
    expect(form.get('top_k')).toBe('5')
    expect(form.get('debug')).toBe('true')
  })

  it('turns an error status into an ApiError with the readable message', async () => {
    vi.stubGlobal('fetch', reply(415, { detail: 'File content is not a JPEG, PNG or WebP image' }))
    const err = await searchText({ text: 'ring' }).catch((e) => e)
    expect(err).toBeInstanceOf(ApiError)
    expect(err.status).toBe(415)
    expect(err.message).toMatch(/^Unsupported file type/)
  })

  it('reports an unreachable backend', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')))
    const err = await searchText({ text: 'ring' }).catch((e) => e)
    expect(err.status).toBe('network')
    expect(err.message).toMatch(/Cannot reach the backend/)
  })

  it('times out', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('fetch', vi.fn((url, { signal }) => new Promise((_, reject) => {
      signal.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')))
    })))
    const pending = searchText({ text: 'ring' }).catch((e) => e)
    await vi.advanceTimersByTimeAsync(60_000)
    expect((await pending).status).toBe('timeout')
  })

  it('returns a degraded health body instead of throwing', async () => {
    const body = { status: 'degraded', dependencies: { embedding_service: { status: 'error' } } }
    vi.stubGlobal('fetch', reply(503, body))
    expect(await health()).toEqual(body)
  })
})

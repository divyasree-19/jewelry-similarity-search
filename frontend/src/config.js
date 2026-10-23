// Runtime settings from frontend/.env (see .env.example)
const env = import.meta.env

export const API_URL = (env.VITE_API_URL || 'http://localhost:8080').replace(/\/+$/, '')
export const MAX_UPLOAD_MB = Number(env.VITE_MAX_UPLOAD_MB) || 10
export const MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1_000_000 // the backend counts 1 MB as 1,000,000 bytes
export const SEARCH_TIMEOUT_MS = Number(env.VITE_SEARCH_TIMEOUT_MS) || 60_000
export const HEALTH_TIMEOUT_MS = Number(env.VITE_HEALTH_TIMEOUT_MS) || 5_000

export const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp']

// Same as data/categories.txt; the backend answers 404 for a category with nothing indexed
export const CATEGORIES = ['rings', 'earrings', 'necklaces', 'pendants', 'bangles', 'bracelets', 'chains', 'nose_pins', 'anklets']

export const imageSrc = (path) => (path ? `${API_URL}${path}` : null)

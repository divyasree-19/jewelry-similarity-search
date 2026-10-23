import { CATEGORIES } from '../config.js'

export default function CategorySelect({ id, value, onChange, autoLabel = 'Auto-detect', disabled }) {
  return (
    <select id={id} className="form-select" value={value} onChange={(e) => onChange(e.target.value)} disabled={disabled}>
      <option value="">{autoLabel}</option>
      {CATEGORIES.map((category) => (
        <option key={category} value={category}>
          {category.replace('_', ' ')}
        </option>
      ))}
    </select>
  )
}

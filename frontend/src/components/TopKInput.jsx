export default function TopKInput({ id, value, onChange, disabled }) {
  return (
    <input
      id={id}
      type="number"
      className="form-control"
      min={1}
      max={50}
      value={value}
      disabled={disabled}
      onChange={(e) => onChange(Math.min(50, Math.max(1, Number(e.target.value) || 1)))}
    />
  )
}

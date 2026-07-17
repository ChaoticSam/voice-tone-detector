export default function IconInput({ icon: Icon, label, ...inputProps }) {
  return (
    <label className="field">
      {label}
      <div className="input-group">
        <Icon size={16} className="input-icon" />
        <input {...inputProps} />
      </div>
    </label>
  );
}

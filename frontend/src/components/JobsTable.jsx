import StatusBadge from "./StatusBadge";

const yn = (v) => (v === true ? "Yes" : v === false ? "No" : "-");

function Field({ label, value }) {
  return (
    <div className="job-field">
      <span className="job-field-label">{label}</span>
      <span className="job-field-value">{value}</span>
    </div>
  );
}

export default function JobsTable({ jobs }) {
  return (
    <div className="job-list">
      {jobs.map((job) => {
        const r = job.result;
        return (
          <div className="job-card" key={job.id}>
            <div className="job-card-header">
              <span className="job-filename">{job.filename}</span>
              <StatusBadge status={job.status} stage={job.stage} />
            </div>
            <div className="job-fields">
              <Field label="Tone" value={r?.emotional_tone ?? "-"} />
              <Field label="Intensity" value={r?.emotional_intensity ?? "-"} />
              <Field label="Confidence" value={r?.confidence ?? "-"} />
              <Field label="Noise" value={yn(r?.background_noise_present)} />
              <Field label="Noise type" value={r?.background_noise_type || "-"} />
              <Field label="Noise severity" value={r?.background_noise_severity ?? "-"} />
              <Field label="Audio quality" value={r?.audio_quality ?? "-"} />
              <Field label="Overlap" value={yn(r?.speaker_overlap_present)} />
              <Field label="Long silence" value={yn(r?.long_silence_present)} />
            </div>
            {job.error && <p className="job-error">{job.error}</p>}
          </div>
        );
      })}
    </div>
  );
}

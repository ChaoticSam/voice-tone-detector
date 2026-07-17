import { useEffect, useRef, useState } from "react";
import { AlertCircle, AlertTriangle, Download, UploadCloud } from "lucide-react";
import { downloadBatch, listBatchJobs, uploadBatch } from "../lib/api";
import JobsTable from "../components/JobsTable";

const POLL_INTERVAL_MS = 3000;

export default function Dashboard() {
  const [batchId, setBatchId] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [dragActive, setDragActive] = useState(false);
  const [validation, setValidation] = useState(null);
  const pollRef = useRef(null);

  useEffect(() => {
    return () => clearInterval(pollRef.current);
  }, []);

  function startPolling(id) {
    clearInterval(pollRef.current);
    const poll = async () => {
      try {
        setJobs(await listBatchJobs(id));
      } catch (err) {
        console.error(err);
      }
    };
    poll();
    pollRef.current = setInterval(poll, POLL_INTERVAL_MS);
  }

  async function handleFiles(files) {
    if (!files.length) return;
    setUploading(true);
    setUploadError("");
    setValidation(null);
    try {
      const { batch_id, validation } = await uploadBatch(files);
      setBatchId(batch_id);
      // Manifest cross-check, if a manifest was uploaded -- ephemeral (this response only,
      // not persisted), so it won't reappear after a page refresh; missing files are also
      // recorded as failed jobs in the table below, which does survive a refresh.
      if (validation?.missing.length || validation?.unmatched.length) setValidation(validation);
      startPolling(batch_id);
    } catch (err) {
      setUploadError(err.message);
    } finally {
      setUploading(false);
    }
  }

  function handleFileChange(e) {
    handleFiles(Array.from(e.target.files ?? []));
    e.target.value = "";
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragActive(false);
    handleFiles(Array.from(e.dataTransfer.files ?? []));
  }

  const allDone =
    jobs.length > 0 && jobs.every((j) => j.status === "finished" || j.status === "failed");

  return (
    <div className="dashboard">
      <section className="upload-section">
        <label
          className={`dropzone${dragActive ? " dropzone-active" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={handleDrop}
        >
          <UploadCloud size={28} />
          <span className="dropzone-label">
            {uploading ? "Uploading..." : "Drop audio files or a .zip folder here, or click to browse"}
          </span>
          <input
            type="file"
            multiple
            accept=".wav,.mp3,.ogg,.flac,.m4a,.zip"
            onChange={handleFileChange}
            disabled={uploading}
            hidden
          />
        </label>
        {uploadError && (
          <p className="error">
            <AlertCircle size={14} /> {uploadError}
          </p>
        )}
        {validation && (
          <div className="validation-warning">
            <AlertTriangle size={14} />
            <div>
              {validation.missing.length > 0 && (
                <p>
                  Listed in manifest but not found in upload: {validation.missing.join(", ")}
                </p>
              )}
              {validation.unmatched.length > 0 && (
                <p>
                  Uploaded but not listed in manifest (still processed): {validation.unmatched.join(", ")}
                </p>
              )}
            </div>
          </div>
        )}
      </section>

      {jobs.length > 0 && (
        <section className="jobs-panel">
          <div className="jobs-header">
            <h2>Batch {batchId}</h2>
            <button className="btn" onClick={() => downloadBatch(batchId)} disabled={!allDone}>
              <Download size={15} /> Download results (CSV)
            </button>
          </div>
          <JobsTable jobs={jobs} />
        </section>
      )}
    </div>
  );
}

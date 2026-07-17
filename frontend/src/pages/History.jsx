import { useEffect, useState } from "react";
import { AlertCircle, ChevronDown, ChevronRight, Download } from "lucide-react";
import { downloadBatch, listBatches, listBatchJobs } from "../lib/api";
import JobsTable from "../components/JobsTable";

export default function History() {
  const [batches, setBatches] = useState(null);
  const [error, setError] = useState("");
  const [expanded, setExpanded] = useState({}); // batch_id -> "loading" | jobs[]

  useEffect(() => {
    listBatches()
      .then(setBatches)
      .catch((err) => setError(err.message));
  }, []);

  async function toggle(batchId) {
    if (expanded[batchId]) {
      setExpanded((e) => ({ ...e, [batchId]: undefined }));
      return;
    }
    setExpanded((e) => ({ ...e, [batchId]: "loading" }));
    try {
      const jobs = await listBatchJobs(batchId);
      setExpanded((e) => ({ ...e, [batchId]: jobs }));
    } catch {
      setExpanded((e) => ({ ...e, [batchId]: [] }));
    }
  }

  return (
    <div className="dashboard">
      <h1 className="page-title">History</h1>

      {error && (
        <p className="error">
          <AlertCircle size={14} /> {error}
        </p>
      )}
      {!error && batches === null && <p className="text-muted">Loading...</p>}
      {!error && batches?.length === 0 && (
        <p className="text-muted">No batches uploaded yet -- go to Home to upload one.</p>
      )}

      <div className="batch-list">
        {batches?.map((b) => {
          const jobs = expanded[b.batch_id];
          const jobsLoaded = Array.isArray(jobs);
          const allDone = jobsLoaded && jobs.length > 0 && jobs.every((j) => j.status === "finished" || j.status === "failed");
          return (
            <div className="batch-card" key={b.batch_id}>
              <button className="batch-summary" onClick={() => toggle(b.batch_id)}>
                {jobs ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                <span className="batch-id">{b.batch_id}</span>
                <span className="batch-date">{new Date(b.created_at).toLocaleString()}</span>
                <span className="batch-counts">
                  {b.file_count} files &middot; {b.finished_count} done &middot; {b.failed_count} failed
                  {b.pending_count > 0 && ` · ${b.pending_count} pending`}
                </span>
              </button>
              {jobs === "loading" && <p className="text-muted batch-loading">Loading jobs...</p>}
              {jobsLoaded && (
                <>
                  <div className="jobs-header">
                    <span />
                    <button className="btn" onClick={() => downloadBatch(b.batch_id)} disabled={!allDone}>
                      <Download size={15} /> Download results (CSV)
                    </button>
                  </div>
                  <JobsTable jobs={jobs} />
                </>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

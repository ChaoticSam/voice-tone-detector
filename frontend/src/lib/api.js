import { getToken } from "./auth";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

function authHeader() {
  const token = getToken();
  if (!token) throw new Error("not logged in");
  return { Authorization: `Bearer ${token}` };
}

export async function uploadBatch(files) {
  const form = new FormData();
  for (const file of files) form.append("files", file);
  const res = await fetch(`${API_BASE_URL}/batches`, {
    method: "POST",
    headers: authHeader(),
    body: form,
  });
  if (!res.ok) throw new Error(`upload failed: ${res.status} ${await res.text()}`);
  return res.json();
}

export async function listBatches() {
  const res = await fetch(`${API_BASE_URL}/batches`, { headers: authHeader() });
  if (!res.ok) throw new Error(`failed to list batches: ${res.status}`);
  return res.json();
}

export async function listBatchJobs(batchId) {
  const res = await fetch(`${API_BASE_URL}/batches/${batchId}/jobs`, {
    headers: authHeader(),
  });
  if (!res.ok) throw new Error(`failed to list jobs: ${res.status}`);
  return res.json();
}

export async function downloadBatch(batchId) {
  // A plain <a href> can't carry the Authorization header this route requires, so fetch
  // the CSV as a blob and trigger the download client-side instead.
  const res = await fetch(`${API_BASE_URL}/batches/${batchId}/download`, {
    headers: authHeader(),
  });
  if (!res.ok) throw new Error(`download failed: ${res.status}`);
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `predictions_${batchId}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

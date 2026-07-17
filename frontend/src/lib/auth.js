// Our own auth (no Supabase Auth): the backend issues an opaque session token on
// signup/signin, which we store in localStorage and send as a Bearer token on every
// request. Deliberately simpler than an httpOnly cookie (no CORS-credentials/CSRF
// wiring) -- a reasonable tradeoff for a small internal tool with a handful of known
// users, not a public product. A more XSS-resistant httpOnly-cookie approach is the
// natural next step if this ever needs to harden.

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
const STORAGE_KEY = "vtd_session"; // { token, user: { id, email } }

export function getStoredSession() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function storeSession(session) {
  if (session) localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
  else localStorage.removeItem(STORAGE_KEY);
}

async function postJSON(path, body) {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail ?? `request failed: ${res.status}`);
  return data;
}

export async function signIn(email, password) {
  const session = await postJSON("/auth/signin", { email, password });
  storeSession(session);
  return session;
}

export async function signUp(email, password) {
  const session = await postJSON("/auth/signup", { email, password });
  storeSession(session);
  return session;
}

export async function signOut() {
  const token = getToken();
  storeSession(null);
  if (!token) return;
  // Best-effort -- clear local state regardless of whether the network request succeeds
  // (e.g. offline signout should still log the user out locally).
  try {
    await fetch(`${API_BASE_URL}/auth/signout`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
    });
  } catch {
    // ignore
  }
}

export function getToken() {
  return getStoredSession()?.token ?? null;
}

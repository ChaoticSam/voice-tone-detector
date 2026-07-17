import { useState } from "react";
import { Navigate, Outlet, Route, BrowserRouter, Routes } from "react-router-dom";
import { getStoredSession } from "./lib/auth";
import Navbar from "./components/Navbar";
import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";
import History from "./pages/History";
import "./App.css";

function AuthenticatedLayout({ session, onAuthChange }) {
  return (
    <>
      <Navbar session={session} onAuthChange={onAuthChange} />
      <Outlet />
    </>
  );
}

export default function App() {
  // Synchronous (localStorage), unlike Supabase's async getSession() -- no loading
  // guard/flash-of-login-page needed before the real session state is known.
  const [session, setSession] = useState(() => getStoredSession());

  return (
    <BrowserRouter>
      <Routes>
        <Route
          path="/login"
          element={session ? <Navigate to="/" /> : <Login onAuthChange={setSession} />}
        />
        <Route
          element={
            session ? (
              <AuthenticatedLayout session={session} onAuthChange={setSession} />
            ) : (
              <Navigate to="/login" />
            )
          }
        >
          <Route path="/" element={<Dashboard />} />
          <Route path="/history" element={<History />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

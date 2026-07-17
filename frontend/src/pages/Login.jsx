import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { AlertCircle, Loader2, LogIn, Lock, Mail, UserPlus } from "lucide-react";
import { signIn, signUp } from "../lib/auth";
import IconInput from "../components/IconInput";
import Logo from "../components/Logo";

export default function Login({ onAuthChange }) {
  const [mode, setMode] = useState("signin"); // "signin" | "signup"
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const session = mode === "signin" ? await signIn(email, password) : await signUp(email, password);
      onAuthChange(session);
      navigate("/");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="login-page">
      <form className="login-form" onSubmit={handleSubmit}>
        <Logo />
        <p className="subtitle">
          {mode === "signin" ? "Sign in to access the dashboard" : "Create an account"}
        </p>
        <IconInput
          icon={Mail}
          label="Email"
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
          autoFocus
        />
        <IconInput
          icon={Lock}
          label="Password"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
          minLength={6}
        />
        {error && (
          <p className="error">
            <AlertCircle size={14} /> {error}
          </p>
        )}
        <button type="submit" className="btn" disabled={loading}>
          {loading ? (
            <Loader2 size={16} className="spin" />
          ) : mode === "signin" ? (
            <LogIn size={16} />
          ) : (
            <UserPlus size={16} />
          )}
          {loading ? "Please wait..." : mode === "signin" ? "Sign in" : "Create account"}
        </button>
        <button
          type="button"
          className="link-btn"
          onClick={() => {
            setMode(mode === "signin" ? "signup" : "signin");
            setError("");
          }}
        >
          {mode === "signin" ? "Need an account? Sign up" : "Already have an account? Sign in"}
        </button>
      </form>
    </div>
  );
}

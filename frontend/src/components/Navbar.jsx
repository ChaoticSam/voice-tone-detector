import { NavLink } from "react-router-dom";
import { LogOut } from "lucide-react";
import { signOut } from "../lib/auth";
import Logo from "./Logo";

export default function Navbar({ session, onAuthChange }) {
  async function handleSignOut() {
    await signOut();
    onAuthChange(null);
  }

  return (
    <header className="navbar">
      <Logo />
      <nav className="nav-links">
        <NavLink to="/" end className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}>
          Home
        </NavLink>
        <NavLink to="/history" className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}>
          History
        </NavLink>
      </nav>
      <div className="header-right">
        <span className="user-email">{session.user.email}</span>
        <button className="btn-secondary" onClick={handleSignOut}>
          <LogOut size={15} /> Sign out
        </button>
      </div>
    </header>
  );
}

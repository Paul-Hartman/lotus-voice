import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import { StudioPage } from "./pages/StudioPage";
import { AudiobookPage } from "./pages/AudiobookPage";
import { AncientPage } from "./pages/AncientPage";

export function App() {
  return (
    <BrowserRouter>
      <nav className="navbar navbar-expand-lg navbar-dark bg-dark border-bottom border-secondary">
        <div className="container-fluid">
          <span className="navbar-brand fw-bold">lotus-voice</span>
          <div className="navbar-nav">
            <NavLink className="nav-link" to="/">
              Studio
            </NavLink>
            <NavLink className="nav-link" to="/audiobook">
              Audiobook
            </NavLink>
            <NavLink className="nav-link" to="/ancient">
              Ancient
            </NavLink>
          </div>
        </div>
      </nav>
      <div className="container-fluid py-4">
        <Routes>
          <Route path="/" element={<StudioPage />} />
          <Route path="/audiobook" element={<AudiobookPage />} />
          <Route path="/ancient" element={<AncientPage />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

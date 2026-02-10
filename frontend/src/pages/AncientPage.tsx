import { useState } from "react";

export function AncientPage() {
  const [text, setText] = useState("ša nagba īmuru");
  const [language, setLanguage] = useState("akkadian");
  const [ipa, setIpa] = useState<string | null>(null);
  const [ssml, setSsml] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const transliterate = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/ancient/transliterate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, language }),
      });
      const data = await res.json();
      setIpa(data.ipa);
    } catch {
      // handle error
    } finally {
      setLoading(false);
    }
  };

  const generateSsml = async () => {
    try {
      const res = await fetch("/api/ancient/ssml", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, language }),
      });
      const data = await res.json();
      setIpa(data.ipa);
      setSsml(data.ssml);
    } catch {
      // handle error
    }
  };

  return (
    <div className="row">
      <div className="col-lg-8">
        <h2 className="mb-4">Ancient Language Lab</h2>
        <p className="text-muted">
          Convert Sumerian and Akkadian transliterations to IPA pronunciation
          for audio synthesis.
        </p>

        <div className="mb-3">
          <label className="form-label">Transliteration</label>
          <textarea
            className="form-control bg-dark text-light border-secondary font-monospace"
            rows={3}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter transliterated text..."
          />
        </div>

        <div className="row mb-3">
          <div className="col-auto">
            <select
              className="form-select bg-dark text-light border-secondary"
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
            >
              <option value="akkadian">Akkadian</option>
              <option value="sumerian">Sumerian</option>
            </select>
          </div>
          <div className="col-auto">
            <button
              className="btn btn-primary me-2"
              onClick={transliterate}
              disabled={loading}
            >
              {loading ? "Converting..." : "Convert to IPA"}
            </button>
            <button
              className="btn btn-outline-secondary"
              onClick={generateSsml}
            >
              Generate SSML
            </button>
          </div>
        </div>

        {ipa && (
          <div className="card bg-dark border-success mb-3">
            <div className="card-body">
              <h6 className="card-subtitle text-muted mb-2">IPA Transcription</h6>
              <p className="card-text fs-4 font-monospace">{ipa}</p>
            </div>
          </div>
        )}

        {ssml && (
          <div className="card bg-dark border-info">
            <div className="card-body">
              <h6 className="card-subtitle text-muted mb-2">SSML Output</h6>
              <pre className="text-light mb-0" style={{ whiteSpace: "pre-wrap" }}>
                {ssml}
              </pre>
            </div>
          </div>
        )}
      </div>

      <div className="col-lg-4">
        <h5 className="text-muted">Quick Reference</h5>
        <div className="card bg-dark border-secondary mb-3">
          <div className="card-body">
            <h6>Gilgamesh I.1</h6>
            <p className="font-monospace mb-1">sha nagba imuru</p>
            <p className="text-muted small">"He who saw the deep"</p>
          </div>
        </div>
        <div className="card bg-dark border-secondary">
          <div className="card-body">
            <h6>Special Characters</h6>
            <table className="table table-dark table-sm mb-0">
              <tbody>
                <tr><td>sh</td><td>= /ʃ/</td></tr>
                <tr><td>q</td><td>= /kʼ/ (ejective)</td></tr>
                <tr><td>long vowels</td><td>= aa, ii, uu or macron</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

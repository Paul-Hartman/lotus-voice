import { useState, FormEvent } from "react";

interface SynthesisFormProps {
  onSynthesize: (text: string, backend: string) => Promise<void>;
  loading: boolean;
  backends?: string[];
}

export function SynthesisForm({
  onSynthesize,
  loading,
  backends = ["bark", "edge_tts", "espeak"],
}: SynthesisFormProps) {
  const [text, setText] = useState("");
  const [backend, setBackend] = useState(backends[0] || "bark");

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (text.trim()) {
      onSynthesize(text, backend);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <div className="mb-3">
        <label className="form-label">Text to synthesize</label>
        <textarea
          className="form-control bg-dark text-light border-secondary"
          rows={4}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter text to convert to speech..."
        />
      </div>
      <div className="row mb-3">
        <div className="col-auto">
          <label className="form-label">Backend</label>
          <select
            className="form-select bg-dark text-light border-secondary"
            value={backend}
            onChange={(e) => setBackend(e.target.value)}
          >
            {backends.map((b) => (
              <option key={b} value={b}>
                {b}
              </option>
            ))}
          </select>
        </div>
        <div className="col-auto d-flex align-items-end">
          <button
            type="submit"
            className="btn btn-primary"
            disabled={loading || !text.trim()}
          >
            {loading ? (
              <>
                <span className="spinner-border spinner-border-sm me-2" />
                Generating...
              </>
            ) : (
              "Synthesize"
            )}
          </button>
        </div>
      </div>
    </form>
  );
}

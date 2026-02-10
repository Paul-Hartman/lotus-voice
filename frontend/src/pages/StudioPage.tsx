import { useState, useEffect } from "react";
import { SynthesisForm } from "../components/SynthesisForm";
import { AudioPlayer } from "../components/AudioPlayer";

export function StudioPage() {
  const [loading, setLoading] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [backends, setBackends] = useState<string[]>([]);

  useEffect(() => {
    fetch("/api/backends")
      .then((r) => r.json())
      .then((data) => setBackends(data.map((b: { name: string }) => b.name)))
      .catch(() => setBackends(["bark", "edge_tts", "espeak"]));
  }, []);

  const handleSynthesize = async (text: string, backend: string) => {
    setLoading(true);
    setError(null);
    setAudioUrl(null);

    try {
      const res = await fetch("/api/synthesize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, backend }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || "Synthesis failed");
      }

      const blob = await res.blob();
      setAudioUrl(URL.createObjectURL(blob));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="row">
      <div className="col-lg-8">
        <h2 className="mb-4">Voice Studio</h2>
        <SynthesisForm
          onSynthesize={handleSynthesize}
          loading={loading}
          backends={backends.length ? backends : undefined}
        />
        {error && (
          <div className="alert alert-danger mt-3">{error}</div>
        )}
        <AudioPlayer src={audioUrl} label="Generated audio" />
      </div>
      <div className="col-lg-4">
        <h5 className="text-muted">Available Backends</h5>
        <ul className="list-group list-group-flush">
          {backends.map((b) => (
            <li key={b} className="list-group-item bg-dark text-light border-secondary">
              {b}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

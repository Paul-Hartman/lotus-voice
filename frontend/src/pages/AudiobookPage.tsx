import { useState } from "react";

interface AudiobookProject {
  id: string;
  title: string;
  status: string;
}

export function AudiobookPage() {
  const [title, setTitle] = useState("");
  const [projects, setProjects] = useState<AudiobookProject[]>([]);
  const [creating, setCreating] = useState(false);

  const createProject = async () => {
    if (!title.trim()) return;
    setCreating(true);

    try {
      const res = await fetch("/api/audiobook/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title }),
      });
      const project = await res.json();
      setProjects((prev) => [...prev, project]);
      setTitle("");
    } catch {
      // handle error
    } finally {
      setCreating(false);
    }
  };

  return (
    <div>
      <h2 className="mb-4">Audiobook Producer</h2>

      <div className="card bg-dark border-secondary mb-4">
        <div className="card-body">
          <h5 className="card-title">New Audiobook Project</h5>
          <div className="input-group">
            <input
              type="text"
              className="form-control bg-dark text-light border-secondary"
              placeholder="Audiobook title..."
              value={title}
              onChange={(e) => setTitle(e.target.value)}
            />
            <button
              className="btn btn-primary"
              onClick={createProject}
              disabled={creating || !title.trim()}
            >
              {creating ? "Creating..." : "Create Project"}
            </button>
          </div>
        </div>
      </div>

      {projects.length > 0 && (
        <div className="card bg-dark border-secondary">
          <div className="card-body">
            <h5 className="card-title">Projects</h5>
            <table className="table table-dark table-striped">
              <thead>
                <tr>
                  <th>Title</th>
                  <th>ID</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {projects.map((p) => (
                  <tr key={p.id}>
                    <td>{p.title}</td>
                    <td><code>{p.id}</code></td>
                    <td>
                      <span className="badge bg-info">{p.status}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

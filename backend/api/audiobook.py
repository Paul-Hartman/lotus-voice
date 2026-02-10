"""Audiobook production API endpoints."""

import uuid

from config import AUDIOBOOK_OUTPUT_DIR
from flask import Blueprint, jsonify, request

audiobook_bp = Blueprint("audiobook", __name__)

# In-memory project tracking (replace with DB later)
_projects: dict = {}


@audiobook_bp.route("/api/audiobook/create", methods=["POST"])
def create_audiobook():
    """Create a new audiobook project.

    Request JSON:
        title: str - Audiobook title
        text: str (optional) - Full text content
        source_file: str (optional) - Path to source text file
        backend: str (optional) - TTS backend preference
    """
    data = request.get_json(force=True)
    title = data.get("title", "untitled")

    project_id = uuid.uuid4().hex[:12]
    output_dir = AUDIOBOOK_OUTPUT_DIR / project_id
    output_dir.mkdir(parents=True, exist_ok=True)

    project = {
        "id": project_id,
        "title": title,
        "status": "created",
        "output_dir": str(output_dir),
        "chapters": [],
    }
    _projects[project_id] = project

    return jsonify(project), 201


@audiobook_bp.route("/api/audiobook/<project_id>")
def get_audiobook(project_id: str):
    """Get audiobook project status."""
    project = _projects.get(project_id)
    if not project:
        return jsonify({"error": "Project not found"}), 404
    return jsonify(project)


@audiobook_bp.route("/api/audiobook/list")
def list_audiobooks():
    """List all audiobook projects."""
    return jsonify(list(_projects.values()))

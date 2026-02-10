"""Speech synthesis API endpoint."""

import uuid

from config import AUDIOBOOK_OUTPUT_DIR
from core.tts_manager import TTSBackend, TTSManager
from flask import Blueprint, jsonify, request, send_file

synthesis_bp = Blueprint("synthesis", __name__)

# Lazy-initialized manager
_manager: TTSManager | None = None


def get_manager() -> TTSManager:
    global _manager
    if _manager is None:
        _manager = TTSManager()
    return _manager


@synthesis_bp.route("/api/synthesize", methods=["POST"])
def synthesize():
    """Synthesize speech from text.

    Request JSON:
        text: str - Text to synthesize
        backend: str (optional) - Backend name (bark, edge_tts, espeak, etc.)
        voice_id: str (optional) - Voice identifier
        emotion: str (optional) - Emotion tag
    """
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400

    backend_name = data.get("backend")
    voice_id = data.get("voice_id")
    emotion = data.get("emotion")

    manager = get_manager()

    # Resolve backend
    backend_enum = None
    if backend_name:
        try:
            backend_enum = TTSBackend(backend_name)
        except ValueError:
            return jsonify({"error": f"Unknown backend: {backend_name}"}), 400

    # Generate to temp file
    output_dir = AUDIOBOOK_OUTPUT_DIR / "tmp"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{uuid.uuid4().hex}.wav"

    success = manager.synthesize(
        text=text,
        output_path=output_path,
        backend=backend_enum,
        voice_id=voice_id,
        emotion=emotion,
    )

    if not success or not output_path.exists():
        return jsonify({"error": "Synthesis failed"}), 500

    return send_file(
        output_path,
        mimetype="audio/wav",
        as_attachment=True,
        download_name="synthesis.wav",
    )

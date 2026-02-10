"""TTS backends discovery API."""

from core.tts_manager import TTSManager
from flask import Blueprint, jsonify

backends_bp = Blueprint("backends", __name__)

_manager: TTSManager | None = None


def get_manager() -> TTSManager:
    global _manager
    if _manager is None:
        _manager = TTSManager()
    return _manager


@backends_bp.route("/api/backends")
def list_backends():
    """List all available TTS backends and their capabilities."""
    manager = get_manager()
    backends = []
    for name, backend in manager.backends.items():
        backends.append({
            "name": name.value,
            "supports_emotion": backend.supports_emotion,
            "supports_voice_cloning": backend.supports_voice_cloning,
            "supports_multispeaker": backend.supports_multispeaker,
            "voices": backend.list_voices(),
        })
    return jsonify(backends)

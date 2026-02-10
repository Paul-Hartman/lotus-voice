"""Voice library API endpoints."""

from config import VOICE_LIBRARY_PATH
from core.voice_library import VoiceGender, VoiceLibrary
from flask import Blueprint, jsonify, request

voices_bp = Blueprint("voices", __name__)

_library: VoiceLibrary | None = None


def get_library() -> VoiceLibrary:
    global _library
    if _library is None:
        _library = VoiceLibrary(VOICE_LIBRARY_PATH)
    return _library


@voices_bp.route("/api/voices")
def list_voices():
    """List all voice profiles."""
    library = get_library()
    voices = []
    for voice in library.voices.values():
        voices.append({
            "voice_id": voice.voice_id,
            "name": voice.name,
            "gender": voice.gender.value,
            "archetype": voice.archetype.value if voice.archetype else None,
            "description": voice.description,
            "backend": voice.backend,
            "tags": voice.tags,
        })
    return jsonify(voices)


@voices_bp.route("/api/voices/<voice_id>")
def get_voice(voice_id: str):
    """Get a specific voice profile."""
    library = get_library()
    voice = library.get_voice(voice_id)
    if not voice:
        return jsonify({"error": "Voice not found"}), 404
    return jsonify({
        "voice_id": voice.voice_id,
        "name": voice.name,
        "gender": voice.gender.value,
        "archetype": voice.archetype.value if voice.archetype else None,
        "description": voice.description,
        "backend": voice.backend,
        "tags": voice.tags,
        "characteristics": voice.characteristics,
    })


@voices_bp.route("/api/voices", methods=["POST"])
def add_voice():
    """Add a new voice profile."""
    data = request.get_json(force=True)
    name = data.get("name")
    gender = data.get("gender", "neutral")

    if not name:
        return jsonify({"error": "name is required"}), 400

    library = get_library()
    voice_id = data.get("voice_id", name.lower().replace(" ", "_"))

    voice = library.add_voice(
        voice_id=voice_id,
        name=name,
        gender=VoiceGender(gender),
        description=data.get("description", ""),
        backend=data.get("backend", "bark"),
        tags=data.get("tags", []),
    )

    return jsonify({"voice_id": voice.voice_id, "name": voice.name}), 201

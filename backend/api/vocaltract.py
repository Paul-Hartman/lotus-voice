"""Vocal tract simulator API blueprint.

Provides endpoints for:
- IPA/text → audio synthesis
- Single phone synthesis with state return
- Direct articulatory parameter control
- Vocal tract shape queries
- Glottal pulse generation
- IPA mapping database
"""

import logging
from flask import Blueprint, Response, jsonify, request

logger = logging.getLogger(__name__)

vocaltract_bp = Blueprint("vocaltract", __name__, url_prefix="/api/vocaltract")

# Lazy-loaded backend
_backend = None


def _get_backend():
    """Lazy-load the vocal tract backend."""
    global _backend
    if _backend is None:
        from core.backends.vocaltract_backend import VocalTractBackend
        _backend = VocalTractBackend()
    return _backend


@vocaltract_bp.route("/synthesize", methods=["POST"])
def synthesize():
    """Synthesize audio from IPA string.

    Request JSON:
        ipa: str - IPA transcription (required)
        f0: float - Fundamental frequency (optional, default 120)
        Rd: float - Voice quality (optional, default 1.0)
        phone_duration: float - Default phone duration in sec (optional, default 0.15)

    Returns:
        WAV audio file (audio/wav)
    """
    data = request.get_json(silent=True) or {}
    ipa = data.get("ipa")
    if not ipa:
        return jsonify({"error": "Missing 'ipa' field"}), 400

    backend = _get_backend()
    wav_bytes = backend.synthesize_to_bytes(
        ipa,
        f0=data.get("f0"),
        Rd=data.get("Rd"),
        phone_duration=data.get("phone_duration", 0.15),
    )

    if wav_bytes is None:
        return jsonify({"error": "Synthesis failed"}), 500

    return Response(wav_bytes, mimetype="audio/wav")


@vocaltract_bp.route("/phone", methods=["POST"])
def synthesize_phone():
    """Synthesize a single phone and return audio + state.

    Request JSON:
        phone: str - IPA symbol (required)
        duration: float - Duration in seconds (optional, default 0.3)
        f0: float - Fundamental frequency (optional)
        Rd: float - Voice quality (optional)

    Returns JSON:
        wav_base64: str - Base64-encoded WAV audio
        states: list - State snapshots at 100Hz
        formants: list - Estimated formant frequencies
    """
    import base64

    data = request.get_json(silent=True) or {}
    phone = data.get("phone")
    if not phone:
        return jsonify({"error": "Missing 'phone' field"}), 400

    backend = _get_backend()
    result = backend.synthesize_phone(
        phone,
        duration=data.get("duration", 0.3),
        f0=data.get("f0"),
        Rd=data.get("Rd"),
    )

    if result is None:
        return jsonify({"error": f"Failed to synthesize phone '{phone}'"}), 500

    return jsonify({
        "phone": phone,
        "wav_base64": base64.b64encode(result["wav_bytes"]).decode("ascii"),
        "states": result["states"],
        "formants": result["formants"],
    })


@vocaltract_bp.route("/articulate", methods=["POST"])
def articulate():
    """Synthesize from direct articulatory parameters.

    Request JSON:
        jaw: float (-3 to 3)
        tongue_dorsal_pos: float (-3 to 3)
        tongue_dorsal_shape: float (-3 to 3)
        tongue_tip: float (-3 to 3)
        lip_height: float (-3 to 3)
        lip_protrusion: float (-3 to 3)
        larynx_height: float (-3 to 3)
        velum: float (0 to 1)
        f0: float (optional, default 120)
        Rd: float (optional, default 1.0)
        duration: float (optional, default 0.3)

    Returns:
        WAV audio file (audio/wav)
    """
    data = request.get_json(silent=True) or {}

    backend = _get_backend()
    backend._ensure_loaded()

    from vocaltract.area_function import articulators_to_area_function
    from vocaltract.articulators import ArticulatorTarget

    target = ArticulatorTarget(
        jaw=data.get("jaw", 0.0),
        tongue_dorsal_pos=data.get("tongue_dorsal_pos", 0.0),
        tongue_dorsal_shape=data.get("tongue_dorsal_shape", 0.0),
        tongue_tip=data.get("tongue_tip", 0.0),
        lip_height=data.get("lip_height", 0.0),
        lip_protrusion=data.get("lip_protrusion", 0.0),
        larynx_height=data.get("larynx_height", 0.0),
        velum=data.get("velum", 0.0),
        f0=data.get("f0", 120.0),
        Rd=data.get("Rd", 1.0),
    )

    audio, states = backend._synthesizer.synthesize_with_articulators(
        target, duration_sec=data.get("duration", 0.3),
    )

    wav_bytes = backend._synthesizer.audio_to_wav_bytes(audio)
    return Response(wav_bytes, mimetype="audio/wav")


@vocaltract_bp.route("/tract-shape", methods=["POST"])
def tract_shape():
    """Query the area function for a phone or articulatory config.

    Request JSON (one of):
        phone: str - IPA symbol
        OR
        jaw, tongue_dorsal_pos, etc. - Direct articulatory params

    Returns JSON:
        area_function_cm2: list[float] - 44-section area function
        constriction: dict - Primary constriction info
        formants: list[float] - Estimated formant frequencies
        articulators: dict - Articulatory parameters used
    """
    data = request.get_json(silent=True) or {}

    backend = _get_backend()
    backend._ensure_loaded()

    phone = data.get("phone")
    if phone:
        info = backend.get_phone_info(phone)
        if info is None:
            return jsonify({"error": f"Unknown phone '{phone}'"}), 404
        return jsonify(info)

    # Direct articulatory params
    from vocaltract.area_function import articulators_to_area_function, find_constriction
    from vocaltract.articulators import ArticulatorTarget

    target = ArticulatorTarget(
        jaw=data.get("jaw", 0.0),
        tongue_dorsal_pos=data.get("tongue_dorsal_pos", 0.0),
        tongue_dorsal_shape=data.get("tongue_dorsal_shape", 0.0),
        tongue_tip=data.get("tongue_tip", 0.0),
        lip_height=data.get("lip_height", 0.0),
        lip_protrusion=data.get("lip_protrusion", 0.0),
        larynx_height=data.get("larynx_height", 0.0),
        velum=data.get("velum", 0.0),
    )

    areas = articulators_to_area_function(target)
    backend._synthesizer.tube.set_area_function(areas)
    constriction = find_constriction(areas)
    formants = backend._synthesizer.tube.estimate_formants()

    result = {
        "area_function_cm2": areas.tolist(),
        "formants": formants,
        "articulators": {
            "jaw": target.jaw,
            "tongue_dorsal_pos": target.tongue_dorsal_pos,
            "tongue_dorsal_shape": target.tongue_dorsal_shape,
            "tongue_tip": target.tongue_tip,
            "lip_height": target.lip_height,
            "lip_protrusion": target.lip_protrusion,
            "larynx_height": target.larynx_height,
            "velum": target.velum,
        },
    }

    if constriction:
        result["constriction"] = {
            "section": constriction.section,
            "area_cm2": constriction.area_cm2,
            "position_cm": constriction.position_cm,
            "region": constriction.region,
        }

    return jsonify(result)


@vocaltract_bp.route("/glottal-pulse", methods=["POST"])
def glottal_pulse():
    """Generate a single glottal pulse waveform.

    Request JSON:
        f0: float - Fundamental frequency (default 120)
        Rd: float - Voice quality parameter (default 1.0)

    Returns JSON:
        pulse: list[float] - One cycle of the LF glottal pulse
        f0: float - Actual f0 used
        Rd: float - Actual Rd used
        period_ms: float - Pulse period in milliseconds
        oq: float - Open quotient
    """
    import base64

    data = request.get_json(silent=True) or {}
    f0 = data.get("f0", 120.0)
    Rd = data.get("Rd", 1.0)

    backend = _get_backend()
    backend._ensure_loaded()

    source = backend._synthesizer.source
    source.set_params(f0=f0, Rd=Rd)
    cycle = source.generate_cycle()

    return jsonify({
        "pulse": cycle.tolist(),
        "f0": f0,
        "Rd": Rd,
        "period_ms": 1000.0 / f0,
        "oq": source.state.oq,
        "speed_quotient": source.state.speed_quotient,
        "num_samples": len(cycle),
    })


@vocaltract_bp.route("/sing", methods=["POST"])
def sing():
    """Synthesize singing from a sequence of notes.

    Request JSON:
        notes: list of objects, each with:
            phone: str - IPA vowel/consonant
            midi_note: int - MIDI note number (60 = middle C)
            duration: float - Duration in seconds
            register: str - "chest", "head", "falsetto", "mixed" (optional)
            vibrato: str - Vibrato preset name or null (optional)
            dynamics: float - 0=pp, 1=ff (optional, default 0.7)
        voice_type: str - "soprano", "tenor", "bass", etc. (optional)

    Returns:
        WAV audio file (audio/wav)
    """
    data = request.get_json(silent=True) or {}
    notes = data.get("notes", [])
    if not notes:
        return jsonify({"error": "Missing 'notes' array"}), 400

    backend = _get_backend()
    backend._ensure_loaded()

    from vocaltract.singing import (
        SingingController,
        SingingNote,
        VibratoConfig,
        VocalRegister,
        midi_to_hz,
    )

    controller = SingingController(backend._synthesizer.sample_rate)

    # Load vibrato presets
    import json
    from pathlib import Path
    presets_path = Path(__file__).parent.parent.parent / "data" / "vocaltract" / "singing_presets.json"
    vibrato_presets = {}
    if presets_path.exists():
        with open(presets_path) as f:
            presets_data = json.load(f)
        vibrato_presets = presets_data.get("vibrato_presets", {})

    all_audio = []

    for note_data in notes:
        phone = note_data.get("phone", "a")
        midi_note = note_data.get("midi_note", 60)
        duration = note_data.get("duration", 0.5)
        register_name = note_data.get("register", "chest")
        vibrato_name = note_data.get("vibrato", "classical")
        dynamics = note_data.get("dynamics", 0.7)

        f0 = midi_to_hz(midi_note)

        # Get vibrato config
        vibrato = None
        if vibrato_name and vibrato_name in vibrato_presets:
            vp = vibrato_presets[vibrato_name]
            vibrato = VibratoConfig(
                rate_hz=vp.get("rate_hz", 5.0),
                extent_semitones=vp.get("extent_semitones", 1.0),
                delay_sec=vp.get("delay_sec", 0.2),
                rise_time_sec=vp.get("rise_time_sec", 0.15),
            )

        # Generate vibrato f0 contour (sample-rate resolution)
        if vibrato and vibrato.rate_hz > 0:
            f0_contour = controller.generate_vibrato_f0(f0, duration, vibrato)
            # Use the mean f0 for synthesis (vibrato will be approximated)
            mean_f0 = float(f0_contour.mean())
        else:
            mean_f0 = f0

        # Get register Rd
        try:
            register = VocalRegister(register_name)
        except ValueError:
            register = VocalRegister.CHEST

        from vocaltract.singing import REGISTER_CONFIGS
        config = REGISTER_CONFIGS.get(register, REGISTER_CONFIGS[VocalRegister.CHEST])
        Rd = config.Rd

        audio, _ = backend._synthesizer.synthesize_phone(
            phone, duration_sec=duration, f0=mean_f0, Rd=Rd,
        )
        all_audio.append(audio)

    if not all_audio:
        return jsonify({"error": "No audio generated"}), 500

    import numpy as np
    combined = np.concatenate(all_audio)
    wav_bytes = backend._synthesizer.audio_to_wav_bytes(combined)

    return Response(wav_bytes, mimetype="audio/wav")


@vocaltract_bp.route("/ipa-map", methods=["GET"])
def ipa_map():
    """Get the complete IPA-to-articulation mapping database.

    Returns JSON:
        vowels: list[str] - Available vowel phones
        consonants: list[str] - Available consonant phones
    """
    backend = _get_backend()
    return jsonify(backend.get_ipa_map())

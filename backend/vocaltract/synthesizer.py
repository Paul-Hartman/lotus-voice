"""Top-level vocal tract synthesizer: IPA in -> audio + state out.

Orchestrates the source-filter pipeline:
  1. IPA phone → articulatory target (Maeda 7-param)
  2. Articulatory target → area function (PCA basis)
  3. Glottal source generates excitation signal
  4. Tube model filters through the vocal tract shape
  5. Nasal tract processes side-branch (if velum open)
  6. Radiation filter converts volume velocity to pressure
  7. State snapshots captured for introspection

Phase 1: Preset area functions for cardinal vowels.
Phase 2: Articulatory parameters -> area functions -> synthesis.
Phase 3+: Connected speech with coarticulation.
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from vocaltract.area_function import articulators_to_area_function, find_constriction
from vocaltract.articulators import ArticulatorTarget
from vocaltract.glottal_source import GlottalSource
from vocaltract.ipa_to_articulation import IPAToArticulation, parse_ipa_to_phones
from vocaltract.nasal_tract import COUPLING_SECTION, NasalTract
from vocaltract.radiation import RadiationFilter
from vocaltract.state import (
    ArticulatorState,
    GlottalState,
    SubglottalState,
    TubeState,
    VocalTractState,
)
from vocaltract.tube_model import TubeModel

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 44100
DEFAULT_F0 = 120.0
DEFAULT_RD = 1.0

# Path to data files (relative to project root)
_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "vocaltract"


def _load_presets() -> Dict:
    """Load tube geometry presets from JSON."""
    preset_path = _DATA_DIR / "tube_geometry_presets.json"
    if not preset_path.exists():
        logger.warning(f"Preset file not found: {preset_path}")
        return {}
    with open(preset_path) as f:
        return json.load(f)


class VocalTractSynthesizer:
    """Research-grade articulatory synthesis engine.

    Synthesizes audio from IPA phones using a physical model of the
    human vocal tract. Returns both audio and full state snapshots
    for educational introspection.

    Pipeline: IPA → articulators → area function → waveguide → audio
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ):
        self.sample_rate = sample_rate
        # Area functions are always 44-element (Maeda PCA model)
        # The tube model uses 23 waveguide sections and auto-resamples
        self.num_sections = 44

        # Core modules
        self.source = GlottalSource(sample_rate=sample_rate)
        self.tube = TubeModel(sample_rate=sample_rate)  # Uses default 23 sections
        self.nasal = NasalTract(sample_rate=sample_rate)
        self.radiation = RadiationFilter()
        self.nasal_radiation = RadiationFilter(radiation_coefficient=0.95)

        # Articulatory pathway (Phase 2)
        self.ipa_mapper = IPAToArticulation()

        # Presets (Phase 1 fallback)
        self._presets = _load_presets()
        self._vowel_presets = self._presets.get("vowels", {})

        # Default voice parameters
        self._f0 = DEFAULT_F0
        self._Rd = DEFAULT_RD

    def set_voice(self, f0: float = 120.0, Rd: float = 1.0,
                  phonation_type: str = "modal") -> None:
        """Set voice parameters.

        Args:
            f0: Fundamental frequency in Hz (80-400 typical)
            Rd: Voice quality (0.3=pressed, 1.0=modal, 2.7=breathy)
            phonation_type: Label string
        """
        self._f0 = f0
        self._Rd = Rd
        self.source.set_params(f0=f0, Rd=Rd, phonation_type=phonation_type)

    def get_articulator_target(self, phone: str) -> Optional[ArticulatorTarget]:
        """Get the articulatory target for an IPA phone.

        Args:
            phone: IPA symbol

        Returns:
            ArticulatorTarget or None
        """
        return self.ipa_mapper.get_target(phone)

    def get_area_function(self, phone: str) -> Tuple[Optional[np.ndarray], Optional[ArticulatorTarget]]:
        """Compute area function for an IPA phone.

        Uses optimized preset area functions for known vowels (these are
        numerically optimized to produce correct Peterson & Barney formants).
        Falls back to the PCA articulatory model for other phones.

        Args:
            phone: IPA symbol or vowel letter

        Returns:
            Tuple of (area_function, articulator_target)
            area_function may be None if phone is unknown.
        """
        # Try optimized preset first (vowels have formant-accurate area functions)
        areas = self._lookup_preset(phone)
        if areas is not None:
            # Still get the articulatory target for state snapshots
            target = self.ipa_mapper.get_target(phone)
            return areas, target

        # Articulatory pathway for consonants and non-preset vowels
        target = self.ipa_mapper.get_target(phone)
        if target is not None:
            areas = articulators_to_area_function(target, self.num_sections)
            return areas, target

        return None, None

    def _lookup_preset(self, phone: str) -> Optional[np.ndarray]:
        """Look up preset area function (Phase 1 fallback)."""
        if phone in self._vowel_presets:
            return np.array(self._vowel_presets[phone]["areas"], dtype=np.float64)
        for key, preset in self._vowel_presets.items():
            if preset.get("ipa") == phone:
                return np.array(preset["areas"], dtype=np.float64)
        if phone in ("ə", "schwa", "neutral"):
            neutral = self._presets.get("neutral", {})
            if "areas" in neutral:
                return np.array(neutral["areas"], dtype=np.float64)
        return None

    def synthesize_phone(
        self,
        phone: str,
        duration_sec: float = 0.3,
        f0: Optional[float] = None,
        Rd: Optional[float] = None,
        aspiration: float = 0.0,
        area_function: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[VocalTractState]]:
        """Synthesize a single IPA phone.

        Args:
            phone: IPA symbol (e.g., "a", "i", "ɑ", "ʃ", "n")
            duration_sec: Duration in seconds
            f0: Override fundamental frequency
            Rd: Override voice quality
            aspiration: Override aspiration noise level (0.0-1.0)
            area_function: Override area function (44 values in cm²)

        Returns:
            Tuple of (audio_samples, state_snapshots)
        """
        # Resolve area function and articulatory target
        target = None
        if area_function is None:
            area_function, target = self.get_area_function(phone)
        if area_function is None:
            logger.warning(f"No area function for phone '{phone}', using neutral")
            area_function = np.ones(self.num_sections, dtype=np.float64)

        # Determine glottal parameters from target (or defaults)
        phone_f0 = f0 or (target.f0 if target and target.f0 else self._f0)
        phone_Rd = Rd or (target.Rd if target and target.Rd else self._Rd)
        phone_aspiration = aspiration or (target.aspiration if target else 0.0)
        phonation_type = target.phonation_type if target else "modal"
        noise_section = target.noise_source_section if target else None
        noise_amplitude = target.noise_amplitude if target else 0.0
        velum = target.velum if target else 0.0

        # Handle voiceless sounds — use noise excitation only
        is_voiceless = phonation_type == "voiceless"

        # Set source parameters
        if not is_voiceless:
            self.source.set_params(f0=phone_f0, Rd=phone_Rd, phonation_type=phonation_type)

        # Set tract shape
        self.tube.set_area_function(area_function)

        # Set nasal coupling
        self.nasal.set_velum(velum)

        num_samples = int(duration_sec * self.sample_rate)
        state_interval = self.sample_rate // 100  # 100 Hz state snapshots

        # Reset for clean synthesis
        self.tube.reset()
        self.nasal.reset()
        self.nasal.set_velum(velum)
        self.radiation.reset()
        self.nasal_radiation.reset()

        # Generate
        audio = np.zeros(num_samples, dtype=np.float64)
        states: List[VocalTractState] = []

        # Build articulator state for snapshots
        art_state = ArticulatorState()
        if target is not None:
            art_state = ArticulatorState(
                jaw=target.jaw,
                tongue_dorsal_pos=target.tongue_dorsal_pos,
                tongue_dorsal_shape=target.tongue_dorsal_shape,
                tongue_tip=target.tongue_tip,
                lip_height=target.lip_height,
                lip_protrusion=target.lip_protrusion,
                larynx_height=target.larynx_height,
                velum=target.velum,
            )

        # Process sample-by-sample for nasal coupling
        use_nasal = velum > 0.01
        oral_area_at_coupling = float(area_function[COUPLING_SECTION]) if COUPLING_SECTION < len(area_function) else 1.0

        chunk_size = 512
        pos = 0

        while pos < num_samples:
            end = min(pos + chunk_size, num_samples)
            n = end - pos

            # Generate excitation
            if is_voiceless:
                source_samples = np.zeros(n, dtype=np.float64)
            else:
                source_samples = self.source.generate_samples(n)
                if phone_aspiration > 0:
                    source_samples = self.source.add_noise(source_samples, phone_aspiration)

            # Add fricative noise at constriction point
            if noise_section is not None and noise_amplitude > 0:
                noise = np.random.randn(n) * noise_amplitude * 0.3
                # Inject noise at the constriction section
                # (simplified: add to source with spectral shaping)
                source_samples = source_samples + noise

            if use_nasal:
                # Sample-by-sample for nasal coupling
                for j in range(n):
                    # Process oral tract
                    oral_out = self.tube.process_sample(source_samples[j])

                    # Nasal coupling at the velum section
                    oral_fwd = self.tube.forward[COUPLING_SECTION]
                    oral_bwd = self.tube.backward[COUPLING_SECTION]
                    oral_fwd_out, oral_bwd_out, nasal_in = self.nasal.compute_coupling(
                        oral_fwd, oral_bwd, oral_area_at_coupling,
                    )
                    # Feed back modified waves
                    self.tube.forward[COUPLING_SECTION] = oral_fwd_out
                    self.tube.backward[COUPLING_SECTION] = oral_bwd_out

                    # Process nasal branch
                    nasal_out = self.nasal.process_sample(nasal_in)

                    # Radiate both outputs
                    oral_rad = self.radiation.process_sample(oral_out)
                    nasal_rad = self.nasal_radiation.process_sample(nasal_out)

                    audio[pos + j] = oral_rad + nasal_rad * 0.5  # Nasal is quieter
            else:
                # No nasal — use faster block processing
                lip_output = self.tube.process_block(source_samples)
                radiated = self.radiation.process_block(lip_output)
                audio[pos:end] = radiated

            # Capture state snapshots at 100Hz
            for i in range(pos, end, state_interval):
                if i < num_samples:
                    state = VocalTractState(
                        time_sec=i / self.sample_rate,
                        phone=phone,
                        phone_progress=i / num_samples,
                        glottal=GlottalState(
                            f0=phone_f0,
                            Rd=phone_Rd,
                            oq=self.source.state.oq,
                            speed_quotient=self.source.state.speed_quotient,
                            phonation_type=phonation_type,
                        ),
                        articulators=art_state,
                        tube=TubeState(
                            num_sections=self.num_sections,
                            area_function_cm2=area_function.tolist(),
                            reflection_coefficients=self.tube.reflection_coefficients.tolist(),
                            nasal_area_function_cm2=self.nasal.areas.tolist() if use_nasal else [],
                            nasal_coupling_area_cm2=self.nasal.velum_area if use_nasal else 0.0,
                        ),
                        subglottal=SubglottalState(),
                        output_sample=audio[i] if i < len(audio) else 0.0,
                    )
                    states.append(state)

            pos = end

        # Normalize audio to prevent clipping
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.8

        # Apply gentle fade-in/out to avoid clicks
        fade_samples = min(int(0.01 * self.sample_rate), num_samples // 4)
        if fade_samples > 0:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            audio[:fade_samples] *= fade_in
            audio[-fade_samples:] *= fade_out

        # Estimate formants
        formants = self.tube.estimate_formants()
        for state in states:
            state.formants_hz = formants

        return audio, states

    def synthesize_phones(
        self,
        phones: List[str],
        durations: Optional[List[float]] = None,
        f0: Optional[float] = None,
        Rd: Optional[float] = None,
    ) -> Tuple[np.ndarray, List[VocalTractState]]:
        """Synthesize a sequence of IPA phones.

        Phase 2: Per-phone articulatory synthesis.
        Phase 3+: Will add coarticulation transitions.

        Args:
            phones: List of IPA symbols
            durations: Per-phone durations in seconds (default 0.15 each)
            f0: Fundamental frequency
            Rd: Voice quality

        Returns:
            Tuple of (audio, states)
        """
        if durations is None:
            durations = [0.15] * len(phones)

        all_audio = []
        all_states = []

        for phone, dur in zip(phones, durations):
            audio, states = self.synthesize_phone(
                phone, duration_sec=dur, f0=f0, Rd=Rd
            )
            all_audio.append(audio)
            all_states.extend(states)

        if not all_audio:
            return np.array([], dtype=np.float64), []

        combined = np.concatenate(all_audio)
        return combined, all_states

    def synthesize_ipa_string(
        self,
        ipa_string: str,
        f0: Optional[float] = None,
        Rd: Optional[float] = None,
        phone_duration: float = 0.15,
        long_vowel_multiplier: float = 1.8,
    ) -> Tuple[np.ndarray, List[VocalTractState]]:
        """Synthesize an IPA string using the articulatory pipeline.

        Parses the string into individual phones (handling digraphs
        and diacritics) and synthesizes each.

        Args:
            ipa_string: IPA transcription (e.g., "ʃa naqba iːmuru")
            f0: Fundamental frequency
            Rd: Voice quality
            phone_duration: Default phone duration in seconds
            long_vowel_multiplier: Duration multiplier for long vowels (ː)

        Returns:
            Tuple of (audio, states)
        """
        # Use the improved IPA parser
        raw_phones = parse_ipa_to_phones(ipa_string)

        phones = []
        durations = []

        for raw in raw_phones:
            if raw == " ":
                phones.append("_silence")
                durations.append(0.05)
            elif raw.endswith("ː"):
                phones.append(raw.rstrip("ː"))
                durations.append(phone_duration * long_vowel_multiplier)
            else:
                phones.append(raw)
                # Shorter duration for stops and taps
                manner = self.ipa_mapper.get_manner(raw)
                if manner == "stop":
                    durations.append(0.08)
                elif manner == "tap":
                    durations.append(0.04)
                elif manner == "affricate":
                    durations.append(0.12)
                else:
                    durations.append(phone_duration)

        # Synthesize
        all_audio = []
        all_states = []

        for phone, dur in zip(phones, durations):
            if phone == "_silence":
                silence = np.zeros(int(dur * self.sample_rate), dtype=np.float64)
                all_audio.append(silence)
            else:
                audio, states = self.synthesize_phone(
                    phone, duration_sec=dur, f0=f0, Rd=Rd
                )
                all_audio.append(audio)
                all_states.extend(states)

        if not all_audio:
            return np.array([], dtype=np.float64), []

        combined = np.concatenate(all_audio)
        return combined, all_states

    def phone_info(self, phone: str) -> Optional[Dict]:
        """Get information about a phone's vocal tract configuration.

        Returns area function, expected formants, articulatory params,
        and description without synthesizing audio.

        Args:
            phone: IPA symbol

        Returns:
            Dict with area function and metadata, or None
        """
        areas, target = self.get_area_function(phone)

        if areas is None:
            return None

        self.tube.set_area_function(areas)
        constriction = find_constriction(areas)

        info = {
            "phone": phone,
            "estimated_formants_hz": self.tube.estimate_formants(),
            "area_function_cm2": areas.tolist(),
            "num_sections": self.num_sections,
            "tract_length_cm": self.tube.get_total_length_cm(),
            "source": "articulatory" if target else "preset",
        }

        # Add articulatory details if available
        if target is not None:
            info["articulators"] = {
                "jaw": target.jaw,
                "tongue_dorsal_pos": target.tongue_dorsal_pos,
                "tongue_dorsal_shape": target.tongue_dorsal_shape,
                "tongue_tip": target.tongue_tip,
                "lip_height": target.lip_height,
                "lip_protrusion": target.lip_protrusion,
                "larynx_height": target.larynx_height,
                "velum": target.velum,
            }

        if constriction:
            info["constriction"] = {
                "section": constriction.section,
                "area_cm2": constriction.area_cm2,
                "position_cm": constriction.position_cm,
                "region": constriction.region,
            }

        # Add preset info if it exists
        for key, preset in self._vowel_presets.items():
            if key == phone or preset.get("ipa") == phone:
                info["ipa"] = preset.get("ipa", phone)
                info["description"] = preset.get("description", "")
                info["expected_formants_hz"] = preset.get("expected_formants_hz", {})
                break

        return info

    def synthesize_with_articulators(
        self,
        target: ArticulatorTarget,
        duration_sec: float = 0.3,
        f0: Optional[float] = None,
        Rd: Optional[float] = None,
    ) -> Tuple[np.ndarray, List[VocalTractState]]:
        """Synthesize directly from articulatory parameters.

        Bypasses IPA lookup — useful for exploring the parameter space
        or implementing custom articulations.

        Args:
            target: ArticulatorTarget with all parameters set
            duration_sec: Duration in seconds
            f0: Override f0 (or use target.f0)
            Rd: Override Rd (or use target.Rd)

        Returns:
            Tuple of (audio, states)
        """
        areas = articulators_to_area_function(target, self.num_sections)
        use_f0 = f0 or target.f0 or self._f0
        use_Rd = Rd or target.Rd or self._Rd

        return self.synthesize_phone(
            phone="[direct]",
            duration_sec=duration_sec,
            f0=use_f0,
            Rd=use_Rd,
            aspiration=target.aspiration,
            area_function=areas,
        )

    def state_to_dict(self, state: VocalTractState) -> Dict:
        """Convert a VocalTractState to a JSON-serializable dict."""
        return asdict(state)

    def audio_to_wav_bytes(self, audio: np.ndarray) -> bytes:
        """Convert audio array to WAV file bytes.

        Args:
            audio: Float64 audio samples normalized to [-1, 1]

        Returns:
            WAV file as bytes
        """
        import io
        import struct

        # Convert to 16-bit PCM
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

        buf = io.BytesIO()
        num_samples = len(audio_int16)
        data_size = num_samples * 2  # 16-bit = 2 bytes per sample

        # WAV header
        buf.write(b'RIFF')
        buf.write(struct.pack('<I', 36 + data_size))
        buf.write(b'WAVE')
        buf.write(b'fmt ')
        buf.write(struct.pack('<I', 16))            # Chunk size
        buf.write(struct.pack('<H', 1))              # PCM format
        buf.write(struct.pack('<H', 1))              # Mono
        buf.write(struct.pack('<I', self.sample_rate))
        buf.write(struct.pack('<I', self.sample_rate * 2))  # Byte rate
        buf.write(struct.pack('<H', 2))              # Block align
        buf.write(struct.pack('<H', 16))             # Bits per sample
        buf.write(b'data')
        buf.write(struct.pack('<I', data_size))
        buf.write(audio_int16.tobytes())

        return buf.getvalue()

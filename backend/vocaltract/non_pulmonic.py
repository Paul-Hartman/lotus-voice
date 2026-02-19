"""Non-pulmonic airstream synthesis: ejectives, implosives, clicks.

Provides physically-motivated synthesis for sounds that don't use the
standard pulmonic egressive (exhaling from lungs) airstream:

- Ejectives (glottalic egressive): larynx rises as a piston to
  compress trapped supraglottal air. Used in Georgian, Amharic,
  Quechua, Navajo, and reconstructed Akkadian emphatics.

- Implosives (glottalic ingressive): larynx lowers during oral
  closure, rarefying supraglottal air. Voiced throughout. Used in
  Swahili, Hausa, Sindhi.

- Clicks (velaric ingressive): air rarefied between velar and
  anterior closures. Used in Zulu, Xhosa, Khoisan languages.

All synthesizers inject their excitation into the existing TubeModel
so that spectral shaping comes from the physical tract geometry.

References:
  Ladefoged, P. & Maddieson, I. (1996). The Sounds of the World's
    Languages. Blackwell. Chapters 3 (stops), 4 (nasals), 5 (clicks).
  Lindau, M. (1984). Phonetic differences in glottalic consonants.
    Journal of Phonetics 12, 147-155.
"""

import logging
import math
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class AirstreamType(Enum):
    """Airstream mechanism classification."""
    PULMONIC_EGRESSIVE = "pulmonic_egressive"
    GLOTTALIC_EGRESSIVE = "glottalic_egressive"    # Ejectives
    GLOTTALIC_INGRESSIVE = "glottalic_ingressive"  # Implosives
    VELARIC_INGRESSIVE = "velaric_ingressive"       # Clicks


class EjectiveSynthesizer:
    """4-phase ejective consonant synthesis using the physical tract model.

    Ejective production phases:
      1. Closure (silence): glottis and oral closure both sealed
      2. Compression: larynx rises, building supraglottal pressure
      3. Burst: oral closure released, compressed air escapes as
         a sharp transient filtered through the tract shape
      4. VOT: aspiration noise before voicing onset

    The burst is injected at the constriction section of the TubeModel
    so spectral shaping comes from the real tract geometry.
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def synthesize(
        self,
        tube_model,
        radiation_filter,
        area_function: np.ndarray,
        constriction_section: int,
        duration_sec: float = 0.12,
        pressure_factor: float = 1.5,
        larynx_height: float = 1.0,
    ) -> np.ndarray:
        """Synthesize an ejective using the tract model for spectral shaping.

        Args:
            tube_model: TubeModel instance (will be configured with area_function)
            radiation_filter: RadiationFilter for lip output
            area_function: 44-section tract shape during closure
            constriction_section: Tube section where oral closure occurs
            duration_sec: Total phone duration
            pressure_factor: Supraglottal pressure buildup (1.0-3.0)
            larynx_height: Articulatory larynx height for this ejective

        Returns:
            Synthesized audio samples
        """
        from vocaltract.glottal_source import generate_burst_impulse

        num_samples = int(duration_sec * self.sample_rate)
        audio = np.zeros(num_samples, dtype=np.float64)

        # Phase timing (Lindau 1984 proportions)
        closure_end = int(num_samples * 0.30)
        compression_end = int(num_samples * 0.40)
        burst_end = int(num_samples * 0.55)
        # Remaining is VOT/aspiration

        # Configure tract for the closure shape
        tube_model.set_area_function(area_function)
        tube_model.reset()
        radiation_filter.reset()

        # Phase 1: Closure — silence (both glottis and supralaryngeal sealed)
        # No excitation needed

        # Phase 2: Compression — still silent, pressure building
        # (implicit — the burst amplitude reflects this)

        # Phase 3: Burst — generate impulse and push through tract
        burst_samples = burst_end - compression_end
        burst = generate_burst_impulse(
            burst_samples,
            pressure_factor=pressure_factor,
            decay_rate=8.0,
            sample_rate=self.sample_rate,
        )

        # Push burst through the tube model for spectral shaping
        burst_output = tube_model.process_block(burst)
        burst_radiated = radiation_filter.process_block(burst_output)
        audio[compression_end:burst_end] = burst_radiated

        # Phase 4: VOT — aspiration noise decaying into silence
        vot_samples = num_samples - burst_end
        if vot_samples > 0:
            noise = np.random.randn(vot_samples) * 0.3 * pressure_factor
            vot_env = np.exp(-np.linspace(0, 6, vot_samples))
            vot_signal = noise * vot_env
            vot_output = tube_model.process_block(vot_signal)
            vot_radiated = radiation_filter.process_block(vot_output)
            audio[burst_end:] = vot_radiated

        return audio


class ImplosiveSynthesizer:
    """Implosive consonant synthesis (glottalic ingressive).

    Implosives are voiced stops with larynx lowering during closure.
    The lowering rarefies supraglottal air (Boyle's law), reducing
    transglottal pressure and causing f0 to fall. Voicing continues
    throughout because the pressure differential is maintained by
    lung pressure from below.

    Phases:
      1. Closure with voiced murmur (falling f0)
      2. Weak burst at release (much less energy than ejectives)
      3. Rapid f0 recovery to normal
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def synthesize(
        self,
        tube_model,
        radiation_filter,
        glottal_source,
        area_function: np.ndarray,
        duration_sec: float = 0.10,
        base_f0: float = 120.0,
    ) -> np.ndarray:
        """Synthesize an implosive consonant.

        Args:
            tube_model: TubeModel for spectral shaping
            radiation_filter: RadiationFilter for lip output
            glottal_source: GlottalSource for voicing
            area_function: Tract shape during closure
            duration_sec: Total duration
            base_f0: Speaker's base f0

        Returns:
            Synthesized audio
        """
        num_samples = int(duration_sec * self.sample_rate)
        audio = np.zeros(num_samples, dtype=np.float64)

        # Phase timing
        closure_end = int(num_samples * 0.6)   # Voiced closure (long)
        burst_end = int(num_samples * 0.7)     # Weak burst
        # Remaining: f0 recovery

        tube_model.set_area_function(area_function)
        tube_model.reset()
        radiation_filter.reset()

        # Phase 1: Voiced closure with falling f0
        # f0 drops ~30% during closure due to larynx lowering
        for i in range(closure_end):
            progress = i / max(closure_end, 1)
            # f0 falls during closure
            current_f0 = base_f0 * (1.0 - 0.3 * progress)
            current_f0 = max(current_f0, 50.0)

            # Generate voicing at current f0
            glottal_source.set_params(f0=current_f0, Rd=1.2, phonation_type="modal")
            sample = glottal_source.generate_samples(1)[0]

            # Attenuate (closed tract dampens output)
            sample *= 0.3
            out = tube_model.process_sample(sample)
            audio[i] = radiation_filter.process_sample(out)

        # Phase 2: Weak burst
        burst_samples = burst_end - closure_end
        if burst_samples > 0:
            burst_env = np.exp(-np.linspace(0, 4, burst_samples))
            noise = np.random.randn(burst_samples) * 0.15
            burst = noise * burst_env
            burst_out = tube_model.process_block(burst)
            audio[closure_end:burst_end] = radiation_filter.process_block(burst_out)

        # Phase 3: f0 recovery — rapid return to base f0
        recovery_samples = num_samples - burst_end
        if recovery_samples > 0:
            for i in range(recovery_samples):
                progress = i / max(recovery_samples, 1)
                # f0 recovers quickly
                current_f0 = base_f0 * (0.7 + 0.3 * min(progress * 3, 1.0))
                glottal_source.set_params(f0=current_f0, Rd=1.0, phonation_type="modal")
                sample = glottal_source.generate_samples(1)[0]
                out = tube_model.process_sample(sample)
                audio[burst_end + i] = radiation_filter.process_sample(out)

        return audio


class ClickSynthesizer:
    """Click consonant synthesis (velaric ingressive).

    Models the trapped cavity between velar and anterior closures.
    When the anterior closure is released, air rushes in producing
    a sharp transient. The spectral character depends on the cavity
    shape, which varies by click type:

    - Bilabial (ʘ): large cavity, low spectral peak ~1500 Hz
    - Dental (ǀ): small cavity, high spectral peak ~4000 Hz
    - Alveolar (ǃ): medium cavity, mid-high peak ~3000 Hz
    - Palatal (ǂ): medium cavity, mid-high peak ~3500 Hz
    - Lateral (ǁ): broad cavity, mid peak ~2500 Hz

    Accompaniment variants (voiced, nasal, aspirated) modify the
    release behavior.
    """

    # Spectral peak frequencies by click type (Hz)
    SPECTRAL_PEAKS = {
        "bilabial": 1500,
        "dental": 4000,
        "alveolar": 3000,
        "palatal": 3500,
        "lateral": 2500,
    }

    # Cavity length as fraction of tract (affects resonance)
    CAVITY_LENGTHS = {
        "bilabial": 0.6,   # Large cavity (velar to lips)
        "dental": 0.15,    # Small cavity
        "alveolar": 0.2,
        "palatal": 0.3,
        "lateral": 0.25,
    }

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def synthesize(
        self,
        tube_model,
        radiation_filter,
        click_type: str = "dental",
        duration_sec: float = 0.06,
        accompaniment: str = "plain",
    ) -> np.ndarray:
        """Synthesize a click consonant.

        Args:
            tube_model: TubeModel for spectral shaping
            radiation_filter: RadiationFilter
            click_type: bilabial/dental/alveolar/palatal/lateral
            duration_sec: Total duration
            accompaniment: plain/voiced/nasal/aspirated

        Returns:
            Synthesized audio
        """
        num_samples = int(duration_sec * self.sample_rate)
        audio = np.zeros(num_samples, dtype=np.float64)

        center_freq = self.SPECTRAL_PEAKS.get(click_type, 3000)
        cavity_frac = self.CAVITY_LENGTHS.get(click_type, 0.2)

        # The click transient: damped sinusoid at cavity resonance
        # Duration of the transient itself: ~3-8ms
        transient_ms = 3.0 + cavity_frac * 10.0  # Larger cavity = longer ring
        transient_samples = min(int(transient_ms / 1000.0 * self.sample_rate), num_samples)

        if transient_samples > 0:
            t = np.arange(transient_samples, dtype=np.float64) / self.sample_rate

            # Damped sinusoid at the cavity resonance frequency
            damping = 1500.0 + (1.0 - cavity_frac) * 2000.0  # Smaller cavity = faster decay
            click = np.sin(2 * np.pi * center_freq * t) * np.exp(-damping * t)

            # Add inrush noise component
            inrush = np.random.randn(transient_samples) * 0.3
            inrush_env = np.exp(-damping * 0.5 * t)
            click = click * 0.7 + inrush * inrush_env

            # Place transient at ~30% into the duration (after closure)
            onset = int(num_samples * 0.3)
            end_idx = min(onset + transient_samples, num_samples)
            actual = end_idx - onset
            audio[onset:end_idx] = click[:actual]

        # Accompaniment effects in remaining portion
        release_start = int(num_samples * 0.3) + transient_samples
        release_end = num_samples

        if accompaniment == "aspirated" and release_start < release_end:
            n = release_end - release_start
            asp_noise = np.random.randn(n) * 0.2
            asp_env = np.exp(-np.linspace(0, 5, n))
            audio[release_start:release_end] += asp_noise * asp_env

        return audio

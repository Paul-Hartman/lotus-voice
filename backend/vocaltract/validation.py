"""Validation tools for vocal tract synthesis quality.

Compares synthesized formants against known acoustic targets:
1. Peterson & Barney (1952) vowel formant database
2. Expected formants from our area function presets
3. Custom formant targets

Also provides mechanisms for non-pulmonic airstream synthesis:
- Ejectives (glottalic egressive) for Akkadian emphatics
- Implosives (glottalic ingressive)
- Clicks (velaric ingressive)

References:
  Peterson, G.E. & Barney, H.L. (1952). Control methods used in a
    study of the vowels. JASA 24(2), 175-184.
  Ladefoged, P. & Maddieson, I. (1996). The Sounds of the World's
    Languages. Blackwell.
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from vocaltract.synthesizer import VocalTractSynthesizer

logger = logging.getLogger(__name__)


# Peterson & Barney (1952) male speaker formant targets (Hz)
PETERSON_BARNEY_MALE = {
    "i": {"F1": 270, "F2": 2290, "F3": 3010},
    "ɪ": {"F1": 390, "F2": 1990, "F3": 2550},
    "e": {"F1": 400, "F2": 2100, "F3": 2650},
    "ɛ": {"F1": 530, "F2": 1840, "F3": 2480},
    "æ": {"F1": 660, "F2": 1720, "F3": 2410},
    "ɑ": {"F1": 730, "F2": 1090, "F3": 2440},
    "ɔ": {"F1": 570, "F2": 840, "F3": 2410},
    "ʊ": {"F1": 440, "F2": 1020, "F3": 2240},
    "u": {"F1": 300, "F2": 870, "F3": 2240},
    "ʌ": {"F1": 640, "F2": 1190, "F3": 2390},
    "ə": {"F1": 500, "F2": 1500, "F3": 2500},
}

# Female targets (Peterson & Barney)
PETERSON_BARNEY_FEMALE = {
    "i": {"F1": 310, "F2": 2790, "F3": 3310},
    "ɪ": {"F1": 430, "F2": 2480, "F3": 2990},
    "ɛ": {"F1": 610, "F2": 2330, "F3": 2990},
    "æ": {"F1": 860, "F2": 2050, "F3": 2850},
    "ɑ": {"F1": 850, "F2": 1220, "F3": 2810},
    "ɔ": {"F1": 590, "F2": 920, "F3": 2710},
    "u": {"F1": 370, "F2": 950, "F3": 2670},
}


@dataclass
class FormantComparison:
    """Result of comparing synthesized formants to targets."""
    phone: str
    target_formants: Dict[str, float]
    estimated_formants: List[float]
    errors_hz: Dict[str, float]      # Absolute error per formant
    errors_percent: Dict[str, float]  # Percentage error per formant
    mean_error_percent: float


class FormantValidator:
    """Validates synthesized formant accuracy against reference data."""

    def __init__(self, synthesizer: Optional[VocalTractSynthesizer] = None):
        self.synthesizer = synthesizer or VocalTractSynthesizer()

    def validate_vowel(
        self,
        phone: str,
        reference: str = "peterson_barney_male",
    ) -> Optional[FormantComparison]:
        """Validate a single vowel's formants against reference.

        Args:
            phone: IPA vowel symbol
            reference: Reference database ("peterson_barney_male" or "female")

        Returns:
            FormantComparison or None if phone not in reference
        """
        targets = PETERSON_BARNEY_MALE if "male" in reference else PETERSON_BARNEY_FEMALE
        if phone not in targets:
            return None

        target = targets[phone]

        # Synthesize and estimate formants
        audio, states = self.synthesizer.synthesize_phone(phone, duration_sec=0.3)
        if not states:
            return None

        estimated = states[0].formants_hz
        if not estimated:
            # Try from tube model directly
            areas, _ = self.synthesizer.get_area_function(phone)
            if areas is not None:
                self.synthesizer.tube.set_area_function(areas)
                estimated = self.synthesizer.tube.estimate_formants()

        # Compare formants
        errors_hz = {}
        errors_pct = {}
        formant_names = ["F1", "F2", "F3"]

        for i, fname in enumerate(formant_names):
            if fname in target and i < len(estimated):
                target_val = target[fname]
                est_val = estimated[i]
                errors_hz[fname] = abs(est_val - target_val)
                errors_pct[fname] = abs(est_val - target_val) / target_val * 100

        mean_pct = sum(errors_pct.values()) / max(len(errors_pct), 1)

        return FormantComparison(
            phone=phone,
            target_formants=target,
            estimated_formants=estimated,
            errors_hz=errors_hz,
            errors_percent=errors_pct,
            mean_error_percent=mean_pct,
        )

    def validate_all_vowels(
        self,
        reference: str = "peterson_barney_male",
    ) -> List[FormantComparison]:
        """Validate all vowels in the reference database."""
        targets = PETERSON_BARNEY_MALE if "male" in reference else PETERSON_BARNEY_FEMALE
        results = []
        for phone in targets:
            result = self.validate_vowel(phone, reference)
            if result:
                results.append(result)
        return results

    def report(self, reference: str = "peterson_barney_male") -> str:
        """Generate a human-readable validation report."""
        results = self.validate_all_vowels(reference)
        if not results:
            return "No vowels validated."

        lines = ["Vowel Formant Validation Report", "=" * 50, ""]
        lines.append(f"{'Phone':<6} {'F1 err%':<10} {'F2 err%':<10} {'F3 err%':<10} {'Mean':<10}")
        lines.append("-" * 50)

        total_mean = 0.0
        for r in results:
            f1 = f"{r.errors_percent.get('F1', 0):.1f}%"
            f2 = f"{r.errors_percent.get('F2', 0):.1f}%"
            f3 = f"{r.errors_percent.get('F3', 0):.1f}%"
            mean = f"{r.mean_error_percent:.1f}%"
            lines.append(f"{r.phone:<6} {f1:<10} {f2:<10} {f3:<10} {mean:<10}")
            total_mean += r.mean_error_percent

        overall = total_mean / len(results)
        lines.append("-" * 50)
        lines.append(f"Overall mean error: {overall:.1f}%")

        status = "PASS" if overall < 10.0 else "NEEDS TUNING" if overall < 20.0 else "FAIL"
        lines.append(f"Status: {status} (target: <10%)")

        return "\n".join(lines)


# ─── Non-pulmonic airstream mechanisms ───────────────────────────

class EjectiveGenerator:
    """Generates ejective consonants (glottalic egressive airstream).

    Ejectives (/tʼ/, /kʼ/, /sʼ/, /pʼ/, /qʼ/) are produced by:
    1. Glottal closure (glottis acts as piston)
    2. Larynx raising (compresses supraglottal air)
    3. Oral closure at place of articulation
    4. Release: sharp burst from compressed air
    5. Brief VOT before voicing onset

    Used for Akkadian emphatic consonants in their reconstructed
    pronunciation.
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def generate_ejective(
        self,
        base_phone: str,
        duration_sec: float = 0.12,
        synthesizer: Optional[VocalTractSynthesizer] = None,
    ) -> Tuple[np.ndarray, List]:
        """Generate an ejective consonant.

        Models the ejective mechanism as three phases:
        1. Closure (30%): silence with building pressure
        2. Burst (10%): sharp transient
        3. Aspiration/VOT (60%): noisy release into following vowel

        Args:
            base_phone: The oral stop to ejectivize (e.g., "t", "k", "s")
            duration_sec: Total duration
            synthesizer: Synthesizer instance (creates one if needed)

        Returns:
            Tuple of (audio, states)
        """
        synth = synthesizer or VocalTractSynthesizer()
        num_samples = int(duration_sec * self.sample_rate)

        # Phase timing
        closure_end = int(num_samples * 0.3)
        burst_end = int(num_samples * 0.4)

        audio = np.zeros(num_samples, dtype=np.float64)

        # Phase 1: Closure — silence (both glottis and supralaryngeal closed)
        # Just silence

        # Phase 2: Burst — sharp transient noise
        burst_samples = burst_end - closure_end
        burst_noise = np.random.randn(burst_samples) * 0.8
        # Sharp attack envelope
        burst_env = np.exp(-np.linspace(0, 5, burst_samples))
        audio[closure_end:burst_end] = burst_noise * burst_env

        # Phase 3: Aspiration/VOT — filtered noise transitioning to voicing
        vot_samples = num_samples - burst_end
        if vot_samples > 0:
            # Start with noise, fade to voiced
            noise = np.random.randn(vot_samples) * 0.3
            # Decaying noise envelope
            noise_env = np.exp(-np.linspace(0, 4, vot_samples))
            audio[burst_end:] = noise * noise_env

        # Normalize
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.7

        return audio, []


class ImplosiveGenerator:
    """Generates implosive consonants (glottalic ingressive).

    Implosives (/ɓ/, /ɗ/, /ɠ/) are produced by lowering the
    larynx while maintaining oral closure, creating a partial
    vacuum that produces a characteristic hollow quality.
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def generate_implosive(
        self,
        base_phone: str,
        duration_sec: float = 0.1,
    ) -> np.ndarray:
        """Generate an implosive consonant."""
        num_samples = int(duration_sec * self.sample_rate)
        audio = np.zeros(num_samples, dtype=np.float64)

        # Implosives have a brief low-frequency "pop"
        # followed by voicing with falling f0
        pop_end = int(num_samples * 0.2)

        # Low-frequency pop (voiced closure with falling f0)
        t = np.linspace(0, 0.02, pop_end)
        f0_falling = np.linspace(80, 60, pop_end)
        for i in range(pop_end):
            audio[i] = 0.3 * math.sin(2 * math.pi * f0_falling[i] * t[i])

        # Smooth release into voiced sound
        release_env = np.linspace(0.3, 0.1, num_samples - pop_end)
        for i in range(pop_end, num_samples):
            j = i - pop_end
            t_val = j / self.sample_rate
            audio[i] = release_env[j] * math.sin(2 * math.pi * 90 * t_val)

        return audio


class ClickGenerator:
    """Generates click consonants (velaric ingressive).

    Clicks are produced by rarefying air between two closures
    (velar and a forward closure), then releasing the forward
    closure to produce a sharp transient.

    Not used in Sumerian/Akkadian but included for completeness
    (e.g., for potential African language support).
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def generate_click(
        self,
        click_type: str = "dental",
        duration_sec: float = 0.05,
    ) -> np.ndarray:
        """Generate a click consonant.

        Args:
            click_type: "dental", "lateral", "palatal", "bilabial", "retroflex"
            duration_sec: Duration of the click transient
        """
        num_samples = int(duration_sec * self.sample_rate)
        audio = np.zeros(num_samples, dtype=np.float64)

        # Click is a very brief transient
        transient_samples = min(int(0.005 * self.sample_rate), num_samples)

        # Different click types have different spectral characteristics
        spectral_center = {
            "dental": 4000,     # /ǀ/ — bright, high frequency
            "lateral": 2500,    # /ǁ/ — broader, lateral release
            "palatal": 3500,    # /ǂ/ — moderate-high
            "bilabial": 1500,   # /ʘ/ — lower frequency
            "retroflex": 3000,  # /‼/ — mid-high
        }
        center = spectral_center.get(click_type, 3000)

        # Sharp transient: bandpass-filtered impulse
        t = np.arange(transient_samples) / self.sample_rate
        # Damped sinusoid at the spectral center
        click = np.sin(2 * np.pi * center * t) * np.exp(-t * 2000)
        audio[:transient_samples] = click * 0.8

        return audio

"""Singing synthesis extensions.

Adds singing-specific vocal control to the articulatory synthesizer:

1. Vibrato: periodic f0 modulation at ~5Hz, ±1 semitone (adjustable)
2. Vocal registers: chest, head, falsetto with different Rd/glottal configs
3. Singer's formant: 3kHz spectral peak via epilaryngeal tube narrowing
4. F1:F0 tracking: jaw widening at high pitches to keep F1 ≥ f0

These modifications overlay on the existing articulatory model.

References:
  Sundberg, J. (1987). The Science of the Singing Voice. Northern
    Illinois University Press.
  Titze, I.R. (2000). Principles of Voice Production. National Center
    for Voice and Speech.
  Joliveau, E., Smith, J., & Wolfe, J. (2004). Vocal tract resonances
    in singing: The soprano voice. JASA 116(4), 2434-2439.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from vocaltract.articulators import ArticulatorTarget


class VocalRegister(Enum):
    """Vocal registers with distinct phonation patterns."""
    CHEST = "chest"         # Modal phonation, strong subglottals
    HEAD = "head"           # Lighter phonation, more falsetto-like
    FALSETTO = "falsetto"   # Very light, minimal vocal fold contact
    MIXED = "mixed"         # Blend of chest and head (contemporary singing)
    WHISTLE = "whistle"     # Extreme high range (flageolet)


@dataclass
class RegisterConfig:
    """Phonation configuration for a vocal register."""
    Rd: float             # Voice quality parameter
    pressure_mult: float  # Subglottal pressure multiplier
    glottal_closure: float  # 0=breathy, 1=fully pressed
    singer_formant: float  # Singer's formant strength (0-1)
    oq_target: float      # Target open quotient


# Register configurations from Sundberg (1987)
REGISTER_CONFIGS = {
    VocalRegister.CHEST: RegisterConfig(
        Rd=0.8, pressure_mult=1.0, glottal_closure=0.7,
        singer_formant=0.8, oq_target=0.5,
    ),
    VocalRegister.HEAD: RegisterConfig(
        Rd=1.3, pressure_mult=0.8, glottal_closure=0.5,
        singer_formant=0.5, oq_target=0.6,
    ),
    VocalRegister.FALSETTO: RegisterConfig(
        Rd=2.0, pressure_mult=0.6, glottal_closure=0.3,
        singer_formant=0.2, oq_target=0.75,
    ),
    VocalRegister.MIXED: RegisterConfig(
        Rd=1.0, pressure_mult=0.9, glottal_closure=0.6,
        singer_formant=0.6, oq_target=0.55,
    ),
    VocalRegister.WHISTLE: RegisterConfig(
        Rd=2.5, pressure_mult=0.5, glottal_closure=0.2,
        singer_formant=0.0, oq_target=0.9,
    ),
}


@dataclass
class VibratoConfig:
    """Vibrato parameters."""
    rate_hz: float = 5.0        # Vibrato frequency (typically 5-7 Hz)
    extent_semitones: float = 1.0  # Peak deviation in semitones
    delay_sec: float = 0.2      # Delay before vibrato onset
    rise_time_sec: float = 0.15  # Time to reach full vibrato depth


@dataclass
class SingingNote:
    """A single note in a singing sequence."""
    phone: str              # IPA vowel/consonant
    midi_note: int          # MIDI note number (60 = middle C)
    duration_sec: float     # Note duration
    register: VocalRegister = VocalRegister.CHEST
    vibrato: Optional[VibratoConfig] = None
    dynamics: float = 0.7   # 0=pianissimo, 1=fortissimo
    legato: bool = True     # Connected to next note?


def midi_to_hz(midi_note: int) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def hz_to_semitones(f1: float, f2: float) -> float:
    """Compute interval in semitones between two frequencies."""
    if f1 <= 0 or f2 <= 0:
        return 0.0
    return 12.0 * math.log2(f2 / f1)


class SingingController:
    """Controls singing-specific vocal parameters.

    Generates f0 contours with vibrato, adjusts register-specific
    phonation, applies singer's formant and F1:F0 tracking.
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.default_vibrato = VibratoConfig()

    def generate_vibrato_f0(
        self,
        base_f0: float,
        duration_sec: float,
        vibrato: Optional[VibratoConfig] = None,
    ) -> np.ndarray:
        """Generate an f0 contour with vibrato.

        Args:
            base_f0: Target pitch in Hz
            duration_sec: Duration in seconds
            vibrato: Vibrato config (or default)

        Returns:
            f0 array at sample_rate
        """
        vib = vibrato or self.default_vibrato
        n = int(duration_sec * self.sample_rate)
        f0 = np.full(n, base_f0, dtype=np.float64)

        for i in range(n):
            t = i / self.sample_rate

            # Vibrato onset: ramp up after delay
            if t < vib.delay_sec:
                depth = 0.0
            elif t < vib.delay_sec + vib.rise_time_sec:
                depth = (t - vib.delay_sec) / vib.rise_time_sec
            else:
                depth = 1.0

            # Sinusoidal f0 modulation in semitones
            semitone_deviation = depth * vib.extent_semitones * math.sin(2.0 * math.pi * vib.rate_hz * t)
            f0[i] = base_f0 * (2.0 ** (semitone_deviation / 12.0))

        return f0

    def apply_singer_formant(
        self,
        target: ArticulatorTarget,
        strength: float = 0.8,
    ) -> ArticulatorTarget:
        """Modify articulation to produce the singer's formant.

        The singer's formant (~2.5-3.5 kHz) is created by narrowing
        the epilaryngeal tube — the region between the false vocal
        folds and the epiglottis tip. This creates a resonance cluster
        that projects the voice above an orchestra.

        In our model, this corresponds to narrowing the laryngeal
        sections (0-7) while keeping the pharynx wide.

        Args:
            target: Base articulatory target
            strength: 0-1 singer's formant strength

        Returns:
            Modified ArticulatorTarget
        """
        # Lower larynx creates more space above the glottis,
        # which paradoxically creates the narrow epilaryngeal tube
        # by stretching the aryepiglottic fold
        modified = ArticulatorTarget(
            jaw=target.jaw,
            tongue_dorsal_pos=target.tongue_dorsal_pos,
            tongue_dorsal_shape=target.tongue_dorsal_shape,
            tongue_tip=target.tongue_tip,
            lip_height=target.lip_height,
            lip_protrusion=target.lip_protrusion,
            larynx_height=target.larynx_height - strength * 1.5,  # Lower larynx
            velum=target.velum,
            f0=target.f0,
            Rd=target.Rd,
            aspiration=target.aspiration,
            phonation_type=target.phonation_type,
            noise_source_section=target.noise_source_section,
            noise_amplitude=target.noise_amplitude,
        )
        return modified

    def apply_f1_f0_tracking(
        self,
        target: ArticulatorTarget,
        f0: float,
        f1_threshold: float = 500.0,
    ) -> ArticulatorTarget:
        """Widen jaw to keep F1 above f0 at high pitches.

        When f0 exceeds F1, the first harmonic falls below the first
        formant, causing dramatic loudness loss. Trained singers
        compensate by opening the jaw to raise F1.

        This is why all high soprano vowels sound like [ɑ].

        Args:
            target: Base articulatory target
            f0: Current fundamental frequency
            f1_threshold: Approximate F1 of the vowel (Hz)

        Returns:
            Modified ArticulatorTarget with jaw widening
        """
        if f0 <= f1_threshold:
            return target

        # Need to raise F1: widen jaw proportional to the deficit
        deficit_semitones = hz_to_semitones(f1_threshold, f0)
        jaw_opening = min(deficit_semitones * 0.3, 2.0)  # Cap at +2

        modified = ArticulatorTarget(
            jaw=min(target.jaw + jaw_opening, 3.0),
            tongue_dorsal_pos=target.tongue_dorsal_pos,
            tongue_dorsal_shape=target.tongue_dorsal_shape,
            tongue_tip=target.tongue_tip,
            lip_height=min(target.lip_height + jaw_opening * 0.5, 3.0),
            lip_protrusion=target.lip_protrusion,
            larynx_height=target.larynx_height,
            velum=target.velum,
            f0=target.f0,
            Rd=target.Rd,
            aspiration=target.aspiration,
            phonation_type=target.phonation_type,
            noise_source_section=target.noise_source_section,
            noise_amplitude=target.noise_amplitude,
        )
        return modified

    def select_register(self, f0: float, dynamics: float = 0.7) -> VocalRegister:
        """Auto-select vocal register based on pitch and dynamics.

        Approximate register boundaries (male voice):
          Chest: up to ~330 Hz (E4)
          Mixed: 250-440 Hz (B3 to A4)
          Head: 330-523 Hz (E4 to C5)
          Falsetto: 440-700 Hz (A4 to F5)
          Whistle: above 700 Hz

        Args:
            f0: Fundamental frequency in Hz
            dynamics: 0=soft, 1=loud

        Returns:
            Appropriate vocal register
        """
        if f0 > 700:
            return VocalRegister.WHISTLE
        elif f0 > 440:
            return VocalRegister.FALSETTO if dynamics < 0.5 else VocalRegister.HEAD
        elif f0 > 330:
            return VocalRegister.HEAD if dynamics < 0.5 else VocalRegister.MIXED
        elif f0 > 250:
            return VocalRegister.MIXED if dynamics > 0.7 else VocalRegister.CHEST
        else:
            return VocalRegister.CHEST

    def prepare_singing_note(
        self,
        note: SingingNote,
    ) -> Tuple[float, float, ArticulatorTarget]:
        """Prepare phonation parameters for a singing note.

        Returns:
            Tuple of (f0, Rd, modified_target)
        """
        f0 = midi_to_hz(note.midi_note)
        register = note.register
        config = REGISTER_CONFIGS[register]

        Rd = config.Rd

        # Get base articulatory target (placeholder — will be looked
        # up via the synthesizer's IPA mapper in practice)
        target = ArticulatorTarget(f0=f0, Rd=Rd)

        # Apply singer's formant
        if config.singer_formant > 0:
            target = self.apply_singer_formant(target, config.singer_formant)

        # Apply F1:F0 tracking for high notes
        target = self.apply_f1_f0_tracking(target, f0)

        return f0, Rd, target

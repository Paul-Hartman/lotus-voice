"""Phonation control: f0 contours and voice quality dynamics.

Generates time-varying fundamental frequency (f0) and voice quality
(Rd) contours for natural-sounding speech. Controls:

1. Intonation contours: declarative (falling), interrogative (rising),
   continuation rise, exclamatory, etc.
2. Microprosody: consonant perturbations on f0 (voiceless consonants
   cause brief f0 dips, high vowels raise f0 slightly).
3. Emotion-to-Rd mapping: converts EmotionalBeat parameters from
   VoiceActingDirector into physical voice quality settings.
4. Declination: global f0 downtrend over the course of an utterance.

References:
  Pierrehumbert, J. (1980). The phonology and phonetics of English
    intonation. PhD thesis, MIT.
  Ladd, D.R. (2008). Intonational Phonology. Cambridge University Press.
  Hanson, H.M. (1997). Glottal characteristics of female speakers:
    Acoustic correlates. JASA 101(1), 466-481.
"""

import math
from enum import Enum
from typing import List, Optional

import numpy as np


class IntonationPattern(Enum):
    """Common intonation patterns."""
    DECLARATIVE = "declarative"       # Falling at end
    INTERROGATIVE = "interrogative"   # Rising at end (yes/no question)
    WH_QUESTION = "wh_question"       # Falling (like declarative)
    CONTINUATION = "continuation"     # Slight rise (more to come)
    EXCLAMATORY = "exclamatory"       # Expanded range, falling
    LISTING = "listing"               # Rise for each item, fall on last
    VOCATIVE = "vocative"             # High-low (calling someone)


# Emotional states mapped to voice quality parameters
# Based on Scherer (2003) vocal expression of emotion
EMOTION_RD_MAP = {
    # Emotion: (Rd, f0_shift_semitones, f0_range_multiplier)
    "neutral": (1.0, 0.0, 1.0),
    "devastated": (2.0, -3.0, 0.6),      # Breathy, low, narrow range
    "terrified": (0.6, 4.0, 1.8),         # Pressed, high, wide range
    "elated": (1.2, 3.0, 1.5),            # Slightly breathy, high, wide
    "resigned": (1.8, -2.0, 0.5),         # Breathy, low, flat
    "defiant": (0.5, 0.0, 0.8),           # Pressed, steady
    "vulnerable": (2.2, 1.0, 1.2),        # Very breathy, slightly high
    "furious": (0.4, 2.0, 2.0),           # Very pressed, high, extreme range
    "calculating": (0.8, -1.0, 0.7),      # Slightly pressed, low, narrow
    "happy": (1.1, 2.0, 1.3),
    "sad": (1.8, -2.0, 0.6),
    "angry": (0.5, 2.0, 1.6),
    "fearful": (0.7, 3.0, 1.5),
    "whisper": (3.5, 0.0, 0.3),           # Very breathy, minimal range
    "creaky": (0.3, -4.0, 0.4),           # Very pressed, very low
}


class PhonationController:
    """Generates f0 and Rd contours for natural speech.

    Combines intonation pattern, stress, microprosody, and emotional
    state into smooth f0 and Rd trajectories.
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._base_f0 = 120.0
        self._base_Rd = 1.0
        self._emotion = "neutral"

    def set_base_voice(self, f0: float = 120.0, Rd: float = 1.0) -> None:
        """Set the speaker's baseline f0 and Rd."""
        self._base_f0 = f0
        self._base_Rd = Rd

    def set_emotion(self, emotion: str) -> None:
        """Set emotional state (affects Rd and f0 range)."""
        self._emotion = emotion

    def get_emotion_params(self) -> tuple:
        """Get (Rd, f0_shift_semitones, f0_range_multiplier) for current emotion."""
        return EMOTION_RD_MAP.get(self._emotion, EMOTION_RD_MAP["neutral"])

    def generate_f0_contour(
        self,
        total_duration_sec: float,
        pattern: IntonationPattern = IntonationPattern.DECLARATIVE,
        stress_positions: Optional[List[float]] = None,
        declination_rate: float = 0.5,
    ) -> np.ndarray:
        """Generate an f0 contour for an utterance.

        Args:
            total_duration_sec: Total duration in seconds
            pattern: Intonation pattern
            stress_positions: List of time positions (sec) for stress peaks
            declination_rate: f0 drop rate in Hz/sec

        Returns:
            f0 array at sample_rate
        """
        num_samples = int(total_duration_sec * self.sample_rate)
        f0 = np.full(num_samples, self._base_f0, dtype=np.float64)

        # Apply emotion
        Rd_emotion, f0_shift_st, f0_range_mult = self.get_emotion_params()
        # Convert semitone shift to Hz ratio
        f0_shifted = self._base_f0 * (2.0 ** (f0_shift_st / 12.0))
        f0[:] = f0_shifted

        # Declination: gradual f0 lowering over utterance
        for i in range(num_samples):
            t = i / self.sample_rate
            f0[i] -= declination_rate * t

        # Intonation pattern
        self._apply_intonation(f0, pattern, total_duration_sec, f0_range_mult)

        # Stress peaks
        if stress_positions:
            for stress_t in stress_positions:
                self._add_stress_peak(f0, stress_t, total_duration_sec)

        # Clamp to reasonable range
        f0 = np.clip(f0, 50.0, 600.0)

        return f0

    def _apply_intonation(self, f0: np.ndarray, pattern: IntonationPattern,
                          duration: float, range_mult: float) -> None:
        """Apply intonation pattern to f0 contour."""
        n = len(f0)
        base = f0[0]
        range_hz = base * 0.15 * range_mult  # ±15% of base, scaled by emotion

        if pattern == IntonationPattern.DECLARATIVE:
            # Gentle rise in first third, fall in last third
            for i in range(n):
                t = i / n
                if t < 0.3:
                    f0[i] += range_hz * 0.3 * (t / 0.3)
                elif t > 0.7:
                    fall = (t - 0.7) / 0.3
                    f0[i] -= range_hz * 0.5 * fall

        elif pattern == IntonationPattern.INTERROGATIVE:
            # Rise in final 30%
            for i in range(n):
                t = i / n
                if t > 0.7:
                    rise = (t - 0.7) / 0.3
                    f0[i] += range_hz * 0.8 * rise

        elif pattern == IntonationPattern.WH_QUESTION:
            # High start, falling (similar to declarative)
            for i in range(n):
                t = i / n
                f0[i] += range_hz * 0.3 * (1.0 - t)

        elif pattern == IntonationPattern.EXCLAMATORY:
            # Expanded range: high start, dramatic fall
            for i in range(n):
                t = i / n
                f0[i] += range_hz * 1.5 * max(0, 0.8 - t)

        elif pattern == IntonationPattern.CONTINUATION:
            # Slight rise at end
            for i in range(n):
                t = i / n
                if t > 0.8:
                    f0[i] += range_hz * 0.3 * (t - 0.8) / 0.2

        elif pattern == IntonationPattern.VOCATIVE:
            # High-low pattern (calling: "Hel-LO!")
            mid = n // 2
            for i in range(n):
                if i < mid:
                    f0[i] += range_hz * 0.5
                else:
                    f0[i] -= range_hz * 0.3

    def _add_stress_peak(self, f0: np.ndarray, stress_time: float,
                         total_duration: float) -> None:
        """Add a stress-related f0 peak at the given time."""
        n = len(f0)
        peak_idx = int(stress_time / total_duration * n)
        peak_idx = max(0, min(peak_idx, n - 1))
        peak_hz = f0[peak_idx] * 0.08  # +8% f0 boost

        # Gaussian peak shape (σ = 30ms)
        sigma_samples = int(0.03 * self.sample_rate)
        for i in range(max(0, peak_idx - 3 * sigma_samples),
                       min(n, peak_idx + 3 * sigma_samples)):
            dist = (i - peak_idx) / max(sigma_samples, 1)
            f0[i] += peak_hz * math.exp(-0.5 * dist * dist)

    def generate_Rd_contour(
        self,
        total_duration_sec: float,
        phone_types: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Generate Rd (voice quality) contour.

        Args:
            total_duration_sec: Total duration
            phone_types: List of phone manner types at each sample point

        Returns:
            Rd array at sample_rate
        """
        num_samples = int(total_duration_sec * self.sample_rate)

        # Base Rd from emotion
        Rd_emotion = self.get_emotion_params()[0]
        Rd = np.full(num_samples, Rd_emotion, dtype=np.float64)

        # Clamp to physical range
        Rd = np.clip(Rd, 0.3, 6.0)

        return Rd

    def apply_microprosody(
        self,
        f0: np.ndarray,
        phone_boundaries: List[tuple],
        phone_types: List[str],
    ) -> np.ndarray:
        """Apply microprosodic perturbations to f0.

        Consonants perturb f0 of adjacent vowels:
        - Voiceless stops: brief f0 dip after release
        - Voiced stops: f0 rise after release
        - High vowels: slightly higher f0 than low vowels

        Args:
            f0: Base f0 contour
            phone_boundaries: List of (start_sample, end_sample) per phone
            phone_types: List of phone manner types

        Returns:
            Modified f0 contour
        """
        f0_mod = f0.copy()

        for i, (start, end) in enumerate(phone_boundaries):
            if i >= len(phone_types):
                break
            ptype = phone_types[i]

            if ptype == "voiceless_stop":
                # f0 dip in the first 20ms after stop release
                dip_end = min(end + int(0.02 * self.sample_rate), len(f0_mod))
                for j in range(end, dip_end):
                    progress = (j - end) / max(dip_end - end, 1)
                    f0_mod[j] -= 10.0 * (1.0 - progress)  # -10 Hz dip, recovering

            elif ptype == "voiced_stop":
                # f0 rise in first 20ms
                rise_end = min(end + int(0.02 * self.sample_rate), len(f0_mod))
                for j in range(end, rise_end):
                    progress = (j - end) / max(rise_end - end, 1)
                    f0_mod[j] += 5.0 * (1.0 - progress)

        return f0_mod

    def emotion_from_beat(self, beat) -> tuple:
        """Convert a VoiceActingDirector.EmotionalBeat to phonation params.

        Args:
            beat: EmotionalBeat from voice_acting_director.py

        Returns:
            Tuple of (f0, Rd, intonation_pattern)
        """
        feeling = getattr(beat, "feeling", "neutral")
        self.set_emotion(feeling)

        Rd_emotion, f0_shift, f0_range = self.get_emotion_params()
        f0 = self._base_f0 * (2.0 ** (f0_shift / 12.0))

        # Determine intonation from text
        text = getattr(beat, "text", "")
        if text.rstrip().endswith("?"):
            pattern = IntonationPattern.INTERROGATIVE
        elif text.rstrip().endswith("!"):
            pattern = IntonationPattern.EXCLAMATORY
        else:
            pattern = IntonationPattern.DECLARATIVE

        return f0, Rd_emotion, pattern

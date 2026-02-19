"""Respiratory / subglottal pressure model.

Models the lungs and trachea as a pressure source driving the
glottal source. Lung pressure varies over the course of a phrase
due to:

1. Declination: pressure naturally falls during an utterance as
   lung volume decreases (Ladefoged 1967).
2. Stress: stressed syllables receive a pressure pulse.
3. Phrase boundaries: speaker takes a new breath.

The pressure-to-airflow relationship follows from Bernoulli's
principle at the glottis: airflow depends on the pressure
differential across the glottal constriction.

References:
  Ladefoged, P. (1967). Three Areas of Experimental Phonetics. OUP.
  Draper, M.H., Ladefoged, P., & Whitteridge, D. (1960). Expiratory
    pressures and air flow during speech. British Medical Journal.
  Titze, I.R. (1994). Principles of Voice Production. Prentice Hall.
"""

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from vocaltract.state import SubglottalState


@dataclass
class BreathGroup:
    """A group of phones between breaths."""
    start_phone_idx: int
    end_phone_idx: int
    num_syllables: int
    duration_sec: float


class SubglottalModel:
    """Models respiratory pressure driving the vocal folds.

    Provides time-varying subglottal pressure that accounts for
    lung volume depletion, stress patterns, and breath planning.
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

        # Physiological defaults (adult male)
        self.resting_pressure_cmH2O = 8.0     # Normal conversational speech
        self.max_pressure_cmH2O = 20.0        # Loud speech / stress peaks
        self.min_pressure_cmH2O = 3.0         # Barely phonating
        self.initial_lung_volume = 3.5        # Liters (after inhalation)

        # Declination rate: pressure drops linearly per second
        self.declination_rate = 0.8  # cmH2O per second

        # Current state
        self._pressure = self.resting_pressure_cmH2O
        self._lung_volume = self.initial_lung_volume
        self._airflow = 0.15  # liters/sec at normal phonation

    @property
    def state(self) -> SubglottalState:
        return SubglottalState(
            lung_pressure_cmH2O=self._pressure,
            lung_volume_liters=self._lung_volume,
            airflow_liters_per_sec=self._airflow,
            diaphragm_tension=self._pressure / self.max_pressure_cmH2O,
        )

    def get_pressure(self, time_in_phrase: float, phrase_duration: float,
                     stress: float = 0.0) -> float:
        """Get subglottal pressure at a point in a phrase.

        Args:
            time_in_phrase: Seconds since phrase start
            phrase_duration: Total phrase duration in seconds
            stress: Stress level (0=unstressed, 1=primary stress)

        Returns:
            Subglottal pressure in cmH2O
        """
        # Base pressure with declination
        progress = time_in_phrase / max(phrase_duration, 0.01)
        base = self.resting_pressure_cmH2O - self.declination_rate * time_in_phrase

        # Stress pulse: brief pressure increase for stressed syllables
        stress_boost = stress * 4.0  # Up to +4 cmH2O for strong stress

        # Phrase-final lowering (utterance-final declination)
        if progress > 0.85:
            final_drop = (progress - 0.85) / 0.15 * 2.0  # Extra 2 cmH2O drop
            base -= final_drop

        pressure = base + stress_boost
        return max(self.min_pressure_cmH2O, min(pressure, self.max_pressure_cmH2O))

    def generate_pressure_contour(
        self,
        total_duration_sec: float,
        stress_pattern: Optional[List[float]] = None,
        breath_groups: Optional[List[BreathGroup]] = None,
    ) -> np.ndarray:
        """Generate a complete subglottal pressure contour.

        Args:
            total_duration_sec: Total duration in seconds
            stress_pattern: Per-sample stress values (0-1), or None for flat
            breath_groups: Breath planning, or None for single breath

        Returns:
            Pressure contour array at sample_rate
        """
        num_samples = int(total_duration_sec * self.sample_rate)
        pressure = np.zeros(num_samples, dtype=np.float64)

        if breath_groups is None:
            # Single breath group
            for i in range(num_samples):
                t = i / self.sample_rate
                stress = stress_pattern[i] if stress_pattern is not None and i < len(stress_pattern) else 0.0
                pressure[i] = self.get_pressure(t, total_duration_sec, stress)
        else:
            # Multiple breath groups
            sample_pos = 0
            for bg in breath_groups:
                bg_samples = int(bg.duration_sec * self.sample_rate)
                for j in range(bg_samples):
                    if sample_pos + j >= num_samples:
                        break
                    t = j / self.sample_rate
                    stress = 0.0
                    if stress_pattern is not None and sample_pos + j < len(stress_pattern):
                        stress = stress_pattern[sample_pos + j]
                    pressure[sample_pos + j] = self.get_pressure(t, bg.duration_sec, stress)
                sample_pos += bg_samples

        return pressure

    def pressure_to_airflow(self, pressure_cmH2O: float,
                            glottal_area_cm2: float = 0.1) -> float:
        """Convert subglottal pressure to glottal airflow via Bernoulli.

        U = A_g * sqrt(2 * ΔP / ρ)

        where A_g is glottal area, ΔP is transglottal pressure,
        ρ is air density.

        Args:
            pressure_cmH2O: Subglottal pressure
            glottal_area_cm2: Glottal opening area

        Returns:
            Airflow in liters/second
        """
        # Convert cmH2O to dyne/cm²: 1 cmH2O = 980.665 dyne/cm²
        delta_p = pressure_cmH2O * 980.665
        rho = 1.14e-3  # g/cm³ air density at body temperature

        if delta_p <= 0 or glottal_area_cm2 <= 0:
            return 0.0

        velocity = math.sqrt(2.0 * delta_p / rho)
        flow_cm3_per_s = glottal_area_cm2 * velocity
        return flow_cm3_per_s / 1000.0  # Convert to liters/sec

    def plan_breaths(self, phone_durations: List[float],
                     max_phrase_sec: float = 4.0) -> List[BreathGroup]:
        """Plan breath group boundaries for a sequence of phones.

        Inserts breaths at natural phrase boundaries (silences) or
        when the phrase exceeds maximum duration.

        Args:
            phone_durations: Duration of each phone in seconds
            max_phrase_sec: Maximum duration before forced breath

        Returns:
            List of BreathGroups
        """
        groups = []
        current_start = 0
        current_duration = 0.0
        current_syllables = 0

        for i, dur in enumerate(phone_durations):
            current_duration += dur
            current_syllables += 1  # Simplified syllable counting

            # Force breath at max duration
            if current_duration >= max_phrase_sec:
                groups.append(BreathGroup(
                    start_phone_idx=current_start,
                    end_phone_idx=i,
                    num_syllables=current_syllables,
                    duration_sec=current_duration,
                ))
                current_start = i + 1
                current_duration = 0.0
                current_syllables = 0

        # Final group
        if current_start < len(phone_durations):
            groups.append(BreathGroup(
                start_phone_idx=current_start,
                end_phone_idx=len(phone_durations) - 1,
                num_syllables=current_syllables,
                duration_sec=current_duration,
            ))

        return groups

    def reset(self) -> None:
        """Reset to initial breath state."""
        self._pressure = self.resting_pressure_cmH2O
        self._lung_volume = self.initial_lung_volume
        self._airflow = 0.15

"""Coarticulation engine for connected speech.

Produces smooth articulatory trajectories by blending between
phone targets, incorporating:

1. Anticipatory coarticulation: lookahead 2-3 phones, articulators
   begin moving toward upcoming targets before the current phone ends.
2. Carryover effects: exponential decay from previous phone's
   articulatory influence.
3. Transition timing: biomechanically constrained, with shorter
   transitions for faster articulators.
4. Locus equations: F2 transition targets for CV sequences
   (Sussman et al. 1991).

The output is a time-varying articulatory trajectory sampled at
the articulator update rate (200 Hz), which drives the area
function and tube model for smooth, natural-sounding speech.

References:
  Sussman, H.M., McCaffrey, H.A., & Matthews, S.A. (1991).
    An investigation of locus equations as a source of relational
    invariance for stop place categorization. JASA 90(3), 1309-1325.
  Recasens, D. (1999). Lingual coarticulation. In W.J. Hardcastle
    & N. Hewlett (Eds.), Coarticulation, pp. 80-104.
  Öhman, S.E.G. (1966). Coarticulation in VCV utterances: spectrographic
    measurements. JASA 39(1), 151-168.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from vocaltract.articulators import (
    ArticulatorTarget,
    compute_transition_duration,
    interpolate_targets,
)

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "vocaltract"

# Articulator update rate (Hz) — controls trajectory resolution
ARTICULATOR_RATE = 200

# Number of phones to look ahead for anticipatory coarticulation
LOOKAHEAD = 2

# Carryover decay time constant (seconds)
CARRYOVER_TAU = 0.04  # 40ms — fast exponential decay


@dataclass
class CoarticulationRules:
    """Rules governing coarticulatory blending."""

    # Locus equations: F2 onset = slope * F2_vowel + intercept
    # Keyed by consonant place of articulation
    locus_equations: Dict[str, Dict[str, float]]

    # Per-articulator resistance to coarticulation (0=fully coarticulated, 1=fixed)
    # Higher values mean the articulator resists being influenced by neighbors
    resistance: Dict[str, Dict[str, float]]

    # Default transition duration multipliers by manner
    transition_multipliers: Dict[str, float]


def load_coarticulation_rules() -> CoarticulationRules:
    """Load coarticulation rules from JSON data file."""
    rules_path = _DATA_DIR / "coarticulation_rules.json"
    if rules_path.exists():
        with open(rules_path) as f:
            data = json.load(f)
        return CoarticulationRules(
            locus_equations=data.get("locus_equations", {}),
            resistance=data.get("resistance", {}),
            transition_multipliers=data.get("transition_multipliers", {}),
        )

    # Default rules if no file exists
    return CoarticulationRules(
        locus_equations={
            "bilabial": {"slope": 0.85, "intercept": 300},
            "alveolar": {"slope": 0.45, "intercept": 1050},
            "velar": {"slope": 0.95, "intercept": 100},
            "palatal": {"slope": 0.55, "intercept": 900},
            "uvular": {"slope": 0.90, "intercept": 150},
        },
        resistance={
            "stop": {
                "jaw": 0.9, "tongue_dorsal_pos": 0.8, "tongue_dorsal_shape": 0.7,
                "tongue_tip": 0.9, "lip_height": 0.9, "lip_protrusion": 0.3,
                "larynx_height": 0.2, "velum": 0.95,
            },
            "fricative": {
                "jaw": 0.7, "tongue_dorsal_pos": 0.7, "tongue_dorsal_shape": 0.6,
                "tongue_tip": 0.8, "lip_height": 0.5, "lip_protrusion": 0.3,
                "larynx_height": 0.2, "velum": 0.9,
            },
            "vowel": {
                "jaw": 0.3, "tongue_dorsal_pos": 0.3, "tongue_dorsal_shape": 0.3,
                "tongue_tip": 0.2, "lip_height": 0.3, "lip_protrusion": 0.3,
                "larynx_height": 0.2, "velum": 0.5,
            },
            "nasal": {
                "jaw": 0.5, "tongue_dorsal_pos": 0.7, "tongue_dorsal_shape": 0.5,
                "tongue_tip": 0.8, "lip_height": 0.8, "lip_protrusion": 0.3,
                "larynx_height": 0.2, "velum": 0.95,
            },
        },
        transition_multipliers={
            "stop": 0.6,        # Fast transitions into/out of stops
            "fricative": 0.8,   # Moderate
            "nasal": 0.9,
            "vowel": 1.0,       # Full transition time
            "approximant": 1.0,
            "lateral": 0.9,
            "trill": 0.7,
            "tap": 0.4,         # Very fast
            "affricate": 0.7,
        },
    )


@dataclass
class PhoneSegment:
    """A phone with its articulatory target and timing."""
    phone: str
    target: ArticulatorTarget
    duration_sec: float
    manner: Optional[str] = None  # Manner of articulation


class CoarticulationEngine:
    """Produces smooth articulatory trajectories for connected speech.

    Given a sequence of PhoneSegments with targets and durations,
    generates a time-varying articulator trajectory that incorporates
    anticipatory and carryover coarticulation.
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.rules = load_coarticulation_rules()

    def generate_trajectory(
        self,
        segments: List[PhoneSegment],
    ) -> List[Tuple[ArticulatorTarget, float]]:
        """Generate coarticulated articulatory trajectory.

        Returns a list of (target, timestamp) pairs sampled at
        ARTICULATOR_RATE Hz. Each target represents the blended
        articulatory configuration at that instant.

        Args:
            segments: Ordered list of phone segments with targets

        Returns:
            List of (ArticulatorTarget, time_sec) tuples
        """
        if not segments:
            return []
        if len(segments) == 1:
            n_frames = max(1, int(segments[0].duration_sec * ARTICULATOR_RATE))
            return [(segments[0].target, i / ARTICULATOR_RATE) for i in range(n_frames)]

        trajectory = []
        time = 0.0

        for idx, seg in enumerate(segments):
            n_frames = max(1, int(seg.duration_sec * ARTICULATOR_RATE))

            # Get neighboring targets for coarticulation
            prev_target = segments[idx - 1].target if idx > 0 else None
            next_targets = [
                segments[idx + k].target
                for k in range(1, LOOKAHEAD + 1)
                if idx + k < len(segments)
            ]

            # Get coarticulation resistance for this phone's manner
            manner = seg.manner or "vowel"
            resistance = self.rules.resistance.get(manner, self.rules.resistance.get("vowel", {}))

            for frame_i in range(n_frames):
                t_in_phone = frame_i / n_frames  # 0..1 progress

                # Start with the phone's own target
                blended = seg.target.to_array().copy()

                # Carryover from previous phone (exponential decay)
                if prev_target is not None:
                    carryover_strength = np.exp(-t_in_phone * seg.duration_sec / CARRYOVER_TAU)
                    prev_arr = prev_target.to_array()
                    for p in range(len(blended)):
                        param_name = seg.target.PARAM_NAMES[p] if p < len(seg.target.PARAM_NAMES) else ""
                        r = resistance.get(param_name, 0.5)
                        # Less resistant articulators are more affected by carryover
                        blend_amount = carryover_strength * (1.0 - r)
                        blended[p] = blended[p] * (1.0 - blend_amount) + prev_arr[p] * blend_amount

                # Anticipatory: blend toward upcoming targets
                for look_i, next_tgt in enumerate(next_targets):
                    # Anticipation increases toward end of phone
                    anticipation = t_in_phone * 0.3 / (look_i + 1)
                    next_arr = next_tgt.to_array()
                    for p in range(len(blended)):
                        param_name = seg.target.PARAM_NAMES[p] if p < len(seg.target.PARAM_NAMES) else ""
                        r = resistance.get(param_name, 0.5)
                        blend_amount = anticipation * (1.0 - r)
                        blended[p] = blended[p] * (1.0 - blend_amount) + next_arr[p] * blend_amount

                # Clamp to valid ranges
                blended[:7] = np.clip(blended[:7], -3.0, 3.0)
                blended[7] = np.clip(blended[7], 0.0, 1.0)

                frame_target = ArticulatorTarget.from_array(blended)
                # Preserve glottal params from the original target
                frame_target.f0 = seg.target.f0
                frame_target.Rd = seg.target.Rd
                frame_target.aspiration = seg.target.aspiration
                frame_target.phonation_type = seg.target.phonation_type
                frame_target.noise_source_section = seg.target.noise_source_section
                frame_target.noise_amplitude = seg.target.noise_amplitude

                trajectory.append((frame_target, time))
                time += 1.0 / ARTICULATOR_RATE

        return trajectory

    def compute_transitions(
        self,
        segments: List[PhoneSegment],
    ) -> List[float]:
        """Compute transition durations between each pair of segments.

        Uses biomechanical speed constraints and manner-dependent
        multipliers.

        Args:
            segments: Phone segments in order

        Returns:
            List of N-1 transition durations in seconds
        """
        durations = []
        for i in range(len(segments) - 1):
            base_dur = compute_transition_duration(
                segments[i].target, segments[i + 1].target
            )
            # Apply manner-dependent multiplier
            manner = segments[i + 1].manner or "vowel"
            mult = self.rules.transition_multipliers.get(manner, 1.0)
            durations.append(base_dur * mult)
        return durations

"""Maeda (1990) articulatory model — 7-parameter PCA representation.

Models the human vocal tract articulators as 7 principal components
derived from midsagittal X-ray tracings. Each parameter controls one
degree of freedom, normalized to [-3, +3] standard deviations.

Parameters:
  1. Jaw position (positive = open)
  2. Tongue dorsal position (positive = front)
  3. Tongue dorsal shape (positive = arched/bunched)
  4. Tongue tip (positive = raised)
  5. Lip height (positive = open)
  6. Lip protrusion (positive = protruded)
  7. Larynx height (positive = raised)
  + Velum (0 = closed/oral, 1 = fully open/nasal)

Biomechanical constraints limit articulator speed based on measured
maximum velocities from electromagnetic articulography data.

References:
  Maeda, S. (1990). Compensatory articulation during speech: Evidence
    from the analysis and synthesis of vocal-tract shapes using an
    articulatory model. In W.J. Hardcastle & A. Marchal (Eds.),
    Speech Production and Speech Modelling, pp. 131-149. Kluwer.
  Perkell, J.S., et al. (2002). Economy of effort in different speaking
    conditions. JASA, 112(4), 1627-1641.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class ArticulatorTarget:
    """Target articulatory configuration for a phone.

    All parameters normalized to [-3, +3] (standard deviations from
    the neutral/mean configuration). Velum is [0, 1].
    """
    jaw: float = 0.0
    tongue_dorsal_pos: float = 0.0
    tongue_dorsal_shape: float = 0.0
    tongue_tip: float = 0.0
    lip_height: float = 0.0
    lip_protrusion: float = 0.0
    larynx_height: float = 0.0
    velum: float = 0.0  # 0=closed (oral), 1=open (nasal)

    # Glottal parameters (carried along for convenience)
    f0: Optional[float] = None
    Rd: Optional[float] = None
    aspiration: float = 0.0
    phonation_type: str = "modal"

    # Fricative noise generation
    noise_source_section: Optional[int] = None  # Tube section for turbulence
    noise_amplitude: float = 0.0

    def to_array(self) -> np.ndarray:
        """Convert to 8-element array [jaw, tdp, tds, tt, lh, lp, lx, vel]."""
        return np.array([
            self.jaw, self.tongue_dorsal_pos, self.tongue_dorsal_shape,
            self.tongue_tip, self.lip_height, self.lip_protrusion,
            self.larynx_height, self.velum,
        ], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "ArticulatorTarget":
        """Create from 8-element array."""
        return cls(
            jaw=float(arr[0]),
            tongue_dorsal_pos=float(arr[1]),
            tongue_dorsal_shape=float(arr[2]),
            tongue_tip=float(arr[3]),
            lip_height=float(arr[4]),
            lip_protrusion=float(arr[5]),
            larynx_height=float(arr[6]),
            velum=float(arr[7]) if len(arr) > 7 else 0.0,
        )

    PARAM_NAMES: list = field(default=None, repr=False, init=False)

    def __post_init__(self):
        self.PARAM_NAMES = [
            "jaw", "tongue_dorsal_pos", "tongue_dorsal_shape",
            "tongue_tip", "lip_height", "lip_protrusion",
            "larynx_height", "velum",
        ]


# Maximum articulator velocities in parameter-units per second.
# Derived from EMA studies (Perkell et al. 2002, Tasko & Westbury 2002).
# Jaw and lips are fastest; tongue body is slower; larynx is slowest.
MAX_VELOCITIES = {
    "jaw": 20.0,             # Fast — small mass, strong muscles
    "tongue_dorsal_pos": 12.0,  # Moderate — large mass
    "tongue_dorsal_shape": 14.0,  # Moderate
    "tongue_tip": 25.0,      # Very fast — low mass, agile
    "lip_height": 22.0,      # Fast
    "lip_protrusion": 18.0,  # Moderate-fast
    "larynx_height": 8.0,    # Slow — heavy structure
    "velum": 15.0,           # Moderate
}


def interpolate_targets(
    start: ArticulatorTarget,
    end: ArticulatorTarget,
    num_frames: int,
    sample_rate: int = 44100,
    frame_rate: int = 200,
) -> List[ArticulatorTarget]:
    """Interpolate between two articulatory targets with biomechanical rate limiting.

    Uses critically-damped interpolation clamped to maximum articulator
    velocities. This prevents unrealistic instantaneous jumps while
    preserving natural articulator dynamics.

    Args:
        start: Starting configuration
        end: Target configuration
        num_frames: Number of interpolation frames to produce
        sample_rate: Audio sample rate (for computing time)
        frame_rate: Articulator update rate in Hz

    Returns:
        List of ArticulatorTarget at each interpolation frame
    """
    if num_frames <= 0:
        return []
    if num_frames == 1:
        return [end]

    dt = 1.0 / frame_rate  # Time step between frames
    start_arr = start.to_array()
    end_arr = end.to_array()
    delta = end_arr - start_arr

    # Max change per frame for each parameter
    max_vel_arr = np.array([
        MAX_VELOCITIES["jaw"],
        MAX_VELOCITIES["tongue_dorsal_pos"],
        MAX_VELOCITIES["tongue_dorsal_shape"],
        MAX_VELOCITIES["tongue_tip"],
        MAX_VELOCITIES["lip_height"],
        MAX_VELOCITIES["lip_protrusion"],
        MAX_VELOCITIES["larynx_height"],
        MAX_VELOCITIES["velum"],
    ], dtype=np.float64)
    max_step = max_vel_arr * dt

    frames = []
    current = start_arr.copy()

    for i in range(num_frames):
        # Progress fraction with smooth ease-in/ease-out (cosine interpolation)
        t = (i + 1) / num_frames
        smooth_t = 0.5 * (1.0 - np.cos(np.pi * t))

        # Target position at this time
        target = start_arr + delta * smooth_t

        # Rate-limit: clamp step to max velocity
        step = target - current
        step = np.clip(step, -max_step, max_step)
        current = current + step

        # Clamp parameters to valid ranges
        current[:7] = np.clip(current[:7], -3.0, 3.0)
        current[7] = np.clip(current[7], 0.0, 1.0)  # velum

        target_frame = ArticulatorTarget.from_array(current)

        # Interpolate glottal parameters linearly (no biomechanical limit)
        if start.f0 is not None and end.f0 is not None:
            target_frame.f0 = start.f0 + (end.f0 - start.f0) * t
        else:
            target_frame.f0 = end.f0

        if start.Rd is not None and end.Rd is not None:
            target_frame.Rd = start.Rd + (end.Rd - start.Rd) * t
        else:
            target_frame.Rd = end.Rd

        target_frame.aspiration = start.aspiration + (end.aspiration - start.aspiration) * t
        target_frame.phonation_type = end.phonation_type
        target_frame.noise_source_section = end.noise_source_section
        target_frame.noise_amplitude = start.noise_amplitude + (end.noise_amplitude - start.noise_amplitude) * t

        frames.append(target_frame)

    return frames


def compute_transition_duration(
    start: ArticulatorTarget,
    end: ArticulatorTarget,
    min_duration_sec: float = 0.02,
    max_duration_sec: float = 0.15,
) -> float:
    """Estimate minimum transition duration based on articulator distances.

    The transition takes as long as the slowest articulator needs to
    travel its required distance at maximum speed.

    Args:
        start: Starting configuration
        end: Target configuration
        min_duration_sec: Minimum transition time (20ms)
        max_duration_sec: Maximum transition time (150ms)

    Returns:
        Estimated transition duration in seconds
    """
    start_arr = start.to_array()
    end_arr = end.to_array()
    delta = np.abs(end_arr - start_arr)

    max_vel_arr = np.array([
        MAX_VELOCITIES["jaw"],
        MAX_VELOCITIES["tongue_dorsal_pos"],
        MAX_VELOCITIES["tongue_dorsal_shape"],
        MAX_VELOCITIES["tongue_tip"],
        MAX_VELOCITIES["lip_height"],
        MAX_VELOCITIES["lip_protrusion"],
        MAX_VELOCITIES["larynx_height"],
        MAX_VELOCITIES["velum"],
    ], dtype=np.float64)

    # Time each articulator needs at max speed
    times = np.where(max_vel_arr > 0, delta / max_vel_arr, 0.0)
    # The slowest articulator determines the minimum duration
    required = float(np.max(times))

    # Apply limits
    return np.clip(required, min_duration_sec, max_duration_sec)

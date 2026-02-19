"""Nasal tract side-branch model.

Models the nasal cavity as a 15-section fixed-geometry waveguide
coupled to the oral tract at the velopharyngeal port (velum).
The velum aperture controls nasal coupling, creating the characteristic
zeros (anti-resonances) of nasal consonants and nasalized vowels.

The nasal tract geometry is fixed — unlike the oral tract, nasal
cavity shape doesn't change during speech. Area values from
Dang & Honda (1996) MRI measurements of the nasal passages.

The coupling point is at oral tract section ~16 (uvular region),
where the velum connects the oropharynx to the nasopharynx.

Processing:
  1. At each sample, the velum coupling splits energy between
     oral and nasal paths.
  2. The nasal tract processes its branch independently.
  3. Nasal output radiates from the nostrils (with a different
     radiation characteristic than the lips).

References:
  Dang, J. & Honda, K. (1996). Acoustic characteristics of the
    human paranasal sinuses derived from transmission characteristic
    measurement and morphological observation. JASA, 100(5), 3374.
  Maeda, S. (1982). A digital simulation method of the vocal-tract
    system. Speech Communication, 1(3-4), 199-229.
"""

import numpy as np

# Number of nasal tract sections
NUM_NASAL_SECTIONS = 15

# Oral tract section where velum couples (uvular region)
COUPLING_SECTION = 16

# Maximum velopharyngeal port area in cm² (fully open velum)
MAX_VP_AREA_CM2 = 2.0

# Fixed nasal tract area function (cm²), from Dang & Honda (1996).
# Section 0 = nasopharynx (near velum), section 14 = nostrils.
# The complex turbinate structures create irregular geometry.
NASAL_AREAS = np.array([
    2.0,   # Nasopharynx (wide)
    1.8,   # Upper nasopharynx
    1.5,   # Choanae (posterior nares)
    1.2,   # Posterior turbinate region
    0.9,   # Middle turbinate region (narrowing)
    0.7,   # Inferior turbinate constriction
    0.6,   # Minimum area (turbinate squeeze)
    0.7,   # Anterior to turbinates
    0.9,   # Nasal valve region
    1.0,   # Internal nasal valve
    1.1,   # Vestibule entrance
    1.3,   # Nasal vestibule
    1.5,   # External naris approach
    1.8,   # External naris
    2.0,   # Nostril opening
], dtype=np.float64)


def compute_nasal_reflection_coefficients(areas: np.ndarray) -> np.ndarray:
    """Compute reflection coefficients for nasal tract sections."""
    N = len(areas)
    k = np.zeros(N - 1)
    for i in range(N - 1):
        a_sum = areas[i] + areas[i + 1]
        if a_sum > 0:
            k[i] = (areas[i + 1] - areas[i]) / a_sum
    return k


class NasalTract:
    """Fixed-geometry nasal tract waveguide with velum coupling.

    The nasal tract runs in parallel with the oral tract. Energy
    enters through the velopharyngeal port and exits through the
    nostrils. The velum aperture (0-1) controls how much energy
    is diverted into the nasal branch.
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.num_sections = NUM_NASAL_SECTIONS

        # Waveguide delay lines
        self.forward = np.zeros(self.num_sections, dtype=np.float64)
        self.backward = np.zeros(self.num_sections, dtype=np.float64)

        # Fixed geometry
        self._areas = NASAL_AREAS.copy()
        self._reflection_coefficients = compute_nasal_reflection_coefficients(self._areas)

        # Nostril reflection (similar to lip radiation but slightly more closed)
        self.nostril_reflection = -0.4

        # Wall losses (slightly higher than oral tract due to mucosal lining)
        self.wall_loss_factor = 0.995

        # Velum state
        self._velum_area = 0.0  # Current velopharyngeal port area in cm²

    def set_velum(self, velum_opening: float) -> None:
        """Set velum opening.

        Args:
            velum_opening: 0.0 (closed/oral) to 1.0 (fully open/nasal)
        """
        self._velum_area = velum_opening * MAX_VP_AREA_CM2

    @property
    def velum_area(self) -> float:
        return self._velum_area

    @property
    def areas(self) -> np.ndarray:
        return self._areas.copy()

    def compute_coupling(
        self,
        oral_forward: float,
        oral_backward: float,
        oral_area: float,
    ) -> tuple:
        """Compute three-way junction at the velopharyngeal port.

        At the velum, the oral tract branches into the continued oral
        path and the nasal side-branch. This is a three-way scattering
        junction with areas: oral_in, oral_out (same area), and nasal.

        Args:
            oral_forward: Forward wave in oral tract at coupling section
            oral_backward: Backward wave in oral tract at coupling section
            oral_area: Oral tract area at coupling section (cm²)

        Returns:
            Tuple of:
              - oral_forward_out: Modified forward wave continuing in oral tract
              - oral_backward_out: Modified backward wave continuing in oral tract
              - nasal_input: Wave entering the nasal tract
        """
        if self._velum_area < 0.001:
            # Velum closed — no nasal coupling
            return oral_forward, oral_backward, 0.0

        nasal_area = self._areas[0]  # Nasopharynx area
        vp_area = self._velum_area

        # Three-port junction scattering
        # Total area at junction: oral + nasal (weighted by coupling)
        total_area = oral_area + vp_area
        if total_area < 0.001:
            return oral_forward, oral_backward, 0.0

        # Coupling coefficient
        coupling = vp_area / total_area

        # Energy splits at the junction
        # Oral path continues with reduced amplitude
        oral_forward_out = oral_forward * (1.0 - coupling) + self.backward[0] * coupling
        oral_backward_out = oral_backward * (1.0 - coupling)

        # Nasal branch receives coupled energy
        nasal_input = oral_forward * coupling - self.backward[0] * (1.0 - coupling)

        return oral_forward_out, oral_backward_out, nasal_input

    def process_sample(self, nasal_input: float) -> float:
        """Process one sample through the nasal tract.

        Args:
            nasal_input: Input wave from velopharyngeal coupling

        Returns:
            Nasal radiation output (volume velocity at nostrils)
        """
        N = self.num_sections
        k = self._reflection_coefficients
        fwd = self.forward
        bwd = self.backward

        # Inject at nasopharynx (section 0)
        fwd[0] = nasal_input

        # Scattering at internal junctions
        new_fwd = np.zeros(N, dtype=np.float64)
        new_bwd = np.zeros(N, dtype=np.float64)
        new_fwd[0] = fwd[0]

        for i in range(N - 1):
            k_plus = (1.0 + k[i]) * 0.5
            k_minus = (1.0 - k[i]) * 0.5
            new_fwd[i + 1] = k_plus * fwd[i] + k_minus * bwd[i + 1]
            new_bwd[i] = k_minus * fwd[i] + k_plus * bwd[i + 1]

        # Nostril reflection
        new_bwd[N - 1] = self.nostril_reflection * fwd[N - 1]

        # Wall losses
        new_fwd *= self.wall_loss_factor
        new_bwd *= self.wall_loss_factor

        # Update delay lines
        self.forward[:] = new_fwd
        self.backward[:] = new_bwd

        # Output: volume velocity at nostrils
        output = fwd[N - 1] * (1.0 + self.nostril_reflection)
        return output

    def reset(self) -> None:
        """Reset nasal tract delay lines."""
        self.forward[:] = 0.0
        self.backward[:] = 0.0
        self._velum_area = 0.0

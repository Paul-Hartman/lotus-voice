"""Kelly-Lochbaum acoustic tube model for the vocal tract.

Models the vocal tract as a cascade of cylindrical tube sections.
Sound propagates as forward (+) and backward (-) traveling pressure waves.
At each junction between sections of different cross-sectional area,
waves partially reflect and partially transmit (Smith one-multiply form).

The waveguide uses N sections where each section-to-section transition
contributes one sample of propagation delay per direction. For N sections
and N-1 junctions, the one-way delay is N-1 samples and the round trip
is 2*(N-1) samples. For a closed-open tube (vocal tract), the fundamental
is at fs/(4*(N-1)).

With N=23 sections: f1 = 44100/(4*22) = 501 Hz, matching the ~17.5 cm
adult male vocal tract.

The articulatory model produces 44-element area functions (Maeda PCA),
which are resampled to 23 waveguide sections internally.

Losses:
  - Per-section wall loss factor (gentle, physically motivated)
  - Area-dependent: narrower sections lose slightly more energy

References:
  Kelly, J.L. & Lochbaum, C.C. (1962). Speech synthesis.
  Smith, J.O. (2010). Physical Audio Signal Processing. W3K Publishing.
  Maeda, S. (1982). A digital simulation method of the vocal-tract system.
  Story, B.H. et al. (1996). Vocal tract area functions from MRI. JASA.
"""

import math

import numpy as np

# Speed of sound in the vocal tract (warm, humid air)
SPEED_OF_SOUND_CM_S = 35000.0

# Default sample rate
DEFAULT_SAMPLE_RATE = 44100

# Number of waveguide sections (gives f1 ≈ 501 Hz for uniform tube)
DEFAULT_NUM_SECTIONS = 23

# Number of area function measurement points (from Maeda PCA model)
NUM_AREA_POINTS = 44


def compute_section_length(sample_rate: int = DEFAULT_SAMPLE_RATE) -> float:
    """Compute the physical length of one waveguide section.

    Each section-to-section transition is one sample of delay.
    With N=23 sections, the tube length is (N-1) * section_length.

    Returns:
        Section length in cm
    """
    return SPEED_OF_SOUND_CM_S / sample_rate


def resample_area_function(
    areas: np.ndarray,
    target_sections: int = DEFAULT_NUM_SECTIONS,
) -> np.ndarray:
    """Resample an area function to a different number of sections.

    Typically resamples from 44 PCA measurement points to 23 waveguide
    sections using linear interpolation.

    Args:
        areas: Input area function (any length)
        target_sections: Desired number of output sections

    Returns:
        Resampled area function of length target_sections
    """
    if len(areas) == target_sections:
        return np.asarray(areas, dtype=np.float64)
    src_x = np.linspace(0, 1, len(areas))
    dst_x = np.linspace(0, 1, target_sections)
    return np.interp(dst_x, src_x, areas)


def compute_reflection_coefficients(areas: np.ndarray) -> np.ndarray:
    """Compute reflection coefficients from cross-sectional areas.

    At junction between section i (area A_i) and section i+1 (area A_{i+1}):
      k_i = (A_{i+1} - A_i) / (A_{i+1} + A_i)

    Using the convention where positive k means area INCREASES (expansion).
    k = 0 means perfect transmission (uniform tube).

    Args:
        areas: Array of N cross-sectional areas in cm²

    Returns:
        Array of N-1 reflection coefficients
    """
    N = len(areas)
    k = np.zeros(N - 1)
    for i in range(N - 1):
        a_sum = areas[i] + areas[i + 1]
        if a_sum > 0:
            k[i] = (areas[i + 1] - areas[i]) / a_sum
        else:
            k[i] = 0.0
    return k


def compute_section_losses(
    areas: np.ndarray,
    base_loss: float = 0.999,
) -> np.ndarray:
    """Compute per-section loss factors based on cross-sectional area.

    Narrower sections have slightly higher losses (more wall interaction
    per unit volume). The loss factor is the base loss scaled by the
    inverse square root of the area, clamped to a reasonable range.

    Target formant bandwidth is ~60-100 Hz, corresponding to a total
    round-trip loss that gives exp(-pi*B/fs) ≈ 0.993 per round trip.
    Distributed across 2*(N-1) section traversals, each section
    contributes base_loss ≈ 0.999.

    Args:
        areas: Cross-sectional areas in cm²
        base_loss: Loss factor for a 1 cm² section

    Returns:
        Per-section loss factors (multiply waves by this each sample)
    """
    N = len(areas)
    losses = np.ones(N, dtype=np.float64)
    for i in range(N):
        area = max(areas[i], 0.05)
        # Narrower sections lose more: loss scales with 1/sqrt(area)
        # Normalized so that 1 cm² section gets exactly base_loss
        losses[i] = base_loss ** (1.0 / math.sqrt(area))
    # Clamp to prevent instability or excessive damping
    np.clip(losses, 0.995, 0.9999, out=losses)
    return losses


class TubeModel:
    """Kelly-Lochbaum waveguide vocal tract model.

    Forward (+) waves travel from glottis to lips.
    Backward (-) waves travel from lips to glottis.

    Uses the Smith one-multiply scattering junction, which gives
    full transmission for uniform tubes (k=0) and correct partial
    reflection for area discontinuities.

    Accepts area functions of any length — resamples to the
    waveguide section count internally.
    """

    def __init__(
        self,
        num_sections: int = DEFAULT_NUM_SECTIONS,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        wall_loss_factor: float = 0.999,
    ):
        """
        Args:
            num_sections: Number of waveguide sections (default 23)
            sample_rate: Audio sample rate in Hz
            wall_loss_factor: Base per-section energy loss factor
        """
        self.num_sections = num_sections
        self.sample_rate = sample_rate
        self.wall_loss_factor = wall_loss_factor
        self.section_length_cm = compute_section_length(sample_rate)

        # Traveling wave delay lines
        self.forward = np.zeros(num_sections, dtype=np.float64)
        self.backward = np.zeros(num_sections, dtype=np.float64)

        # Area function and derived coefficients
        self._areas = np.ones(num_sections, dtype=np.float64)
        self._reflection_coefficients = np.zeros(num_sections - 1, dtype=np.float64)
        # Precomputed scattering: cos(θ) = sqrt(1-k²) for energy conservation
        self._scatter_cos = np.ones(num_sections - 1, dtype=np.float64)
        # sqrt(area) for normalized ↔ raw conversion at boundaries
        self._sqrt_areas = np.ones(num_sections, dtype=np.float64)

        # Per-section loss factors (area-dependent)
        self._section_losses = np.full(num_sections, wall_loss_factor, dtype=np.float64)

        # Glottal reflection coefficient (partially closed end)
        self.glottal_reflection = 0.75

        # Lip reflection coefficient (open end → negative/inverted)
        self.lip_reflection = -0.5

    def set_area_function(self, areas: np.ndarray) -> None:
        """Set the cross-sectional area function.

        Accepts area functions of any length — resamples to match
        the waveguide section count if needed. Typically receives
        44-element arrays from the Maeda PCA model and resamples
        to 23 waveguide sections.

        Args:
            areas: Array of cross-sectional areas in cm²
        """
        areas = resample_area_function(areas, self.num_sections)
        self._areas = np.asarray(areas, dtype=np.float64)
        self._reflection_coefficients = compute_reflection_coefficients(self._areas)

        # Precompute orthogonal scattering coefficients
        k = self._reflection_coefficients
        self._scatter_cos = np.sqrt(1.0 - k * k)

        # sqrt(area) for normalized wave variable conversion
        self._sqrt_areas = np.sqrt(np.maximum(self._areas, 0.01))

        # Per-section losses (area-dependent)
        self._section_losses = compute_section_losses(
            self._areas, self.wall_loss_factor,
        )

        # Lip reflection from lip area.
        # Even for wide-open vowels, the lip radiation impedance reflects
        # a significant fraction of energy. The reflection only drops
        # substantially for very large openings (e.g., bilabial release).
        # Range: -0.8 (near-closed) to -0.3 (wide open)
        lip_area = self._areas[-1]
        self.lip_reflection = -0.8 + 0.5 * min(lip_area / 10.0, 1.0)

    @property
    def areas(self) -> np.ndarray:
        return self._areas.copy()

    @property
    def reflection_coefficients(self) -> np.ndarray:
        return self._reflection_coefficients.copy()

    def process_sample(self, glottal_input: float) -> float:
        """Process one sample through the tube model.

        Uses normalized wave variables (p̃ = p * √A) with orthogonal
        scattering to guarantee energy conservation and numerical stability:

          new_fwd[i+1] = cos(θ)*fwd[i] + k[i]*bwd[i+1]
          new_bwd[i]   = -k[i]*fwd[i] + cos(θ)*bwd[i+1]

        where cos(θ) = √(1 - k²) and k = (A_{i+1}-A_i)/(A_{i+1}+A_i).

        The delay lines store normalized variables. Conversion to/from
        raw pressure happens only at the boundaries (glottis and lips).

        Args:
            glottal_input: Glottal source sample (flow derivative)

        Returns:
            Volume velocity at the lips
        """
        N = self.num_sections
        k = self._reflection_coefficients
        c = self._scatter_cos
        fwd = self.forward
        bwd = self.backward

        # 1. Glottal boundary: inject source in normalized coordinates
        #    Raw: p_fwd = input + R_g * p_bwd
        #    Normalized: p̃_fwd = √A₀ * input + R_g * p̃_bwd
        glottal_fwd = self._sqrt_areas[0] * glottal_input + self.glottal_reflection * bwd[0]

        # 2. Orthogonal scattering at each junction (energy-conserving)
        new_fwd = np.zeros(N, dtype=np.float64)
        new_bwd = np.zeros(N, dtype=np.float64)
        new_fwd[0] = glottal_fwd

        for i in range(N - 1):
            new_fwd[i + 1] = c[i] * fwd[i] + k[i] * bwd[i + 1]
            new_bwd[i] = -k[i] * fwd[i] + c[i] * bwd[i + 1]

        # 3. Lip reflection on the newly-arrived forward wave
        #    Same formula in normalized coordinates: p̃_bwd = R_l * p̃_fwd
        new_bwd[N - 1] = self.lip_reflection * new_fwd[N - 1]

        # 4. Apply per-section wall losses
        new_fwd *= self._section_losses
        new_bwd *= self._section_losses

        # 5. Output: convert from normalized back to raw at the lips
        #    Raw output = p̃_fwd / √A_{N-1} * (1 + R_l)
        lip_output = new_fwd[N - 1] / self._sqrt_areas[N - 1] * (1.0 + self.lip_reflection)

        # 6. Update delay lines
        self.forward[:] = new_fwd
        self.backward[:] = new_bwd

        return lip_output

    def process_block(self, glottal_input: np.ndarray) -> np.ndarray:
        """Process a block of samples through the tube model.

        Args:
            glottal_input: Array of glottal source samples

        Returns:
            Array of volume velocity at the lips
        """
        output = np.zeros(len(glottal_input), dtype=np.float64)
        for i in range(len(glottal_input)):
            output[i] = self.process_sample(glottal_input[i])
        return output

    def reset(self) -> None:
        """Reset all delay lines to zero."""
        self.forward[:] = 0.0
        self.backward[:] = 0.0

    def get_total_length_cm(self) -> float:
        """Return total vocal tract length in cm."""
        return self.num_sections * self.section_length_cm

    def estimate_formants(self, num_formants: int = 5) -> list[float]:
        """Estimate formant frequencies from the current area function.

        Uses two methods and returns the best:
        1. LPC (linear prediction) root-finding on the impulse response
        2. Peak-picking on the FFT magnitude spectrum (fallback)

        LPC models the vocal tract transfer function as an all-pole filter,
        whose pole frequencies correspond to formants.

        Args:
            num_formants: Number of formants to extract

        Returns:
            List of formant frequencies in Hz
        """
        # Save state
        saved_fwd = self.forward.copy()
        saved_bwd = self.backward.copy()
        self.reset()

        # Generate impulse response
        ir_length = 4096
        ir = np.zeros(ir_length, dtype=np.float64)
        ir[0] = self.process_sample(1.0)
        for i in range(1, ir_length):
            ir[i] = self.process_sample(0.0)

        # Restore state
        self.forward[:] = saved_fwd
        self.backward[:] = saved_bwd

        # Try LPC method first
        formants = self._lpc_formants(ir, num_formants)
        if len(formants) >= min(num_formants, 3):
            return formants[:num_formants]

        # Fallback: FFT peak picking with improved detection
        return self._fft_formants(ir, num_formants)

    def _lpc_formants(self, signal: np.ndarray, num_formants: int) -> list[float]:
        """Extract formants via LPC (Linear Predictive Coding).

        Fits an all-pole model to the signal, then finds the pole
        frequencies. Poles near the unit circle with positive frequency
        correspond to formants.
        """
        # LPC order: 2 per expected formant + 2 for spectral tilt
        lpc_order = 2 * num_formants + 4

        # Pre-emphasis to flatten spectrum
        pre = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])

        # Window
        windowed = pre[:min(len(pre), 2048)] * np.hamming(min(len(pre), 2048))

        # Autocorrelation method for LPC coefficients
        n = len(windowed)
        r = np.correlate(windowed, windowed, mode='full')
        r = r[n - 1:]  # Keep positive lags

        if r[0] == 0:
            return []

        # Levinson-Durbin recursion
        try:
            a = self._levinson_durbin(r, lpc_order)
        except (ValueError, np.linalg.LinAlgError):
            return []

        # Find roots of the LPC polynomial
        roots = np.roots(a)

        # Keep roots inside unit circle (stable) with positive frequency
        formant_freqs = []
        for root in roots:
            if np.imag(root) >= 0:  # Positive frequency only
                freq = np.abs(np.arctan2(np.imag(root), np.real(root))) * self.sample_rate / (2.0 * np.pi)
                bw = -0.5 * self.sample_rate / (2.0 * np.pi) * np.log(np.abs(root) + 1e-10)

                # Formant criteria: reasonable frequency and bandwidth
                # Allow wider bandwidths for open vowels with low lip reflection
                if 80 < freq < 5500 and bw < 800:
                    formant_freqs.append(freq)

        formant_freqs.sort()
        return formant_freqs[:num_formants]

    @staticmethod
    def _levinson_durbin(r: np.ndarray, order: int) -> np.ndarray:
        """Levinson-Durbin recursion for LPC coefficients.

        Returns the polynomial coefficients [1, a1, a2, ..., a_order].
        """
        a = np.zeros(order + 1)
        a[0] = 1.0
        e = r[0]

        for i in range(1, order + 1):
            # Compute reflection coefficient
            acc = sum(a[j] * r[i - j] for j in range(1, i))
            k = -(r[i] + acc) / e if e != 0 else 0

            # Update coefficients
            a_new = a.copy()
            for j in range(1, i):
                a_new[j] = a[j] + k * a[i - j]
            a_new[i] = k
            a = a_new

            e *= (1.0 - k * k)
            if e <= 0:
                break

        return a

    def _fft_formants(self, ir: np.ndarray, num_formants: int) -> list[float]:
        """Extract formants via FFT peak picking (fallback method)."""
        n_fft = 16384  # More zero-padding for frequency resolution
        spectrum = np.abs(np.fft.rfft(ir, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.sample_rate)

        spectrum_db = 20.0 * np.log10(spectrum + 1e-10)

        # Light smoothing to preserve formant peaks
        kernel_size = 9
        kernel = np.ones(kernel_size) / kernel_size
        spectrum_smooth = np.convolve(spectrum_db, kernel, mode='same')

        # Find peaks with relaxed prominence (some formants have low Q)
        formants = []
        min_prominence = 1.5  # dB — lowered for broad formants

        for i in range(2, len(spectrum_smooth) - 2):
            if (80 < freqs[i] < 5500
                    and spectrum_smooth[i] > spectrum_smooth[i - 1]
                    and spectrum_smooth[i] > spectrum_smooth[i + 1]
                    and spectrum_smooth[i] > spectrum_smooth[i - 2]
                    and spectrum_smooth[i] > spectrum_smooth[i + 2]):
                left_min = np.min(spectrum_smooth[max(0, i - 40):i])
                right_min = np.min(spectrum_smooth[i + 1:min(len(spectrum_smooth), i + 40)])
                prominence = spectrum_smooth[i] - max(left_min, right_min)
                if prominence > min_prominence:
                    formants.append(float(freqs[i]))
                    if len(formants) >= num_formants:
                        break

        return formants

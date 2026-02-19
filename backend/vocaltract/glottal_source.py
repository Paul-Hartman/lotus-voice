"""LF glottal pulse model (Fant, Liljencrants & Lin 1985).

Generates the glottal flow derivative waveform sample-by-sample.
Uses Fant's (1995) Rd parametrization to control voice quality with
a single perceptual parameter.

The LF model defines the differentiated glottal flow as two phases:
  Opening phase (0 <= t <= te):  E(t) = E0 * exp(alpha*t) * sin(omega_g*t)
  Return phase  (te < t <= T0):  E(t) = -E_e/(epsilon*ta) * (exp(-epsilon*(t-te)) - exp(-epsilon*(T0-te)))

Key parameters derived from Rd:
  Rd: 0.3 (pressed) -> 1.0 (modal) -> 2.7 (breathy) -> 6.0 (extreme breathy)

References:
  Fant, G., Liljencrants, J., & Lin, Q. (1985). A four-parameter model
    of glottal flow. STL-QPSR, 26(4), 1-13.
  Fant, G. (1995). The LF-model revisited. Transformations and frequency
    domain analysis. STL-QPSR, 36(2-3), 119-156.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from vocaltract.state import GlottalState


@dataclass
class LFParams:
    """Resolved LF model timing parameters for one glottal cycle.

    All times normalized to the fundamental period T0 = 1/f0.
    """
    T0: float        # Fundamental period (seconds)
    te: float        # Instant of maximum excitation (seconds)
    tp: float        # Instant of maximum glottal flow (seconds)
    ta: float        # Effective duration of return phase (seconds)
    E_e: float       # Amplitude of excitation at te
    alpha: float     # Exponential growth factor (opening phase)
    omega_g: float   # Angular frequency of opening phase
    epsilon: float   # Decay rate of return phase


def rd_to_lf_params(Rd: float, f0: float) -> LFParams:
    """Convert Rd voice quality parameter to LF timing parameters.

    Follows Fant (1995) regression equations relating Rd to the
    normalized timing parameters Rk, Rg, Ra.

    Args:
        Rd: Voice quality (0.3=pressed, 1.0=modal, 2.7=breathy)
        f0: Fundamental frequency in Hz

    Returns:
        Fully resolved LF parameters for one glottal cycle
    """
    Rd = max(0.3, min(Rd, 6.0))
    T0 = 1.0 / f0

    # Fant (1995) regression: Rd -> (Ra, Rk, Rg)
    Ra = (-1.0 + 4.8 * Rd) / 100.0
    Ra = max(0.001, min(Ra, 0.2))

    Rk = (22.4 + 11.8 * Rd) / 100.0
    Rk = max(0.2, min(Rk, 0.7))

    # Rg relates to open quotient via Rg = 1 / (2 * OQ)
    # From Fant (1997): approximation for typical Rd range
    OQ = 1.0 / (2.17 * (0.5 + 1.2 * Rd) / (1.0 + 1.2 * Rd))
    OQ = max(0.3, min(OQ, 0.95))
    Rg = 1.0 / (2.0 * OQ)

    # Derive timing from R-parameters
    tp = T0 / (2.0 * Rg)               # Peak flow time
    te = tp * (1.0 + Rk)               # Excitation instant
    ta = Ra * T0                         # Return phase effective duration

    # Ensure physical constraints
    te = min(te, 0.98 * T0)
    ta = max(ta, 0.001 * T0)
    ta = min(ta, T0 - te - 0.001 * T0)

    # Opening phase: find alpha and omega_g
    omega_g = math.pi / tp

    # Alpha is found by solving the implicit equation:
    #   alpha * sin(omega_g * te) + omega_g * cos(omega_g * te) = 0
    # (the derivative of the opening phase is zero at te)
    # Iterative solution using Newton's method
    alpha = _solve_alpha(omega_g, te, T0)

    # Return phase: epsilon from ta
    epsilon = 1.0 / ta

    # E_e: normalized amplitude of excitation
    E_e = 1.0

    return LFParams(
        T0=T0, te=te, tp=tp, ta=ta,
        E_e=E_e, alpha=alpha, omega_g=omega_g, epsilon=epsilon
    )


def _solve_alpha(omega_g: float, te: float, T0: float, max_iter: int = 50) -> float:
    """Solve for alpha using Newton's method.

    The constraint is that the opening phase waveform
    E(t) = exp(alpha*t) * sin(omega_g*t) has its negative peak at t=te.
    This means dE/dt = 0 at t=te:
      alpha*sin(omega_g*te) + omega_g*cos(omega_g*te) = 0
    => alpha = -omega_g * cos(omega_g*te) / sin(omega_g*te)
    """
    sin_te = math.sin(omega_g * te)
    cos_te = math.cos(omega_g * te)

    if abs(sin_te) < 1e-10:
        return 0.0

    alpha = -omega_g * cos_te / sin_te
    return alpha


class GlottalSource:
    """Sample-by-sample LF glottal source generator.

    Produces the differentiated glottal flow waveform that drives
    the vocal tract tube model.
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._phase: float = 0.0  # Phase within current cycle (0..T0)
        self._params: Optional[LFParams] = None
        self._state = GlottalState()

    @property
    def state(self) -> GlottalState:
        return self._state

    def set_params(self, f0: float, Rd: float = 1.0,
                   phonation_type: str = "modal") -> None:
        """Update glottal source parameters.

        Args:
            f0: Fundamental frequency (Hz)
            Rd: Voice quality parameter (0.3-6.0)
            phonation_type: Label for the phonation type
        """
        self._params = rd_to_lf_params(Rd, f0)
        self._state.f0 = f0
        self._state.Rd = Rd
        self._state.phonation_type = phonation_type

        # Derive perceptual parameters for state introspection
        oq = self._params.te / self._params.T0
        sq = self._params.tp / (self._params.te - self._params.tp) if self._params.te > self._params.tp else 2.0
        self._state.oq = oq
        self._state.speed_quotient = sq

    def generate_cycle(self) -> np.ndarray:
        """Generate one complete glottal cycle.

        Returns:
            Array of samples for one fundamental period
        """
        if self._params is None:
            self.set_params(f0=120.0)

        p = self._params
        num_samples = max(1, int(round(p.T0 * self.sample_rate)))
        output = np.zeros(num_samples, dtype=np.float64)

        te_sample = int(p.te * self.sample_rate)
        te_sample = min(te_sample, num_samples - 1)

        for i in range(num_samples):
            t = i / self.sample_rate
            if i <= te_sample:
                # Opening phase: E(t) = E0 * exp(alpha*t) * sin(omega_g*t)
                output[i] = math.exp(p.alpha * t) * math.sin(p.omega_g * t)
            else:
                # Return phase: exponential recovery
                t_rel = t - p.te
                output[i] = (-p.E_e / (p.epsilon * p.ta)) * (
                    math.exp(-p.epsilon * t_rel) - math.exp(-p.epsilon * (p.T0 - p.te))
                )

        # Normalize so the negative peak (at te) has amplitude E_e
        peak = np.min(output)
        if abs(peak) > 1e-10:
            output = output * (p.E_e / abs(peak))

        return output

    def generate_samples(self, num_samples: int) -> np.ndarray:
        """Generate a block of glottal source samples.

        Concatenates cycles to fill the requested number of samples.
        Handles fractional cycles by tracking phase.

        Args:
            num_samples: Number of samples to generate

        Returns:
            Array of glottal flow derivative samples
        """
        if self._params is None:
            self.set_params(f0=120.0)

        output = np.zeros(num_samples, dtype=np.float64)
        pos = 0

        while pos < num_samples:
            # Generate one cycle
            cycle = self.generate_cycle()
            cycle_len = len(cycle)

            # Handle phase offset within cycle
            start_in_cycle = int(self._phase * self.sample_rate)
            remaining_in_cycle = cycle_len - start_in_cycle
            remaining_to_fill = num_samples - pos

            n_copy = min(remaining_in_cycle, remaining_to_fill)
            output[pos:pos + n_copy] = cycle[start_in_cycle:start_in_cycle + n_copy]
            pos += n_copy

            if n_copy >= remaining_in_cycle:
                # Completed this cycle, reset phase
                self._phase = 0.0
            else:
                # Didn't finish the cycle, update phase
                self._phase = (start_in_cycle + n_copy) / self.sample_rate

        return output

    def add_noise(self, signal: np.ndarray, aspiration_level: float = 0.0) -> np.ndarray:
        """Add aspiration noise for breathy phonation.

        Aspiration noise is modulated by the glottal opening — strongest
        during the open phase, absent during the closed phase.

        Args:
            signal: Glottal source signal
            aspiration_level: 0.0 (none) to 1.0 (full aspiration)

        Returns:
            Signal with aspiration noise added
        """
        if aspiration_level <= 0.0:
            return signal

        noise = np.random.randn(len(signal)) * aspiration_level
        # Simple modulation: noise is stronger where signal is less negative
        envelope = np.clip(signal + 0.5, 0.0, 1.0)
        return signal * (1.0 - aspiration_level * 0.3) + noise * envelope

"""Lip radiation filter — Flanagan (1972) first-difference approximation.

The vocal tract produces volume velocity at the lips, but what we hear
is sound pressure. The conversion from volume velocity to radiated
pressure at a distance involves a radiation impedance that acts as a
high-pass filter with approximately +6 dB/octave slope.

The simplest and most standard approximation is a first-order difference:
  p_rad[n] = u_lip[n] - u_lip[n-1]

This is the time-domain equivalent of multiplying by jω (differentiation),
which gives the +6 dB/octave spectral tilt that converts the source
spectrum from -12 dB/octave (glottal flow derivative through the tract)
to the familiar -6 dB/octave of radiated speech.

A radiation coefficient R can model the lip aperture effect:
  p_rad[n] = R * u_lip[n] - u_lip[n-1]
  where R depends on the lip opening area (larger opening → more radiation)

References:
  Flanagan, J.L. (1972). Speech Analysis Synthesis and Perception.
    Springer-Verlag, 2nd edition.
"""

import numpy as np


class RadiationFilter:
    """First-difference lip radiation filter.

    Converts volume velocity at the lips to radiated sound pressure.
    """

    def __init__(self, radiation_coefficient: float = 0.97):
        """
        Args:
            radiation_coefficient: Controls high-frequency damping at the
                lip opening. 0.97 is standard for an average lip area.
                Range [0.9, 1.0]. Lower values attenuate high frequencies more.
        """
        self.R = radiation_coefficient
        self._prev_sample: float = 0.0

    def process_sample(self, sample: float) -> float:
        """Process a single sample through the radiation filter.

        Args:
            sample: Volume velocity at the lips

        Returns:
            Radiated sound pressure sample
        """
        output = sample - self.R * self._prev_sample
        self._prev_sample = sample
        return output

    def process_block(self, samples: np.ndarray) -> np.ndarray:
        """Process a block of samples through the radiation filter.

        Args:
            samples: Array of volume velocity samples

        Returns:
            Array of radiated sound pressure samples
        """
        output = np.zeros_like(samples)
        for i in range(len(samples)):
            output[i] = self.process_sample(samples[i])
        return output

    def reset(self) -> None:
        """Reset filter state."""
        self._prev_sample = 0.0

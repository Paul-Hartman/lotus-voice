"""Full introspection state for the vocal tract simulator.

Every module writes into these dataclasses so the API can return
complete snapshots of the simulator at any point in time.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GlottalState:
    """Instantaneous state of the glottal source."""
    f0: float = 120.0                  # Fundamental frequency (Hz)
    Rd: float = 1.0                    # LF shape parameter (0.3=pressed, 1.0=modal, 2.7=breathy)
    oq: float = 0.5                    # Open quotient (fraction of cycle glottis is open)
    speed_quotient: float = 2.0        # Ratio of opening phase to closing phase
    subglottal_pressure_cmH2O: float = 8.0
    phonation_type: str = "modal"      # modal | breathy | pressed | falsetto | whisper | creaky


@dataclass
class ArticulatorState:
    """Maeda 7-parameter articulatory configuration.

    All parameters normalized to [-3, +3] (standard deviations from mean).
    Based on Maeda (1990) PCA analysis of midsagittal X-ray data.
    """
    jaw: float = 0.0
    tongue_dorsal_pos: float = 0.0     # Front-back position
    tongue_dorsal_shape: float = 0.0   # Flat vs arched
    tongue_tip: float = 0.0            # Raised vs lowered
    lip_height: float = 0.0            # Open vs closed
    lip_protrusion: float = 0.0        # Protruded vs retracted
    larynx_height: float = 0.0         # Raised vs lowered
    velum: float = 0.0                 # 0=closed (oral), 1=open (nasal)


@dataclass
class TubeState:
    """State of the acoustic tube model."""
    num_sections: int = 44
    section_length_cm: float = 0.795   # c / (2 * sample_rate) at 44.1kHz
    area_function_cm2: List[float] = field(default_factory=list)
    reflection_coefficients: List[float] = field(default_factory=list)
    forward_pressure: List[float] = field(default_factory=list)
    backward_pressure: List[float] = field(default_factory=list)
    nasal_area_function_cm2: List[float] = field(default_factory=list)
    nasal_coupling_area_cm2: float = 0.0


@dataclass
class SubglottalState:
    """Respiratory system state."""
    lung_pressure_cmH2O: float = 8.0
    lung_volume_liters: float = 3.0
    airflow_liters_per_sec: float = 0.15
    diaphragm_tension: float = 0.5     # 0=relaxed, 1=fully contracted


@dataclass
class VocalTractState:
    """Complete simulator state at a single time instant.

    Returned by the synthesizer for full introspection at up to
    100 Hz (every ~441 samples at 44.1 kHz).
    """
    time_sec: float = 0.0
    phone: str = ""                    # Current IPA phone being synthesized
    phone_progress: float = 0.0        # 0.0-1.0 progress through current phone

    glottal: GlottalState = field(default_factory=GlottalState)
    articulators: ArticulatorState = field(default_factory=ArticulatorState)
    tube: TubeState = field(default_factory=TubeState)
    subglottal: SubglottalState = field(default_factory=SubglottalState)

    output_sample: float = 0.0        # Last audio sample value
    formants_hz: List[float] = field(default_factory=list)  # Estimated F1-F5

"""Articulatory parameters to vocal tract area function conversion.

Uses Maeda's (1990) PCA basis functions to map 7 articulatory parameters
to a 44-section cross-sectional area function. The area function is:

  A(x) = A_mean(x) + Σ_j  p_j × φ_j(x)

where:
  A_mean(x) = mean area function (from the neutral configuration)
  p_j       = j-th articulatory parameter value (in standard deviations)
  φ_j(x)    = j-th PCA basis function (change in area per SD)

The basis functions below are derived from published Maeda (1990) data
and Story et al. (1996) MRI measurements, discretized to 44 sections
(glottis at section 0, lips at section 43).

References:
  Maeda, S. (1990). Compensatory articulation during speech.
  Story, B.H., Titze, I.R., & Hoffman, E.A. (1996). Vocal tract area
    functions from MRI. JASA 100(1), 537-554.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from vocaltract.articulators import ArticulatorTarget

NUM_SECTIONS = 44

# Mean area function (cm²) — neutral/schwa configuration
# Approximately uniform tube, slight pharyngeal widening
MEAN_AREA = np.array([
    0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1,
    1.2, 1.3, 1.4, 1.5, 1.6, 1.6, 1.6, 1.6,
    1.5, 1.5, 1.4, 1.4, 1.3, 1.3, 1.3, 1.3,
    1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.4, 1.4,
    1.5, 1.5, 1.6, 1.6, 1.7, 1.7, 1.8, 1.8,
    1.9, 1.9, 2.0, 2.0,
], dtype=np.float64)

# PCA basis functions (φ_j): each is a 44-element vector showing how
# a +1 SD change in parameter j alters the area function.
#
# Sections 0-7: laryngeal tube / lower pharynx
# Sections 8-15: upper pharynx
# Sections 16-23: velar/uvular region
# Sections 24-31: palatal region
# Sections 32-39: alveolar region
# Sections 40-43: lip tube

# φ_1: Jaw opening — lowers tongue, opens lips, widens oral cavity
BASIS_JAW = np.array([
    0.00, 0.00, 0.00, 0.00, 0.02, 0.05, 0.08, 0.10,
    0.12, 0.15, 0.18, 0.22, 0.26, 0.30, 0.34, 0.38,
    0.42, 0.46, 0.50, 0.52, 0.54, 0.55, 0.55, 0.54,
    0.52, 0.50, 0.48, 0.46, 0.44, 0.42, 0.40, 0.38,
    0.36, 0.35, 0.34, 0.34, 0.35, 0.38, 0.42, 0.48,
    0.55, 0.60, 0.65, 0.70,
], dtype=np.float64)

# φ_2: Tongue dorsal position (front-back)
# Positive = fronted (constriction moves forward, pharynx widens)
BASIS_TONGUE_DORSAL_POS = np.array([
    0.00, 0.00, 0.00, 0.02, 0.05, 0.10, 0.18, 0.28,
    0.38, 0.48, 0.55, 0.58, 0.55, 0.45, 0.30, 0.10,
   -0.10,-0.30,-0.50,-0.65,-0.72,-0.70,-0.60,-0.45,
   -0.30,-0.15,-0.05, 0.02, 0.05, 0.05, 0.03, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00,
], dtype=np.float64)

# φ_3: Tongue dorsal shape (flat vs arched)
# Positive = arched/bunched (narrower at constriction, wider elsewhere)
BASIS_TONGUE_DORSAL_SHAPE = np.array([
    0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.05, 0.10,
    0.15, 0.18, 0.15, 0.08,-0.02,-0.15,-0.28,-0.38,
   -0.42,-0.40,-0.32,-0.20,-0.08, 0.05, 0.15, 0.22,
    0.25, 0.22, 0.15, 0.08, 0.02, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00,
], dtype=np.float64)

# φ_4: Tongue tip (raised vs lowered)
# Positive = raised tip (narrows alveolar region)
BASIS_TONGUE_TIP = np.array([
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.05,
    0.10, 0.15, 0.18, 0.15, 0.05,-0.10,-0.28,-0.42,
   -0.50,-0.48,-0.38,-0.25,-0.12,-0.02, 0.05, 0.08,
    0.08, 0.05, 0.02, 0.00,
], dtype=np.float64)

# φ_5: Lip height (opening)
# Positive = more open lips
BASIS_LIP_HEIGHT = np.array([
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.02, 0.05, 0.10, 0.18, 0.28,
    0.40, 0.52, 0.62, 0.70,
], dtype=np.float64)

# φ_6: Lip protrusion
# Positive = protruded (lengthens lip tube, narrows aperture)
BASIS_LIP_PROTRUSION = np.array([
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00,-0.05,-0.12,
   -0.22,-0.32,-0.40,-0.45,
], dtype=np.float64)

# φ_7: Larynx height
# Positive = raised larynx (shortens pharynx, raises formants)
BASIS_LARYNX_HEIGHT = np.array([
    0.10, 0.15, 0.18, 0.18, 0.15, 0.10, 0.05, 0.00,
   -0.05,-0.08,-0.08,-0.05,-0.02, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00,
], dtype=np.float64)

# Stack into matrix: shape (7, 44)
# Scale factor: the raw basis vectors are scaled so that a ±2 SD parameter
# change produces realistic area deviations from the mean. The scaling must
# balance sufficient vowel contrast against preventing area clamping at 0.
_BASIS_SCALE = 1.0

PCA_BASIS = np.stack([
    BASIS_JAW * _BASIS_SCALE,
    BASIS_TONGUE_DORSAL_POS * _BASIS_SCALE,
    BASIS_TONGUE_DORSAL_SHAPE * _BASIS_SCALE,
    BASIS_TONGUE_TIP * _BASIS_SCALE,
    BASIS_LIP_HEIGHT * _BASIS_SCALE,
    BASIS_LIP_PROTRUSION * _BASIS_SCALE,
    BASIS_LARYNX_HEIGHT * _BASIS_SCALE,
], axis=0)

# Minimum area (cm²) — prevents total closure for vowels/sonorants.
# For stops, constriction area is driven to near-zero explicitly.
MIN_AREA_CM2 = 0.05


@dataclass
class ConstrictionInfo:
    """Information about a vocal tract constriction point."""
    section: int           # Tube section index (0=glottis, 43=lips)
    area_cm2: float        # Cross-sectional area at constriction
    position_cm: float     # Distance from glottis in cm
    region: str            # Anatomical region name


def articulators_to_area_function(
    target: ArticulatorTarget,
    num_sections: int = NUM_SECTIONS,
) -> np.ndarray:
    """Convert articulatory parameters to vocal tract area function.

    Computes: A(x) = A_mean(x) + Σ_j  p_j × φ_j(x)
    Then applies minimum area floor and smooth clamping.

    Args:
        target: ArticulatorTarget with parameter values
        num_sections: Number of tube sections (must be 44)

    Returns:
        Array of cross-sectional areas in cm² (length num_sections)
    """
    if num_sections != NUM_SECTIONS:
        raise ValueError(f"Only {NUM_SECTIONS} sections supported, got {num_sections}")

    params = np.array([
        target.jaw,
        target.tongue_dorsal_pos,
        target.tongue_dorsal_shape,
        target.tongue_tip,
        target.lip_height,
        target.lip_protrusion,
        target.larynx_height,
    ], dtype=np.float64)

    # Linear combination: mean + params @ basis
    areas = MEAN_AREA.copy() + params @ PCA_BASIS

    # Smooth clamping: use softplus to avoid hard discontinuity at zero
    # softplus(x) = ln(1 + exp(x/scale)) * scale
    # This ensures areas stay positive while keeping gradients smooth
    scale = 0.1  # Controls sharpness of the floor
    areas = np.log1p(np.exp((areas - MIN_AREA_CM2) / scale)) * scale + MIN_AREA_CM2

    return areas


def find_constriction(
    areas: np.ndarray,
    section_length_cm: float = 0.795,
) -> Optional[ConstrictionInfo]:
    """Find the primary constriction point in the vocal tract.

    The constriction is where the tongue (or lips) comes closest to
    the palate/pharynx wall — the minimum area in the oral cavity
    (sections 8-43, excluding the laryngeal tube).

    Args:
        areas: Area function (44 sections)
        section_length_cm: Physical length of each section

    Returns:
        ConstrictionInfo or None if no significant constriction
    """
    # Search only oral cavity (sections 8+)
    oral_areas = areas[8:]
    min_idx = int(np.argmin(oral_areas)) + 8
    min_area = float(areas[min_idx])
    position_cm = min_idx * section_length_cm

    # Classify region
    if min_idx < 12:
        region = "pharyngeal"
    elif min_idx < 18:
        region = "uvular"
    elif min_idx < 24:
        region = "velar"
    elif min_idx < 30:
        region = "palatal"
    elif min_idx < 36:
        region = "alveolar"
    elif min_idx < 40:
        region = "dental"
    else:
        region = "labial"

    return ConstrictionInfo(
        section=min_idx,
        area_cm2=min_area,
        position_cm=position_cm,
        region=region,
    )


def find_all_constrictions(
    areas: np.ndarray,
    threshold_cm2: float = 0.5,
    section_length_cm: float = 0.795,
) -> List[ConstrictionInfo]:
    """Find all constriction points below a threshold area.

    Useful for detecting double articulations (e.g., /w/ has both
    labial and velar constrictions).

    Args:
        areas: Area function (44 sections)
        threshold_cm2: Area threshold for constriction detection
        section_length_cm: Physical length of each section

    Returns:
        List of ConstrictionInfo, sorted from glottis to lips
    """
    constrictions = []
    in_constriction = False
    min_area = float("inf")
    min_idx = 0

    for i in range(8, len(areas)):
        if areas[i] < threshold_cm2:
            if not in_constriction:
                in_constriction = True
                min_area = areas[i]
                min_idx = i
            elif areas[i] < min_area:
                min_area = areas[i]
                min_idx = i
        else:
            if in_constriction:
                # End of constriction region — record the minimum
                info = ConstrictionInfo(
                    section=min_idx,
                    area_cm2=float(min_area),
                    position_cm=min_idx * section_length_cm,
                    region=_section_to_region(min_idx),
                )
                constrictions.append(info)
                in_constriction = False
                min_area = float("inf")

    # Handle constriction at the very end (lips)
    if in_constriction:
        info = ConstrictionInfo(
            section=min_idx,
            area_cm2=float(min_area),
            position_cm=min_idx * section_length_cm,
            region=_section_to_region(min_idx),
        )
        constrictions.append(info)

    return constrictions


def _section_to_region(section: int) -> str:
    """Map tube section index to anatomical region."""
    if section < 8:
        return "laryngeal"
    elif section < 12:
        return "pharyngeal"
    elif section < 18:
        return "uvular"
    elif section < 24:
        return "velar"
    elif section < 30:
        return "palatal"
    elif section < 36:
        return "alveolar"
    elif section < 40:
        return "dental"
    else:
        return "labial"

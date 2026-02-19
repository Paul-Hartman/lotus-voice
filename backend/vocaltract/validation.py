"""Validation tools for vocal tract synthesis quality.

Compares synthesized formants against known acoustic targets:
1. Peterson & Barney (1952) vowel formant database
2. Expected formants from our area function presets
3. Custom formant targets

Non-pulmonic airstream synthesis (ejectives, implosives, clicks)
has been moved to vocaltract.non_pulmonic.

References:
  Peterson, G.E. & Barney, H.L. (1952). Control methods used in a
    study of the vowels. JASA 24(2), 175-184.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from vocaltract.synthesizer import VocalTractSynthesizer

logger = logging.getLogger(__name__)


# Peterson & Barney (1952) male speaker formant targets (Hz)
PETERSON_BARNEY_MALE = {
    "i": {"F1": 270, "F2": 2290, "F3": 3010},
    "ɪ": {"F1": 390, "F2": 1990, "F3": 2550},
    "e": {"F1": 400, "F2": 2100, "F3": 2650},
    "ɛ": {"F1": 530, "F2": 1840, "F3": 2480},
    "æ": {"F1": 660, "F2": 1720, "F3": 2410},
    "ɑ": {"F1": 730, "F2": 1090, "F3": 2440},
    "ɔ": {"F1": 570, "F2": 840, "F3": 2410},
    "ʊ": {"F1": 440, "F2": 1020, "F3": 2240},
    "u": {"F1": 300, "F2": 870, "F3": 2240},
    "ʌ": {"F1": 640, "F2": 1190, "F3": 2390},
    "ə": {"F1": 500, "F2": 1500, "F3": 2500},
}

# Female targets (Peterson & Barney)
PETERSON_BARNEY_FEMALE = {
    "i": {"F1": 310, "F2": 2790, "F3": 3310},
    "ɪ": {"F1": 430, "F2": 2480, "F3": 2990},
    "ɛ": {"F1": 610, "F2": 2330, "F3": 2990},
    "æ": {"F1": 860, "F2": 2050, "F3": 2850},
    "ɑ": {"F1": 850, "F2": 1220, "F3": 2810},
    "ɔ": {"F1": 590, "F2": 920, "F3": 2710},
    "u": {"F1": 370, "F2": 950, "F3": 2670},
}


@dataclass
class FormantComparison:
    """Result of comparing synthesized formants to targets."""
    phone: str
    target_formants: Dict[str, float]
    estimated_formants: List[float]
    errors_hz: Dict[str, float]      # Absolute error per formant
    errors_percent: Dict[str, float]  # Percentage error per formant
    mean_error_percent: float


class FormantValidator:
    """Validates synthesized formant accuracy against reference data."""

    def __init__(self, synthesizer: Optional[VocalTractSynthesizer] = None):
        self.synthesizer = synthesizer or VocalTractSynthesizer()

    def validate_vowel(
        self,
        phone: str,
        reference: str = "peterson_barney_male",
    ) -> Optional[FormantComparison]:
        """Validate a single vowel's formants against reference.

        Args:
            phone: IPA vowel symbol
            reference: Reference database ("peterson_barney_male" or "female")

        Returns:
            FormantComparison or None if phone not in reference
        """
        targets = PETERSON_BARNEY_MALE if "male" in reference else PETERSON_BARNEY_FEMALE
        if phone not in targets:
            return None

        target = targets[phone]

        # Synthesize and estimate formants
        audio, states = self.synthesizer.synthesize_phone(phone, duration_sec=0.3)
        if not states:
            return None

        estimated = states[0].formants_hz
        if not estimated:
            # Try from tube model directly
            areas, _ = self.synthesizer.get_area_function(phone)
            if areas is not None:
                self.synthesizer.tube.set_area_function(areas)
                estimated = self.synthesizer.tube.estimate_formants()

        # Compare formants
        errors_hz = {}
        errors_pct = {}
        formant_names = ["F1", "F2", "F3"]

        for i, fname in enumerate(formant_names):
            if fname in target and i < len(estimated):
                target_val = target[fname]
                est_val = estimated[i]
                errors_hz[fname] = abs(est_val - target_val)
                errors_pct[fname] = abs(est_val - target_val) / target_val * 100

        mean_pct = sum(errors_pct.values()) / max(len(errors_pct), 1)

        return FormantComparison(
            phone=phone,
            target_formants=target,
            estimated_formants=estimated,
            errors_hz=errors_hz,
            errors_percent=errors_pct,
            mean_error_percent=mean_pct,
        )

    def validate_all_vowels(
        self,
        reference: str = "peterson_barney_male",
    ) -> List[FormantComparison]:
        """Validate all vowels in the reference database."""
        targets = PETERSON_BARNEY_MALE if "male" in reference else PETERSON_BARNEY_FEMALE
        results = []
        for phone in targets:
            result = self.validate_vowel(phone, reference)
            if result:
                results.append(result)
        return results

    def report(self, reference: str = "peterson_barney_male") -> str:
        """Generate a human-readable validation report."""
        results = self.validate_all_vowels(reference)
        if not results:
            return "No vowels validated."

        lines = ["Vowel Formant Validation Report", "=" * 50, ""]
        lines.append(f"{'Phone':<6} {'F1 err%':<10} {'F2 err%':<10} {'F3 err%':<10} {'Mean':<10}")
        lines.append("-" * 50)

        total_mean = 0.0
        for r in results:
            f1 = f"{r.errors_percent.get('F1', 0):.1f}%"
            f2 = f"{r.errors_percent.get('F2', 0):.1f}%"
            f3 = f"{r.errors_percent.get('F3', 0):.1f}%"
            mean = f"{r.mean_error_percent:.1f}%"
            lines.append(f"{r.phone:<6} {f1:<10} {f2:<10} {f3:<10} {mean:<10}")
            total_mean += r.mean_error_percent

        overall = total_mean / len(results)
        lines.append("-" * 50)
        lines.append(f"Overall mean error: {overall:.1f}%")

        status = "PASS" if overall < 10.0 else "NEEDS TUNING" if overall < 20.0 else "FAIL"
        lines.append(f"Status: {status} (target: <10%)")

        return "\n".join(lines)

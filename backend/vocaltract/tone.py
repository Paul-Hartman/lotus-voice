"""Lexical tone system for tonal language synthesis.

Implements Chao tone letter representation and converts tone targets
to f0 contours. Supports tone sandhi (context-dependent tone changes)
for major tonal languages.

The Chao system uses a 5-level scale:
  5 = highest pitch, 1 = lowest pitch
A tone is a sequence of these levels traced over the syllable:
  [5,5] = high level (Mandarin tone 1)
  [3,5] = rising (Mandarin tone 2)
  [2,1,4] = dipping (Mandarin tone 3)
  [5,1] = falling (Mandarin tone 4)

References:
  Chao, Y.R. (1930). A system of tone-letters. Le Maitre Phonetique.
  Yip, M. (2002). Tone. Cambridge University Press.
  Chen, M.Y. (2000). Tone Sandhi. Cambridge University Press.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "vocaltract"


@dataclass
class ToneTarget:
    """Tone target using Chao pitch levels.

    Attributes:
        chao: Sequence of Chao levels (1-5), e.g. [5,1] for falling
        name: Human-readable name, e.g. "high level", "rising"
        tone_number: Language-specific number, e.g. 1 for Mandarin T1
    """
    chao: List[int]
    name: str = ""
    tone_number: Optional[int] = None

    def __post_init__(self):
        self.chao = [max(1, min(5, c)) for c in self.chao]


@dataclass
class SandhiRule:
    """Context-dependent tone change rule.

    Attributes:
        trigger_tone: Tone number that triggers the change
        target_tone: Tone number being changed
        result_tone: What the target becomes
        context: "before" or "after" — where the trigger is relative to target
        description: Human-readable explanation
    """
    trigger_tone: int
    target_tone: int
    result_tone: int
    context: str = "before"
    description: str = ""


class ToneSystem:
    """Tone inventory and sandhi rules for a language.

    Provides tone lookup by number, IPA tone letter parsing,
    and sandhi application.
    """

    def __init__(self):
        self._systems: Dict[str, dict] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        tone_path = _DATA_DIR / "tone_systems.json"
        if tone_path.exists():
            with open(tone_path) as f:
                self._systems = json.load(f)
            logger.info(f"Loaded tone systems for {len(self._systems)} languages")
        else:
            logger.warning(f"Tone systems not found: {tone_path}")
        self._loaded = True

    def get_tone(self, language: str, tone_number: int) -> Optional[ToneTarget]:
        """Look up a tone by language and number."""
        self._ensure_loaded()
        lang_data = self._systems.get(language, {})
        tones = lang_data.get("tones", {})
        tone_data = tones.get(str(tone_number))
        if tone_data is None:
            return None
        return ToneTarget(
            chao=tone_data["chao"],
            name=tone_data.get("name", ""),
            tone_number=tone_number,
        )

    def get_inventory(self, language: str) -> List[ToneTarget]:
        """Get all tones for a language."""
        self._ensure_loaded()
        lang_data = self._systems.get(language, {})
        tones = lang_data.get("tones", {})
        return [
            ToneTarget(
                chao=t["chao"],
                name=t.get("name", ""),
                tone_number=int(num),
            )
            for num, t in tones.items()
        ]

    def apply_sandhi(
        self, tones: List[int], language: str
    ) -> List[int]:
        """Apply tone sandhi rules to a sequence of tone numbers.

        Processes left-to-right, applying each rule once per pair.

        Args:
            tones: List of tone numbers
            language: Language code

        Returns:
            Modified tone number list
        """
        self._ensure_loaded()
        lang_data = self._systems.get(language, {})
        rules = lang_data.get("sandhi_rules", [])
        if not rules:
            return tones

        result = list(tones)
        for rule_data in rules:
            rule = SandhiRule(
                trigger_tone=rule_data["trigger_tone"],
                target_tone=rule_data["target_tone"],
                result_tone=rule_data["result_tone"],
                context=rule_data.get("context", "before"),
            )
            for i in range(len(result) - 1):
                if rule.context == "before":
                    # Target is at i, trigger is at i+1
                    if result[i] == rule.target_tone and result[i + 1] == rule.trigger_tone:
                        result[i] = rule.result_tone
                elif rule.context == "after":
                    # Trigger is at i, target is at i+1
                    if result[i] == rule.trigger_tone and result[i + 1] == rule.target_tone:
                        result[i + 1] = rule.result_tone

        return result

    def available_languages(self) -> List[str]:
        """List all languages with tone data."""
        self._ensure_loaded()
        return list(self._systems.keys())


# ── IPA tone letter parsing ─────────────────────────────────

# IPA tone letters (Chao letters) map to pitch levels
TONE_LETTERS = {
    "˥": 5, "˦": 4, "˧": 3, "˨": 2, "˩": 1,
}

# Combining tone diacritics on vowels
TONE_DIACRITICS = {
    "\u0301": [3, 5],   # ◌́  acute accent = rising / high
    "\u0300": [5, 1],   # ◌̀  grave accent = falling / low
    "\u0302": [5, 3, 1],  # ◌̂  circumflex = falling (high-low)
    "\u030C": [2, 1, 4],  # ◌̌  caron = rising (dipping)
    "\u0304": [3, 3],   # ◌̄  macron = mid level
}


def parse_tone_letters(tone_str: str) -> Optional[List[int]]:
    """Parse IPA tone letters into Chao levels.

    Args:
        tone_str: String of tone letters like "˥˩" (falling) or "˧˥" (rising)

    Returns:
        List of Chao levels, or None if no tone letters found
    """
    levels = []
    for ch in tone_str:
        if ch in TONE_LETTERS:
            levels.append(TONE_LETTERS[ch])
    return levels if levels else None


def chao_to_f0(
    chao: List[int],
    base_f0: float,
    duration_sec: float,
    sample_rate: int = 44100,
    f0_range_semitones: float = 10.0,
) -> np.ndarray:
    """Convert Chao tone levels to an f0 contour.

    Maps the 5-level Chao scale to a pitch range centered on base_f0.
    Level 3 = base_f0, level 5 = base_f0 + half_range, level 1 = base_f0 - half_range.

    Args:
        chao: Sequence of Chao levels [1-5]
        base_f0: Speaker's base f0 in Hz
        duration_sec: Syllable duration in seconds
        sample_rate: Audio sample rate
        f0_range_semitones: Total pitch range in semitones

    Returns:
        f0 contour array at sample_rate
    """
    num_samples = int(duration_sec * sample_rate)
    if num_samples <= 0 or not chao:
        return np.full(max(1, num_samples), base_f0, dtype=np.float64)

    # Map Chao levels 1-5 to semitone offsets from base_f0
    # Level 3 = 0 semitones (base), Level 5 = +half_range, Level 1 = -half_range
    half_range = f0_range_semitones / 2.0
    targets_st = [(level - 3) / 2.0 * f0_range_semitones for level in chao]
    targets_hz = [base_f0 * (2.0 ** (st / 12.0)) for st in targets_st]

    # Create interpolation points evenly spaced across the syllable
    n_targets = len(targets_hz)
    if n_targets == 1:
        return np.full(num_samples, targets_hz[0], dtype=np.float64)

    # Interpolate with cubic-like smoothing
    target_positions = np.linspace(0, num_samples - 1, n_targets)
    sample_positions = np.arange(num_samples, dtype=np.float64)

    # Use numpy interp for smooth interpolation
    f0 = np.interp(sample_positions, target_positions, targets_hz)

    # Apply gentle smoothing (low-pass) to avoid discontinuities
    # Use a small kernel to avoid edge artifacts
    if num_samples > 40:
        kernel_size = min(num_samples // 10, int(0.01 * sample_rate))
        if kernel_size > 2:
            kernel = np.hanning(kernel_size)
            kernel /= kernel.sum()
            # Pad with edge values to avoid smoothing artifacts at boundaries
            padded = np.pad(f0, kernel_size, mode="edge")
            smoothed = np.convolve(padded, kernel, mode="same")
            f0 = smoothed[kernel_size:-kernel_size]

    return f0

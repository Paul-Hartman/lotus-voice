"""IPA symbol to articulatory target mapping.

Converts IPA phones to ArticulatorTarget configurations by:
1. Looking up pre-defined targets for known phones
2. Generating targets from phonetic features for unknown phones
3. Handling diacritics (nasalization, aspiration, length, etc.)

The mapping covers the full IPA chart relevant to natural languages,
with special attention to Sumerian/Akkadian phonemes.

References:
  Ladefoged, P. & Maddieson, I. (1996). The Sounds of the World's
    Languages. Blackwell.
  International Phonetic Association (1999). Handbook of the IPA.
    Cambridge University Press.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from vocaltract.articulators import ArticulatorTarget

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "vocaltract"

# IPA tone letters and diacritics
_TONE_LETTERS = {"˥", "˦", "˧", "˨", "˩"}
_TONE_DIACRITICS = {
    "\u0301",  # ◌́  acute = high/rising
    "\u0300",  # ◌̀  grave = low/falling
    "\u0302",  # ◌̂  circumflex = falling
    "\u030C",  # ◌̌  caron = rising/dipping
    "\u0304",  # ◌̄  macron = mid level
}


@dataclass
class PhoneToken:
    """Structured phone token preserving suprasegmental information.

    Replaces raw strings in the IPA parser output while maintaining
    backwards compatibility via __str__ and __eq__.

    Attributes:
        phone: Base phone string (may include diacritics, e.g. "tʰ")
        tone: Chao-level tone contour if present, e.g. [5, 1] for falling
        stress: 0=unstressed, 1=primary, 2=secondary
        syllable_boundary: True if a syllable boundary precedes this token
        is_long: True if length mark ː was attached
    """
    phone: str
    tone: Optional[List[int]] = None
    stress: int = 0
    syllable_boundary: bool = False
    is_long: bool = False

    def __str__(self) -> str:
        """Backwards-compatible string: returns phone + ː if long."""
        if self.is_long:
            return self.phone + "ː"
        return self.phone

    def __eq__(self, other) -> bool:
        """Allow comparison with plain strings for backwards compat."""
        if isinstance(other, str):
            return str(self) == other
        if isinstance(other, PhoneToken):
            return self.phone == other.phone and self.tone == other.tone
        return NotImplemented

    def __hash__(self) -> int:
        return hash(str(self))


# Multi-character IPA sequences that should be treated as single phones
# (Must be checked before single-character lookup)
DIGRAPHS = [
    # Affricates
    "tʃ", "dʒ", "ts", "dz", "tɕ", "dʑ", "pf",
    "tʂ", "dʐ", "ʈʂ", "ɖʐ", "pɸ", "bβ",
    # Double articulations
    "kp", "ɡb",
    # Ejective affricates
    "tsʼ", "tʃʼ",
    # Ejective stops (base + ʼ)
    "tʼ", "kʼ", "sʼ", "pʼ", "qʼ",
    # Click accompaniments (prefix + click base)
    "ɡʘ", "ŋʘ", "ɡǀ", "ŋǀ", "ɡǃ", "ŋǃ", "ɡǂ", "ŋǂ", "ɡǁ", "ŋǁ",
    # Prenasalized stops (prefix ⁿ + stop)
    "ⁿb", "ⁿd", "ⁿɡ", "ⁿdʒ", "ⁿdz",
]

# Click base symbols for parser recognition
CLICK_BASES = {"ʘ", "ǀ", "ǃ", "ǂ", "ǁ"}


class IPAToArticulation:
    """Maps IPA symbols to articulatory targets.

    Loads vowel and consonant targets from JSON data files and
    provides lookup with diacritic handling.
    """

    def __init__(self):
        self._vowel_targets: Dict[str, dict] = {}
        self._consonant_targets: Dict[str, dict] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Lazy-load target data from JSON files."""
        if self._loaded:
            return

        # Load vowel targets
        vowel_path = _DATA_DIR / "vowel_targets.json"
        if vowel_path.exists():
            with open(vowel_path) as f:
                data = json.load(f)
            self._vowel_targets = data.get("vowels", {})
            logger.info(f"Loaded {len(self._vowel_targets)} vowel targets")
        else:
            logger.warning(f"Vowel targets not found: {vowel_path}")

        # Load consonant targets
        consonant_path = _DATA_DIR / "consonant_targets.json"
        if consonant_path.exists():
            with open(consonant_path) as f:
                data = json.load(f)
            self._consonant_targets = data.get("consonants", {})
            logger.info(f"Loaded {len(self._consonant_targets)} consonant targets")
        else:
            logger.warning(f"Consonant targets not found: {consonant_path}")

        self._loaded = True

    def get_target(self, phone: str) -> Optional[ArticulatorTarget]:
        """Get articulatory target for an IPA phone.

        Handles diacritics by modifying the base target:
        - ̃  (nasalization): sets velum = 0.6
        - ʰ (aspiration): increases aspiration
        - ʼ (ejective): handled in Phase 5
        - ː (length): not articulatory — handled by duration

        Args:
            phone: IPA symbol (possibly with diacritics)

        Returns:
            ArticulatorTarget or None if phone unknown
        """
        self._ensure_loaded()

        # Strip diacritics and remember them
        base_phone, diacritics = self._strip_diacritics(phone)

        # Look up base phone
        target = self._lookup_phone(base_phone)
        if target is None:
            return None

        # Apply diacritics
        self._apply_diacritics(target, diacritics)

        return target

    def _lookup_phone(self, phone: str) -> Optional[ArticulatorTarget]:
        """Look up a base phone (no diacritics) in the target databases."""
        # Check consonants first (digraphs, then single)
        if phone in self._consonant_targets:
            return self._dict_to_target(self._consonant_targets[phone])

        # Check vowels
        if phone in self._vowel_targets:
            return self._dict_to_target(self._vowel_targets[phone])

        # Check by IPA field
        for data in self._consonant_targets.values():
            if data.get("ipa") == phone:
                return self._dict_to_target(data)
        for data in self._vowel_targets.values():
            if data.get("ipa") == phone:
                return self._dict_to_target(data)

        return None

    def _dict_to_target(self, data: dict) -> ArticulatorTarget:
        """Convert a JSON target dict to an ArticulatorTarget."""
        target = ArticulatorTarget(
            jaw=data.get("jaw", 0.0),
            tongue_dorsal_pos=data.get("tongue_dorsal_pos", 0.0),
            tongue_dorsal_shape=data.get("tongue_dorsal_shape", 0.0),
            tongue_tip=data.get("tongue_tip", 0.0),
            lip_height=data.get("lip_height", 0.0),
            lip_protrusion=data.get("lip_protrusion", 0.0),
            larynx_height=data.get("larynx_height", 0.0),
            velum=data.get("velum", 0.0),
        )

        # Glottal and noise properties
        target.Rd = data.get("Rd")
        target.aspiration = data.get("aspiration", 0.0)
        target.noise_source_section = data.get("noise_section")
        target.noise_amplitude = data.get("noise_amplitude", 0.0)

        # Airstream mechanism
        target.airstream = data.get("airstream", "pulmonic_egressive")

        # Set phonation type from voicing and airstream
        voicing = data.get("voicing", True)
        manner = data.get("manner", "vowel")
        if target.airstream == "glottalic_egressive":
            target.phonation_type = "ejective"
        elif manner == "stop" and not voicing:
            target.phonation_type = "voiceless"
        elif manner == "fricative" and not voicing:
            target.phonation_type = "voiceless"
        else:
            target.phonation_type = "modal"

        return target

    def _strip_diacritics(self, phone: str) -> tuple:
        """Separate base phone from IPA diacritics.

        Returns:
            Tuple of (base_phone, set of diacritic names)
        """
        diacritics = set()
        base = phone

        # Combining diacritics (applied after base character)
        diacritic_map = {
            "\u0303": "nasalized",     # ̃  combining tilde
            "\u0325": "voiceless",     # ̥  combining ring below
            "\u032C": "voiced",        # ̬  combining caron below
            "\u0324": "breathy",       # ̤  combining diaeresis below
            "\u0330": "creaky",        # ̰  combining tilde below
            "\u031A": "unreleased",    # ̚  combining left angle above
            "\u0339": "more_rounded",  # ̹  combining right half ring below
            "\u031C": "less_rounded",  # ̜  combining left half ring below
            "\u0318": "advanced_root", # ̘  combining left tack below
            "\u0319": "retracted_root",# ̙  combining right tack below
        }

        # Superscript modifiers (after base)
        suffix_map = {
            "ʰ": "aspirated",
            "ʷ": "labialized",
            "ʲ": "palatalized",
            "ˠ": "velarized",
            "ˤ": "pharyngealized",
            "ʼ": "ejective",
            "ⁿ": "prenasalized",
            "˞": "rhotacized",
        }

        # Strip combining diacritics
        cleaned = []
        for char in base:
            if char in diacritic_map:
                diacritics.add(diacritic_map[char])
            elif char in suffix_map:
                diacritics.add(suffix_map[char])
            elif char == "ː":
                diacritics.add("long")
            else:
                cleaned.append(char)

        return "".join(cleaned), diacritics

    def _apply_diacritics(self, target: ArticulatorTarget, diacritics: set) -> None:
        """Modify a target based on diacritics."""
        if "nasalized" in diacritics:
            target.velum = 0.6  # Partial opening for nasalized vowels

        if "aspirated" in diacritics:
            target.aspiration = max(target.aspiration, 0.5)

        if "breathy" in diacritics:
            if target.Rd is not None:
                target.Rd = max(target.Rd, 2.5)
            target.aspiration = max(target.aspiration, 0.3)

        if "creaky" in diacritics:
            if target.Rd is not None:
                target.Rd = min(target.Rd, 0.5)
            target.phonation_type = "creaky"

        if "voiceless" in diacritics:
            target.phonation_type = "voiceless"
            if target.Rd is not None:
                target.Rd = 0.3

        if "labialized" in diacritics:
            target.lip_protrusion = min(target.lip_protrusion + 1.5, 3.0)
            target.lip_height = max(target.lip_height - 0.5, -3.0)

        if "palatalized" in diacritics:
            target.tongue_dorsal_pos = min(target.tongue_dorsal_pos + 1.5, 3.0)
            target.tongue_dorsal_shape = min(target.tongue_dorsal_shape + 0.5, 3.0)

        if "velarized" in diacritics:
            target.tongue_dorsal_pos = max(target.tongue_dorsal_pos - 1.0, -3.0)
            target.tongue_dorsal_shape = min(target.tongue_dorsal_shape + 0.5, 3.0)

        if "pharyngealized" in diacritics:
            target.tongue_dorsal_pos = max(target.tongue_dorsal_pos - 0.5, -3.0)
            target.larynx_height = max(target.larynx_height - 0.5, -3.0)

        if "rhotacized" in diacritics:
            target.tongue_tip = min(target.tongue_tip + 1.5, 3.0)
            target.tongue_dorsal_shape = max(target.tongue_dorsal_shape - 0.5, -3.0)

        if "prenasalized" in diacritics:
            target.velum = max(target.velum, 0.8)

        if "ejective" in diacritics:
            target.phonation_type = "ejective"
            target.airstream = "glottalic_egressive"
            target.larynx_height = min(target.larynx_height + 1.5, 3.0)

    def is_vowel(self, phone: str) -> bool:
        """Check if a phone is a vowel."""
        self._ensure_loaded()
        base, _ = self._strip_diacritics(phone)
        if base in self._vowel_targets:
            return True
        for data in self._vowel_targets.values():
            if data.get("ipa") == base:
                return True
        return False

    def is_consonant(self, phone: str) -> bool:
        """Check if a phone is a consonant."""
        self._ensure_loaded()
        base, _ = self._strip_diacritics(phone)
        if base in self._consonant_targets:
            return True
        for data in self._consonant_targets.values():
            if data.get("ipa") == base:
                return True
        return False

    def get_manner(self, phone: str) -> Optional[str]:
        """Get the manner of articulation for a consonant."""
        self._ensure_loaded()
        base, _ = self._strip_diacritics(phone)
        if base in self._consonant_targets:
            return self._consonant_targets[base].get("manner")
        for data in self._consonant_targets.values():
            if data.get("ipa") == base:
                return data.get("manner")
        return None

    def list_phones(self) -> dict:
        """List all available phone mappings.

        Returns:
            Dict with 'vowels' and 'consonants' keys
        """
        self._ensure_loaded()
        return {
            "vowels": list(self._vowel_targets.keys()),
            "consonants": list(self._consonant_targets.keys()),
        }


def parse_ipa_to_phones(ipa_string: str) -> list:
    """Parse an IPA string into PhoneToken segments.

    Handles digraphs, diacritics, tone letters, stress marks,
    and syllable boundaries. Returns PhoneToken objects that are
    backwards-compatible with plain strings via __str__ and __eq__.

    Args:
        ipa_string: IPA transcription string

    Returns:
        List of PhoneToken (or " " for word boundaries)
    """
    phones: list = []
    i = 0
    length = len(ipa_string)
    pending_stress = 0
    pending_syllable_boundary = False
    pending_tone_letters: List[int] = []

    while i < length:
        char = ipa_string[i]

        # Whitespace = word boundary (kept as raw string for backwards compat)
        if char == " ":
            _flush_tone(phones, pending_tone_letters)
            pending_tone_letters = []
            phones.append(" ")
            i += 1
            continue

        # Stress marks — attach to next phone
        if char == "ˈ":
            pending_stress = 1
            i += 1
            continue
        if char == "ˌ":
            pending_stress = 2
            i += 1
            continue

        # Syllable boundary
        if char == ".":
            pending_syllable_boundary = True
            i += 1
            continue

        # Prosodic boundaries
        if char in ("|", "‖"):
            i += 1
            continue

        # Tone letters — collect and attach to previous or next phone
        if char in _TONE_LETTERS:
            from vocaltract.tone import TONE_LETTERS
            pending_tone_letters.append(TONE_LETTERS[char])
            i += 1
            continue

        # Length mark extends previous phone
        if char == "ː":
            if phones and isinstance(phones[-1], PhoneToken):
                phones[-1].is_long = True
            i += 1
            continue

        # Check for digraphs (longest match first)
        matched_digraph = None
        for dg in sorted(DIGRAPHS, key=len, reverse=True):
            if ipa_string[i:i + len(dg)] == dg:
                matched_digraph = dg
                break

        if matched_digraph:
            phone_str = matched_digraph
            i += len(matched_digraph)
        else:
            phone_str = char
            i += 1

        # Collect combining diacritics and modifiers (but not tone diacritics)
        tone_from_diacritics = None
        while i < length and (_is_diacritic(ipa_string[i]) or ipa_string[i] in _TONE_DIACRITICS):
            if ipa_string[i] in _TONE_DIACRITICS:
                from vocaltract.tone import TONE_DIACRITICS
                tone_from_diacritics = TONE_DIACRITICS[ipa_string[i]]
                i += 1
            else:
                phone_str += ipa_string[i]
                i += 1

        # Flush any pending tone letters to the previous phone
        _flush_tone(phones, pending_tone_letters)
        pending_tone_letters = []

        # Create token
        token = PhoneToken(
            phone=phone_str,
            tone=tone_from_diacritics,
            stress=pending_stress,
            syllable_boundary=pending_syllable_boundary,
        )
        pending_stress = 0
        pending_syllable_boundary = False

        phones.append(token)

    # Flush any trailing tone letters
    _flush_tone(phones, pending_tone_letters)

    return phones


def _flush_tone(phones: list, tone_letters: List[int]) -> None:
    """Attach accumulated tone letters to the most recent PhoneToken."""
    if not tone_letters or not phones:
        return
    # Walk back to find the last PhoneToken
    for j in range(len(phones) - 1, -1, -1):
        if isinstance(phones[j], PhoneToken):
            if phones[j].tone is None:
                phones[j].tone = tone_letters
            break


def _is_diacritic(char: str) -> bool:
    """Check if a character is an IPA diacritic or modifier."""
    # Combining diacritics (Unicode combining marks)
    cp = ord(char)
    if 0x0300 <= cp <= 0x036F:  # Combining Diacritical Marks
        return True
    if 0x1DC0 <= cp <= 0x1DFF:  # Combining Diacritical Marks Supplement
        return True

    # Superscript modifiers
    # Note: ⁿ is NOT here — it's a prefix (prenasalized), handled via DIGRAPHS
    return char in ("ʰ", "ʷ", "ʲ", "ˠ", "ˤ", "ʼ", "˞")

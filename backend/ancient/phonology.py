"""Phonological rules for Sumerian and Akkadian.

Based on:
- Sumerian: Jagersma (2010) "A Descriptive Grammar of Sumerian"
- Akkadian: Huehnergard (2011) "A Grammar of Akkadian"
- Edzard (2003) "Sumerian Grammar"

These are scholarly reconstructions - pronunciation is approximate
but follows the academic consensus.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Sumerian phoneme inventory (Jagersma 2010)
#
# Standard transliteration uses: š, ĝ, ḫ
# ETCSL (etcsl.orinst.ox.ac.uk) uses ASCII: c=š, j=ĝ, h=ḫ
# Both conventions are mapped below for interoperability.
SUMERIAN_PHONOLOGY = {
    "vowels": {
        "a": "a",    # open central
        "e": "e",    # mid front
        "i": "i",    # close front
        "u": "u",    # close back
        # Long vowels (written doubled or with macron)
        "aa": "aː", "ā": "aː",
        "ee": "eː", "ē": "eː",
        "ii": "iː", "ī": "iː",
        "uu": "uː", "ū": "uː",
    },
    "consonants": {
        "b": "b",
        "d": "d",
        "g": "ɡ",
        "ĝ": "ŋ",    # velar nasal (NG in 'sing')
        "ŋ": "ŋ",    # alternate notation
        "j": "ŋ",    # ETCSL convention: j = ĝ = [ŋ]
        "h": "x",    # voiceless velar fricative (as in German 'Bach')
        "ḫ": "x",    # standard transliteration notation
        "k": "k",
        "l": "l",
        "m": "m",
        "n": "n",
        "p": "p",
        "r": "r",    # likely alveolar trill
        "s": "s",
        "c": "ʃ",    # ETCSL convention: c = š = [ʃ]
        "š": "ʃ",    # voiceless postalveolar fricative
        "sz": "ʃ",   # ASCII transliteration of š
        "t": "t",
        "z": "z",
    },
    "syllable_structure": "CV, CVC, VC preferred",
}

# Akkadian phoneme inventory (Huehnergard 2011)
AKKADIAN_PHONOLOGY = {
    "vowels": {
        "a": "a",
        "e": "e",
        "i": "i",
        "u": "u",
        # Long vowels
        "ā": "aː", "aa": "aː",
        "ē": "eː", "ee": "eː",
        "ī": "iː", "ii": "iː",
        "ū": "uː", "uu": "uː",
    },
    "consonants": {
        # Labials
        "b": "b",
        "p": "p",
        "m": "m",
        "w": "w",
        # Dentals/Alveolars
        "d": "d",
        "t": "t",
        "ṭ": "tʼ",   # emphatic (ejective)
        "n": "n",
        "l": "l",
        "r": "r",
        "s": "s",
        "z": "z",
        "ṣ": "sʼ",   # emphatic (ejective)
        "š": "ʃ",
        "sz": "ʃ",    # ASCII fallback
        # Velars
        "g": "ɡ",
        "k": "k",
        "q": "kʼ",   # emphatic (ejective/uvular)
        # Pharyngeals/Laryngeals
        "ʾ": "ʔ",    # glottal stop (aleph)
        "'": "ʔ",     # ASCII fallback
        "h": "x",
        "ḫ": "x",     # voiceless velar/pharyngeal
    },
    # Akkadian allows more complex syllables than Sumerian
    "syllable_structure": "CV, CVC, CVCC possible",
}


def get_phonology(language: str) -> Dict:
    """Get phonology data for a language.

    Args:
        language: "sumerian" or "akkadian"

    Returns:
        Dict with vowels, consonants, and syllable structure
    """
    if language.lower() == "sumerian":
        return SUMERIAN_PHONOLOGY
    elif language.lower() == "akkadian":
        return AKKADIAN_PHONOLOGY
    else:
        raise ValueError(f"Unsupported language: {language}")


def load_phonology_json(language: str, data_dir: Optional[Path] = None) -> Dict:
    """Load extended phonology from JSON data file.

    Falls back to built-in data if JSON file not found.
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "data" / "ancient"

    json_file = data_dir / f"{language.lower()}_phonology.json"
    if json_file.exists():
        with open(json_file) as f:
            return json.load(f)

    logger.info(f"No JSON phonology file for {language}, using built-in data")
    return get_phonology(language)

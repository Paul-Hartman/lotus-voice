"""Transliteration to IPA converter for Sumerian and Akkadian.

Converts scholarly transliteration (ATF or plain) into IPA notation
suitable for TTS synthesis via eSpeak-NG or SSML phoneme tags.

Supports both standard Assyriology transliteration (š, ĝ, ḫ) and
ETCSL ASCII conventions (c=š, j=ĝ, h=ḫ).
"""

import re
from typing import List

from ancient.phonology import get_phonology


class IPAConverter:
    """Convert transliterated cuneiform text to IPA."""

    def __init__(self):
        self._phonologies = {}

    def _get_phonology(self, language: str) -> dict:
        if language not in self._phonologies:
            self._phonologies[language] = get_phonology(language)
        return self._phonologies[language]

    def clean_etcsl(self, text: str) -> str:
        """Preprocess ETCSL transliteration for phonological conversion.

        Removes non-phonological markup:
        - Subscript numbers (sign reading indices: jectug2 → jectug)
        - Determinatives (d before divine names, ki after places)
        - Damage markers ([restored], /partial\\, X)
        - HTML entities
        - Line numbers
        """
        # Remove HTML entities
        text = re.sub(r'&\w+;', '', text)

        # Remove damage markers
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'/([^\\]*?)\\', r'\1', text)
        text = re.sub(r'[/\\]', '', text)
        text = re.sub(r'\bX\b', '', text)

        # Uppercase sign names → lowercase (KEC→kec, DU→du)
        text = re.sub(r'\b([A-Z]{2,})\b', lambda m: m.group(1).lower(), text)

        # Remove line numbers at line start
        text = re.sub(r'^\d+[A-Z]?\s+', '', text, flags=re.MULTILINE)

        # Remove subscript numbers (sign reading indices).
        # ETCSL writes subscripts as separate tokens: "e 2", "jectug 2"
        # Multi-digit: "ju 1 0" = ju₁₀. Handle multi-digit first.
        text = re.sub(
            r'([a-zšĝḫ])\s+(\d)\s+(\d)(?=[\s\-.,;:!?]|$)', r'\1', text
        )
        # Single digit subscripts: "ce 3" → "ce", "la 2" → "la"
        text = re.sub(
            r'([a-zšĝḫ])\s+(\d)(?=[\s\-.,;:!?]|$)', r'\1', text
        )
        # Subscripts glued to word: "jectug2" → "jectug"
        text = re.sub(r'([a-zšĝḫ])(\d+)(?=[\s\-.,;:!?]|$)', r'\1', text)

        # Remove determinatives:
        # 'd' before divine names (standalone 'd' followed by space + word)
        text = re.sub(r'\bd\s+(?=[a-z])', '', text)
        # Note: 'ki' after place names is NOT removed because 'ki' is also
        # a common Sumerian word meaning "earth/place". Determinatives were
        # likely silent but the distinction is hard to detect automatically.

        # Remove hyphens (morpheme boundaries, not pronounced)
        text = text.replace('-', '')

        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def to_ipa(self, text: str, language: str = "akkadian") -> str:
        """Convert transliterated text to IPA.

        Args:
            text: Transliterated text (e.g. "ša nagba īmuru")
            language: "sumerian" or "akkadian"

        Returns:
            IPA string (e.g. "ʃa naqba iːmuru")
        """
        phonology = self._get_phonology(language)
        vowels = phonology["vowels"]
        consonants = phonology["consonants"]

        # Combine all mappings, longest first for greedy matching
        all_mappings = {**vowels, **consonants}
        sorted_keys = sorted(all_mappings.keys(), key=len, reverse=True)

        words = text.strip().split()
        ipa_words = []

        for word in words:
            ipa_word = self._convert_word(word, sorted_keys, all_mappings)
            if ipa_word:
                ipa_words.append(ipa_word)

        return " ".join(ipa_words)

    def _convert_word(self, word: str, sorted_keys: List[str], mappings: dict) -> str:
        """Convert a single word to IPA by greedy matching."""
        result = []
        i = 0
        # Remove hyphens (sign boundaries) for phonological processing
        word = word.replace("-", "")

        while i < len(word):
            matched = False
            for key in sorted_keys:
                if word[i:i + len(key)] == key:
                    result.append(mappings[key])
                    i += len(key)
                    matched = True
                    break
            if not matched:
                char = word[i]
                # Skip digits and punctuation silently
                if char.isdigit() or char in '(){}.,;:!?':
                    i += 1
                    continue
                result.append(char)
                i += 1

        return "".join(result)

    def to_ipa_words(self, text: str, language: str = "akkadian") -> List[dict]:
        """Convert text to IPA with word-level mapping.

        Returns:
            List of dicts with 'original' and 'ipa' keys
        """
        words = text.strip().split()
        results = []
        for word in words:
            ipa = self.to_ipa(word, language)
            results.append({"original": word, "ipa": ipa})
        return results

    def ipa_to_espeak_italian(self, ipa: str) -> str:
        """Map IPA notation to Italian-readable text for eSpeak synthesis.

        The Italian voice in eSpeak-NG has pure vowels [a, e, i, u]
        matching reconstructed Sumerian, and handles most consonants.

        IPA → Italian spelling:
            ʃ → she/sha/shi/shu (Italian reads 'sc'+front vowel as [ʃ])
            ŋ → ng (Italian reads as [ŋɡ], acceptably close)
            x → kh (aspirated approximation; true [x] unavailable)
            ɡ → g (Italian voice handles correctly)
        """
        result = ipa
        # Map IPA symbols to Italian-readable spellings
        # Order matters: longest replacements first
        result = result.replace("ʃe", "she")
        result = result.replace("ʃa", "sha")
        result = result.replace("ʃi", "shi")
        result = result.replace("ʃu", "shu")
        result = result.replace("ʃ", "sh")
        result = result.replace("ŋ", "ng")
        result = result.replace("x", "kh")
        result = result.replace("ɡ", "g")
        # Long vowels: add double
        result = result.replace("aː", "aa")
        result = result.replace("eː", "ee")
        result = result.replace("iː", "ii")
        result = result.replace("uː", "uu")
        return result

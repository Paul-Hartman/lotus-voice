"""Transliteration to IPA converter for Sumerian and Akkadian.

Converts scholarly transliteration (ATF or plain) into IPA notation
suitable for TTS synthesis via eSpeak-NG or SSML phoneme tags.
"""

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
        # Sort by key length descending so longer sequences match first
        sorted_keys = sorted(all_mappings.keys(), key=len, reverse=True)

        words = text.strip().split()
        ipa_words = []

        for word in words:
            ipa_word = self._convert_word(word, sorted_keys, all_mappings)
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
                # Pass through unknown characters (numbers, punctuation)
                result.append(word[i])
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

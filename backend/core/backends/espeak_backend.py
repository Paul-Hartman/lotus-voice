"""eSpeak-NG backend - critical for ancient language synthesis.

eSpeak-NG supports custom language definitions, IPA input, and runs
locally without any API. This is the primary backend for Sumerian/Akkadian
audio generation.
"""

import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from core.backends.base import EmotionTag, TTSBackendBase

logger = logging.getLogger(__name__)


class ESpeakBackend(TTSBackendBase):
    """eSpeak-NG - open source speech synthesizer with IPA support.

    Key for ancient languages because it:
    - Accepts IPA input directly via [[...]] notation
    - Supports custom language definitions
    - Runs fully offline
    - Available via `brew install espeak-ng`
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._espeak_path = config.get("espeak_path", "espeak-ng") if config else "espeak-ng"
        self._available = self._check_available()

    def _check_available(self) -> bool:
        """Check if espeak-ng is installed."""
        try:
            result = subprocess.run(
                [self._espeak_path, "--version"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                logger.info(f"eSpeak-NG found: {result.stdout.strip()}")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        logger.warning("eSpeak-NG not found. Install with: brew install espeak-ng")
        return False

    def synthesize(
        self,
        text: str,
        output_path: Path,
        voice_id: Optional[str] = None,
        emotion: Optional[EmotionTag] = None,
        **kwargs,
    ) -> bool:
        """Synthesize speech with eSpeak-NG.

        Special input modes:
        - Plain text: normal synthesis
        - IPA input: wrap in [[...]] for phoneme-level control
        - language kwarg: use specific eSpeak language voice
        """
        if not self._available:
            logger.error("eSpeak-NG not available")
            return False

        language = kwargs.get("language", "en")
        speed = kwargs.get("speed", 150)
        pitch = kwargs.get("pitch", 50)
        is_ipa = kwargs.get("ipa", False)

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            cmd = [
                self._espeak_path,
                "-v", voice_id or language,
                "-s", str(speed),
                "-p", str(pitch),
                "-w", str(output_path),
            ]

            # Note: --ipa is an OUTPUT flag (prints IPA to stdout).
            # For IPA input, text should be wrapped in [[ ]] using
            # eSpeak's own phoneme notation, not raw IPA Unicode.
            # The synthesize_ipa() method handles this conversion.

            cmd.append(text)

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                logger.info(f"eSpeak synthesis complete: {output_path}")
                return True
            else:
                logger.error(f"eSpeak error: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"eSpeak synthesis failed: {e}")
            return False

    def synthesize_ipa(self, ipa_text: str, output_path: Path, **kwargs) -> bool:
        """Synthesize directly from IPA notation.

        eSpeak uses its own phoneme notation (e.g. h@l'oU for "hello"),
        NOT standard IPA Unicode. This method converts common IPA symbols
        to eSpeak's Kirshenbaum-like notation before synthesis.

        Args:
            ipa_text: IPA string (e.g. "ʃa naqba iːmuru")
            output_path: Output WAV path
        """
        espeak_phonemes = self._ipa_to_espeak(ipa_text)
        bracketed = f"[[{espeak_phonemes}]]"
        return self.synthesize(bracketed, output_path, **kwargs)

    @staticmethod
    def _ipa_to_espeak(ipa: str) -> str:
        """Convert IPA Unicode to eSpeak phoneme notation.

        This is a best-effort mapping. eSpeak's phoneme set is English-centric
        and cannot represent clicks, ejectives, implosives, or tones.
        """
        # IPA → eSpeak phoneme mapping (English voice)
        table = {
            # Vowels
            "i": "i:", "ɪ": "I", "e": "e", "ɛ": "E", "æ": "a",
            "ɑ": "A:", "ɒ": "0", "ɔ": "O:", "ʊ": "U", "u": "u:",
            "ʌ": "V", "ə": "@", "ɜ": "3:", "ɝ": "3:",
            # Consonants
            "p": "p", "b": "b", "t": "t", "d": "d", "k": "k", "ɡ": "g",
            "g": "g", "f": "f", "v": "v", "θ": "T", "ð": "D",
            "s": "s", "z": "z", "ʃ": "S", "ʒ": "Z", "h": "h",
            "m": "m", "n": "n", "ŋ": "N", "l": "l", "r": "r",
            "ɹ": "r", "w": "w", "j": "j", "ɾ": "t#",
            # Affricates
            "tʃ": "tS", "dʒ": "dZ", "ts": "ts",
            # Diacritics/modifiers
            "ː": ":", "ˈ": "'", "ˌ": ",",
            # Rhotacized
            "˞": "r",
        }
        result = []
        i = 0
        text = ipa
        while i < len(text):
            # Try 2-char sequences first
            if i + 1 < len(text) and text[i:i+2] in table:
                result.append(table[text[i:i+2]])
                i += 2
            elif text[i] in table:
                result.append(table[text[i]])
                i += 1
            elif text[i] == " ":
                result.append(" ")
                i += 1
            else:
                # Pass through unknown symbols (eSpeak will ignore them)
                i += 1
        return "".join(result)

    def list_voices(self) -> List[Dict]:
        if not self._available:
            return []

        try:
            result = subprocess.run(
                [self._espeak_path, "--voices"],
                capture_output=True, text=True, timeout=10,
            )
            voices = []
            for line in result.stdout.strip().split("\n")[1:]:  # Skip header
                parts = line.split()
                if len(parts) >= 4:
                    voices.append({
                        "id": parts[4] if len(parts) > 4 else parts[3],
                        "language": parts[1],
                        "name": parts[3],
                        "type": "espeak",
                    })
            return voices[:20]  # Limit to first 20
        except Exception:
            return []

    @property
    def supports_ipa(self) -> bool:
        return True

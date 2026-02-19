"""Vocal tract physical model TTS backend.

Uses the articulatory synthesis engine from vocaltract/ to generate
speech from IPA strings or direct articulatory control. This is the
only backend that models the physical vocal tract rather than using
neural networks or concatenative synthesis.

Capabilities:
- Direct IPA input (no text-to-phoneme conversion needed)
- Emotion support via phonation control (f0 + Rd mapping)
- Direct articulatory parameter control
- Full state introspection (area function, formants, etc.)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from core.backends.base import EmotionTag, TTSBackendBase

logger = logging.getLogger(__name__)


class VocalTractBackend(TTSBackendBase):
    """Physical vocal tract model TTS backend.

    Synthesizes speech by simulating the human vocal tract:
    IPA → articulatory targets → area function → waveguide → audio.
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._synthesizer = None

    def _ensure_loaded(self):
        """Lazy-load the vocal tract synthesizer."""
        if self._synthesizer is not None:
            return

        from vocaltract.synthesizer import VocalTractSynthesizer
        self._synthesizer = VocalTractSynthesizer()

        # Apply config
        f0 = (self.config or {}).get("f0", 120.0)
        Rd = (self.config or {}).get("Rd", 1.0)
        self._synthesizer.set_voice(f0=f0, Rd=Rd)

        logger.info("Vocal tract synthesizer initialized")

    def synthesize(
        self,
        text: str,
        output_path: Path,
        voice_id: Optional[str] = None,
        emotion: Optional[EmotionTag] = None,
        **kwargs,
    ) -> bool:
        """Synthesize speech from IPA text to a WAV file.

        Args:
            text: IPA string (e.g., "ʃa naqba iːmuru")
            output_path: Path to write WAV file
            voice_id: Ignored (single physical model)
            emotion: EmotionTag for voice quality adjustment
            **kwargs:
                f0: Override fundamental frequency
                Rd: Override voice quality
                phone_duration: Default phone duration
                ipa: If True, text is treated as IPA (default True)

        Returns:
            True on success
        """
        try:
            self._ensure_loaded()

            f0 = kwargs.get("f0")
            Rd = kwargs.get("Rd")

            # Apply emotion to Rd
            if emotion and Rd is None:
                Rd = self._emotion_to_Rd(emotion)

            phone_duration = kwargs.get("phone_duration", 0.15)

            audio, states = self._synthesizer.synthesize_ipa_string(
                text, f0=f0, Rd=Rd, phone_duration=phone_duration
            )

            if len(audio) == 0:
                logger.warning(f"No audio generated for: {text[:50]}")
                return False

            # Write WAV file
            wav_bytes = self._synthesizer.audio_to_wav_bytes(audio)
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(wav_bytes)

            logger.info(f"Synthesized {len(audio)} samples to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Vocal tract synthesis failed: {e}")
            return False

    def synthesize_to_bytes(self, ipa_string: str,
                            f0: Optional[float] = None,
                            Rd: Optional[float] = None,
                            phone_duration: float = 0.15) -> Optional[bytes]:
        """Synthesize IPA string directly to WAV bytes (for API use).

        Returns:
            WAV file bytes, or None on failure
        """
        try:
            self._ensure_loaded()
            audio, states = self._synthesizer.synthesize_ipa_string(
                ipa_string, f0=f0, Rd=Rd, phone_duration=phone_duration,
            )
            if len(audio) == 0:
                return None
            return self._synthesizer.audio_to_wav_bytes(audio)
        except Exception as e:
            logger.error(f"Synthesis to bytes failed: {e}")
            return None

    def synthesize_phone(self, phone: str, duration: float = 0.3,
                         f0: Optional[float] = None,
                         Rd: Optional[float] = None) -> Optional[dict]:
        """Synthesize a single phone and return audio + state.

        Returns:
            Dict with 'wav_bytes', 'states', 'formants', or None
        """
        try:
            self._ensure_loaded()
            audio, states = self._synthesizer.synthesize_phone(
                phone, duration_sec=duration, f0=f0, Rd=Rd,
            )
            return {
                "wav_bytes": self._synthesizer.audio_to_wav_bytes(audio),
                "states": [self._synthesizer.state_to_dict(s) for s in states],
                "formants": states[0].formants_hz if states else [],
            }
        except Exception as e:
            logger.error(f"Phone synthesis failed: {e}")
            return None

    def get_phone_info(self, phone: str) -> Optional[dict]:
        """Get vocal tract info for a phone without synthesizing."""
        self._ensure_loaded()
        return self._synthesizer.phone_info(phone)

    def get_ipa_map(self) -> dict:
        """Get the complete IPA-to-articulation mapping."""
        self._ensure_loaded()
        return self._synthesizer.ipa_mapper.list_phones()

    @staticmethod
    def _emotion_to_Rd(emotion: EmotionTag) -> float:
        """Map EmotionTag to Rd voice quality parameter."""
        mapping = {
            EmotionTag.NEUTRAL: 1.0,
            EmotionTag.HAPPY: 1.1,
            EmotionTag.SAD: 1.8,
            EmotionTag.ANGRY: 0.5,
            EmotionTag.EXCITED: 0.8,
            EmotionTag.FEARFUL: 0.7,
            EmotionTag.WHISPER: 3.5,
            EmotionTag.LAUGH: 1.5,
            EmotionTag.SIGH: 2.5,
            EmotionTag.GASP: 0.6,
        }
        return mapping.get(emotion, 1.0)

    def list_voices(self) -> List[Dict]:
        return [{
            "id": "vocaltract_default",
            "name": "Physical Vocal Tract Model",
            "description": "Research-grade articulatory synthesis based on Maeda (1990) model",
            "supports_ipa": True,
            "supports_emotion": True,
        }]

    @property
    def supports_ipa(self) -> bool:
        return True

    @property
    def supports_emotion(self) -> bool:
        return True

"""ElevenLabs backend - commercial premium TTS with SSML/IPA support.

Extracted from lotus-books/data-pipeline/tts/tts_manager.py.
ElevenLabs is important for ancient languages because it supports
SSML <phoneme> tags for precise IPA pronunciation.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from core.backends.base import EmotionTag, TTSBackendBase

logger = logging.getLogger(__name__)


class ElevenLabsBackend(TTSBackendBase):
    """ElevenLabs v3 - commercial API with SSML phoneme support."""

    EMOTION_MAP = {
        EmotionTag.HAPPY: "[excited]",
        EmotionTag.SAD: "[somber]",
        EmotionTag.ANGRY: "[angry]",
        EmotionTag.WHISPER: "[whispers]",
        EmotionTag.LAUGH: "[laughs]",
        EmotionTag.SIGH: "[sighs]",
    }

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.api_key = (config or {}).get(
            "api_key", os.getenv("ELEVEN_LABS_API_KEY")
        )
        self._client = None

    def _ensure_loaded(self):
        if self._client is not None:
            return
        if not self.api_key:
            raise RuntimeError("ElevenLabs API key not configured")
        try:
            from elevenlabs.client import ElevenLabs
            self._client = ElevenLabs(api_key=self.api_key)
            logger.info("ElevenLabs client initialized")
        except ImportError:
            raise RuntimeError("elevenlabs not installed. Run: pip install elevenlabs")

    def synthesize(
        self,
        text: str,
        output_path: Path,
        voice_id: Optional[str] = None,
        emotion: Optional[EmotionTag] = None,
        **kwargs,
    ) -> bool:
        self._ensure_loaded()

        try:
            synth_text = text
            if emotion:
                tag = self.EMOTION_MAP.get(emotion, "")
                if tag:
                    synth_text = f"{tag} {synth_text}"

            voice = voice_id or "21m00Tcm4TlvDq8ikWAM"  # Rachel

            audio = self._client.text_to_speech.convert(
                text=synth_text,
                voice_id=voice,
                model_id="eleven_multilingual_v2",
            )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                for chunk in audio:
                    f.write(chunk)

            logger.info(f"ElevenLabs synthesis complete: {output_path}")
            return True

        except Exception as e:
            logger.error(f"ElevenLabs synthesis failed: {e}")
            return False

    def clone_voice(self, audio_path: Path, voice_name: str, **kwargs) -> str:
        self._ensure_loaded()
        logger.info(f"ElevenLabs voice clone: {voice_name}")
        return voice_name

    def list_voices(self) -> List[Dict]:
        return [
            {"id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel", "gender": "female"},
            {"id": "EXAVITQu4vr4xnSDxMaL", "name": "Bella", "gender": "female"},
        ]

    @property
    def supports_multispeaker(self) -> bool:
        return True

    @property
    def supports_emotion(self) -> bool:
        return True

    @property
    def supports_voice_cloning(self) -> bool:
        return True

    @property
    def supports_ssml(self) -> bool:
        return True

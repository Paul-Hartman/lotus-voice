"""XTTS v2 (Coqui TTS) backend - voice cloning with 6 seconds of audio.

Extracted from lotus-books/data-pipeline/tts/tts_manager.py.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from core.backends.base import EmotionTag, TTSBackendBase

logger = logging.getLogger(__name__)


class XTTSv2Backend(TTSBackendBase):
    """XTTS v2 - 17-language voice cloning from 6 seconds of audio."""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._tts = None

    def _ensure_loaded(self):
        if self._tts is not None:
            return
        try:
            from TTS.api import TTS
            self._tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            logger.info("XTTS v2 model loaded")
        except ImportError:
            raise RuntimeError("Coqui TTS not installed. Run: pip install TTS")

    def synthesize(
        self,
        text: str,
        output_path: Path,
        voice_id: Optional[str] = None,
        emotion: Optional[EmotionTag] = None,
        **kwargs,
    ) -> bool:
        self._ensure_loaded()

        speaker_wav = kwargs.get("speaker_wav")
        language = kwargs.get("language", "en")

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            self._tts.tts_to_file(
                text=text,
                file_path=str(output_path),
                speaker_wav=speaker_wav,
                language=language,
            )

            logger.info(f"XTTS synthesis complete: {output_path}")
            return True
        except Exception as e:
            logger.error(f"XTTS synthesis failed: {e}")
            return False

    def clone_voice(self, audio_path: Path, voice_name: str, **kwargs) -> str:
        logger.info(f"XTTS voice clone registered: {voice_name} from {audio_path}")
        return str(audio_path)

    def list_voices(self) -> List[Dict]:
        return [{"id": "clone", "name": "Voice Cloning (6s audio)", "type": "clone"}]

    @property
    def supports_multispeaker(self) -> bool:
        return True

    @property
    def supports_emotion(self) -> bool:
        return True

    @property
    def supports_voice_cloning(self) -> bool:
        return True

"""Orpheus TTS backend - placeholder for future integration."""

import logging
from pathlib import Path
from typing import Optional

from core.backends.base import EmotionTag, TTSBackendBase

logger = logging.getLogger(__name__)


class OrpheusBackend(TTSBackendBase):
    """Orpheus TTS - emotional speech synthesis."""

    def synthesize(
        self,
        text: str,
        output_path: Path,
        voice_id: Optional[str] = None,
        emotion: Optional[EmotionTag] = None,
        **kwargs,
    ) -> bool:
        logger.warning("Orpheus backend not yet implemented")
        return False

    @property
    def supports_emotion(self) -> bool:
        return True

"""Chatterbox TTS backend - placeholder for future integration."""

import logging
from pathlib import Path
from typing import Optional

from core.backends.base import EmotionTag, TTSBackendBase

logger = logging.getLogger(__name__)


class ChatterboxBackend(TTSBackendBase):
    """Chatterbox TTS."""

    def synthesize(
        self,
        text: str,
        output_path: Path,
        voice_id: Optional[str] = None,
        emotion: Optional[EmotionTag] = None,
        **kwargs,
    ) -> bool:
        logger.warning("Chatterbox backend not yet implemented")
        return False

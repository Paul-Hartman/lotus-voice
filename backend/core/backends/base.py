"""Base classes and enums for TTS backends."""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class TTSBackend(Enum):
    """Available TTS backends."""
    BARK = "bark"
    EDGE_TTS = "edge_tts"
    ESPEAK = "espeak"
    ELEVENLABS = "elevenlabs"
    XTTS_V2 = "xtts_v2"
    ORPHEUS = "orpheus"
    CHATTERBOX = "chatterbox"
    VOCALTRACT = "vocaltract"


class EmotionTag(Enum):
    """Emotion tags for expressive TTS."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    EXCITED = "excited"
    FEARFUL = "fearful"
    WHISPER = "whisper"
    LAUGH = "laugh"
    SIGH = "sigh"
    GASP = "gasp"


class TTSBackendBase(ABC):
    """Abstract base class for all TTS backends."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

    @abstractmethod
    def synthesize(
        self,
        text: str,
        output_path: Path,
        voice_id: Optional[str] = None,
        emotion: Optional[EmotionTag] = None,
        **kwargs,
    ) -> bool:
        """Synthesize speech from text. Returns True on success."""
        ...

    def clone_voice(self, audio_path: Path, voice_name: str, **kwargs) -> str:
        """Clone voice from audio sample. Returns voice ID. Override if supported."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support voice cloning")

    def list_voices(self) -> List[Dict]:
        """List available voices."""
        return []

    @property
    def supports_multispeaker(self) -> bool:
        return False

    @property
    def supports_emotion(self) -> bool:
        return False

    @property
    def supports_voice_cloning(self) -> bool:
        return False

    @property
    def supports_ssml(self) -> bool:
        """Whether this backend supports SSML input (for ancient language phonemes)."""
        return False

    @property
    def supports_ipa(self) -> bool:
        """Whether this backend can directly process IPA input."""
        return False

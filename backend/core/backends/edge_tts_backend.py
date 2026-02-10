"""Edge TTS backend - free, high-quality Microsoft neural TTS."""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional

from core.backends.base import EmotionTag, TTSBackendBase

logger = logging.getLogger(__name__)


class EdgeTTSBackend(TTSBackendBase):
    """Microsoft Edge TTS - free neural voices with excellent English quality.

    Uses the edge-tts Python package which interfaces with Microsoft's
    online neural TTS service. No API key required.
    """

    # Popular voice selections
    DEFAULT_VOICES = {
        "narrator_male": "en-US-GuyNeural",
        "narrator_female": "en-US-JennyNeural",
        "british_male": "en-GB-RyanNeural",
        "british_female": "en-GB-SoniaNeural",
        "dramatic": "en-US-DavisNeural",
    }

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._edge_tts = None

    def _ensure_loaded(self):
        if self._edge_tts is not None:
            return
        try:
            import edge_tts
            self._edge_tts = edge_tts
            logger.info("Edge TTS loaded")
        except ImportError:
            raise RuntimeError("edge-tts not installed. Run: pip install edge-tts")

    def synthesize(
        self,
        text: str,
        output_path: Path,
        voice_id: Optional[str] = None,
        emotion: Optional[EmotionTag] = None,
        **kwargs,
    ) -> bool:
        self._ensure_loaded()

        voice = voice_id or self.DEFAULT_VOICES.get("narrator_male", "en-US-GuyNeural")
        # Resolve friendly names
        if voice in self.DEFAULT_VOICES:
            voice = self.DEFAULT_VOICES[voice]

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            communicate = self._edge_tts.Communicate(text, voice)
            asyncio.get_event_loop().run_until_complete(
                communicate.save(str(output_path))
            )

            logger.info(f"Edge TTS synthesis complete: {output_path}")
            return True

        except RuntimeError:
            # No event loop running
            loop = asyncio.new_event_loop()
            try:
                communicate = self._edge_tts.Communicate(text, voice)
                loop.run_until_complete(communicate.save(str(output_path)))
                logger.info(f"Edge TTS synthesis complete: {output_path}")
                return True
            except Exception as e:
                logger.error(f"Edge TTS synthesis failed: {e}")
                return False
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Edge TTS synthesis failed: {e}")
            return False

    def list_voices(self) -> List[Dict]:
        return [
            {"id": voice_id, "name": name, "type": "neural"}
            for name, voice_id in self.DEFAULT_VOICES.items()
        ]

    @property
    def supports_multispeaker(self) -> bool:
        return True

    @property
    def supports_ssml(self) -> bool:
        return True

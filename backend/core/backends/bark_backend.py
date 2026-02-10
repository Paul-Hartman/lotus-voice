"""Bark TTS backend - free, open source, emotional speech synthesis.

Extracted from lotus-books/data-pipeline/tts/audiobook_generator_bark.py
and tts_manager.py. Bark is the confirmed working backend and our priority
for fast audio generation.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from core.backends.base import EmotionTag, TTSBackendBase

logger = logging.getLogger(__name__)


class BarkBackend(TTSBackendBase):
    """Bark TTS - generative model with natural emotion and non-verbal sounds.

    Bark generates audio from text prompts and supports:
    - Emotion tags: [laughs], [sighs], [gasps], [whispers], etc.
    - Speaker presets: v2/en_speaker_0 through v2/en_speaker_9
    - Non-verbal sounds naturally embedded in speech
    """

    # Speaker presets mapped to vocal qualities
    SPEAKER_PRESETS = {
        "warm_narrator": "v2/en_speaker_6",
        "dramatic_male": "v2/en_speaker_9",
        "gentle_female": "v2/en_speaker_0",
        "authoritative": "v2/en_speaker_5",
        "youthful": "v2/en_speaker_3",
        "aged": "v2/en_speaker_7",
        "mysterious": "v2/en_speaker_4",
        "cheerful": "v2/en_speaker_1",
        "serious": "v2/en_speaker_8",
        "emotional": "v2/en_speaker_2",
    }

    # Emotion to Bark tag mapping
    EMOTION_TAGS = {
        EmotionTag.HAPPY: "[laughs]",
        EmotionTag.SAD: "[sighs sadly]",
        EmotionTag.ANGRY: "[angry]",
        EmotionTag.EXCITED: "[excited]",
        EmotionTag.FEARFUL: "[gasps]",
        EmotionTag.WHISPER: "[whispers]",
        EmotionTag.LAUGH: "[laughs]",
        EmotionTag.SIGH: "[sighs]",
        EmotionTag.GASP: "[gasps]",
    }

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._bark = None
        self._sample_rate = None
        self._loaded = False

    def _ensure_loaded(self):
        """Lazy-load Bark models on first use."""
        if self._loaded:
            return
        try:
            from bark import SAMPLE_RATE, generate_audio, preload_models

            self._generate_audio = generate_audio
            self._sample_rate = SAMPLE_RATE
            preload_models()
            self._loaded = True
            logger.info("Bark models loaded successfully")
        except ImportError:
            raise RuntimeError("Bark not installed. Run: pip install bark")

    def synthesize(
        self,
        text: str,
        output_path: Path,
        voice_id: Optional[str] = None,
        emotion: Optional[EmotionTag] = None,
        **kwargs,
    ) -> bool:
        """Synthesize speech with Bark.

        Args:
            text: Text to synthesize (can include Bark tags like [laughs])
            output_path: Path to save WAV file
            voice_id: Speaker preset name or Bark speaker ID (e.g. "v2/en_speaker_6")
            emotion: Emotion tag to prepend
        """
        self._ensure_loaded()

        try:
            from scipy.io.wavfile import write as write_wav

            # Add emotion tag if specified
            synth_text = text
            if emotion and emotion != EmotionTag.NEUTRAL:
                tag = self.EMOTION_TAGS.get(emotion, "")
                if tag:
                    synth_text = f"{tag} {synth_text}"

            # Auto-detect emotions from text
            if kwargs.get("auto_emotion", True):
                synth_text = self._auto_emotion_tags(synth_text)

            # Resolve speaker preset
            history_prompt = None
            if voice_id:
                if voice_id.startswith("v2/"):
                    history_prompt = voice_id
                elif voice_id in self.SPEAKER_PRESETS:
                    history_prompt = self.SPEAKER_PRESETS[voice_id]

            # Generate audio
            audio_array = self._generate_audio(
                synth_text,
                history_prompt=history_prompt,
            )

            # Save
            output_path.parent.mkdir(parents=True, exist_ok=True)
            write_wav(str(output_path), self._sample_rate, audio_array)

            logger.info(f"Bark synthesis complete: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Bark synthesis failed: {e}")
            return False

    def _auto_emotion_tags(self, text: str) -> str:
        """Auto-detect and insert Bark emotion tags based on text content."""
        result = text

        # Skip if already has tags
        if result.strip().startswith("["):
            return result

        # Laughter
        if re.search(r"\b(laugh|chuckle|giggle|haha)\b", text, re.IGNORECASE):
            result = re.sub(
                r"\b(laugh|chuckle|giggle|haha)\w*\b",
                "[laughs]",
                result,
                flags=re.IGNORECASE,
            )

        # Sighs
        if re.search(r"\b(sigh|exhale)\b", text, re.IGNORECASE):
            result = re.sub(
                r"\b(sigh|exhale)\w*\b", "[sighs]", result, flags=re.IGNORECASE
            )

        # Gasps/shock
        if re.search(r"\b(gasp|shock|sudden|surprise)\b", text, re.IGNORECASE):
            if not result.strip().startswith("["):
                result = "[gasps] " + result

        # Sadness
        sad_words = ["sad", "cry", "tears", "died", "death", "lost", "alone", "sorrow"]
        if any(word in text.lower() for word in sad_words):
            if not result.strip().startswith("["):
                result = "[sighs sadly] " + result

        # Whispers
        if re.search(r"\b(whisper|quietly|softly)\b", text, re.IGNORECASE):
            if not result.strip().startswith("["):
                result = "[whispers] " + result

        return result

    def list_voices(self) -> List[Dict]:
        return [
            {"id": bark_id, "name": name, "type": "preset"}
            for name, bark_id in self.SPEAKER_PRESETS.items()
        ]

    @property
    def supports_multispeaker(self) -> bool:
        return True

    @property
    def supports_emotion(self) -> bool:
        return True

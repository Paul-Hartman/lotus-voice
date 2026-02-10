"""Audiobook production orchestrator.

Coordinates the full pipeline: text segmentation -> voice assignment ->
emotional analysis -> TTS synthesis -> audio concatenation -> metadata.

Adapted from lotus-books/data-pipeline/tts/complete_voice_acting_system.py,
refactored to use the modular backend system.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from core.audio_utils import concatenate_wav_files, create_m3u_playlist
from core.dialogue_detector import DialogueDetector
from core.tts_manager import TTSBackend, TTSManager
from core.voice_acting_director import VoiceActingDirector
from core.voice_library import VoiceLibrary

from audiobook.chapter_splitter import ChapterSplitter

logger = logging.getLogger(__name__)


class AudiobookProducer:
    """Orchestrates audiobook production from text to finished audio."""

    def __init__(
        self,
        tts_manager: Optional[TTSManager] = None,
        voice_library: Optional[VoiceLibrary] = None,
        preferred_backend: Optional[TTSBackend] = None,
    ):
        self.tts = tts_manager or TTSManager()
        self.voice_library = voice_library or VoiceLibrary()
        self.director = VoiceActingDirector()
        self.dialogue_detector = DialogueDetector()
        self.chapter_splitter = ChapterSplitter()
        self.preferred_backend = preferred_backend

    def produce(
        self,
        text: str,
        output_dir: Path,
        title: str = "audiobook",
        backend: Optional[TTSBackend] = None,
        voice_id: Optional[str] = None,
        chunk_size: int = 500,
    ) -> Dict:
        """Produce a complete audiobook from text.

        Args:
            text: Full text content
            output_dir: Output directory for audio files
            title: Audiobook title
            backend: Override TTS backend
            voice_id: Override narrator voice
            chunk_size: Max characters per synthesis chunk

        Returns:
            Dict with production results
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        use_backend = backend or self.preferred_backend

        logger.info(f"Starting audiobook production: {title}")

        # Split into chapters
        chapters = self.chapter_splitter.split(text)
        logger.info(f"Split into {len(chapters)} chapters")

        all_files = []
        results = {
            "title": title,
            "output_dir": str(output_dir),
            "chapters": [],
        }

        for ch_idx, chapter in enumerate(chapters):
            ch_name = chapter.get("title", f"chapter_{ch_idx + 1:02d}")
            ch_text = chapter["text"]
            ch_dir = output_dir / ch_name
            ch_dir.mkdir(exist_ok=True)

            logger.info(f"Processing chapter {ch_idx + 1}: {ch_name}")

            # Analyze for emotional beats
            self.dialogue_detector.detect_dialogue(ch_text)
            beats = self.director.prepare_script(ch_text)

            # Generate audio chunks
            chunk_files = []
            chunks = self._split_into_chunks(ch_text, chunk_size)

            for i, chunk in enumerate(chunks):
                output_path = ch_dir / f"{title}_{ch_name}_{i:04d}.wav"

                beat = beats[min(i, len(beats) - 1)] if beats else None
                emotion_tag = None
                if beat:
                    # Map feeling to emotion tag for backend
                    from core.backends.base import EmotionTag
                    feeling_map = {
                        "devastated": EmotionTag.SAD,
                        "terrified": EmotionTag.FEARFUL,
                        "elated": EmotionTag.HAPPY,
                        "defiant": EmotionTag.ANGRY,
                        "furious": EmotionTag.ANGRY,
                        "vulnerable": EmotionTag.SAD,
                    }
                    emotion_tag = feeling_map.get(beat.feeling)

                success = self.tts.synthesize(
                    text=chunk,
                    output_path=output_path,
                    backend=use_backend,
                    voice_id=voice_id or (beat.speaker_preset if beat else None),
                    emotion=emotion_tag,
                )

                if success and output_path.exists():
                    chunk_files.append(output_path)

            # Concatenate chapter
            if chunk_files:
                chapter_file = output_dir / f"{title}_{ch_name}.wav"
                concatenate_wav_files(chunk_files, chapter_file)
                all_files.append(chapter_file)

                results["chapters"].append({
                    "name": ch_name,
                    "chunks": len(chunk_files),
                    "file": str(chapter_file),
                })

        # Create playlist
        if all_files:
            playlist_path = output_dir / f"{title}.m3u"
            create_m3u_playlist(all_files, playlist_path, title)
            results["playlist"] = str(playlist_path)
            results["total_files"] = len(all_files)

        logger.info(f"Audiobook production complete: {len(all_files)} chapters")
        return results

    def _split_into_chunks(self, text: str, max_chars: int) -> List[str]:
        """Split text into TTS-friendly chunks at sentence boundaries."""
        import re

        sentences = re.split(r'([.!?]+[\s\n]+)', text)
        chunks = []
        current = ""

        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            punct = sentences[i + 1] if i + 1 < len(sentences) else ""
            full = (sentence + punct).strip()

            if not full:
                continue

            if current and len(current) + len(full) > max_chars:
                chunks.append(current.strip())
                current = full + " "
            else:
                current += full + " "

        if current.strip():
            chunks.append(current.strip())

        return chunks

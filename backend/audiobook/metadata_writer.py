"""Metadata writer for audiobook files - ID3 tags, chapter markers."""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def write_id3_tags(
    audio_path: Path,
    title: str,
    artist: str = "lotus-voice",
    album: Optional[str] = None,
    track_number: Optional[int] = None,
    genre: str = "Audiobook",
) -> bool:
    """Write ID3 tags to an audio file.

    Args:
        audio_path: Path to MP3 file
        title: Track title
        artist: Artist name
        album: Album/book title
        track_number: Track number
        genre: Genre tag
    """
    try:
        from mutagen.id3 import TALB, TCON, TIT2, TPE1, TRCK
        from mutagen.mp3 import MP3

        audio = MP3(str(audio_path))
        if audio.tags is None:
            audio.add_tags()

        audio.tags.add(TIT2(encoding=3, text=title))
        audio.tags.add(TPE1(encoding=3, text=artist))
        audio.tags.add(TCON(encoding=3, text=genre))

        if album:
            audio.tags.add(TALB(encoding=3, text=album))
        if track_number:
            audio.tags.add(TRCK(encoding=3, text=str(track_number)))

        audio.save()
        logger.info(f"ID3 tags written: {audio_path}")
        return True

    except ImportError:
        logger.warning("mutagen not installed, skipping ID3 tags")
        return False
    except Exception as e:
        logger.error(f"Failed to write ID3 tags: {e}")
        return False

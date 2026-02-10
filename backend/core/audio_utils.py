"""Audio processing utilities."""

import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def concatenate_wav_files(input_files: List[Path], output_path: Path) -> bool:
    """Concatenate multiple WAV files into one using pydub."""
    try:
        from pydub import AudioSegment

        combined = AudioSegment.empty()
        for f in input_files:
            if f.exists():
                segment = AudioSegment.from_wav(str(f))
                combined += segment

        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.export(str(output_path), format="wav")
        logger.info(f"Concatenated {len(input_files)} files -> {output_path}")
        return True
    except Exception as e:
        logger.error(f"Concatenation failed: {e}")
        return False


def convert_to_mp3(wav_path: Path, mp3_path: Optional[Path] = None, bitrate: str = "192k") -> bool:
    """Convert WAV to MP3."""
    try:
        from pydub import AudioSegment

        mp3_path = mp3_path or wav_path.with_suffix(".mp3")
        audio = AudioSegment.from_wav(str(wav_path))
        audio.export(str(mp3_path), format="mp3", bitrate=bitrate)
        logger.info(f"Converted {wav_path} -> {mp3_path}")
        return True
    except Exception as e:
        logger.error(f"MP3 conversion failed: {e}")
        return False


def normalize_audio(
    input_path: Path, output_path: Optional[Path] = None, target_dbfs: float = -20.0
) -> bool:
    """Normalize audio volume."""
    try:
        from pydub import AudioSegment

        output_path = output_path or input_path
        audio = AudioSegment.from_file(str(input_path))

        change_in_dbfs = target_dbfs - audio.dBFS
        normalized = audio.apply_gain(change_in_dbfs)

        normalized.export(str(output_path), format=input_path.suffix.lstrip("."))
        return True
    except Exception as e:
        logger.error(f"Normalization failed: {e}")
        return False


def create_m3u_playlist(audio_files: List[Path], output_path: Path, title: str = "Audiobook"):
    """Create an M3U playlist file."""
    with open(output_path, "w") as f:
        f.write("#EXTM3U\n")
        f.write(f"#PLAYLIST:{title}\n")
        for audio_file in audio_files:
            f.write(f"{audio_file.name}\n")
    logger.info(f"Created playlist: {output_path}")

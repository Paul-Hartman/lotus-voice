"""Configuration for lotus-voice backend."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
AUDIOBOOK_OUTPUT_DIR = Path(os.getenv("AUDIOBOOK_OUTPUT_DIR", str(DATA_DIR / "audiobooks")))
VOICE_LIBRARY_PATH = Path(
    os.getenv("VOICE_LIBRARY_PATH", str(DATA_DIR / "voices" / "library.json"))
)
ANCIENT_DATA_DIR = DATA_DIR / "ancient"

# Server
FLASK_PORT = int(os.getenv("FLASK_PORT", "5031"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "true").lower() == "true"

# TTS Backend settings
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")
ESPEAK_NG_PATH = os.getenv("ESPEAK_NG_PATH", "espeak-ng")
BARK_MODELS_DIR = os.getenv("BARK_MODELS_DIR")

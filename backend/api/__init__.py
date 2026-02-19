"""API blueprints for lotus-voice."""

from api.ancient import ancient_bp
from api.audiobook import audiobook_bp
from api.backends import backends_bp
from api.health import health_bp
from api.synthesis import synthesis_bp
from api.vocaltract import vocaltract_bp
from api.voices import voices_bp

__all__ = [
    "health_bp",
    "synthesis_bp",
    "audiobook_bp",
    "voices_bp",
    "backends_bp",
    "ancient_bp",
    "vocaltract_bp",
]

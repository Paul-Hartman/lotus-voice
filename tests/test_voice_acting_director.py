"""Test voice acting director."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from core.voice_acting_director import VoiceActingDirector


def test_detect_devastation():
    director = VoiceActingDirector()
    beat = director.analyze_emotional_subtext("She gasped. Everything was gone.")
    assert beat.feeling == "devastated"


def test_detect_defiance():
    director = VoiceActingDirector()
    beat = director.analyze_emotional_subtext("I will never surrender!")
    assert beat.feeling == "defiant"


def test_detect_vulnerability():
    director = VoiceActingDirector()
    beat = director.analyze_emotional_subtext("I... I don't know how to say this.")
    assert beat.feeling == "vulnerable"


def test_prepare_script():
    director = VoiceActingDirector()
    beats = director.prepare_script("She gasped. Everything was gone. No! Never!")
    assert len(beats) >= 2


def test_neutral_fallback():
    director = VoiceActingDirector()
    beat = director.analyze_emotional_subtext("The weather today is mild.")
    assert beat.feeling in ("calculating", "neutral")

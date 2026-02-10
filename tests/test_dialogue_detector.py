"""Test dialogue detection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from core.dialogue_detector import DialogueDetector, SegmentType, TheatricalScriptParser


def test_dialogue_detection():
    detector = DialogueDetector()
    text = '"I can\'t believe it," said Alice.'
    segments = detector.detect_dialogue(text)
    assert any(s.segment_type == SegmentType.DIALOGUE for s in segments)


def test_speaker_identification():
    detector = DialogueDetector()
    text = '"Hello there," said Bob. "How are you?"'
    segments = detector.detect_dialogue(text)
    dialogue_segments = [s for s in segments if s.segment_type == SegmentType.DIALOGUE]
    assert len(dialogue_segments) >= 1
    assert dialogue_segments[0].speaker == "Bob"


def test_theatrical_parser():
    parser = TheatricalScriptParser()
    script = "HAMLET: To be or not to be."
    segments = parser.parse_script(script)
    assert len(segments) == 1
    assert segments[0].speaker == "HAMLET"
    assert segments[0].segment_type == SegmentType.DIALOGUE


def test_theatrical_stage_direction():
    parser = TheatricalScriptParser()
    script = "[They exit]"
    segments = parser.parse_script(script)
    assert segments[0].segment_type == SegmentType.STAGE_DIRECTION

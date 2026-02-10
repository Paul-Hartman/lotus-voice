"""Dialogue Detection and Speaker Identification.

Extracted from lotus-books/data-pipeline/tts/dialogue_detector.py.
Detects quoted dialogue in text and assigns speakers for multi-voice synthesis.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class SegmentType(Enum):
    NARRATION = "narration"
    DIALOGUE = "dialogue"
    STAGE_DIRECTION = "stage_direction"


@dataclass
class TextSegment:
    text: str
    segment_type: SegmentType
    speaker: Optional[str] = None
    emotion: Optional[str] = None
    start_pos: int = 0
    end_pos: int = 0


class DialogueDetector:
    """Detect and extract dialogue from prose text."""

    QUOTE_PATTERNS = [
        r'\u201c([^\u201d]+)\u201d',  # Smart double quotes (check first)
        r'\u2018([^\u2019]+)\u2019',  # Smart single quotes
        r'"([^"]+)"',                  # Standard double quotes
    ]

    ATTRIBUTION_PATTERNS = [
        r'(\w+)\s+said', r'(\w+)\s+asked', r'(\w+)\s+shouted',
        r'(\w+)\s+whispered', r'(\w+)\s+replied', r'(\w+)\s+answered',
        r'(\w+)\s+exclaimed', r'said\s+(\w+)', r'asked\s+(\w+)',
    ]

    def __init__(self):
        self.speakers: set = set()
        self.speaker_history: list = []

    def detect_dialogue(self, text: str) -> List[TextSegment]:
        segments = []
        current_pos = 0

        for pattern in self.QUOTE_PATTERNS:
            for match in re.finditer(pattern, text):
                quote_start = match.start()
                quote_end = match.end()
                dialogue_text = match.group(1)

                if current_pos < quote_start:
                    narration = text[current_pos:quote_start].strip()
                    if narration:
                        segments.append(TextSegment(
                            text=narration,
                            segment_type=SegmentType.NARRATION,
                            start_pos=current_pos,
                            end_pos=quote_start,
                        ))

                speaker = self._identify_speaker(text, quote_start, quote_end)
                emotion = self._detect_emotion(text, quote_start)

                segments.append(TextSegment(
                    text=dialogue_text,
                    segment_type=SegmentType.DIALOGUE,
                    speaker=speaker,
                    emotion=emotion,
                    start_pos=quote_start,
                    end_pos=quote_end,
                ))
                current_pos = quote_end

        if current_pos < len(text):
            remaining = text[current_pos:].strip()
            if remaining:
                segments.append(TextSegment(
                    text=remaining,
                    segment_type=SegmentType.NARRATION,
                    start_pos=current_pos,
                    end_pos=len(text),
                ))

        return segments

    def _identify_speaker(self, text: str, quote_start: int, quote_end: int) -> Optional[str]:
        context_before = text[max(0, quote_start - 100):quote_start]
        context_after = text[quote_end:min(len(text), quote_end + 100)]

        for pattern in self.ATTRIBUTION_PATTERNS:
            for context in (context_before, context_after):
                match = re.search(pattern, context, re.IGNORECASE)
                if match:
                    speaker = match.group(1).strip()
                    self.speakers.add(speaker)
                    self.speaker_history.append(speaker)
                    return speaker

        if self.speaker_history:
            return self.speaker_history[-1]
        return None

    def _detect_emotion(self, text: str, quote_start: int) -> Optional[str]:
        context = text[max(0, quote_start - 150):quote_start].lower()
        emotion_keywords = {
            "angry": ["angrily", "furiously", "shouted", "yelled", "roared"],
            "sad": ["sadly", "tearfully", "sobbed", "cried", "mournfully"],
            "happy": ["happily", "joyfully", "laughed", "cheerfully"],
            "whisper": ["whispered", "murmured", "muttered", "quietly"],
            "excited": ["excitedly", "enthusiastically", "exclaimed"],
            "fearful": ["fearfully", "nervously", "trembling", "shakily"],
        }
        for emotion, keywords in emotion_keywords.items():
            if any(kw in context for kw in keywords):
                return emotion
        return None

    def get_unique_speakers(self) -> List[str]:
        return sorted(self.speakers)


class TheatricalScriptParser:
    """Parse theatrical scripts (SPEAKER: dialogue format)."""

    CHARACTER_LINE = re.compile(r'^([A-Z][A-Z\s]+):\s*(.+)$')
    STAGE_DIRECTION = re.compile(r'\[([^\]]+)\]|\(([^\)]+)\)')

    def __init__(self):
        self.characters: set = set()

    def parse_script(self, script_text: str) -> List[TextSegment]:
        segments = []
        for line_num, line in enumerate(script_text.split("\n")):
            line = line.strip()
            if not line:
                continue

            stage_match = self.STAGE_DIRECTION.match(line)
            if stage_match:
                direction = stage_match.group(1) or stage_match.group(2)
                segments.append(TextSegment(
                    text=direction,
                    segment_type=SegmentType.STAGE_DIRECTION,
                    start_pos=line_num,
                    end_pos=line_num,
                ))
                continue

            char_match = self.CHARACTER_LINE.match(line)
            if char_match:
                character = char_match.group(1).strip()
                dialogue = char_match.group(2).strip()
                self.characters.add(character)

                # Extract inline emotion
                emotion = None
                em_match = self.STAGE_DIRECTION.search(dialogue)
                if em_match:
                    direction = (em_match.group(1) or em_match.group(2)).lower()
                    emotion_map = {
                        "angrily": "angry", "sadly": "sad", "happily": "happy",
                        "whispers": "whisper", "shouts": "angry", "laughing": "laugh",
                    }
                    for kw, em in emotion_map.items():
                        if kw in direction:
                            emotion = em
                            break

                dialogue = self.STAGE_DIRECTION.sub("", dialogue).strip()
                segments.append(TextSegment(
                    text=dialogue,
                    segment_type=SegmentType.DIALOGUE,
                    speaker=character,
                    emotion=emotion,
                    start_pos=line_num,
                    end_pos=line_num,
                ))
            else:
                segments.append(TextSegment(
                    text=line,
                    segment_type=SegmentType.NARRATION,
                    start_pos=line_num,
                    end_pos=line_num,
                ))

        return segments

    def get_characters(self) -> List[str]:
        return sorted(self.characters)

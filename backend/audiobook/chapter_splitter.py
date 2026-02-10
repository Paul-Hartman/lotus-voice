"""Intelligent text segmentation for audiobooks.

Splits text into chapters/sections based on structural markers
like headings, tablet numbers, chapter numbers, scene breaks, etc.
"""

import re
from typing import Dict, List


class ChapterSplitter:
    """Split text into chapters for audiobook production."""

    # Patterns that indicate chapter/section boundaries
    SPLIT_PATTERNS = [
        # "TABLET I", "Tablet II", etc. (Gilgamesh)
        re.compile(r'^(?:TABLET|Tablet)\s+([IVXLCDM]+|\d+)', re.MULTILINE),
        # "Chapter 1", "CHAPTER ONE", etc.
        re.compile(r'^(?:CHAPTER|Chapter)\s+(\w+)', re.MULTILINE),
        # "Part I", "PART 1", etc.
        re.compile(r'^(?:PART|Part)\s+(\w+)', re.MULTILINE),
        # "Act I", "Scene 1", etc. (theatrical)
        re.compile(r'^(?:ACT|Act|SCENE|Scene)\s+(\w+)', re.MULTILINE),
        # Section break markers
        re.compile(r'^(?:---|\*\*\*|===)\s*$', re.MULTILINE),
    ]

    def split(self, text: str) -> List[Dict]:
        """Split text into chapters.

        Args:
            text: Full text content

        Returns:
            List of dicts with 'title' and 'text' keys
        """
        # Try each pattern
        for pattern in self.SPLIT_PATTERNS:
            matches = list(pattern.finditer(text))
            if len(matches) >= 2:
                return self._split_at_matches(text, matches, pattern)

        # No structural markers found - treat as single chapter
        return [{"title": "full_text", "text": text.strip()}]

    def _split_at_matches(self, text: str, matches: list, pattern) -> List[Dict]:
        """Split text at matched positions."""
        chapters = []

        # Check if there's text before the first match
        if matches[0].start() > 50:
            preamble = text[:matches[0].start()].strip()
            if preamble:
                chapters.append({"title": "preamble", "text": preamble})

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            chapter_text = text[start:end].strip()
            title = self._extract_title(match)

            chapters.append({"title": title, "text": chapter_text})

        return chapters

    def _extract_title(self, match: re.Match) -> str:
        """Extract a clean title from a match."""
        full = match.group(0).strip()
        # Clean for filename use
        clean = re.sub(r'[^\w\s-]', '', full).strip()
        clean = re.sub(r'\s+', '_', clean).lower()
        return clean or "section"

    def split_by_line_count(self, text: str, lines_per_chapter: int = 50) -> List[Dict]:
        """Fallback: split by line count."""
        lines = text.split("\n")
        chapters = []

        for i in range(0, len(lines), lines_per_chapter):
            chunk = "\n".join(lines[i:i + lines_per_chapter]).strip()
            if chunk:
                chapters.append({
                    "title": f"section_{i // lines_per_chapter + 1:02d}",
                    "text": chunk,
                })

        return chapters

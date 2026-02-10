"""ATF (ASCII Transliteration Format) parser for cuneiform texts.

ATF is the standard format used by ORACC and CDLI for encoding
cuneiform transliterations. This module parses ATF into structured
data suitable for phonological processing.

ATF conventions:
- Signs separated by hyphens: e.g., "lugal-e" (the king)
- Determinatives in curly braces: {d}utu (the god Utu)
- Damaged signs in brackets: [lugal]
- Numeric subscripts: ka₂ (gate)
- Line numbers: 1. ša₂ naq-ba i-mu-ru
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ATFLine:
    """A single line of cuneiform text."""
    line_number: str
    raw: str
    signs: List[str]
    language: str = "akkadian"
    damaged: bool = False
    notes: str = ""


@dataclass
class ATFDocument:
    """A parsed ATF document (tablet, composition)."""
    identifier: str = ""
    lines: List[ATFLine] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)


class ATFParser:
    """Parse ATF-format cuneiform transliterations."""

    # ATF structure patterns
    RE_DOCUMENT_ID = re.compile(r'^&(\w+)\s*=\s*(.+)$')
    RE_LINE_NUMBER = re.compile(r'^(\d+[\'.]*)\.\s+(.+)$')
    RE_PROTOCOL = re.compile(r'^#(\w+):\s*(.+)$')
    RE_DETERMINATIVE = re.compile(r'\{([^}]+)\}')
    RE_DAMAGED = re.compile(r'\[([^\]]+)\]')
    RE_SUBSCRIPT = re.compile(r'([a-zšŋ]+)([₀-₉]+)')

    # Unicode subscript to ASCII mapping
    SUBSCRIPT_MAP = {
        '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
        '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9',
    }

    def parse(self, text: str) -> Dict:
        """Parse ATF text into structured data.

        Args:
            text: ATF format text (can be full document or plain transliteration)

        Returns:
            Dict with parsed structure
        """
        lines_out = []
        metadata = {}
        doc_id = ""

        for raw_line in text.split("\n"):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                # Protocol line or comment
                proto = self.RE_PROTOCOL.match(line)
                if proto:
                    metadata[proto.group(1)] = proto.group(2)
                continue

            if line.startswith("&"):
                doc_match = self.RE_DOCUMENT_ID.match(line)
                if doc_match:
                    doc_id = doc_match.group(1)
                    metadata["title"] = doc_match.group(2)
                continue

            # Skip ATF structural lines
            if line.startswith(("@", "$")):
                continue

            # Try to parse as numbered line
            line_match = self.RE_LINE_NUMBER.match(line)
            if line_match:
                line_num = line_match.group(1)
                content = line_match.group(2)
            else:
                # Treat as plain transliteration
                line_num = str(len(lines_out) + 1)
                content = line

            # Parse signs
            signs = self._parse_signs(content)
            damaged = bool(self.RE_DAMAGED.search(content))

            lines_out.append({
                "line_number": line_num,
                "raw": content,
                "signs": signs,
                "damaged": damaged,
            })

        return {
            "id": doc_id,
            "metadata": metadata,
            "lines": lines_out,
            "plain_text": self._to_plain_text(lines_out),
        }

    def _parse_signs(self, content: str) -> List[str]:
        """Parse a line of transliteration into individual signs."""
        # Remove determinatives for phonological purposes
        clean = self.RE_DETERMINATIVE.sub("", content)
        # Remove damage markers but keep content
        clean = self.RE_DAMAGED.sub(r"\1", clean)
        # Normalize subscripts
        for sub, digit in self.SUBSCRIPT_MAP.items():
            clean = clean.replace(sub, digit)

        # Split on hyphens and spaces
        signs = re.split(r'[-\s]+', clean)
        return [s.strip() for s in signs if s.strip()]

    def _to_plain_text(self, lines: List[Dict]) -> str:
        """Convert parsed lines to plain readable text."""
        return " ".join(
            " ".join(line["signs"]) for line in lines
        )

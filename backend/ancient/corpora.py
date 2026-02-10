"""ORACC/CDLI corpus integration for ancient text sources.

Provides utilities for fetching and processing texts from:
- ORACC (Open Richly Annotated Cuneiform Corpus)
- CDLI (Cuneiform Digital Library Initiative)
- ETCSL (Electronic Text Corpus of Sumerian Literature)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class CorpusManager:
    """Manage cuneiform text corpora."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path(__file__).parent.parent.parent / "data" / "ancient"

    def list_available_texts(self, collection: str = "gilgamesh") -> List[Dict]:
        """List available texts in a collection."""
        collection_dir = self.data_dir / collection
        if not collection_dir.exists():
            return []

        texts = []
        for subdir in ("tablets", "translations"):
            path = collection_dir / subdir
            if path.exists():
                for f in sorted(path.iterdir()):
                    if f.suffix in (".txt", ".atf", ".json"):
                        texts.append({
                            "name": f.stem,
                            "type": subdir.rstrip("s"),
                            "path": str(f),
                            "format": f.suffix.lstrip("."),
                        })

        return texts

    def load_text(self, path: str) -> str:
        """Load a text file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Text not found: {path}")
        return p.read_text(encoding="utf-8")

    def load_tablet(self, collection: str, tablet_name: str) -> Optional[Dict]:
        """Load a specific tablet with both source and translation."""
        base = self.data_dir / collection

        result = {"name": tablet_name, "source": None, "translation": None}

        # Try to find source
        for ext in (".atf", ".txt"):
            source_path = base / "tablets" / f"{tablet_name}{ext}"
            if source_path.exists():
                result["source"] = source_path.read_text(encoding="utf-8")
                break

        # Try to find translation
        for ext in (".txt", ".md"):
            trans_path = base / "translations" / f"{tablet_name}{ext}"
            if trans_path.exists():
                result["translation"] = trans_path.read_text(encoding="utf-8")
                break

        if result["source"] or result["translation"]:
            return result
        return None

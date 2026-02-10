"""Test chapter splitter."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from audiobook.chapter_splitter import ChapterSplitter


def test_split_by_tablet():
    splitter = ChapterSplitter()
    text = """TABLET I - He Who Saw the Deep

He who saw the deep, the foundation of the land.

TABLET II - The Forest Journey

They set out on the forest journey together.

TABLET III - Words of Encouragement

The elders of Uruk spoke words of encouragement."""

    chapters = splitter.split(text)
    assert len(chapters) == 3
    assert "tablet_i" in chapters[0]["title"]


def test_no_markers_single_chapter():
    splitter = ChapterSplitter()
    text = "Just a plain paragraph with no structural markers at all."
    chapters = splitter.split(text)
    assert len(chapters) == 1
    assert chapters[0]["title"] == "full_text"

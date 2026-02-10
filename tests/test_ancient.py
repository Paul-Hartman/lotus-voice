"""Test ancient language pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from ancient.transliteration import ATFParser
from ancient.ipa_converter import IPAConverter
from ancient.ssml_builder import SSMLBuilder
from ancient.phonology import get_phonology


def test_akkadian_phonology():
    phonology = get_phonology("akkadian")
    assert "a" in phonology["vowels"]
    assert "š" in phonology["consonants"]
    assert phonology["consonants"]["š"] == "ʃ"
    assert phonology["consonants"]["q"] == "kʼ"


def test_sumerian_phonology():
    phonology = get_phonology("sumerian")
    assert "ĝ" in phonology["consonants"]
    assert phonology["consonants"]["ĝ"] == "ŋ"


def test_ipa_converter_akkadian():
    converter = IPAConverter()
    # "ša nagba īmuru" - Gilgamesh I.1 opening
    ipa = converter.to_ipa("ša nagba", "akkadian")
    assert "ʃ" in ipa  # š -> ʃ
    assert "a" in ipa


def test_ipa_converter_gilgamesh_opening():
    converter = IPAConverter()
    ipa = converter.to_ipa("ša nagba imuru", "akkadian")
    # Should produce: ʃa naɡba imuru
    assert ipa.startswith("ʃa")


def test_atf_parser_plain():
    parser = ATFParser()
    result = parser.parse("ša naq-ba i-mu-ru")
    assert "lines" in result
    assert len(result["lines"]) > 0
    assert "ša" in result["lines"][0]["signs"]


def test_atf_parser_numbered():
    parser = ATFParser()
    result = parser.parse("1. ša naq-ba i-mu-ru\n2. i-šid ma-a-ti")
    assert len(result["lines"]) == 2
    assert result["lines"][0]["line_number"] == "1"
    assert result["lines"][1]["line_number"] == "2"


def test_ssml_builder():
    builder = SSMLBuilder()
    ssml = builder.build("ša nagba", "ʃa naɡba", "akkadian")
    assert '<phoneme alphabet="ipa"' in ssml
    assert 'ph="ʃa"' in ssml
    assert "</speak>" in ssml


def test_ipa_word_mapping():
    converter = IPAConverter()
    words = converter.to_ipa_words("ša nagba imuru", "akkadian")
    assert len(words) == 3
    assert words[0]["original"] == "ša"
    assert "ʃ" in words[0]["ipa"]

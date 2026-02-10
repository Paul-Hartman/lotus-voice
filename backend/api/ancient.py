"""Ancient language processing API endpoints."""

from ancient.ipa_converter import IPAConverter
from ancient.ssml_builder import SSMLBuilder
from ancient.transliteration import ATFParser
from flask import Blueprint, jsonify, request

ancient_bp = Blueprint("ancient", __name__)

_parser: ATFParser | None = None
_converter: IPAConverter | None = None


def get_parser() -> ATFParser:
    global _parser
    if _parser is None:
        _parser = ATFParser()
    return _parser


def get_converter() -> IPAConverter:
    global _converter
    if _converter is None:
        _converter = IPAConverter()
    return _converter


@ancient_bp.route("/api/ancient/transliterate", methods=["POST"])
def transliterate():
    """Convert transliteration to IPA.

    Request JSON:
        text: str - Transliterated text (ATF format or plain)
        language: str - "sumerian" or "akkadian"
    """
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    language = data.get("language", "akkadian")

    if not text:
        return jsonify({"error": "text is required"}), 400

    converter = get_converter()
    ipa = converter.to_ipa(text, language)

    return jsonify({
        "input": text,
        "language": language,
        "ipa": ipa,
    })


@ancient_bp.route("/api/ancient/ssml", methods=["POST"])
def to_ssml():
    """Convert transliteration to SSML with phoneme tags.

    Request JSON:
        text: str - Transliterated text
        language: str - "sumerian" or "akkadian"
    """
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    language = data.get("language", "akkadian")

    if not text:
        return jsonify({"error": "text is required"}), 400

    converter = get_converter()
    ipa = converter.to_ipa(text, language)

    builder = SSMLBuilder()
    ssml = builder.build(text, ipa, language)

    return jsonify({
        "input": text,
        "language": language,
        "ipa": ipa,
        "ssml": ssml,
    })


@ancient_bp.route("/api/ancient/parse-atf", methods=["POST"])
def parse_atf():
    """Parse ATF-format cuneiform transliteration.

    Request JSON:
        text: str - ATF format text
    """
    data = request.get_json(force=True)
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "text is required"}), 400

    parser = get_parser()
    result = parser.parse(text)

    return jsonify(result)

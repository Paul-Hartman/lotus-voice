"""SSML builder for ancient language TTS.

Generates SSML (Speech Synthesis Markup Language) with <phoneme> tags
for precise IPA pronunciation. This is the Path A output from the
ancient language pipeline, targeting ElevenLabs/Azure/Google TTS.
"""

from typing import Optional


class SSMLBuilder:
    """Build SSML documents with IPA phoneme tags."""

    def build(self, text: str, ipa: str, language: str = "akkadian") -> str:
        """Build SSML from text and IPA transcription.

        Args:
            text: Original transliterated text
            ipa: IPA transcription
            language: Source language for metadata

        Returns:
            SSML string with phoneme tags
        """
        # Split into words and pair with IPA
        text_words = text.strip().split()
        ipa_words = ipa.strip().split()

        phoneme_tags = []
        for i, word in enumerate(text_words):
            ipa_word = ipa_words[i] if i < len(ipa_words) else word
            phoneme_tags.append(
                f'<phoneme alphabet="ipa" ph="{ipa_word}">{word}</phoneme>'
            )

        body = " ".join(phoneme_tags)

        return (
            f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
            f'xml:lang="en-US">\n'
            f'  <prosody rate="slow" pitch="-5%">\n'
            f'    {body}\n'
            f'  </prosody>\n'
            f'</speak>'
        )

    def build_with_context(
        self,
        text: str,
        ipa: str,
        language: str = "akkadian",
        preamble: Optional[str] = None,
        postamble: Optional[str] = None,
    ) -> str:
        """Build SSML with optional English context around the ancient text.

        Useful for audiobooks: "In the original Akkadian: [phoneme-tagged text]"
        """
        text_words = text.strip().split()
        ipa_words = ipa.strip().split()

        phoneme_tags = []
        for i, word in enumerate(text_words):
            ipa_word = ipa_words[i] if i < len(ipa_words) else word
            phoneme_tags.append(
                f'<phoneme alphabet="ipa" ph="{ipa_word}">{word}</phoneme>'
            )

        ancient_body = " ".join(phoneme_tags)

        speak_open = (
            '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"'
            ' xml:lang="en-US">'
        )
        parts = [speak_open]

        if preamble:
            parts.append(f"  <p>{preamble}</p>")

        parts.append(f'  <p><prosody rate="slow" pitch="-5%">{ancient_body}</prosody></p>')

        if postamble:
            parts.append(f"  <p>{postamble}</p>")

        parts.append("</speak>")

        return "\n".join(parts)

"""Voice Acting Director - emotional subtext analysis for TTS performance.

Extracted from lotus-books/data-pipeline/tts/voice_acting_director.py.
Analyzes text like a professional voice acting coach, determining
emotional stakes, objectives, and vocal direction.
"""

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class EmotionalBeat:
    """A beat is a unit of action with one emotional objective."""
    text: str
    feeling: str
    stakes: str
    objective: str
    obstacle: str
    tactic: str
    pitch: str
    tempo: str
    volume: str
    timbre: str
    pauses: List[str] = field(default_factory=list)
    speaker_context: str = ""
    speaker_preset: str = "v2/en_speaker_6"


class VoiceActingDirector:
    """Prepare scripts for authentic voice acting, like a professional director."""

    SPEAKER_PRESETS = {
        "warm_narrator": "v2/en_speaker_6",
        "dramatic_male": "v2/en_speaker_9",
        "gentle_female": "v2/en_speaker_0",
        "authoritative": "v2/en_speaker_5",
        "youthful": "v2/en_speaker_3",
        "aged": "v2/en_speaker_7",
        "mysterious": "v2/en_speaker_4",
        "cheerful": "v2/en_speaker_1",
        "serious": "v2/en_speaker_8",
        "emotional": "v2/en_speaker_2",
    }

    FEELINGS = {
        "devastated": {
            "stakes": "losing something irreplaceable",
            "vocal": "voice cracking, struggling to speak, breathy",
            "tempo": "slower, with painful pauses",
            "pitch": "lower, strained",
        },
        "terrified": {
            "stakes": "survival, immediate danger",
            "vocal": "shallow breathing, higher pitch, rushed",
            "tempo": "faster, urgent, breathless",
            "pitch": "higher, trembling",
        },
        "elated": {
            "stakes": "achieving something deeply desired",
            "vocal": "clear, energized, laugh-adjacent",
            "tempo": "faster, energetic, bouncing",
            "pitch": "higher, rising",
        },
        "resigned": {
            "stakes": "accepting defeat or loss",
            "vocal": "exhaling, quiet, dropping energy",
            "tempo": "slower, measured, final",
            "pitch": "lower, falling",
        },
        "defiant": {
            "stakes": "maintaining dignity under pressure",
            "vocal": "firm, clear, controlled intensity",
            "tempo": "measured, deliberate, unyielding",
            "pitch": "steady, slightly lower",
        },
        "vulnerable": {
            "stakes": "risking rejection by opening up",
            "vocal": "softer, wavering, intimate",
            "tempo": "slower, hesitant, searching",
            "pitch": "variable, less controlled",
        },
        "furious": {
            "stakes": "justice, boundaries violated",
            "vocal": "sharp, forceful, explosive",
            "tempo": "faster, clipped, building",
            "pitch": "lower, intensifying",
        },
        "calculating": {
            "stakes": "gaining advantage, control",
            "vocal": "smooth, controlled, measured",
            "tempo": "deliberate, strategic pauses",
            "pitch": "steady, modulated",
        },
    }

    OBJECTIVES = {
        "devastated": "to make them understand the depth of loss",
        "terrified": "to escape or survive",
        "elated": "to share joy or celebrate",
        "resigned": "to accept and move forward",
        "defiant": "to resist or maintain dignity",
        "vulnerable": "to connect or be seen",
        "furious": "to punish or establish boundaries",
        "calculating": "to persuade or manipulate",
    }

    PRESET_MAP = {
        "devastated": "emotional",
        "terrified": "youthful",
        "elated": "cheerful",
        "resigned": "aged",
        "defiant": "authoritative",
        "vulnerable": "gentle_female",
        "furious": "dramatic_male",
        "calculating": "mysterious",
    }

    def analyze_emotional_subtext(self, text: str, context: str = "") -> EmotionalBeat:
        feeling = self._detect_feeling(text)
        feel_data = self.FEELINGS.get(feeling)

        if feel_data:
            pauses = self._determine_pauses(text, feeling)
            speaker_context = f"[{feel_data['stakes']}] [{feel_data['vocal']}] {text}"
            preset_key = self.PRESET_MAP.get(feeling, "warm_narrator")
            speaker_preset = self.SPEAKER_PRESETS[preset_key]

            return EmotionalBeat(
                text=text,
                feeling=feeling,
                stakes=feel_data["stakes"],
                objective=self.OBJECTIVES.get(feeling, "to be heard"),
                obstacle=self._identify_obstacle(text),
                tactic=self._choose_tactic(feeling),
                pitch=feel_data["pitch"],
                tempo=feel_data["tempo"],
                volume=self._determine_volume(feeling),
                timbre=self._extract_timbre(feel_data["vocal"]),
                pauses=pauses,
                speaker_context=speaker_context,
                speaker_preset=speaker_preset,
            )

        return EmotionalBeat(
            text=text, feeling="neutral",
            stakes="maintaining engagement", objective="to inform clearly",
            obstacle="listener attention", tactic="stating clearly",
            pitch="steady", tempo="measured", volume="moderate", timbre="clear",
            pauses=["natural phrase boundaries"],
            speaker_context=text,
            speaker_preset=self.SPEAKER_PRESETS["warm_narrator"],
        )

    def _detect_feeling(self, text: str) -> str:
        t = text.lower()

        if any(w in t for w in ["gone", "nothing", "lost", "empty"]):
            if any(w in t for w in ["gasped", "breath", "everything"]):
                return "devastated"

        if any(w in t for w in ["scared", "afraid", "terrified", "fear"]):
            return "terrified"

        if "..." in text:
            if any(w in t for w in ["don't know", "can't", "scared"]):
                return "vulnerable"

        if any(w in t for w in ["never", "refuse", "won't"]) and "!" in text:
            return "defiant"

        if any(w in t for w in ["screamed", "shouted", "roared", "bellowed"]):
            return "furious"

        if any(w in t for w in ["laughed", "wonderful", "amazing"]) and "!" in text:
            return "elated"

        if any(w in t for w in ["nothing matters", "it's over", "anymore"]):
            return "resigned"

        return "calculating"

    def _identify_obstacle(self, text: str) -> str:
        t = text.lower()
        if "but" in t:
            return "internal conflict"
        if "can't" in t or "couldn't" in t:
            return "powerlessness"
        if "?" in text:
            return "uncertainty"
        return "unspoken resistance"

    def _choose_tactic(self, feeling: str) -> str:
        tactics = {
            "devastated": "pleading, revealing pain",
            "terrified": "warning, seeking safety",
            "elated": "celebrating, inviting",
            "resigned": "surrendering, releasing",
            "defiant": "challenging, standing firm",
            "vulnerable": "confessing, reaching out",
            "furious": "attacking, demanding",
            "calculating": "analyzing, probing",
        }
        return tactics.get(feeling, "stating, informing")

    def _determine_pauses(self, text: str, feeling: str) -> List[str]:
        pauses = []
        if feeling == "devastated":
            pauses.append("before painful words")
        elif feeling == "vulnerable":
            pauses.append("before admissions")
        elif feeling == "calculating":
            pauses.append("strategic - building suspense")
        if "..." in text:
            pauses.append("ellipsis - trailing off")
        if "—" in text or "--" in text:
            pauses.append("dash - sharp turn")
        return pauses

    def _determine_volume(self, feeling: str) -> str:
        volume_map = {
            "devastated": "softer",
            "terrified": "rising with panic",
            "elated": "fuller, projected",
            "resigned": "quieter",
            "defiant": "controlled intensity",
            "vulnerable": "intimate",
            "furious": "building, explosive",
            "calculating": "measured",
        }
        return volume_map.get(feeling, "moderate")

    def _extract_timbre(self, vocal_desc: str) -> str:
        for timbre in ("breathy", "creaky", "strained", "clear"):
            if timbre in vocal_desc:
                return timbre
        return "modal"

    def prepare_script(self, text: str, context: str = "") -> List[EmotionalBeat]:
        """Break text into sentences and analyze each for emotional subtext."""
        sentences = re.split(r"([.!?]+)", text)
        beats = []
        current_context = context

        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i].strip()
            punct = sentences[i + 1] if i + 1 < len(sentences) else ""
            if sentence:
                full = sentence + punct
                beat = self.analyze_emotional_subtext(full, current_context)
                beats.append(beat)
                current_context += f" Previous: {beat.feeling}"

        return beats

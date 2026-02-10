"""Voice Library Manager - voice profiles, archetypes, and character assignment.

Extracted from lotus-books/data-pipeline/tts/voice_library.py with additions
for ancient character archetypes (Gilgamesh, Enkidu, etc.).
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class VoiceGender(Enum):
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


class VoiceArchetype(Enum):
    # Narrators
    NARRATOR_AUTHORITATIVE = "narrator_authoritative"
    NARRATOR_WARM = "narrator_warm"
    NARRATOR_NEUTRAL = "narrator_neutral"
    NARRATOR_DRAMATIC = "narrator_dramatic"
    # Characters
    HERO_YOUNG_MALE = "hero_young_male"
    HERO_OLDER_MALE = "hero_older_male"
    HERO_FEMALE = "hero_female"
    VILLAIN_MENACING = "villain_menacing"
    VILLAIN_SOPHISTICATED = "villain_sophisticated"
    COMIC_RELIEF = "comic_relief"
    ELDER_WISE = "elder_wise"
    CHILD = "child"
    # Ancient/Mythological
    DEITY = "deity"
    KING_EPIC = "king_epic"
    WARRIOR_WILD = "warrior_wild"
    PRIESTESS = "priestess"
    DEMON = "demon"


@dataclass
class VoiceProfile:
    voice_id: str
    name: str
    gender: VoiceGender
    archetype: Optional[VoiceArchetype] = None
    description: str = ""
    audio_sample_path: Optional[str] = None
    backend: str = "bark"
    backend_voice_id: Optional[str] = None
    characteristics: Dict = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


class VoiceLibrary:
    """Manage collection of voice profiles."""

    def __init__(self, library_path: Optional[Path] = None):
        self.library_path = library_path or Path("data/voices/library.json")
        self.voices: Dict[str, VoiceProfile] = {}
        self._load_library()

    def _load_library(self):
        if self.library_path.exists():
            try:
                with open(self.library_path) as f:
                    data = json.load(f)
                for voice_data in data.get("voices", []):
                    voice_data["gender"] = VoiceGender(voice_data["gender"])
                    if voice_data.get("archetype"):
                        voice_data["archetype"] = VoiceArchetype(voice_data["archetype"])
                    profile = VoiceProfile(**voice_data)
                    self.voices[profile.voice_id] = profile
                logger.info(f"Loaded {len(self.voices)} voices from library")
            except Exception as e:
                logger.error(f"Error loading voice library: {e}")
                self._create_default_library()
        else:
            self._create_default_library()

    def _create_default_library(self):
        """Create default voice profiles including ancient character voices."""
        defaults = [
            VoiceProfile(
                voice_id="narrator_1",
                name="Classic Narrator",
                gender=VoiceGender.MALE,
                archetype=VoiceArchetype.NARRATOR_AUTHORITATIVE,
                description="Deep, authoritative narrator voice",
                backend_voice_id="v2/en_speaker_6",
                tags=["narrator", "audiobook", "authoritative"],
            ),
            VoiceProfile(
                voice_id="narrator_2",
                name="Warm Narrator",
                gender=VoiceGender.FEMALE,
                archetype=VoiceArchetype.NARRATOR_WARM,
                description="Warm, engaging narrator voice",
                backend_voice_id="v2/en_speaker_0",
                tags=["narrator", "audiobook", "warm"],
            ),
            VoiceProfile(
                voice_id="gilgamesh",
                name="Gilgamesh",
                gender=VoiceGender.MALE,
                archetype=VoiceArchetype.KING_EPIC,
                description="King of Uruk - powerful, commanding, seeking immortality",
                backend_voice_id="v2/en_speaker_5",
                tags=["character", "ancient", "hero", "king"],
            ),
            VoiceProfile(
                voice_id="enkidu",
                name="Enkidu",
                gender=VoiceGender.MALE,
                archetype=VoiceArchetype.WARRIOR_WILD,
                description="Wild man of the steppe - raw, untamed, loyal",
                backend_voice_id="v2/en_speaker_9",
                tags=["character", "ancient", "warrior", "wild"],
            ),
            VoiceProfile(
                voice_id="shamhat",
                name="Shamhat",
                gender=VoiceGender.FEMALE,
                archetype=VoiceArchetype.PRIESTESS,
                description="Temple priestess - wise, sensual, civilizing",
                backend_voice_id="v2/en_speaker_0",
                tags=["character", "ancient", "priestess"],
            ),
            VoiceProfile(
                voice_id="ishtar",
                name="Ishtar",
                gender=VoiceGender.FEMALE,
                archetype=VoiceArchetype.DEITY,
                description="Goddess of love and war - imperious, passionate, dangerous",
                backend_voice_id="v2/en_speaker_2",
                tags=["character", "ancient", "deity", "goddess"],
            ),
            VoiceProfile(
                voice_id="utnapishtim",
                name="Utnapishtim",
                gender=VoiceGender.MALE,
                archetype=VoiceArchetype.ELDER_WISE,
                description="The immortal sage - ancient, measured, knowing",
                backend_voice_id="v2/en_speaker_7",
                tags=["character", "ancient", "elder", "wise"],
            ),
        ]

        for voice in defaults:
            self.voices[voice.voice_id] = voice
        self.save_library()

    def save_library(self):
        try:
            self.library_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "voices": [
                    {
                        **asdict(voice),
                        "gender": voice.gender.value,
                        "archetype": voice.archetype.value if voice.archetype else None,
                    }
                    for voice in self.voices.values()
                ]
            }
            with open(self.library_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.voices)} voices")
        except Exception as e:
            logger.error(f"Error saving voice library: {e}")

    def add_voice(self, voice_id: str, name: str, gender: VoiceGender, **kwargs) -> VoiceProfile:
        profile = VoiceProfile(voice_id=voice_id, name=name, gender=gender, **kwargs)
        self.voices[voice_id] = profile
        self.save_library()
        return profile

    def get_voice(self, voice_id: str) -> Optional[VoiceProfile]:
        return self.voices.get(voice_id)

    def find_voices(
        self,
        gender: Optional[VoiceGender] = None,
        archetype: Optional[VoiceArchetype] = None,
        tags: Optional[List[str]] = None,
    ) -> List[VoiceProfile]:
        results = []
        for voice in self.voices.values():
            if gender and voice.gender != gender:
                continue
            if archetype and voice.archetype != archetype:
                continue
            if tags and not any(tag in voice.tags for tag in tags):
                continue
            results.append(voice)
        return results

    def get_narrator_voice(self) -> Optional[VoiceProfile]:
        narrators = self.find_voices(tags=["narrator"])
        return narrators[0] if narrators else None


class CharacterVoiceAssigner:
    """Assign voices to characters in a work."""

    def __init__(self, voice_library: VoiceLibrary):
        self.library = voice_library
        self.assignments: Dict[str, str] = {}

    def assign_voice(self, character: str, voice_id: str):
        if voice_id in self.library.voices:
            self.assignments[character] = voice_id

    def auto_assign_voices(self, characters: List[str]):
        narrator = self.library.get_narrator_voice()
        if narrator:
            self.assignments["NARRATOR"] = narrator.voice_id

        available = [v for v in self.library.voices.values() if "character" in v.tags]
        for i, character in enumerate(characters):
            # Check for exact match by voice_id
            char_lower = character.lower()
            match = self.library.get_voice(char_lower)
            if match:
                self.assignments[character] = match.voice_id
            elif available:
                self.assignments[character] = available[i % len(available)].voice_id

    def get_voice_for_character(self, character: str) -> Optional[VoiceProfile]:
        voice_id = self.assignments.get(character)
        return self.library.get_voice(voice_id) if voice_id else None

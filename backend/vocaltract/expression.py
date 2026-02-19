"""Emotional expression via vocal tract configuration.

Maps emotional states from VoiceActingDirector's EmotionalBeat to
physical vocal tract modifications. Each emotion affects:

1. Pharynx width: tension → narrowing (anger, fear), relaxation → widening (sadness)
2. Larynx height: arousal → raised (fear, joy), low arousal → lowered (sadness, calm)
3. Lip configuration: smile → retracted (joy), pout → protruded (sadness)
4. Jaw tension: high tension → clenched (anger), low → relaxed (calm)
5. Voice quality (Rd): pressed (anger) vs breathy (sadness, vulnerability)
6. f0 range: expanded (high arousal) vs compressed (low arousal)

Based on Scherer's Component Process Model of vocal emotion expression
and acoustic measurements by Banse & Scherer (1996).

References:
  Scherer, K.R. (2003). Vocal communication of emotion: A review of
    research paradigms. Speech Communication, 40(1-2), 227-256.
  Banse, R. & Scherer, K.R. (1996). Acoustic profiles in vocal emotion
    expression. Journal of Personality and Social Psychology, 70(3), 614-636.
  Johnstone, T. & Scherer, K.R. (2000). Vocal communication of emotion.
    In M. Lewis & J. Haviland (Eds.), Handbook of Emotions, pp. 220-235.
"""

from dataclasses import dataclass
from typing import Optional

from vocaltract.articulators import ArticulatorTarget


@dataclass
class EmotionTractConfig:
    """Vocal tract modifications for an emotional state."""
    # Articulatory offsets (added to the phone's base target)
    jaw_offset: float = 0.0          # Positive = more open
    tongue_dorsal_pos_offset: float = 0.0
    tongue_dorsal_shape_offset: float = 0.0
    tongue_tip_offset: float = 0.0
    lip_height_offset: float = 0.0   # Positive = more open
    lip_protrusion_offset: float = 0.0  # Positive = more protruded
    larynx_height_offset: float = 0.0  # Positive = raised

    # Voice quality
    Rd: float = 1.0
    f0_shift_semitones: float = 0.0
    f0_range_multiplier: float = 1.0
    pressure_multiplier: float = 1.0
    aspiration_offset: float = 0.0

    # Tension (0=relaxed, 1=extremely tense)
    overall_tension: float = 0.3


# Emotion → tract configuration mapping
# Based on Scherer (2003) and Banse & Scherer (1996)
EMOTION_CONFIGS = {
    "neutral": EmotionTractConfig(
        Rd=1.0, f0_shift_semitones=0.0, f0_range_multiplier=1.0,
        overall_tension=0.3,
    ),
    "devastated": EmotionTractConfig(
        jaw_offset=-0.3,  # Slight jaw clenching
        lip_height_offset=-0.2,
        larynx_height_offset=-0.8,  # Lowered larynx (heavy quality)
        Rd=2.2, f0_shift_semitones=-3.0, f0_range_multiplier=0.5,
        pressure_multiplier=0.6,
        aspiration_offset=0.2,
        overall_tension=0.7,  # High tension despite breathy voice
    ),
    "terrified": EmotionTractConfig(
        jaw_offset=0.5,  # Mouth open
        lip_height_offset=0.5,
        lip_protrusion_offset=-0.3,  # Retracted (grimace)
        larynx_height_offset=1.5,  # Raised larynx (constricted)
        tongue_dorsal_shape_offset=0.3,  # Tense tongue
        Rd=0.6, f0_shift_semitones=4.0, f0_range_multiplier=1.8,
        pressure_multiplier=1.3,
        overall_tension=0.9,
    ),
    "elated": EmotionTractConfig(
        jaw_offset=0.3,  # Open, relaxed
        lip_height_offset=0.5,
        lip_protrusion_offset=-0.5,  # Retracted (smile)
        larynx_height_offset=0.8,  # Slightly raised
        Rd=1.1, f0_shift_semitones=3.0, f0_range_multiplier=1.5,
        pressure_multiplier=1.1,
        overall_tension=0.4,
    ),
    "resigned": EmotionTractConfig(
        jaw_offset=-0.2,  # Slightly closed
        lip_height_offset=-0.3,
        larynx_height_offset=-0.5,  # Lowered
        Rd=1.8, f0_shift_semitones=-2.0, f0_range_multiplier=0.4,
        pressure_multiplier=0.5,
        aspiration_offset=0.15,
        overall_tension=0.2,  # Low tension (giving up)
    ),
    "defiant": EmotionTractConfig(
        jaw_offset=0.2,  # Firm jaw
        lip_protrusion_offset=0.3,
        larynx_height_offset=-0.3,
        tongue_dorsal_pos_offset=-0.2,  # Slight pharyngeal narrowing
        Rd=0.5, f0_shift_semitones=0.0, f0_range_multiplier=0.8,
        pressure_multiplier=1.2,
        overall_tension=0.8,
    ),
    "vulnerable": EmotionTractConfig(
        jaw_offset=-0.2,
        lip_height_offset=-0.2,
        larynx_height_offset=0.3,
        Rd=2.2, f0_shift_semitones=1.0, f0_range_multiplier=1.3,
        pressure_multiplier=0.6,
        aspiration_offset=0.25,
        overall_tension=0.5,
    ),
    "furious": EmotionTractConfig(
        jaw_offset=0.8,  # Wide open (shouting)
        lip_height_offset=0.8,
        lip_protrusion_offset=-0.5,  # Bared teeth
        larynx_height_offset=0.5,
        tongue_dorsal_pos_offset=-0.5,  # Pharynx narrowing (constriction)
        tongue_dorsal_shape_offset=0.3,
        Rd=0.4, f0_shift_semitones=2.0, f0_range_multiplier=2.0,
        pressure_multiplier=1.5,
        overall_tension=0.95,
    ),
    "calculating": EmotionTractConfig(
        jaw_offset=-0.1,
        lip_protrusion_offset=0.2,
        larynx_height_offset=-0.2,
        Rd=0.8, f0_shift_semitones=-1.0, f0_range_multiplier=0.6,
        pressure_multiplier=0.8,
        overall_tension=0.4,
    ),
    "happy": EmotionTractConfig(
        jaw_offset=0.2,
        lip_height_offset=0.3,
        lip_protrusion_offset=-0.4,  # Smile
        larynx_height_offset=0.5,
        Rd=1.1, f0_shift_semitones=2.0, f0_range_multiplier=1.3,
        pressure_multiplier=1.0,
        overall_tension=0.3,
    ),
    "sad": EmotionTractConfig(
        jaw_offset=-0.2,
        lip_height_offset=-0.3,
        lip_protrusion_offset=0.2,  # Slight pout
        larynx_height_offset=-0.5,
        Rd=1.8, f0_shift_semitones=-2.0, f0_range_multiplier=0.5,
        pressure_multiplier=0.6,
        aspiration_offset=0.15,
        overall_tension=0.3,
    ),
    "angry": EmotionTractConfig(
        jaw_offset=0.5,
        lip_height_offset=0.4,
        lip_protrusion_offset=-0.3,
        larynx_height_offset=0.3,
        tongue_dorsal_pos_offset=-0.3,
        Rd=0.5, f0_shift_semitones=2.0, f0_range_multiplier=1.5,
        pressure_multiplier=1.3,
        overall_tension=0.85,
    ),
    "whisper": EmotionTractConfig(
        jaw_offset=-0.3,
        lip_height_offset=-0.2,
        larynx_height_offset=0.2,
        Rd=3.5, f0_shift_semitones=0.0, f0_range_multiplier=0.2,
        pressure_multiplier=0.3,
        aspiration_offset=0.6,
        overall_tension=0.2,
    ),
}


class ExpressionController:
    """Maps emotional states to vocal tract modifications.

    Takes an emotion label (from VoiceActingDirector.EmotionalBeat)
    and returns physical vocal tract parameter offsets.
    """

    def __init__(self):
        self._current_config = EMOTION_CONFIGS["neutral"]

    def set_emotion(self, emotion: str) -> None:
        """Set the current emotional state."""
        self._current_config = EMOTION_CONFIGS.get(emotion, EMOTION_CONFIGS["neutral"])

    @property
    def config(self) -> EmotionTractConfig:
        return self._current_config

    def apply_emotion(
        self,
        target: ArticulatorTarget,
        emotion: Optional[str] = None,
    ) -> ArticulatorTarget:
        """Apply emotional modifications to an articulatory target.

        Adds emotional offsets to the phone's base articulatory
        configuration, creating emotion-colored articulation.

        Args:
            target: Base articulatory target for the phone
            emotion: Emotion label (or use previously set emotion)

        Returns:
            Modified ArticulatorTarget with emotional coloring
        """
        if emotion:
            self.set_emotion(emotion)

        cfg = self._current_config

        return ArticulatorTarget(
            jaw=_clamp(target.jaw + cfg.jaw_offset),
            tongue_dorsal_pos=_clamp(target.tongue_dorsal_pos + cfg.tongue_dorsal_pos_offset),
            tongue_dorsal_shape=_clamp(target.tongue_dorsal_shape + cfg.tongue_dorsal_shape_offset),
            tongue_tip=_clamp(target.tongue_tip + cfg.tongue_tip_offset),
            lip_height=_clamp(target.lip_height + cfg.lip_height_offset),
            lip_protrusion=_clamp(target.lip_protrusion + cfg.lip_protrusion_offset),
            larynx_height=_clamp(target.larynx_height + cfg.larynx_height_offset),
            velum=max(0.0, min(1.0, target.velum)),
            f0=target.f0,
            Rd=cfg.Rd if target.Rd is None else target.Rd,
            aspiration=max(0.0, min(1.0, target.aspiration + cfg.aspiration_offset)),
            phonation_type=target.phonation_type,
            noise_source_section=target.noise_source_section,
            noise_amplitude=target.noise_amplitude,
        )

    def get_f0_for_emotion(self, base_f0: float) -> float:
        """Get emotion-adjusted base f0."""
        shift = self._current_config.f0_shift_semitones
        return base_f0 * (2.0 ** (shift / 12.0))

    def get_Rd_for_emotion(self) -> float:
        """Get Rd voice quality for current emotion."""
        return self._current_config.Rd

    def emotion_from_beat(self, beat) -> str:
        """Extract emotion label from a VoiceActingDirector EmotionalBeat."""
        feeling = getattr(beat, "feeling", "neutral")
        # Map VoiceActingDirector feelings to our emotion configs
        if feeling in EMOTION_CONFIGS:
            return feeling
        return "neutral"


def _clamp(val: float, lo: float = -3.0, hi: float = 3.0) -> float:
    """Clamp a parameter to valid range."""
    return max(lo, min(hi, val))

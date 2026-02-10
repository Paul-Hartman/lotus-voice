"""Multi-backend TTS Manager - orchestrates all TTS backends.

Refactored from lotus-books/data-pipeline/tts/tts_manager.py.
The monolithic file was split: this orchestrator stays, each backend
got its own file in core/backends/.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from core.backends.base import EmotionTag, TTSBackend, TTSBackendBase

logger = logging.getLogger(__name__)

# Re-export for convenience
__all__ = ["TTSManager", "TTSBackend", "EmotionTag", "TTSBackendBase"]


class TTSManager:
    """High-level TTS manager that orchestrates multiple backends.

    Automatically discovers and initializes available backends.
    Selects best backend based on content requirements.
    """

    def __init__(self):
        self.backends: Dict[TTSBackend, TTSBackendBase] = {}
        self._initialize_backends()

    def _initialize_backends(self):
        """Try to initialize each backend, skip those with missing dependencies."""
        backend_classes = [
            (TTSBackend.BARK, "core.backends.bark_backend", "BarkBackend"),
            (TTSBackend.EDGE_TTS, "core.backends.edge_tts_backend", "EdgeTTSBackend"),
            (TTSBackend.ESPEAK, "core.backends.espeak_backend", "ESpeakBackend"),
            (TTSBackend.ELEVENLABS, "core.backends.elevenlabs_backend", "ElevenLabsBackend"),
            (TTSBackend.XTTS_V2, "core.backends.xtts_v2_backend", "XTTSv2Backend"),
            (TTSBackend.ORPHEUS, "core.backends.orpheus_backend", "OrpheusBackend"),
            (TTSBackend.CHATTERBOX, "core.backends.chatterbox_backend", "ChatterboxBackend"),
        ]

        for backend_type, module_name, class_name in backend_classes:
            try:
                import importlib
                module = importlib.import_module(module_name)
                backend_class = getattr(module, class_name)
                self.backends[backend_type] = backend_class()
                logger.info(f"Initialized backend: {backend_type.value}")
            except Exception as e:
                logger.debug(f"Backend {backend_type.value} not available: {e}")

        logger.info(f"Available backends: {[b.value for b in self.backends]}")

    def select_backend(self, requirements: Dict) -> TTSBackendBase:
        """Intelligently select best backend based on requirements.

        Args:
            requirements: Dict with keys like:
                - dialogue_heavy: bool
                - emotion_control_needed: bool
                - multilingual: bool
                - long_audiobook: bool
                - premium_quality: bool
                - budget: 'free' or 'paid'
                - ancient_language: bool
                - ipa_input: bool
        """
        budget = requirements.get("budget", "free")
        ancient = requirements.get("ancient_language", False)
        ipa_input = requirements.get("ipa_input", False)

        # Ancient language: eSpeak for IPA, ElevenLabs for SSML
        if ancient or ipa_input:
            if TTSBackend.ESPEAK in self.backends:
                return self.backends[TTSBackend.ESPEAK]

        # Premium + paid: ElevenLabs
        if budget == "paid" and TTSBackend.ELEVENLABS in self.backends:
            if requirements.get("premium_quality") or requirements.get("emotion_control_needed"):
                return self.backends[TTSBackend.ELEVENLABS]

        # Multilingual: XTTS v2
        if requirements.get("multilingual") and TTSBackend.XTTS_V2 in self.backends:
            return self.backends[TTSBackend.XTTS_V2]

        # Default priority: Edge TTS > Bark > eSpeak > anything
        priority = [TTSBackend.EDGE_TTS, TTSBackend.BARK, TTSBackend.ESPEAK]
        for backend_type in priority:
            if backend_type in self.backends:
                return self.backends[backend_type]

        # Last resort: first available
        if self.backends:
            return next(iter(self.backends.values()))

        raise RuntimeError("No TTS backends available!")

    def synthesize(
        self,
        text: str,
        output_path: Path,
        backend: Optional[TTSBackend] = None,
        **kwargs,
    ) -> bool:
        """Synthesize speech using specified or auto-selected backend."""
        if backend and backend in self.backends:
            selected = self.backends[backend]
        else:
            requirements = {
                "emotion_control_needed": kwargs.get("emotion") is not None,
                "budget": kwargs.get("budget", "free"),
                "ancient_language": kwargs.get("ancient_language", False),
                "ipa_input": kwargs.get("ipa", False),
            }
            selected = self.select_backend(requirements)

        logger.info(f"Synthesizing with: {type(selected).__name__}")
        return selected.synthesize(text, output_path, **kwargs)

    def list_available_backends(self) -> List[str]:
        return [backend.value for backend in self.backends]

    def get_backend(self, name: str) -> Optional[TTSBackendBase]:
        """Get a specific backend by name."""
        try:
            backend_type = TTSBackend(name)
            return self.backends.get(backend_type)
        except ValueError:
            return None

# Voice — STATUS

> Last updated: 2026-03-17
> Entity cards: None
> Category: Personal

## Purpose
AI Voice Generation Hub providing multi-backend TTS synthesis (Bark, Edge-TTS, eSpeak-NG, ElevenLabs, XTTS v2), ancient language audio reconstruction (Sumerian/Akkadian via IPA), audiobook production pipeline, and voice library management.

## Current State
| Metric | Value |
|--------|-------|
| Status | SCAFFOLDING |
| Maturity | 25% |
| Tests | 5 test files (test_ancient, test_chapter_splitter, test_dialogue_detector, test_health, test_voice_acting_director) |
| Last commit | 2026-03-06 |
| VPS service | No |

## Tech Stack
| Component | Technology | Port | Entry Point |
|-----------|-----------|------|-------------|
| Backend | Python / Flask | 5031 | `backend/app.py` |
| Frontend | React + Vite + TypeScript | dev server | `frontend/src/` |
| UI framework | Bootstrap 5 (dark theme) | - | - |
| TTS backends | Bark, Edge-TTS, eSpeak-NG, ElevenLabs, XTTS v2 | - | `backend/core/backends/` |
| Ancient pipeline | IPA reconstruction | - | `backend/ancient/` |
| Audiobook | Production orchestrator | - | `backend/audiobook/` |
| Vocal tract | Vocal tract modeling | - | `backend/vocaltract/` |
| Linting | ruff | - | `pyproject.toml` |

## Databases
| Name | Path | Size | Tables | Notes |
|------|------|------|--------|-------|
| None | - | - | - | Voice profiles stored as data files in `data/voices/` |

## Entity Cards Owned
None

## Future Ideas
- [ ] idea-voice-assistant — Full voice assistant integration
- [ ] idea-voice-assistants — Multiple voice assistant persona system
- [ ] idea-voice-navigation — Voice-controlled navigation interface
- [ ] idea-sign-language-recognition-model — Sign language recognition (accessibility)

## Integration Points
| Connected To | Direction | Mechanism | Notes |
|-------------|-----------|-----------|-------|
| Books (lotus-books) | receives | POST /api/audiobook/create | Audiobook generation |
| Deamons | planned | /api/voices/ | Demon voice profiles |
| HomeAssistant | planned | TTS service | Voice command responses |

## Reorganization Notes
### Could absorb from elsewhere:
- TTS functionality scattered in other projects (e.g., lotus-books data-pipeline/tts/) was already extracted here

### Content that better fits elsewhere:
- Nothing identified; this is the canonical voice/TTS service

### Duplicate data/logic to deduplicate:
- Port 5031 CONFLICTS with Alembic (both claim port 5031). One must be reassigned.

## Completed Work
- Flask backend with modular route structure
- Multi-backend TTS architecture with lazy imports
- Ancient language pipeline (Sumerian/Akkadian transliteration to IPA)
- Audiobook production orchestrator
- Vocal tract modeling module
- React + TypeScript frontend scaffolding with Bootstrap 5 dark theme
- Voice library data structure
- Dialogue detection engine
- Chapter splitter for long texts
- Voice acting director module
- 5 test files with pytest

## Known Issues
- **Port conflict**: Port 5031 is also used by Alembic. One project must change.
- `.env.example` exists but unclear if all backends have their API keys configured
- Frontend likely has no node_modules installed (no lock file visible at Voice/ level)
- TTS backends require optional heavy dependencies (Bark, XTTS v2 need GPU)
- No integration tests or CI

## Next Steps
1. Resolve port 5031 conflict with Alembic (reassign one)
2. Run and verify the 5 existing test files pass
3. Install frontend dependencies and verify the React UI builds
4. Test each TTS backend in isolation
5. Wire audiobook pipeline to the Books project
6. Add API key management for ElevenLabs and other paid backends

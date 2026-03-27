# lotus-voice

> **MANDATORY: USE THE EVERYTHING DATABASE.** Do NOT create local databases or JSON data stores. ALL entities go into `hub/data/lotus.db` via API at `http://167.235.139.181/hub/api/lotus/` (no auth needed). Search before building, add cards as you work, use source `claude:session`. See root `CLAUDE.md` for full API reference. This is not optional.

AI Voice Generation Hub for the Lotus Eater 2.0 ecosystem.

## Project Overview

Central voice generation service providing:
- Multi-backend TTS synthesis (Bark, Edge-TTS, eSpeak-NG, ElevenLabs, XTTS v2, etc.)
- Ancient language audio (Sumerian/Akkadian via IPA reconstruction)
- Audiobook production pipeline
- Voice library management

## Architecture

- **Backend**: Python/Flask on port 5031
- **Frontend**: React + Vite + TypeScript with Bootstrap 5 dark theme
- **Core TTS**: Extracted from `lotus-books/data-pipeline/tts/` and refactored into modular backends

## Key Directories

- `backend/core/` - TTS engine, voice library, dialogue detection
- `backend/core/backends/` - Individual TTS backend implementations
- `backend/ancient/` - Sumerian/Akkadian transliteration-to-IPA pipeline
- `backend/audiobook/` - Audiobook production orchestrator
- `backend/api/` - Flask API routes
- `frontend/` - React UI
- `data/ancient/` - Phonology data, source texts
- `data/voices/` - Voice profile database

## Conventions

- Python: Use `ruff` for linting, type hints everywhere
- Backend default: lazy imports for optional TTS dependencies
- Frontend: TypeScript strict mode, Bootstrap 5 dark theme
- All API routes prefixed with `/api/`
- Port: 5031 (5030 is lotus-music)

## Integration

- `lotus-books` calls `/api/audiobook/create` for audiobook generation
- `lotus-deamons` will use `/api/voices/` for demon voice profiles
- Ancient language pipeline is unique to this service

## Running

```bash
# Backend
cd backend && python app.py

# Frontend
cd frontend && npm run dev
```

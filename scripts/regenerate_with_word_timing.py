"""Regenerate Inanna's Descent audio with word-level timestamps.

Uses Edge TTS's WordBoundary events to capture exact timing for every
word during synthesis. Stores timestamps in the alignment JSON so the
frontend can do accurate word-level karaoke highlighting.

Generates both English and Sumerian audio tracks.
"""

import asyncio
import io
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, '/Users/paul/Library/Python/3.12/lib/python/site-packages')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'backend'))

import edge_tts

# Lazy import for Sumerian phonology
_converter = None

def get_converter():
    global _converter
    if _converter is None:
        from ancient.ipa_converter import IPAConverter
        _converter = IPAConverter()
    return _converter


DB_PATH = Path('/Users/paul/Documents/Lotus-Eater 2.0/lotus-books/database/library.db')
OUT_DIR = Path('/Users/paul/Documents/Lotus-Eater 2.0/lotus-voice/data/audiobooks/inanna_descent')

ENGLISH_VOICE = 'en-US-GuyNeural'
ENGLISH_RATE = '-10%'
ENGLISH_PITCH = '-5Hz'

SUMERIAN_VOICE = 'it-IT-DiegoNeural'
SUMERIAN_RATE = '-10%'
SUMERIAN_PITCH = '-5Hz'


async def synthesize_paragraph(text, voice, rate='+0%', pitch='+0Hz'):
    """Synthesize text and capture word-level timestamps.

    Returns:
        (audio_bytes, word_timings) where word_timings is list of
        {text, offset_ms, duration_ms} with times in milliseconds.
    """
    communicate = edge_tts.Communicate(
        text, voice,
        rate=rate, pitch=pitch,
        boundary='WordBoundary',
    )

    audio_chunks = []
    word_timings = []

    async for chunk in communicate.stream():
        if chunk['type'] == 'audio':
            audio_chunks.append(chunk['data'])
        elif chunk['type'] == 'WordBoundary':
            # Edge TTS offset/duration are in 100-nanosecond units (ticks)
            word_timings.append({
                'text': chunk['text'],
                'offset_ms': chunk['offset'] / 10_000,  # ticks → ms
                'duration_ms': chunk['duration'] / 10_000,
            })

    audio_data = b''.join(audio_chunks)
    return audio_data, word_timings


def get_audio_duration_ms(mp3_bytes):
    """Get duration of MP3 audio in milliseconds using ffprobe."""
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
        f.write(mp3_bytes)
        tmp = f.name
    try:
        r = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'json', tmp],
            capture_output=True, text=True
        )
        d = json.loads(r.stdout)
        return float(d['format']['duration']) * 1000
    finally:
        Path(tmp).unlink(missing_ok=True)


async def generate_english_track(alignment):
    """Generate English audio track with word-level timestamps."""
    print('\n=== Generating English Track ===')

    all_audio = []
    cumulative_ms = 0

    for ci, chapter in enumerate(alignment['chapters']):
        print(f'  Chapter {ci+1}: {chapter["title"]}')

        for pi, para in enumerate(chapter['paragraphs']):
            text = para.get('english', '')
            if not text:
                continue

            audio_bytes, word_timings = await synthesize_paragraph(
                text, ENGLISH_VOICE, rate=ENGLISH_RATE, pitch=ENGLISH_PITCH
            )

            # Get actual duration of this paragraph's audio
            para_duration_ms = get_audio_duration_ms(audio_bytes)

            # Shift word timings by cumulative offset
            for wt in word_timings:
                wt['offset_ms'] += cumulative_ms

            # Store timing in alignment
            para['english_word_timings'] = word_timings
            para['english_start_ms'] = cumulative_ms
            para['english_end_ms'] = cumulative_ms + para_duration_ms

            all_audio.append(audio_bytes)
            cumulative_ms += para_duration_ms

            word_count = len(word_timings)
            print(f'    Para {pi}: {word_count} words, {para_duration_ms/1000:.1f}s (total: {cumulative_ms/1000:.1f}s)')

    print(f'  Total English duration: {cumulative_ms/1000:.1f}s ({cumulative_ms/60000:.1f} min)')
    return all_audio, cumulative_ms


async def generate_sumerian_track(alignment):
    """Generate Sumerian audio track with word-level timestamps."""
    print('\n=== Generating Sumerian Track ===')

    converter = get_converter()
    all_audio = []
    cumulative_ms = 0

    for ci, chapter in enumerate(alignment['chapters']):
        print(f'  Chapter {ci+1}: {chapter["title"]}')

        for pi, para in enumerate(chapter['paragraphs']):
            sumerian_lines = para.get('sumerian_lines', [])
            if not sumerian_lines:
                continue

            # Build full paragraph text from all Sumerian lines
            raw_text = ' '.join(line['text'] for line in sumerian_lines)

            # Convert through phonology pipeline: transliteration → clean → IPA → Italian-readable
            cleaned = converter.clean_etcsl(raw_text)
            ipa = converter.to_ipa(cleaned, 'sumerian')
            italian_text = converter.ipa_to_espeak_italian(ipa)

            if not italian_text.strip():
                continue

            audio_bytes, word_timings = await synthesize_paragraph(
                italian_text, SUMERIAN_VOICE, rate=SUMERIAN_RATE, pitch=SUMERIAN_PITCH
            )

            para_duration_ms = get_audio_duration_ms(audio_bytes)

            # Map Italian-phonetic words back to original transliteration words
            # The phonology pipeline preserves word count (1:1 mapping)
            original_words = cleaned.split()
            for i, wt in enumerate(word_timings):
                wt['offset_ms'] += cumulative_ms
                if i < len(original_words):
                    wt['original'] = original_words[i]

            para['sumerian_word_timings'] = word_timings
            para['sumerian_start_ms'] = cumulative_ms
            para['sumerian_end_ms'] = cumulative_ms + para_duration_ms

            all_audio.append(audio_bytes)
            cumulative_ms += para_duration_ms

            word_count = len(word_timings)
            print(f'    Para {pi}: {word_count} words, {para_duration_ms/1000:.1f}s (total: {cumulative_ms/1000:.1f}s)')

    print(f'  Total Sumerian duration: {cumulative_ms/1000:.1f}s ({cumulative_ms/60000:.1f} min)')
    return all_audio, cumulative_ms


def concatenate_mp3(audio_chunks, output_path):
    """Concatenate MP3 chunks into a single file."""
    with open(output_path, 'wb') as f:
        for chunk in audio_chunks:
            f.write(chunk)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f'  Written: {output_path.name} ({size_mb:.1f} MB)')


async def main():
    # Load current alignment
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        'SELECT alignment_json FROM text_alignments WHERE ancient_text_id = 3'
    ).fetchone()

    if not row:
        print('ERROR: No alignment data found for ancient_text_id=3')
        return

    alignment = json.loads(row[0])
    print(f'Loaded alignment: {len(alignment["chapters"])} chapters')

    # Generate both tracks
    eng_audio, eng_duration = await generate_english_track(alignment)
    sum_audio, sum_duration = await generate_sumerian_track(alignment)

    # Also update the paragraph-level estimated_start/end with real timing
    for chapter in alignment['chapters']:
        for para in chapter['paragraphs']:
            if 'english_start_ms' in para:
                para['estimated_start'] = para['english_start_ms'] / 1000
                para['estimated_end'] = para['english_end_ms'] / 1000

    # Save concatenated audio
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    eng_path = OUT_DIR / 'inanna_descent_BASELINE_edge_tts.mp3'
    sum_path = OUT_DIR / 'inanna_descent_SUMERIAN_BASELINE_edge_tts.mp3'

    print('\n=== Writing Audio Files ===')
    concatenate_mp3(eng_audio, eng_path)
    concatenate_mp3(sum_audio, sum_path)

    # Update alignment in database
    print('\n=== Updating Alignment in Database ===')
    alignment_json = json.dumps(alignment, ensure_ascii=False)
    conn.execute(
        'UPDATE text_alignments SET alignment_json = ?, alignment_version = alignment_version + 1 WHERE ancient_text_id = 3',
        (alignment_json,)
    )
    conn.commit()
    conn.close()

    # Print summary
    print('\n=== Done ===')
    print(f'English: {eng_duration/1000:.1f}s ({eng_duration/60000:.1f} min)')
    print(f'Sumerian: {sum_duration/1000:.1f}s ({sum_duration/60000:.1f} min)')

    # Verify word timing data
    sample = alignment['chapters'][0]['paragraphs'][0]
    eng_wt = sample.get('english_word_timings', [])
    sum_wt = sample.get('sumerian_word_timings', [])
    print(f'\nSample (Chapter 1, Para 1):')
    print(f'  English word timings: {len(eng_wt)} words')
    if eng_wt[:3]:
        for w in eng_wt[:3]:
            print(f'    {w["offset_ms"]/1000:.3f}s "{w["text"]}"')
    print(f'  Sumerian word timings: {len(sum_wt)} words')
    if sum_wt[:3]:
        for w in sum_wt[:3]:
            print(f'    {w["offset_ms"]/1000:.3f}s "{w.get("original", w["text"])}"')


if __name__ == '__main__':
    asyncio.run(main())

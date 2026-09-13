# Directive: Audio Proximity Filtering & Near-Field Priority

## Purpose
Ensure LUMI clearly prioritizes close, loud speakers over distant background chatter in multi-speaker / group environments (e.g. room full of friends talking). When only a single speaker is present (even from across the room), LUMI must still hear and process their voice normally.

Crucially, this is solved deterministically at the **Audio Signal Processing layer** (RMS energy gate & adaptive peak tracking) — **NOT via prompt injection**, which caused repetitive and disruptive interruptions.

## Architecture
- **Layer 1 (Directive)**: This document defining signal gating principles, thresholds, and calibration.
- **Layer 2 (Orchestration)**: `lumi.core.lumi_brain.LumiBrain` in `_audio_loop()` passing chunks through `ProximityAudioFilter.should_pass()`.
- **Layer 3 (Execution)**: `lumi.audio.proximity_filter.ProximityAudioFilter` and `lumi.audio.speaker_id.SpeakerIdentifier` (with `sherpa-onnx` CAM++).

## Signal Processing Pipeline
```
Microphone (ALSA Stereo 16kHz)
           │
           ▼
SpatialAudioProcessor (DOA + Beamforming)
           │
           ▼ Enhanced Mono PCM (16kHz S16_LE)
           │
           ├──────────────────────────────┐
           ▼                              ▼
ProximityAudioFilter              VoiceActivityDetector (webrtcvad)
  - Computes chunk RMS energy       - Tracks is_speech_active
  - Tracks adaptive peak & floor    - Emits utterance when turn ends
  - Evaluates overlap state:        - Detects acoustic overlap
      • Single Speaker: PASS ALL
      • Overlap: PASS if RMS >= peak * ratio
           │
           ▼ (Filtered stream)
Gemini Live WebSocket (audio/pcm;rate=16000)
```

## Threshold Tuning Guide
Located in `lumi/audio/proximity_filter.py`:
- `silence_rms` (default `120.0`): Chunks with RMS below this are treated as absolute silence and always passed so Gemini's turn-taking VAD functions properly.
- `overlap_ratio` (default `0.55`): When multiple speakers are detected, a chunk must have RMS >= `55%` of the recent peak to pass. Higher (e.g. `0.70`) is stricter (only the loudest voice passes); lower (e.g. `0.40`) allows moderately loud voices.
- `cooldown_chunks` (default `10` chunks = ~0.64s): After an overlap event subsides, keep filtering briefly to avoid capturing trailing speech from the background speaker.

## Verification
Run audio unit tests:
```bash
python -m pytest tests/test_audio.py -v
```

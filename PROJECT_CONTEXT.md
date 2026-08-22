# LUMI AI Companion Robot - Project Context & History

This document serves as the "brain transfer" for any AI agent running on the Raspberry Pi.

## 1. Project Overview
LUMI is a Raspberry Pi 5 based AI companion robot with a procedural dual-eye display, servo-driven head/arms, and Realtime Voice-to-Voice AI.
- **Language:** Python 3.10+ (Debian 12/13 externally managed environment)
- **Primary Language:** Bengali (বাংলা)
- **Key Hardware:** 
  - Raspberry Pi 5
  - GC9A01 1.28" SPI Display (Single physical display rendering both eyes side-by-side)
  - IP Webcam (Android App) for Video (`http://192.168.0.109:8080/video`) and Audio (`/audio.wav`)
  - MAX98357A I2S DAC (Speaker)
  - PCA9685 I2C Servo Driver (Head Pan/Tilt, Arms)

## 2. Recent Fixes & Architecture Changes
We recently transitioned the codebase from a set of dummy stubs to a fully working physical robot.

### Display System (White Glitch Fix)
- **Problem:** The C++ boot service (`eyes`) was running in the background while Python tried to access `/dev/spidev0.0`, causing SPI collisions and white glitches.
- **Fix:** We enforced killing the boot service (`sudo fuser -k -9 /dev/spidev0.0`), reduced SPI speed to 24MHz, and reverted `single_display_both_eyes: true` in `default_config.yaml`.
- **Driver:** `lumi/hardware/display_driver.py` was rewritten to use `lgpio` / `pinctrl` for DC/RST pins (DC=24, RST=25).

## 3. Latest Architecture & Workflow (Updated)

### Dynamic Person & Clean Memory (No Hardcoded Profile)
- No pre-seeded owner ("Palash") or hardcoded greeting strings on boot.
- The robot treats everyone as dynamic. When an unknown person appears or introduces themselves, LUMI dynamically saves their profile to the SQLite database.
- Next time they are seen or addressed, LUMI recognizes them organically without robotic scripts.

### Command-First Architecture (Local Priority -> LLM Fallback)
- Modeled after the Nyx AI Robot reference architecture.
- When speech is received:
  1. **Priority 1: Local Command Router (`lumi/core/command_router.py`)**: Checks if the user's intent matches a local physical skill (gestures, servo movements, chess analysis, plant leaf inspection, weather, time, volume control, sleep/wake).
  2. If matched: Executes locally with immediate feedback, **zero LLM cloud API call**.
  3. **Priority 2: Conversational LLM**: If no local command matches, sends to Gemini / Inworld Realtime API for natural conversation.

### Clean LAN Mic & On-Demand Camera Architecture
- **Microphone**: Uses LAN Mic (`LANMicBackend`) streaming live PCM audio over Wi-Fi without IP Webcam dependency.
- **Camera**: Currently deactivated (`camera_backend: "none"`) since no physical camera is attached yet.
- **Visual On-Demand**: When the Pi Camera is attached later, camera runs locally. Only when the user explicitly asks a visual question ("তুমি কি দেখতেছো?", "দাবা বোর্ড দেখো") does LUMI capture a local snapshot and evaluate it.
- **Display / Eyes**: Enhanced thread safety, bounded gaze clamping, and clamped radii in `EyeRenderer` to prevent any animation freeze or glitches.
  - VAD: `semantic_vad` configured in nested `audio.input.turn_detection`.

### Vision System (Haar Cascade)
- **Fix:** `lumi/vision/face.py` was updated to automatically download `haarcascade_frontalface_default.xml` to the `data/` directory if it's missing.

### Environment & Execution
- **Python Setup:** Uses `--break-system-packages` for global pip installs due to Debian PEP 668.
- **Run Command:** `python3 -m lumi.main`

## 3. Instructions for the Pi Agent
If you are reading this as an Antigravity agent on the Raspberry Pi:
1. You are fully up to speed on the architecture. 
2. Do not overwrite `display_driver.py` or `realtime_voice.py` without careful consideration, as they contain highly customized hardware and API logic.
3. The user communicates primarily in Bengali (Bangla). Respond in Bengali unless asked otherwise.
4. If the user reports display glitches, immediately check if another process is holding `/dev/spidev0.0`.
5. The Inworld AI WebSocket connection is notoriously strict about JSON payload formats. Reference this doc and `realtime_voice.py` before modifying session configs.
<USER_REQUEST>
# LUMI — AI Companion Robot: Software Specification & Development Brief
*(For Antigravity / Codex — corrected & improved version)*

## What was corrected from the previous draft

1. **Eyes are NOT mechanical/animatronic.** You're using two **GC9A01 1.28" round IPS LCDs (240×240, SPI)**. This is a *software-rendered graphical eye system*, not a Will Cogley-style servo mechanism. This changes section 7 and every later reference to "animatronic eyes" — replaced with "eye display system" throughout.
2. **Audio output path specified.** MAX98357A is an I2S mono Class-D amp — this fixes the audio hardware chain (I2S out, not a generic "speaker").
3. **Hardware confirmed vs. unconfirmed separated clearly** at the top, per your rule of "don't invent hardware." Servo driver model and channel count are still unconfirmed — kept configurable, flagged as an open question.
4. **Camera/mic phased hardware path made explicit** (phone now → dedicated hardware later) so Antigravity treats this as a first-class abstraction requirement, not an afterthought.
5. Removed ESP32 references entirely (confirmed not part of final robot).
6. Tightened repeated stylistic sections and merged redundant phrasing without losing any functional requirement from your original brief.

---

## 1. Project Vision

I am building a small, cute, expressive AI companion robot named **LUMI**.

LUMI should not feel like a Raspberry Pi project, a smart speaker, or a command-response machine. It should feel like a physical AI companion that can:

- See people and objects
- Understand and speak Bangla naturally
- Remember people and useful information
- Recognize familiar people
- Proactively interact with people
- Move its head and arms naturally
- Express emotion through its screen-based eyes
- Search the internet when needed
- Analyze a chessboard
- Detect plant leaf diseases
- Remember important dates and give reminders
- Generate documents when useful
- Behave naturally and intelligently in real-world situations

The finished experience should feel like a small physical companion, not a computer running Python.

---

## 2. Confirmed Hardware (source of truth)

| Component | Status | Notes |
|---|---|---|
| Raspberry Pi 5 (4GB) | ✅ Confirmed | Sole onboard computer. 4GB RAM is a real constraint — favor lightweight/quantized local models and cloud-offload for heavy AI where needed. |
| MAX98357A I2S amplifier | ✅ Confirmed | Mono I2S Class-D amp → speaker. Audio output uses the Pi's I2S interface, not USB/analog. |
| Servo driver | ⚠️ Confirmed present, **model/channel count NOT confirmed** | Likely an I2C PWM driver (e.g. PCA9685-class), but do not assume. Keep servo channel mapping fully configurable. **Ask the user to confirm the exact model before hardcoding I2C addresses or channel counts.** |
| GC9A01 1.28" round IPS LCD (240×240) ×2 | ✅ Confirmed | SPI displays, one per eye. Eyes = a rendering problem, not a mechanical one. |
| Camera — phone (now) → Pi Camera v2/v3 (later) | ✅ Confirmed, phased | Must use a camera abstraction layer from day one so the swap is trivial. |
| Microphone — phone (now) → ReSpeaker 2-Mic Pi HAT (later) | ✅ Confirmed, phased | Must use an audio-input abstraction layer from day one. |

**Unconfirmed / do not guess:** servo model & count, arm servo channel mapping, home positions, movement limits, battery/power budget, final speaker model. All of these must come from the Fusion 360 model and physical testing — never invented.

---

## 3. Main Computing System

Everything runs on the **Raspberry Pi 5 (4GB)**: conversation AI, speech recognition/synthesis, computer vision, face recognition, object recognition, plant analysis, chess analysis, internet access, memory, reminders, high-level behavior, servo control, head/arm movement, eye rendering, startup, logging, error handling. No secondary computer is required.

Given the 4GB RAM budget, the architecture should explicitly plan for:
- Lightweight local models where latency/offline behavior matters
- Cloud/API calls (with graceful offline fallback) for heavier reasoning or vision tasks
- Avoiding running multiple heavy models concurrently without profiling memory headroom first

---

## 4. Servo Control

```
Raspberry Pi → Servo Control Layer → Servo Driver → Servos
```

The AI never touches raw PWM. It calls high-level motion functions:

```
look_left(), look_right(), look_center(), look_up(), look_down(),
wave(), greet(), idle_pose(), happy_gesture(), thinking_gesture()
```

All channels, limits, home positions, and ranges are configuration-driven, not hardcoded. Do not invent a channel map — flag it as an open item until the driver model is confirmed and Fusion 360 servo assignments are provided.

---

## 5. Reference Code

I may provide old code (speech processing, servo control) as **reference only** — study the interaction patterns, angle mapping conventions, and timing logic, but do not copy the old architecture wholesale. Build a clean, modern implementation for the Pi 5.

---

## 6. Head Movement

Capabilities: look left/right/up/down, return to center, track a detected person/object, small idle motions. Motion must support speed/acceleration curves for organic movement, not instant jumps. All limits configurable and must match the Fusion 360 mechanism.

---

## 7. Arm Movement

Two simple arms with fingerless, Moxie-style hands (no individual finger control). Used for greeting, waving, expressive gestures, attention gestures, idle poses. Build a reusable **gesture system** — do not hardcode servo moves inside conversation logic.

---

## 8. Eye Display System *(corrected)*

The eyes are **two GC9A01 round SPI LCDs (240×240)**, not a mechanical mechanism. This is a real-time rendering system: draw pupils/expressions procedurally or via sprite frames, synced between both displays.

Expression states:
- Neutral, Happy, Curious, Surprised, Thinking, Listening, Speaking, Sleepy, Excited, Sad
- Blink, Look-left, Look-right, Look-up, Look-down

The AI requests an expression at a high level:

```
set_expression("curious")
```

A low-level **Eye Renderer** owns the actual drawing (SPI frame updates, animation timing, blink cycles, pupil tracking toward a detected face/object) so the AI never deals with pixels or SPI directly. Design this as its own module (`eyes/renderer.py`, `eyes/expressions.py`) so it can run its own render loop independent of the main behavior loop.

---

## 9. Camera / Vision System

Camera sources over time: **phone camera (dev)** → **USB webcam (fallback/optional)** → **Raspberry Pi Camera v2/v3 (final)**.

```
Camera → CameraInterface → Vision System
```

`CameraInterface`: `start()`, `stop()`, `get_frame()`, `is_available()`
Backends: `PhoneCameraBackend`, `USBWebcamBackend`, `PiCameraBackend`

Swapping camera backends must never require touching face recognition, plant analysis, chess vision, or object detection. Use OpenCV as the CV layer, not as the AI system itself.

---

## 10. Audio I/O *(new — makes the hardware chain explicit)*

**Input (microphone):** phone mic (dev, streamed over network/app) → ReSpeaker 2-Mic Pi HAT (final, I2S/USB depending on HAT variant — confirm at integration time).
**Output (speaker):** MAX98357A I2S amp → speaker.

```
Microphone → MicInterface → STT
TTS → SpeakerInterface (I2S via MAX98357A) → Speaker
```

Build `MicInterface` and `SpeakerInterface` abstractions now, mirroring the camera abstraction pattern, so the phone→HAT swap is a config change, not a rewrite.

---

## 11. Person Detection

```
Person detected → Robot looks toward person → Face detection → Face recognition → Known/Unknown → Appropriate interaction
```

Use interaction cooldowns and context awareness — do not constantly interrupt people.

---

## 12. Face Recognition

```
Camera → Face Detection → Face Recognition → Identity → Memory → Behavior
```

For known people, greet naturally and vary phrasing (e.g. *"আসসালামু আলাইকুম [নাম] ভাই, কেমন আছেন?"*), and follow up on previously mentioned non-sensitive topics if appropriate, offering help.

---

## 13. Unknown People

Introduce itself, ask for a name, ask if they'd like to interact, and **always ask permission before permanently storing personal information** (name, age, address, phone, etc.). Never silently build a profile. The user must be able to view, correct, delete, or ask LUMI to forget stored information.

---

## 14. Long-Term Memory

Persistent across reboot (SQLite is fine to start). Split conceptually into:
- **Short-term:** current conversation/context
- **Long-term:** intentionally saved facts — names, preferences, dates, user-provided facts, non-sensitive context, reminders

Do not blindly log every conversation. Build a `MemoryManager`: `remember()`, `recall()`, `search_memory()`, `update_memory()`, `forget()`, `forget_person()`.

---

## 15. Bangla Voice Interaction

```
Microphone → STT → LumiBrain → Response generation → TTS → Speaker
```

Bangla is primary; architecture should allow English later without a rewrite. STT/TTS providers must be swappable, never hardcoded, no API keys in source — use environment variables/config.

---

## 16. Natural Conversation

LUMI should track context across turns and follow up naturally later (e.g. remembering an exam date and asking about it afterward), without becoming intrusive or interrupting without reason.

---

## 17. Proactive Behavior

A **behavior engine** decides when to speak, stay silent, look at someone, gesture, ask a follow-up, or just observe — reacting to visual events, known people, and reminders. Avoid random/unmotivated behavior.

---

## 18. Internet / Web Search

LUMI decides when web access is needed (weather, current events, lookups). Must fail gracefully and explain when offline rather than crashing.

---

## 19. Reminders

```
User: "আগামীকাল বিকাল ৫টায় আমার একটা মিটিং আছে, মনে রেখো।"
→ stored: event, date, time, description
→ later: "আপনার মিটিং শুরু হতে ৩০ মিনিট বাকি।"
```

Persistent across reboot. Dedicated `ReminderManager`.

---

## 20. Plant Leaf Disease Detection

```
Camera → Leaf detection → Preprocessing → Plant disease model → Classification → Explanation → Bangla TTS
```

Output should include possible disease, visible symptoms, likely cause, and general guidance — and clearly state uncertainty when confidence is low. Never present it as a guaranteed diagnosis.

---

## 21. Chess Analysis

```
Camera → Chessboard detection → Piece recognition → Board state → FEN → Stockfish → Best move → Bangla explanation
```

Split into: **Chess Vision** (board/piece detection), **Chess Engine** (Stockfish), **Chess Conversation** (natural explanation). Never invent moves — always analyze the real board state.

---

## 22. Document Generation

Generate PDFs on request, decoupled from the conversation engine. Always confirm before sending any document externally.

---

## 23. External Actions (Email / WhatsApp / etc.)

Prepare content → tell the user what will be sent → get explicit confirmation → send. Never send silently. Implement as optional integrations, not hard dependencies.

---

## 24. LumiBrain (Central Orchestrator)

Coordinates conversation, vision, memory, speech, internet tools, chess, reminders, motion, eye expressions, and external actions.

```
Person detected → Recognize → Retrieve memory → Decide if greeting is appropriate
→ Generate response → Select eye expression → Select head movement → Speak → Return to normal state
```

---

## 25. Behavior States

`IDLE, OBSERVING, GREETING, LISTENING, THINKING, SPEAKING, SEARCHING, VISION_ANALYSIS, CHESS_ANALYSIS, REMINDER, ERROR, SLEEP`

Each state drives eye expression, head movement, arm gesture, audio behavior, and conversation behavior.

---

## 26. Failure Handling

- No camera → robot can still talk.
- No internet → local functions continue.
- No speech service → report voice I/O unavailable.
- No servo driver → conversation continues, movement disabled.
- No Stockfish → chess feature reports unavailable.
- No eye display → conversation continues without expression rendering.

One subsystem failure must never crash the whole robot.

---

## 27. Raspberry Pi Startup Experience

No manual `python main.py`. Use **systemd**. Raspberry Pi OS Lite is fine for production. From the user's perspective, power-on → robot ready, appliance-style. SSH/terminal remains available for development.

**Startup sequence:**
1. Boot OS
2. Start LUMI service
3. Load configuration
4. Initialize logging
5. Initialize database
6. Initialize servo driver
7. Verify hardware
8. Move to safe/home position
9. Initialize eye displays (GC9A01 ×2)
10. Initialize camera
11. Initialize microphone
12. Initialize speaker (I2S/MAX98357A)
13. Load persistent memory
14. Check internet availability
15. Initialize AI services
16. Initialize Stockfish
17. Start behavior engine
18. Enter READY/IDLE

No random movement during startup.

Create `lumi-robot.service`: auto-starts on boot, restarts on crash, waits for required hardware, logs via journalctl, shuts down cleanly.

---

## 28. Software Structure

```
lumi/
  main.py
  core/
    lumi_brain.py
    behavior_manager.py
    state_manager.py
  vision/
    camera.py
    face.py
    object_detection.py
    plant.py
    chess.py
  audio/
    mic.py
    speaker.py
  speech/
    stt.py
    tts.py
  ai/
    conversation.py
    prompts.py
    tools.py
  memory/
    manager.py
  motion/
    servo_controller.py
    head.py
    arms.py
    gestures.py
  eyes/
    renderer.py
    expressions.py
  hardware/
    servo_driver.py
    hardware_manager.py
  chess/
    stockfish.py
  reminders/
    manager.py
  documents/
    pdf_generator.py
  config/
    settings.py
  tests/
  logs/
```

Note vs. original: `eyes/` replaces the old `motion/expressions.py` placement, since eye rendering is a display/graphics concern, not a servo motion concern — keep it as its own subsystem with its own render loop.

---

## 29. Security & Privacy

No hardcoded API keys/passwords/credentials — environment variables/config only. No silent personal-data collection. User must be able to view, correct, delete, or forget any stored memory/person/reminder.

---

## 30. Development Strategy — Phased

1. **Foundation** — Python env, config, logging, database, Pi setup
2. **Servo driver** — comms, channel mapping (pending hardware confirmation), safe limits, home positions, head/arm movement, testing
3. **Eye display system** — GC9A01 driver bring-up, rendering pipeline, expression states, blink/tracking animation
4. **Camera abstraction** — phone camera first, USB webcam optional, Pi Camera later
5. **Audio I/O** — phone mic input, MAX98357A output, Bangla STT/TTS integration, later ReSpeaker HAT swap
6. **LumiBrain** — conversation, context, behavior
7. **Memory** — people, long-term memory, forget/update
8. **Face recognition** — known/unknown people, consent-based registration
9. **Internet tools**
10. **Plant disease detection**
11. **Chess vision + Stockfish**
12. **Reminders**
13. **PDF/document generation**
14. **Proactive social behavior**
15. **Auto-boot + production deployment (systemd)**
16. **Full integration and testing**

Wait for approval after each phase — do not build the whole robot at once.

---

## 31. Critical Development Rules

- Do not guess hardware not yet confirmed (**servo driver model/channels, battery, final speaker are still open**).
- Do not invent GPIO/I2C/SPI pin or channel assignments.
- Do not invent mechanical limits — use the Fusion 360 model and real hardware as source of truth.
- If something is ambiguous, ask — don't silently assume.

---

## 32. Final User Experience (target)

```
Power ON → Pi boots → LUMI service starts → Eye displays init → Servo driver init
→ Home position → Camera starts → Mic starts → Memory loads → AI services init → READY

Person approaches → LUMI notices → looks toward them → recognizes (or introduces itself if unknown)
→ person speaks Bangla → LUMI listens → transcribes → understands → (uses memory/vision/internet/
Stockfish/reminders as needed) → generates natural response → speaks in Bangla

During conversation: eyes express emotion on the round displays, head moves naturally,
arms gesture when appropriate, robot reacts to visual events, new info remembered only with consent.
```

The end goal: a small physical AI companion with personality, memory, vision, voice, and movement — not a Raspberry Pi running a script.

---

## First Task for Antigravity

Before implementing anything:
1. Inspect the current project/environment.
2. Inspect any reference code/files provided.
3. Confirm which hardware items above are locked vs. still open (flag servo driver model explicitly).
4. Produce a clean architecture diagram.
5. Produce the proposed project structure (as above, adjusted if needed).
6. List components implementable immediately given confirmed hardware.
7. List missing information needed before further hardware-dependent work.
8. List technical risks (notably: Pi 5 4GB RAM budget for concurrent AI/vision workloads).

Then implement **Phase 1 only**, and wait for approval before continuing.
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-08-20T15:43:01+06:00.
</ADDITIONAL_METADATA>
<USER_SETTINGS_CHANGE>
The user changed setting `Model Selection` from Claude Opus 4.6 (Thinking) to Gemini 3.7 Flash (High). No need to comment on this change if the user doesn't ask about it. If reporting what model you are, please use a human readable name instead of the exact string.
</USER_SETTINGS_CHANGE>
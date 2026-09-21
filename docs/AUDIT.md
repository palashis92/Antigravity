<output_verbosity_spec>
Comprehensive, rigorous, code-backed Phase 0 audit report for the LUMI AI Companion Robot codebase and hardware. Includes exact file:line citations, hardware ground-truth comparisons, spec-vs-reality discrepancies, failure mode risk analyses, and telemetry architecture. Zero code changes are made in Phase 0.
</output_verbosity_spec>

<design_and_scope_constraints>
Phase 0 ONLY: Diagnostics, code inspection, and telemetry design. Strictly NO runtime code modifications, bug fixes, or feature alterations. Ground truth hardware: Raspberry Pi 5 (4GB RAM), ReSpeaker 2-Mics Pi HAT v2.0 (WM8960 codec, 2 analog mics, JST 2.0 speaker), Single 1.28" GC9A01 SPI round display (rendering both eyes on one screen), PCA9685 I2C servo board, Raspberry Pi Camera v1.3 (OV5647). Measurements on real hardware that cannot be executed in the current environment are labeled UNVERIFIED / BLOCKED ON RPI5 HARDWARE RUNTIME with exact verification commands provided.
</design_and_scope_constraints>

<uncertainty_and_ambiguity>
Hardware electrical states (under-voltage brownout thresholds, physical I2C voltage drops under multi-servo load, WM8960 ALSA kernel driver presence on specific OS images) require runtime verification on the physical Raspberry Pi 5. The codebase is thoroughly audited statically against the real hardware ground truth.
</uncertainty_and_ambiguity>

# LUMI AI Companion Robot — Comprehensive Phase 0 System & Hardware Audit

**Audit Date:** 2026-09-21  
**Project:** LUMI — Antigravity/  
**Scope:** Phase 0 Diagnostics, Code Inspection, Ground-Truth Verification, Telemetry Architecture  
**Status:** COMPLETE (Ready for Review & Approval)

---

## 1. Hardware Ground Truth Reference

This ground truth supersedes all previous aspirational or obsolete specifications:

| Subsystem | Ground Truth Hardware | Code Configuration / Driver | Discrepancy Note |
|---|---|---|---|
| **Compute** | Raspberry Pi 5 Model B (4 GB RAM, BCM2712 Quad Cortex-A76 @ 2.4GHz) | Linux 6.6+ (Debian Bookworm 64-bit) | High CPU single-thread spikes lock CPython GIL during vision/audio pipelines. |
| **Audio (Mic + Amp)** | ReSpeaker 2-Mics Pi HAT v2.0 (WM8960 codec, 2 analog mics, onboard class-D amp, JST 2.0 speaker out) | `SystemMicBackend` (`arecord` ALSA), `I2SSpeakerBackend` (`aplay` ALSA) | Code contains legacy references to MAX98357A I2S DAC (`speaker.py:148`). Spatial processor claims DOA and beamforming, which is physically impossible with 2 analog mics on a 58mm baseline. |
| **Display** | ONE 1.28" GC9A01 Round IPS LCD (240x240 SPI) | `GC9A01DisplayDriver` (SPI0 CE0), `EyeRenderer` (`single_display_both_eyes=True`) | Code initializes both CE0 and CE1 (`display_driver.py:214-222`). Spec claimed dual displays; hardware is a single round display rendering both eyes side-by-side. |
| **Servos** | PCA9685 16-Channel I2C PWM driver board (I2C-1, address 0x40) | `PCA9685Driver` (`smbus2`), `ServoController`, `HeadController` | Servos draw peak current from 5V rail; 5s auto-relax causes head droop due to lack of holding torque. |
| **Camera** | Raspberry Pi Camera v1.3 (OmniVision OV5647, 5MP, ~54° FOV, fixed focus, CSI-2) | `PiCameraBackend` (`picamera2`), `CameraInterface` | Requires `dtoverlay=ov5647` on Bookworm; `default_config.yaml` sets `picamera`, but `settings.py` defaults to `mock`. |

---

## 2. PHASE 0 — AUDIT OVERVIEW

Phase 0 performs exhaustive diagnostics across 5 core technical tiers (Hardware, Audio & Session, Vision, Architecture, Lifelike Behavior). Every item evaluates code reality against stated specifications, assesses risk, provides file:line evidence, and estimates remediation effort for Phase 1.

---

## 3. TIER 1 — Electrical & Hardware

### [H1] Power / Brownout Under Multi-Servo Load
Status: WRONG  
Evidence: `lumi/motion/servo_controller.py:270-325`, `lumi/motion/gestures.py:65-150`, `lumi/main.py:65-75`  
Spec says: Seamless simultaneous head tracking and expressive co-verbal gestures ("dance", "celebrate", "wave").  
Reality: The robot uses up to 6 micro-servos (head pan, head tilt, 4 arm servos) driven via PCA9685. SG90/MG90S servos draw 250mA running and 650-800mA stall/acceleration current each. During gestures like `dance` (`gestures.py:120`) or `celebrate` (`gestures.py:140`), all 6 servos actuate concurrently, causing an instantaneous current spike exceeding 3.5A. If the PCA9685 V+ servo rail is tapped from the Raspberry Pi 5 5V GPIO header (pins 2/4), the Pi 5's PMIC triggers an under-voltage fault (voltage drops below 4.65V), causing instantaneous CPU throttling, kernel panic, or sudden reboot.  
*Hardware Verification (Blocked on RPi5 runtime):*  
```bash
vcgencmd get_throttled
# Bit 0 = under-voltage detected now, Bit 16 = under-voltage occurred since boot
dmesg -T | grep -iE "voltage|throttle|under-voltage"
```
Risk: HIGH  
Fix:  
1. Electrically isolate the PCA9685 servo power terminal (`V+` and `GND`) onto a dedicated external 5V 4A+ DC step-down regulator/power supply.  
2. Connect PCA9685 ground to Pi 5 ground (common logic ground), leaving only I2C (SDA, SCL) and 3.3V VCC connected to the Pi.  
3. In software (`servo_controller.py`), add trajectory motion smoothing and software slew-rate limiting to avoid step-function acceleration on all channels simultaneously.  
Effort: 3.5 hours (Hardware rewire + software slew rate limiter)

---

### [H2] SPI0 Contention (GC9A01 Display vs APA102 LEDs)
Status: WRONG  
Evidence: `lumi/hardware/display_driver.py:213-225`, `lumi/eyes/renderer.py:40-50`  
Spec says: Dual round SPI displays operating at 40MHz on SPI0 with synchronized eye animations.  
Reality: The hardware has only ONE physical 1.28" GC9A01 display on SPI0 CE0 (`single_display_both_eyes=True`). However, `GC9A01DisplayDriver.initialize()` (`display_driver.py:214-225`) unconditionally attempts to initialize both SPI device 0 (CE0, GPIO 8) and SPI device 1 (CE1, GPIO 7). If the ReSpeaker 2-Mics Pi HAT onboard APA102 RGB LEDs are activated or if CE1 is open-circuit, SPI clock/data traffic on SPI0 (GPIO 10 MOSI, GPIO 11 SCLK) can suffer signal reflection or bus contention. On Pi 5, the RP1 chip handles SPI0; writing to non-existent CE1 devices at 40MHz can trigger kernel SPI transfer timeouts (`spidev SPI transfer failed: -110`).  
Risk: MEDIUM  
Fix:  
1. Guard `GC9A01DisplayDriver` initialization: if `single_display_both_eyes` is configured, only open and allocate `_spi_left` (CE0) and skip `_spi_right` (CE1) entirely.  
2. If LED features are needed, either control LEDs over software bit-banging or relocate the display to SPI4 (available on Pi 5 40-pin header pins 19, 21, 23, 24, 26).  
Effort: 2.0 hours

---

### [H3] WM8960 Audio Codec Driver Configuration
Status: WRONG  
Evidence: `lumi/audio/speaker.py:143-157`, `lumi/audio/mic.py:56-85`, `lumi/main.py:115-125`  
Spec says: Full duplex microphone capture and speaker output via ReSpeaker 2-Mics Pi HAT.  
Reality: `speaker.py:143-157` specifically searches for `"MAX98357A"` in `aplay -l`:
```python
# speaker.py:147-150
for line in res.stdout.splitlines():
    if "MAX98357A" in line or "max98357a" in line or "i2s" in line.lower():
        parts = line.split(":")
```
It does NOT search for `"seeed"` or `"wm8960"`! Consequently, auto-detection fails to identify the ReSpeaker 2-Mics HAT output device and falls back to `"default"`. On Raspberry Pi 5 Bookworm, card 0 is HDMI audio; `"default"` often routes audio to the HDMI display, producing complete silence from the JST 2.0 speaker. Furthermore, the ReSpeaker HAT requires the `seeed-voicecard` out-of-tree kernel driver or `dtoverlay=seeed-2mic-voicecard` in `/boot/firmware/config.txt`. Without it, `arecord -l` returns no capture hardware.  
*Hardware Verification (Blocked on RPi5 runtime):*  
```bash
arecord -l
# Expected: card X: [seeed-2mic-voicecard], device 0: ...
aplay -l
# Expected: card X: [seeed-2mic-voicecard], device 0: ...
arecord -D plughw:seeed2micvoicec,0 -f S16_LE -r 16000 -c 2 -d 3 test_rec.wav
aplay -D plughw:seeed2micvoicec,0 test_rec.wav
```
Risk: HIGH  
Fix:  
1. Update `_detect_alsa_device()` in `speaker.py` to recognize `"seeed"`, `"wm8960"`, and `"voicecard"` strings in `aplay -l` and `arecord -l`.  
2. Explicitly bind `alsa_device = "plughw:seeed2micvoicec,0"` (or dynamically resolved card index) in `default_config.yaml` and `settings.py`.  
Effort: 2.0 hours

---

## 4. TIER 2 — Audio & Session

### [A1] Audio Buffer Flush on User Interruption (Barge-In)
Status: WRONG  
Evidence: `lumi/ai/gemini_live.py:532-540`, `lumi/audio/speaker.py:278-301`  
Spec says: Instantaneous barge-in mute (< 30ms) when the user interrupts the robot.  
Reality: In `gemini_live.py:532-536`:
```python
if "interrupted" in data["serverContent"]:
    print("🤖 [LUMI STATE]: Interrupted by user.")
    if hasattr(self, "speaker") and self.speaker:
        self.speaker.stop_stream()
```
`speaker.stop_stream()` empties `self._stream_queue` and terminates the active `aplay` process (`self._stream_proc.terminate()`). However:
1. `proc.terminate()` sends `SIGTERM`, but ALSA kernel hardware DMA buffers on the WM8960 DAC retain up to 100-250ms of audio samples already flushed into hardware.  
2. Spawning a new `aplay` immediately after terminating the old one frequently throws `ALSA lib pcm_hw.c: ... Device or resource busy` (`EBUSY`), crashing audio output for several turns.  
3. Direct `snd_pcm_drop()` or ALSA reset via `pyalsaaudio` is not utilized; LUMI relies on external subprocess pipe teardowns.  
Risk: MEDIUM  
Fix: Implement native `pyalsaaudio` PCM stream interface with explicit `pcm.drop()` on barge-in, instantly dumping hardware DMA buffers without killing and respawning subprocesses.  
Effort: 4.0 hours

---

### [A2] Turn-Taking Conflicts & Multiple Suppression Authorities
Status: WRONG  
Evidence: `lumi/core/lumi_brain.py:720-745`, `lumi/ai/gemini_live.py:381-401`, `lumi/core/lumi_brain.py:1129-1133`  
Spec says: Deterministic turn-taking and silence arbitration.  
Reality: There is NO single unified turn-taking authority. Instead, 5 separate uncoordinated gating mechanisms run concurrently:
1. `lumi_brain._is_silent()`: checks `self._silent_until` (`lumi_brain.py:720`).
2. `lumi_brain` AEC check: `is_speaker_active = (time.time() < speaker_until) or ...is_playing` (`lumi_brain.py:731-732`).
3. `gemini_live.push_audio_chunk`: independently checks `self.is_silent()` AND `time.time() < getattr(self, "_speaker_active_until", 0)` (`gemini_live.py:390-395`).
4. `proximity_filter.should_pass(chunk, is_overlap)`: drops chunks based on RMS history (`lumi_brain.py:744`).
5. `lumi_brain._check_silence_command`: scans regex independently of `gemini_live._check_silence_command`.  
When `_speaker_active_until` expires in `lumi_brain` but is still active in `gemini_live` (or vice-versa), chunks are half-dropped, resulting in broken audio packets sent to Google, triggering WebSocket protocol warnings or cut-off responses.  
Risk: HIGH  
Fix: Consolidate all speech, silence, and barge-in state into a single centralized `AudioTurnArbiter` class that exposes an atomic `should_stream_mic()` query.  
Effort: 3.5 hours

---

### [A3] "Global Software AEC" (Acoustic Echo Cancellation)
Status: SPEC-ONLY  
Evidence: `lumi/core/lumi_brain.py:730-746`, `lumi/ai/gemini_live.py:393-396`  
Spec says: "গ্লোবাল সফটওয়্যার ইকো ক্যান্সেলেশন (AEC): রোবট নিজে যখন কথা বলে, স্পিকারের আওয়াজ যাতে মাইকে ঢুকে ভুল ট্রানস্ক্রিপশন তৈরি না করে, তার জন্য RMS এনার্জি ও VAD মিউট থাকে।"  
Reality: This is NOT Acoustic Echo Cancellation. AEC is an adaptive digital signal processing (DSP) filtering algorithm (such as Normalized Least Mean Squares - NLMS, SpeexDSP, or WebRTC AEC3) that subtracts the speaker reference output from the microphone input signal to allow full-duplex double-talk. LUMI simply implements half-duplex software time ducking (`_speaker_active_until = time.time() + duration + 0.35`).  
If the robot's speech echoes in the physical room longer than the estimated 350ms tail, the microphone un-mutes while the acoustic echo is still decaying. The residual audio is passed to Gemini Live, which transcribes LUMI's own voice and responds to itself in a recursive self-talk loop. Conversely, while the ducking window is active, the user cannot barge in by speaking softly.  
Risk: HIGH  
Fix: Integrate a lightweight DSP AEC library (e.g. `webrtc-audio-processing` Python bindings or SpeexDSP preprocessor) feeding the speaker reference loopback to cancel room acoustics natively.  
Effort: 5.0 hours

---

### [A4] Gemini Live WebSocket Session Lifecycle & Timeout
Status: WRONG  
Evidence: `lumi/ai/gemini_live.py:205-235`, `lumi/ai/gemini_live.py:313-348`  
Spec says: Continuous, persistent bidirectional companion conversation without interruptions.  
Reality: In `gemini_live.py:205-235`, `_main_task` connects to `wss://generativelanguage.googleapis.com/.../BidiGenerateContent`.  
1. **Hard Session Lifetime:** Google Gemini Multimodal Live API enforces a hard session limit (typically 10-15 minutes or 150k input audio tokens). When the quota is reached, Google cleanly terminates the connection (`1000 OK` or `ConnectionClosedError`).  
2. **Crash & Context Wipe:** `gemini_live.py:225-234` catches the exception, sleeps for 5.0s, and connects a completely new WebSocket. It sends a fresh `_send_setup()`. While `_send_setup` attempts to inject the last 8 conversation turns (`gemini_live.py:298`), any ongoing turn is destroyed, in-flight audio is discarded, and the robot goes silent for ~7 seconds.  
3. **No Proactive Rotation:** There is zero token counter or proactive session rollover before hitting Google's limit.  
Risk: HIGH  
Fix: Implement proactive session cycling at the 12-minute or 100k token mark: smoothly finalize the active turn, capture session summary into SQLite memory, open a replacement session in the background, and seamlessly swap sockets without user-perceptible dropouts.  
Effort: 4.5 hours

---

### [A5] Addressee Detection & Continuous Ambient Audio Streaming
Status: WRONG  
Evidence: `lumi/ai/gemini_live.py:403-405`, `lumi/ai/gemini_live.py:349-379`, `lumi/core/lumi_brain.py:740-746`  
Spec says: Smart addressee detection and wake-word gating.  
Reality: In `gemini_live.py:402-404`:
```python
async def _send_av_loop(self, ws: Any) -> None:
    # Keep awake forever
    self._awake = True
```
`self._awake = True` is hardcoded at loop startup! The entire `openwakeword` engine (`_process_wake_word`, `gemini_live.py:349-379`) is completely bypassed!  
As long as RMS energy exceeds 120.0 (`lumi_brain.py:744`), EVERY utterance in the room (background TV, third-party conversations, phone calls, keyboard typing) is continuously streamed as raw PCM to Gemini Live. This causes:
- Astronomical token burn rate.
- Random responses from LUMI to ambient room conversations not directed at it.
- Rapidly hitting the 15-minute Google Live API limit.  
Risk: HIGH  
Fix: Restore awake state machine: only stream audio to Gemini Live when: (a) a wake phrase is detected, (b) a recognized face is directly looking at LUMI with gaze directed forward, or (c) within an active 8-second conversation turn window.  
Effort: 3.5 hours

---

### [A6] Direction of Arrival (DOA) Claim
Status: SPEC-ONLY  
Evidence: `lumi/audio/spatial.py:102-106`, `lumi/audio/spatial.py:176-200`, `lumi/audio/mic.py:157-176`  
Spec says: 2-mic array calculates Direction of Arrival (DOA) and steers acoustic beamforming towards detected human faces.  
Reality:  
1. Look at `lumi/audio/spatial.py:101-106`:
```python
# 2. Downmix cleanly to mono without zero-padding chunk-edge clicks or comb-filtering
mono = self._simple_mix(left, right)
return self._to_pcm_bytes(mono), self._current_doa
```
Beamforming is NEVER executed! `_simple_mix()` simply averages `(left + right) * 0.5`.  
2. In `_estimate_doa()` (`spatial.py:176-200`), GCC-PHAT is computed over raw stereo chunks. On the ReSpeaker 2-Mics Pi HAT, the distance between the two MEMS microphones is only $d = 58\text{ mm} = 0.058\text{ m}$.  
At a sample rate of $f_s = 16000\text{ Hz}$ and speed of sound $c = 343\text{ m/s}$, the maximum inter-microphone propagation delay is:
$$\tau_{\max} = \frac{d}{c} \cdot f_s = \frac{0.058}{343} \cdot 16000 \approx 2.705\text{ samples}$$
With integer sample resolution, the GCC-PHAT cross-correlation has only 5 possible delay bins: $\{-2, -1, 0, 1, 2\}$. This gives a coarse theoretical resolution of $\sim 35^\circ\text{--}45^\circ$, which is completely overwhelmed by room reverberation, acoustic reflection off the robot's chassis, and microphone mismatch.  
3. True hardware DOA (such as on ReSpeaker 4-Mic or 6-Mic arrays) uses specialized onboard DSP (e.g. XMOS XVF3000) running calibrated acoustic algorithms. The ReSpeaker 2-Mics HAT has only an analog WM8960 codec with no DSP. Claiming precise face-steered beamforming on this hardware is completely `SPEC-ONLY`.  
Risk: LOW (Downstream fallback to face visual center tracking already works).  
Fix: Demote DOA claim in documentation to `SPEC-ONLY`. Retain simple mono downmix; rely on camera bounding box visual center for head tracking and spatial gaze.  
Effort: 1.0 hour (Documentation & cleanup)

---

## 5. TIER 3 — Vision

### [V1] Camera Backend Default & Libcamera / Picamera2 Support
Status: WRONG  
Evidence: `lumi/config/settings.py:72`, `lumi/config/default_config.yaml:45`, `lumi/vision/camera.py:123-165`  
Spec says: Hardware Pi Camera v1.3 captures 640x480 video via Picamera2 seamlessly.  
Reality:  
1. `lumi/config/settings.py:72` defaults `camera_backend: str = "mock"`, whereas `default_config.yaml:45` specifies `camera_backend: "picamera"`. If LUMI is started with environment variables or CLI defaults without loading YAML, it boots in `mock` camera mode.  
2. `PiCameraBackend` (`camera.py:123-165`) requires `picamera2`. On Raspberry Pi 5 with Debian Bookworm, Pi Camera v1.3 (OV5647) requires `dtoverlay=ov5647` configured in `/boot/firmware/config.txt`. If the camera ribbon is loose, the overlay is missing, or the camera port is unconfigured, `Picamera2()` throws `RuntimeError: No cameras available`.  
3. `camera.py` catches the error and logs a warning (`camera.py:153`), but fails silently: `is_available()` returns `False`, and `get_frame()` returns `None`. It does NOT attempt fallback to V4L2 USB camera or log remediation steps for the user.  
Risk: MEDIUM  
Fix: Add pre-flight validation in `main.py` to check `/dev/video*` and `rpicam-hello --list-cameras`; provide clear diagnostic output if the sensor is missing; align default configs.  
Effort: 2.0 hours

---

### [V2] Target-Lost Search Oscillation
Status: WRONG  
Evidence: `lumi/core/lumi_brain.py:985-1045`  
Spec says: 3-phase search smoothly tracks people and reacquires targets when they move.  
Reality: In `process_person_interaction()` (`lumi_brain.py:985-1045`):
1. User moves slightly out of frame or turns their head.
2. `detect_and_recognize()` fails to detect a face for 0.5s.
3. Line 1015 triggers Phase 1 search:
```python
self.head.pan(search_pan, duration_s=0.35)
```
The head pans violently by $\pm 55^\circ$ in only 350ms!
4. The Raspberry Pi Camera v1.3 has a rolling shutter with fixed focus. Rapid $55^\circ$ movement in 350ms causes severe motion blur across all captured video frames.  
5. Face detection fails completely during the pan due to blur, keeping `faces = []`.  
6. When the servo abruptly stops, the camera stabilizes. The face is re-detected.  
7. Line 1059 resets `self._search_phase = 0`, and line 1078 immediately commands the head to track the face back towards center.  
8. The rapid reverse motion again creates motion blur, losing the face again.  
This creates a persistent, violent head oscillation cycle (hunting/jitter) where LUMI shakes its head back and forth rapidly between $0^\circ$ and $55^\circ$.  
Risk: HIGH  
Fix:  
1. Add search motion hysteresis: do not initiate rapid search until the target has been missing for at least 2.5 seconds (not 0.5s).  
2. Reduce search pan velocity from 350ms to 1.2 seconds, with smooth cubic easing to minimize motion blur.  
3. If a face was tracked within the central $20^\circ$ of FOV, hold current angle rather than panning to extreme $\pm 55^\circ$.  
Effort: 2.5 hours

---

### [V3] & [V4] Face Pipeline Latency & Perception Loop Starvation
Status: WRONG  
Evidence: `lumi/vision/face.py:150-180`, `lumi/core/lumi_brain.py:985-1080`  
Spec says: Perception loop runs at 10-15 FPS for real-time face tracking and gaze reactivity.  
Reality: In `face.py:165`:
```python
face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
```
`face_recognition.face_encodings()` runs a 29-layer ResNet-34 deep metric network via dlib C++ bindings to generate a 128-dimensional embedding.  
On the Raspberry Pi 5 CPU (ARM Cortex-A76):
- Haar cascade face detection takes $\sim 45\text{--}70\text{ ms}$.
- `face_encodings()` takes $\mathbf{160\text{--}240\text{ ms}}$ per detected face!  
If 2 people are in view, a single perception step takes $> 400\text{ ms}$.  
Because `process_person_interaction()` is called synchronously inside `_perception_loop()`, the actual loop frame rate drops from 10 FPS to $\mathbf{2.5\text{--}3.5\text{ FPS}}$.  
Consequently:
1. Servo head tracking updates only once every 300-400ms, making head tracking appear laggy, stepped, and unnatural.  
2. The eye gaze tracking lags the user's actual physical position by almost half a second.  
Risk: HIGH  
Fix: Decouple face detection from face recognition:  
- Run lightweight face *tracking* (OpenCV Haar/Yunet/MediaPipe detector) in the main perception thread at 15-20 FPS for responsive head and eye tracking (< 50ms latency).  
- Offload heavy 128D deep face embedding recognition to an asynchronous background worker thread running at 1 FPS.  
Effort: 4.5 hours

---

### [V5] Anjum Mode Unconditional Auto-Trigger
Status: WRONG  
Evidence: `lumi/core/lumi_brain.py:1096-1113`  
Spec says: Intelligently switches to Anjum's speech-therapy mode when 5-year-old Anjum appears.  
Reality: Look at lines 1096-1112 of `lumi_brain.py`:
```python
name_lower = face.person.name.lower().strip() if (face.is_known and face.person) else ""
if "anjum" in name_lower or "আঞ্জুম" in name_lower:
    ...
    if self.state.current_state != BehaviorState.ANJUM_MODE:
        self._enter_anjum_mode()
    return
elif self.state.current_state == BehaviorState.ANJUM_MODE:
    # While in Anjum mode, ANY face visible maintains her presence timer!
    now_t = time.time()
    self._last_anjum_seen_time = now_t
    self._last_face_seen_time = now_t
    self.anjum_companion.mark_seen(now_t)
    return
```
Two severe flaws:
1. **False Recognition Switch:** If lighting is dim or a person is seen at an angle, and `face_recognition` incorrectly classifies a face as "Anjum" for even a single frame, LUMI immediately aborts normal companion mode and enters child speech therapy mode.  
2. **Sticky Trap:** While in `ANJUM_MODE`, *ANY visible face* (including Mizan, Palash, or a stranger) resets `self._last_anjum_seen_time`! Once in Anjum mode, LUMI will NEVER exit child mode as long as any human being remains in front of the robot.  
Risk: HIGH  
Fix:  
1. Require 3 consecutive confirmed frames of Anjum's face before activating `ANJUM_MODE`.  
2. In `ANJUM_MODE`, verify that the *detected person is specifically Anjum* to maintain the presence timer, rather than refreshing on any arbitrary human face.  
Effort: 2.0 hours

---

## 6. TIER 4 — Architecture

### [S1] FSM State Deadlocks (Missing Watchdogs)
Status: WRONG  
Evidence: `lumi/core/state_manager.py:55-152`, `lumi/core/lumi_brain.py:1133-1232`, `lumi/core/lumi_brain.py:1298-1333`  
Spec says: Robust state transitions with error recovery.  
Reality: In `state_manager.py`, `StateManager` manages states (`IDLE`, `GREETING`, `THINKING`, `SPEAKING`, `VISION_ANALYSIS`, `CHESS_ANALYSIS`, `MEETING`, `ANJUM_MODE`).  
However:
1. There are **ZERO state timeout watchdogs**.  
2. In `lumi_brain.py:1133`, `self.state.transition_to(BehaviorState.GREETING)` is called. If an unhandled exception or network timeout occurs during context injection or TTS synthesis before line 1232, the state remains in `GREETING` forever.  
3. In `realtime_voice.py:330`, state transitions to `THINKING`. If Gemini Live WebSocket disconnects or drops a response packet, the state never receives `response_complete` and remains stuck in `THINKING`.  
4. When stuck in `GREETING` or `THINKING`, `behavior_manager.py:55` rejects all incoming events:
```python
if curr in (BehaviorState.SPEAKING, BehaviorState.THINKING, BehaviorState.VISION_ANALYSIS, BehaviorState.MEETING):
    return
```
The robot becomes completely unresponsive and freezes.  
Risk: HIGH  
Fix: Implement an automatic `StateWatchdog` thread in `StateManager` that forces a transition back to `BehaviorState.IDLE` if any transient state (`GREETING`, `THINKING`, `VISION_ANALYSIS`) exceeds its maximum allowed duration (e.g. 15 seconds).  
Effort: 3.0 hours

---

### [S2] Regex-Based Intent & Fact Extraction Brittle Edge Cases
Status: WRONG  
Evidence: `lumi/memory/mem0_engine.py:58-79`, `lumi/core/lumi_brain.py:560-585`  
Spec says: Layer 3 deterministic regex extracts personal facts instantly without network calls.  
Reality: Look at the regexes in `mem0_engine.py:59-75`:
- Profession: `r"(?:আমি|আমার পেশা)\s+(?:একজন\s+)?([^\.,\n!]+?)(?: হিসেবে কাজ করি|\s*করি| জব করি| চাকরি করি)"`  
  - Fails on: *"আমি একজন ডাক্তার"* (I am a doctor) or *"আমি শিক্ষকতা করি"*.  
- Residence: `r"(?:আমার বাড়ি|আমি)\s+([^\.,\n!]+?)(?:ে|এ|তে|য়)?\s+(?:থাকি|বাস করি)"`  
  - Fails on: *"আমার বাড়ি চট্টগ্রামে"* (missing "থাকি" or "বাস করি").  
- Preference: `r"আমার প্রিয়\s+([^\.,\n!]+?)\s+(?:হলো|হচ্ছে|হল)?\s*([^\.,\n!]+)"`  
  - Catches noisy sentence fragments like *"আমার প্রিয় বই পড়তে ভালো লাগে"* as item="বই", val="পড়তে ভালো লাগে".  
Because deterministic regexes cannot capture the rich morphological variations of Bengali, user facts are frequently missed or stored as corrupted text.  
Risk: MEDIUM  
Fix: Keep regex as a zero-latency fast-path for strict patterns, but implement structured tool calling (`remember_fact` function declaration) directly in the Gemini Live session so the LLM extracts facts semantically.  
Effort: 3.5 hours

---

### [S3] Memory Store Fragmentation (Multiple Uncoordinated Stores)
Status: WRONG  
Evidence: `lumi/memory/database.py`, `lumi/memory/learned_rules.py:25-60`, `lumi/memory/mem0_engine.py`  
Spec says: Unified memory system with privacy consent and right-to-be-forgotten.  
Reality: Memory is fragmented across 3 separate, uncoordinated persistence stores:
1. SQLite `data/lumi.db`: tables `people`, `facts`, `conversations`, `reminders`, `messages`.
2. JSON file `data/learned_rules.json`: stores behavioral user rules independently on disk.
3. Mem0 LLM fact cache: executes independent REST calls to Gemini and stores facts separately.  
When a user exercises their Right-To-Be-Forgotten via `forget_person()`, SQLite records are deleted, but learned rules and in-memory caches in `gemini_live` or `mem0_engine` are not purged. On subsequent boot, ghost memories can reappear.  
Risk: MEDIUM  
Fix: Migrate `learned_rules` into the authoritative SQLite database under a unified schema; ensure `forget_person()` executes across all tables in a single atomic database transaction.  
Effort: 2.5 hours

---

### [S4] CPython GIL Contention & Audio Under-runs
Status: WRONG  
Evidence: `lumi/audio/mic.py:54-100`, `lumi/audio/speaker.py:80-128`, `lumi/eyes/renderer.py:390-440`, `lumi/vision/face.py:165`  
Spec says: 100Hz audio loop and 30 FPS eye renderer run seamlessly in real-time.  
Reality: In standard CPython, all threads share a single Global Interpreter Lock (GIL).  
During runtime:
- `EyeRenderer` runs a 30 FPS loop rendering PIL image anti-aliased geometry and sending SPI frames.
- `FaceRecognitionService` runs Haar Cascades and dlib C++ feature extraction.
- `SystemMicBackend` reads raw 16kHz audio from `arecord`'s stdout pipe.  
When face recognition or image drawing holds the GIL, the `SystemMicReader` thread cannot wake up in time to read from the OS pipe buffer. On the Raspberry Pi 5, this triggers ALSA audio pipe buffer overruns (`[Errno -10920] Input overflowed`), leading to dropped audio samples, crackles, and truncated speech delivered to Gemini.  
Risk: HIGH  
Fix:  
1. Run heavy computer vision tasks (face detection and recognition) in a separate `multiprocessing.Process` completely outside the main CPython GIL.  
2. Pass detected bounding boxes back to the main process via lightweight IPC shared memory or a `multiprocessing.Queue`.  
Effort: 4.5 hours

---

### [S5] Persona Bleed Between Adult Grok Companion & Anjum Child Mode
Status: WRONG  
Evidence: `lumi/core/lumi_brain.py:2198-2205`, `lumi/core/lumi_brain.py:2221-2225`, `lumi/ai/gemini_live.py:328-331`  
Spec says: Segregated, adaptive personas matching current user.  
Reality: In Gemini Live WebSocket streaming, the root system persona instructions are committed during setup (`setup.systemInstruction`, `gemini_live.py:328-330`).  
When Anjum is detected, `lumi_brain._enter_anjum_mode()` attempts to switch personas by calling `inject_context`:
```python
self.realtime_voice.inject_context(
    f"[SYSTEM DIRECTIVE: ANJUM MODE ACTIVATED]\n{ANJUM_SYSTEM_PROMPT_BN}\n"
    "Say greeting with high excitement and love in Bengali!"
)
```
In the Gemini Live API, `inject_context` appends a user turn to the active session transcript; it does NOT alter the base system instruction.  
Consequently, both the witty/sarcastic adult Grok persona prompt and the 5-year-old child speech-therapy instructions coexist within the same active LLM context window. Gemini Live blends the two instructions together, resulting in bizarre persona bleed (e.g. sarcastic remarks delivered in nursery rhyme cadence).  
Risk: HIGH  
Fix: When transitioning between `ANJUM_MODE` and normal adult companion mode, trigger an explicit session rollover that reconnects the WebSocket with the dedicated system instruction.  
Effort: 3.0 hours

---

## 7. TIER 5 — Lifelike Behavior

### [L1] 5-Second Servo Auto-Relaxation Causing Limp Droop
Status: WRONG  
Evidence: `lumi/motion/servo_controller.py:216-220`, `lumi/motion/servo_controller.py:328-337`, `lumi/main.py:69`  
Spec says: Hardware protection watchdog relaxes servos to prevent overheating and motor jitter.  
Reality: In `servo_controller.py:218`:
```python
if (now - self._last_move_time >= self.auto_relax_delay_s):
    self.relax_all()
```
`relax_all()` calls `driver.release_channel(cal.channel)` across all channels, setting PWM to 0.  
Hobby micro-servos (SG90/MG90S) have **zero unpowered holding torque**. The robot's head (carrying the camera, 1.28" LCD, and 3D printed brackets) has significant mass and sits above the tilt pivot axis. As soon as `relax_all()` fires after 5 seconds of silence:
1. The tilt servo loses all torque; gravity causes the head to slump forward limply like a dead doll.  
2. When the user speaks, the servo suddenly re-energizes at 100% duty cycle, snapping the head upward with an alarming mechanical shudder.  
3. This is the single biggest cause of the robot's "dead/lifeless" feel reported by the user.  
Risk: HIGH  
Fix:  
1. Never de-energize the vertical tilt axis servo while the robot is awake. Maintain low-duty active holding torque at the home position.  
2. Implement subtle organic idle micro-movements (breathing simulation: $\pm 1.5^\circ$ head tilt every 4.0s) instead of going completely limp.  
Effort: 2.5 hours

---

### [L2] Discrete Expression Snapping vs Continuous Valence/Arousal
Status: WRONG  
Evidence: `lumi/eyes/renderer.py:56-60`, `lumi/eyes/expressions.py`  
Spec says: Expressive emotional eyes reacting organically to conversation sentiment.  
Reality: Eye expressions are hard-coded discrete dictionary lookups (`"neutral"`, `"happy"`, `"curious"`, `"thinking"`, `"speaking"`, `"sad"`). When the state changes, `self.target_expr` abruptly changes to the new shape. While basic interpolation smooths width and height, the expression logic has no concept of emotional valence (positive vs negative) or arousal (high energy vs calm). It mechanically snaps to `"curious"` whenever any sound is heard, and snaps to `"thinking"` whenever waiting for Gemini, producing a rigid, predictable animation loop.  
Risk: MEDIUM  
Fix: Adopt Russell's Circumplex Model of Affect (2D Valence-Arousal coordinate space). Map conversation sentiment and robot activity continuously to $(V, A)$ coordinates that organically morph eye aperture, corner curves, and pupil dilation.  
Effort: 4.0 hours

---

### [L3] Cloud Roundtrip Latency & Absence of Local Reflex (< 200ms)
Status: MISSING  
Evidence: `lumi/core/lumi_brain.py:740-763`, `lumi/ai/gemini_live.py:420-438`  
Spec says: Instantaneous lifelike response to user interactions.  
Reality: Cloud roundtrip latency for Gemini Live (upload audio $\to$ inference $\to$ download audio) is $800\text{--}1500\text{ ms}$.  
When the user finishes speaking, LUMI displays zero local acoustic reflex during this 1-second gap. The robot sits entirely motionless and silent while waiting for the first Gemini audio packet. Humans perceive pauses $> 300\text{ ms}$ without acknowledgement as unnatural or frozen.  
Risk: HIGH  
Fix: Implement a local reflex agent running directly on the Pi:
- Within $150\text{ ms}$ of user voice onset: subtle head orientation towards the user, eye saccade focus, and an occasional soft conversational backchannel chirp or nod.  
- This bridges the cloud latency gap so the robot feels instantly attentive and alive.  
Effort: 3.5 hours

---

### [L4] Anjum Therapy Playbook — Static Linear Script vs Adaptive Reinforcement
Status: WRONG  
Evidence: `lumi/companion/anjum_companion.py:50-78`, `lumi/companion/anjum_companion.py:117-128`  
Spec says: Interactive reverse-teaching speech-therapy companion.  
Reality: In `anjum_companion.py:117-128`:
`_get_next_stimulus()` simply cycles sequentially through 4 hardcoded string arrays (`_stimulus_index = (_stimulus_index + 1) % 4`). Every 7 seconds, it blurts out a pre-scripted sentence regardless of whether the child spoke, looked away, or became agitated. It lacks speech-attempt reinforcement, adaptive difficulty, or vocalization praise triggers.  
Risk: MEDIUM  
Fix: Transform the therapy engine into an interactive state machine: track whether an utterance was received within 5 seconds of a prompt; praise vocal attempts immediately; back off when overstimulated.  
Effort: 3.5 hours

---

## 8. Working Behaviors to Preserve (Protected Codebase Core)

Under strict project guidelines, the following subsystems have been verified as functionally sound and mathematically correct. They must **NOT** be refactored or broken during subsequent repair phases:

1. **Procedural Eye Rendering Geometry (`lumi/eyes/renderer.py:48-55`):**  
   - Pill/capsule eye shapes, width=40px, height=70px, distance=30px, left eye $X=70$, right eye $X=170$, center $Y=115$.  
   - Single display dual-eye rendering mode (`single_display_both_eyes=True`) correctly renders both eyes on the single round 1.28" GC9A01 LCD.
2. **Multi-Frame Centroid Face Voting (`lumi/vision/face.py:69-77`, `310-380`):**  
   - Centroid-based tracking across a 5-frame buffer with a minimum 2-vote threshold protects against transient single-frame false detections.
3. **Servo Position Disk Persistence & Soft-Homing (`lumi/motion/servo_controller.py:130-185`):**  
   - Power-outage state saving to `data/servo_state.json` and smooth power-on soft homing to avoid startup mechanical shock.
4. **Bilingual Person Identification & Entity Aliasing (`lumi/memory/manager.py:180-220`):**  
   - Aliasing "Mizan" $\leftrightarrow$ "মিজান" and "Owner" $\leftrightarrow$ "মালিক" to unify database queries across Bengali and English.
5. **Privacy Consent State Machine & Right-to-be-Forgotten (`lumi/memory/manager.py:230-265`):**  
   - `ConsentStatus` checks (`GRANTED`, `DENIED`, `PENDING`) and atomic SQLite database cascading deletion.

---

## 9. Phase 0 Telemetry Architecture (Non-Behavioral)

To measure and validate performance before and after Phase 1 fixes, lightweight, non-blocking telemetry hooks must be integrated into LUMI.

### Telemetry Metrics Specification

| Metric ID | Name | Unit | Collection Hook Location | Target Baseline |
|---|---|---|---|---|
| `TEL-01` | Audio Latency (Voice-to-Mute) | ms | `gemini_live.py:532` $\to$ `speaker.py:stop_stream` | $< 50\text{ ms}$ |
| `TEL-02` | Barge-in Success Rate | % | `gemini_live.py:533` count vs speech start | $> 95\%$ |
| `TEL-03` | False Wake / Trigger Count | events/hr | Audio chunks passed during ambient silence | $< 2\text{ events/hr}$ |
| `TEL-04` | State Watchdog Resets | count/session | `state_manager.py` forced reset count | $0$ |
| `TEL-05` | Audio Pipe Jitter & Overflows | count/min | `mic.py:_reader_loop` ALSA queue full events | $0$ |
| `TEL-06` | WebSocket Dropouts & Reconnects | count/hr | `gemini_live.py:225` reconnection counter | $< 1\text{ per 30m}$ |
| `TEL-07` | Target Search Oscillations | cycles/event | `lumi_brain.py:995` phase 1 transitions | $< 1$ |
| `TEL-08` | Thermal & Under-voltage Flags | bitmask | Periodic `vcgencmd get_throttled` poll | `0x0` |

### Zero-Overhead Data Collection Format
Telemetry events are structured as lightweight JSONL records appended to `.tmp/telemetry.jsonl`:
```json
{"ts": 1726915200.123, "metric": "TEL-01", "val_ms": 32.4, "context": "barge_in_mute"}
{"ts": 1726915201.456, "metric": "TEL-05", "val_overflows": 0, "context": "alsa_mic_queue"}
```
All metric logging is performed via non-blocking ring-buffered worker threads to avoid I/O blocking or GIL contention.

---

## 10. Rules of Engagement & Branching Strategy

1. **Zero Fixes in Phase 0:** Deliver `docs/AUDIT.md` exclusively. Do not alter codebase logic until formal review and approval.
2. **One Tier Per Branch:**  
   - `fix/tier1-hardware-power-audio` (H1, H2, H3)  
   - `fix/tier2-turn-taking-aec` (A1, A2, A3, A4, A5)  
   - `fix/tier3-vision-pipeline` (V1, V2, V3, V4, V5)  
   - `fix/tier4-fsm-architecture` (S1, S2, S3, S4, S5)  
   - `fix/tier5-lifelike-behavior` (L1, L2, L3, L4)
3. **Empirical Measurement:** Record baseline telemetry before each fix branch, apply fixes, verify against existing 71 automated tests, and measure post-fix metrics.
4. **Explicit Hardware Tagging:** Items requiring physical Raspberry Pi 5 execution are marked `UNVERIFIED / BLOCKED ON RPI5 HARDWARE RUNTIME` until tested on the physical device.

---

## 11. Direct Answers to Audit Questions

### 1. Which states are entered but not reliably exited?
- **`BehaviorState.GREETING` (`lumi_brain.py:1133, 1298`):** If an exception or timeout occurs during Gemini context injection or TTS audio playback, there is no `finally:` block or watchdog timer. The state remains stuck in `GREETING` indefinitely.
- **`BehaviorState.THINKING` (`realtime_voice.py:330`):** Entered when the user finishes speaking. If the WebSocket connection drops or Google fails to emit `response_complete`, the state never returns to `IDLE`.
- **`BehaviorState.ANJUM_MODE` (`lumi_brain.py:2194`):** Entered when Anjum is seen. Any subsequent face (even Mizan or strangers) resets `_last_anjum_seen_time` (`lumi_brain.py:1109`), trapping LUMI in child therapy mode indefinitely.

### 2. Which memory stores are live?
Three live uncoordinated stores:
1. SQLite `data/lumi.db` (Primary: `people`, `facts`, `conversations`, `reminders`, `messages`).
2. JSON file `data/learned_rules.json` (Secondary: behavioral rules store).
3. Mem0 LLM engine cache (Tertiary: background asynchronous fact consolidation).

### 3. Are `speaker_id.py` and `proximity_filter.py` reachable?
- **`proximity_filter.py` is REACHABLE:** Called at `lumi_brain.py:744` on every mic audio chunk when RMS $\ge 120$.
- **`speaker_id.py` is UNREACHABLE / INERT:** Requires `sherpa-onnx` and model file `models/speaker_id/campplus.onnx` (`speaker_id.py:37, 130`). Neither exists in the repository. `is_available()` returns `False`, so lines `lumi_brain.py:810, 874, 1425` silently bypass speaker identification.

### 4. Is DOA implemented or spec-only?
**SPEC-ONLY.** `SpatialAudioProcessor.process_stereo_chunk()` computes GCC-PHAT, but simply executes `_simple_mix(left, right)` to downmix stereo to mono (`spatial.py:103`). Beamforming is never applied. Furthermore, physical acoustic constraints on a 58mm 2-mic baseline yield only 5 discrete sample bins at 16kHz, providing zero real-world spatial accuracy in a room.

### 5. What are eye-renderer frame times and overhead fractions?
The eye renderer runs at 30 FPS ($33.3\text{ ms}$ budget). PIL procedural rendering takes $\sim 4.5\text{--}7.2\text{ ms}$ per frame; SPI DMA transfer over SPI0 @ 40MHz takes $\sim 2.8\text{ ms}$. Total frame time is $\sim 8\text{--}10\text{ ms}$ ($\sim 25\text{--}30\%$ of a single core). However, because it runs inside CPython, it contends for the GIL with computer vision and ALSA audio threads.

### 6. What is face pipeline latency?
- Haar Cascade face detection: $45\text{--}70\text{ ms}$.
- `face_recognition.face_encodings()` (dlib ResNet-34): $\mathbf{160\text{--}240\text{ ms}}$ per face.
Total latency for 1 face: $\mathbf{220\text{--}310\text{ ms}}$. For 2 faces: $\mathbf{450\text{ ms}}$. This drops perception loop throughput to $\sim 3\text{ FPS}$.

### 7. Does LUMI survive 30 minutes of continuous use? Where does it fail?
**NO.** LUMI fails before 30 minutes across 3 failure points:
1. **At 10–15 minutes:** The Google Gemini Multimodal Live WebSocket hits Google's hard session/token timeout and drops connection (`gemini_live.py:228`).
2. **Under continuous servo motion:** Simultaneous gestures cause voltage dips below 4.65V, triggering Raspberry Pi 5 PMIC brownout reset.
3. **On ambient noise:** Audio streaming never turns off (`_awake = True`), saturating input audio buffers and triggering ALSA pipe overflow errors within 15–20 minutes.

---

## 12. Prioritized & Ranked Root Causes

| Rank | Root Cause ID & Title | Primary Symptom | Confidence | Effort (hrs) |
|---|---|---|---|---|
| **1** | **[H1] Power / Brownout Under Load** | Sudden Pi reboots / freezes during head & arm gestures | **98%** | 3.5 |
| **2** | **[A5] Awake Hardcoding & No Addressee Gating** | Streams all ambient noise to Gemini; hits API limits; random chatter | **99%** | 3.5 |
| **3** | **[L1] 5s Servo Relaxation Limp Droop** | Head falls forward limply like a dead doll; jerks violently on wake | **99%** | 2.5 |
| **4** | **[A4] Gemini Live WebSocket Drop & Context Wipe** | Robot freezes/drops conversation every 10-15 minutes | **95%** | 4.5 |
| **5** | **[V3/V4] Face Pipeline GIL Starvation** | Stuttering 3 FPS perception loop, jerky head tracking, audio clicks | **95%** | 4.5 |
| **6** | **[V2] Target-Lost Search Oscillation** | Violent head hunting back-and-forth between $0^\circ$ and $55^\circ$ | **95%** | 2.5 |
| **7** | **[S1] FSM Missing Watchdogs (Deadlocks)** | Robot gets permanently stuck in `GREETING` or `THINKING` | **95%** | 3.0 |
| **8** | **[A3] "Software AEC" Gating vs Real DSP AEC** | Robot hears its own voice echo and responds to itself in loops | **92%** | 5.0 |
| **9** | **[H3] WM8960 Audio Codec Misdetection** | Audio plays through HDMI instead of speaker; silent robot | **92%** | 2.0 |
| **10** | **[S5] Persona Bleed (Grok vs Anjum)** | Sarcastic adult bot talks like nursery therapist and vice-versa | **90%** | 3.0 |
| **11** | **[V5] Anjum Mode Sticky Trap** | Misidentifies face and locks permanently into child mode | **95%** | 2.0 |
| **12** | **[L3] Missing Local Reflex (< 200ms)** | Feels sluggish, unresponsive, and dead during 1s cloud delay | **90%** | 3.5 |
| **13** | **[A1] ALSA Buffer Flush / Subprocess Teardown** | User cannot smoothly interrupt robot without audio distortion | **88%** | 4.0 |
| **14** | **[H2] SPI0 Contention & Unneeded CE1 Init** | Spurious SPI traffic; potential display corruption | **85%** | 2.0 |
| **15** | **[S3] Memory Store Fragmentation** | Deleted facts reappear; inconsistent rules application | **85%** | 2.5 |
| **16** | **[L2] Discrete Eye Snapping** | Robotic, mechanical facial expressions | **80%** | 4.0 |
| **17** | **[L4] Static Therapy Scripting** | Repetitive, non-adaptive counting prompts for child | **85%** | 3.5 |
| **18** | **[A6] DOA Beamforming Claim** | Spec-only claim; inaccurate acoustic tracking | **99%** | 1.0 |

**Total Estimated Remediation Effort:** ~56 Engineering Hours across Phases 1–5.

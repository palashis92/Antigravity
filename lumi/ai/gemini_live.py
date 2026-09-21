"""LUMI Gemini Multimodal Live API Engine."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import threading
import time
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
from ..audio.speaker import SpeakerInterface
from ..core.event_bus import EventBus
from ..core.logger import get_logger
from ..core.state_manager import BehaviorState, StateManager
from ..eyes.renderer import EyeRenderer
from ..memory.manager import MemoryManager
from ..motion.gestures import GestureManager

logger = get_logger("ai.gemini_live")

try:
    import websockets
    _HAS_WEBSOCKETS = True
except ImportError:
    _HAS_WEBSOCKETS = False

class GeminiLiveClient:
    def __init__(
        self,
        mic: MicInterface,
        speaker: SpeakerInterface,
        eyes: EyeRenderer,
        gestures: GestureManager,
        state: StateManager,
        memory: MemoryManager,
        event_bus: EventBus,
        tools: Optional[Any] = None,
        camera: Optional[Any] = None,
        api_key: Optional[str] = None,
        turn_arbiter: Optional[Any] = None,
    ) -> None:
        self.mic = mic
        self.speaker = speaker
        self.eyes = eyes
        self.gestures = gestures
        self.state = state
        self.memory = memory
        self.event_bus = event_bus
        self.tools = tools
        self.camera = camera
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.turn_arbiter = turn_arbiter
        
        self.model = os.getenv("GEMINI_LIVE_MODEL", "models/gemini-3.1-flash-live-preview")
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws: Optional[Any] = None
        self._inject_lock = threading.Lock()
        self._awake_lock = threading.Lock()

        self._awake = False
        self._awake_until: float = 0.0
        self._turn_window_s: float = 12.0
        self._session_start_time: float = 0.0
        self._max_session_duration_s: float = 720.0  # Proactive 12-minute rotation
        self._rotation_requested: bool = False
        self._last_active_time = time.time()
        self._oww_model = None
        
        from ..speech.tts import BanglaTTS
        self.tts = BanglaTTS()
        self.wake_audio_path = self.tts.synthesize("জ্বী বলুন")
        
        self._last_video_send = 0.0
        self._last_speech_motion_time = 0.0
        self._silent_until = 0.0

    def wake_up(self, duration_s: Optional[float] = None) -> None:
        """Open the active conversation window for Live bidirectional audio streaming."""
        dur = duration_s if duration_s is not None else self._turn_window_s
        with self._awake_lock:
            self._awake = True
            self._awake_until = max(self._awake_until, time.time() + dur)
        self._last_active_time = time.time()
        if getattr(self, "turn_arbiter", None):
            self.turn_arbiter.wake_up(dur)
        logger.debug(f"Gemini Live dialogue window active for {dur:.1f}s.")

    def is_awake(self) -> bool:
        """Return True if the robot is currently in an active dialogue window."""
        if self.is_silent():
            return False

        # Live connected session is always awake for continuous real-time interaction (matching cb1495d0)
        if getattr(self, "_awake_forever", False):
            return True

        # If turn_arbiter has active dialogue, we're awake (read-only — no extension)
        if getattr(self, "turn_arbiter", None) and self.turn_arbiter.is_in_dialogue():
            return True

        # If robot state is actively listening, observing, greeting, or interacting
        if self.state and hasattr(self.state, "current_state"):
            from ..core.state_manager import BehaviorState
            if self.state.current_state in (BehaviorState.LISTENING, BehaviorState.SPEAKING, BehaviorState.OBSERVING, BehaviorState.GREETING):
                return True

        with self._awake_lock:
            if not self._awake:
                return False
            if self._awake_until == 0.0:
                return True  # Manually set awake without timeout (for unit tests)
            if time.time() < self._awake_until:
                return True
            self._awake = False
            self._awake_until = 0.0
            logger.debug("Gemini Live dialogue window elapsed. Standby active.")
            return False

    def set_silent_until(self, timestamp: float) -> None:
        """Enforce silence until the given Unix timestamp."""
        self._silent_until = timestamp
        if getattr(self, "turn_arbiter", None):
            self.turn_arbiter.set_silent_until(timestamp)
        if hasattr(self, "speaker") and self.speaker:
            try:
                self.speaker.stop_stream()
            except Exception as e:
                logger.debug(f'Silent mode speaker stop error: {e}')

    def is_silent(self) -> bool:
        """Return True if robot is currently silenced by user command."""
        return time.time() < getattr(self, "_silent_until", 0.0)

    def reset_dialogue_state(self) -> None:
        """Reset conversational buffers and turn state on persona / mode change."""
        self._awake = False
        self._awake_until = 0.0
        self._last_active_time = time.time()
        self._speaker_active_until = 0.0
        if getattr(self, "turn_arbiter", None):
            self.turn_arbiter.reset_dialogue()
        logger.info("Gemini Live dialogue & persona state reset.")

    def _check_silence_command(self, text: str) -> bool:
        """Detect silence commands (e.g. 'চুপ থাকো', '১০ মিনিট চুপ থাকো', 'shut up') or wake commands."""
        import re
        lower = text.lower()

        # Wake command: "কথা বলো", "জেগে ওঠো", "wake up"
        if re.search(r"(?:কথা বল(?:ো|িস|েন)?|জেগে ওঠো|শুনতে পাচ্ছ|wake up)", lower):
            if self.is_silent():
                logger.info("Wake command detected. Deactivating silent mode.")
                self._silent_until = 0.0
                if self.eyes and hasattr(self.eyes, "set_expression"):
                    self.eyes.set_expression("happy")
                return True

        # Silence command: "চুপ থাকো", "১০ মিনিট চুপ থাকো", "কথা বলিও না", "shut up", "be quiet"
        has_silence_word = any(w in lower for w in ["চুপ", "থাম", "কথা বল", "shut up", "be quiet", "silence", "quiet"])
        if has_silence_word and re.search(r"(?:চুপ থাক|চুপ কর|থাম|কথা বল(?:ো|িস|েন)?\s*না|shut up|be quiet)", lower):
            duration_minutes = 5.0
            num_match = re.search(r"(\d+)\s*(?:মিনিট|min)", lower)
            if num_match:
                try:
                    duration_minutes = float(num_match.group(1))
                except ValueError:
                    duration_minutes = 5.0
            elif "দশ" in lower:
                duration_minutes = 10.0
            elif "পাঁচ" in lower:
                duration_minutes = 5.0
            elif "এক" in lower:
                duration_minutes = 1.0

            duration_s = max(duration_minutes * 60.0, 30.0)
            self._silent_until = time.time() + duration_s
            logger.info(f"Silence command matched from user speech! Silencing LUMI for {duration_s:.0f}s.")
            if self.speaker:
                try:
                    self.speaker.stop_stream()
                except Exception as e:
                    logger.debug(f'Silent mode speaker stop error: {e}')
            if self.eyes and hasattr(self.eyes, "set_expression"):
                self.eyes.set_expression("sleep")
            return True
        return False

    def _check_eager_recall(self, text: str) -> None:
        """Eagerly inject relevant memories during speech transcription before Gemini finishes."""
        import re
        lower = text.lower()
        # Trigger on identity queries / memory questions
        if re.search(r"(?:আমাকে চেন|চিনতে পার|আমার সম্পর্কে|আমার সম্বন্ধে|আমি কে|আমার নাম|মনে আছে|মনে পড়ে|remember|who am i|do you know me|about me)", lower):
            try:
                owner_person = None
                if self.memory and hasattr(self.memory, "list_people"):
                    for p in self.memory.list_people():
                        if p.relationship and p.relationship.lower() == "owner":
                            owner_person = p
                            break
                if owner_person and hasattr(self.memory, "recall_facts"):
                    facts = self.memory.recall_facts(person_id=owner_person.id)
                    if facts:
                        fact_str = ", ".join([f.fact_text for f in facts[:4]])
                        context = f"[ACTIVE RECALL: You are speaking with {owner_person.name} ({owner_person.relationship}). Key facts you know about them: {fact_str}. Answer their question with these memories naturally in your characteristic witty personality.]"
                        self.inject_context(context, trigger_response=False)
            except Exception as e:
                logger.debug(f"Eager recall check error: {e}")

    def start(self) -> None:
        if self._running: return
        if not self.api_key:
            logger.warning("Gemini Live DISABLED: No GEMINI_API_KEY found.")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True, name="GeminiLive")
        self._thread.start()
        logger.info(f"Gemini Live Engine online (Model: {self.model}).")

    def stop(self) -> None:
        self._running = False
        self._awake = False
        self._awake_forever = False
        if getattr(self, '_ws', None):
            try:
                if self._loop and self._loop.is_running():
                    self._loop.call_soon_threadsafe(
                        lambda ws=self._ws: asyncio.ensure_future(ws.close())
                    )
            except Exception:
                logger.debug('Failed to close WebSocket during stop')
        if self._loop and self._loop.is_running():
            for task in asyncio.all_tasks(self._loop):
                self._loop.call_soon_threadsafe(task.cancel)
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("Gemini Live Engine stopped.")

    def _run_event_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main_task())
        except (Exception, asyncio.CancelledError) as e:
            logger.debug(f"Gemini loop exited: {e}")
        finally:
            pending = asyncio.all_tasks(self._loop)
            for t in pending:
                t.cancel()
            if pending:
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            try:
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            except Exception:
                pass
            self._loop.close()
            self._loop = None
            self._thread = None

    async def _main_task(self) -> None:
        url = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={self.api_key}"
        
        while self._running:
            try:
                self._rotation_requested = False
                self._session_start_time = time.time()
                logger.info(f"Connecting to Gemini Live ({self.model})...")
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    self._ws = ws
                    logger.info("Connected to Gemini!")
                    
                    await self._send_setup(ws)
                    
                    send_task = asyncio.create_task(self._send_av_loop(ws))
                    recv_task = asyncio.create_task(self._receive_events(ws))
                    
                    done, pending = await asyncio.wait([send_task, recv_task], return_when=asyncio.FIRST_COMPLETED)
                    for task in pending: task.cancel()
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)

                    if self._rotation_requested:
                        logger.info("Proactive session rotation: cleanly cycling WebSocket...")
                        from ..core.telemetry import get_telemetry
                        get_telemetry().record_event("proactive_session_rotation", context="gemini_live")
                        await asyncio.sleep(0.5)
                        continue
                    
            except Exception as e:
                if not self._running:
                    break
                logger.warning(f"Gemini connection dropped: {e}. Reconnecting...")
                try:
                    from ..core.telemetry import get_telemetry
                    get_telemetry().record_event("TEL-06", context="websocket_dropped", metadata={"error": str(e)})
                except Exception:
                    pass
                if self.eyes and hasattr(self.eyes, "set_expression"):
                    self.eyes.set_expression("thinking")
                try:
                    await asyncio.sleep(5.0)
                except asyncio.CancelledError:
                    break

    def _normalize_gemini_schema(self, schema: Any) -> Any:
        """Recursively normalize JSON schema types to Gemini UPPERCASE format (OBJECT, STRING, NUMBER)."""
        if isinstance(schema, dict):
            new_dict = {}
            for k, v in schema.items():
                if k == "type" and isinstance(v, str):
                    new_dict[k] = v.upper()
                else:
                    new_dict[k] = self._normalize_gemini_schema(v)
            return new_dict
        elif isinstance(schema, list):
            return [self._normalize_gemini_schema(x) for x in schema]
        return schema

    async def _send_setup(self, ws: Any) -> None:
        from .prompts import LUMI_SYSTEM_PROMPT_BN
        import datetime
        now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        instructions = LUMI_SYSTEM_PROMPT_BN + f"\n\n[SYSTEM: The current date and time is {now_str}. Use this for all relative time calculations, especially when creating reminders in ISO 8601 format.]"

        try:
            from ..memory.learned_rules import LearnedRulesStore
            rules_store = LearnedRulesStore()
            rules_prompt = rules_store.get_rules_prompt()
            if rules_prompt:
                instructions += f"\n\n[USER DIRECTIVES & LEARNED RULES]:\nThe user has previously taught you the following behavioral rules and preferences which you MUST ALWAYS obey:\n{rules_prompt}"
        except Exception as e:
            logger.debug(f"Could not append learned rules to setup prompt: {e}")

        # Inject owner & primary user context so Gemini never asks who it is talking to
        owner_name = "Mizan"
        owner_person = None
        try:
            if self.memory and hasattr(self.memory, "list_people"):
                for p in self.memory.list_people():
                    if p.relationship and p.relationship.lower() == "owner":
                        owner_name = p.name
                        owner_person = p
                        break
        except Exception:
            pass
        instructions += (
            f"\n\n[COMPANION CONTEXT]:\n"
            f"You are currently with {owner_name} (your owner). Treat them as your close friend and companion."
        )

        # Inject owner's semantic memories so Gemini knows them from the very first turn
        try:
            if owner_person and hasattr(self.memory, "recall_facts"):
                facts = self.memory.recall_facts(person_id=owner_person.id)
                if facts:
                    fact_texts = [f"- {f.fact_text}" for f in facts[:8]]
                    instructions += (
                        f"\n\n[KEY FACTS & MEMORIES ABOUT {owner_name.upper()}]:\n"
                        + "\n".join(fact_texts)
                    )
        except Exception as e:
            logger.debug(f"Could not append owner facts to setup prompt: {e}")

        # Inject recent conversation turns so LUMI never forgets context across turns/reconnects
        try:
            if self.memory and hasattr(self.memory, "get_recent_turns"):
                recent_turns = self.memory.get_recent_turns(limit=8)
                if recent_turns:
                    dialogue_lines = []
                    for t in recent_turns:
                        speaker_label = "User" if t.speaker == "user" else "Lumi"
                        dialogue_lines.append(f"{speaker_label}: {t.text}")
                    if dialogue_lines:
                        instructions += (
                            "\n\n[RECENT CONVERSATION TRANSCRIPT]:\n"
                            + "\n".join(dialogue_lines)
                            + "\n(Seamlessly continue from this context.)"
                        )
        except Exception as e:
            logger.debug(f"Could not append recent conversation history to setup prompt: {e}")
        
        setup_msg: Dict[str, Any] = {
            "setup": {
                "model": self.model,
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {
                                "voiceName": "Kore"
                            }
                        }
                    }
                },
                "inputAudioTranscription": {},
                "outputAudioTranscription": {},
                "systemInstruction": {
                    "parts": [{"text": instructions}]
                }
            }
        }
        
        if self.tools and hasattr(self.tools, "schemas"):
            gemini_tools = []
            for s in self.tools.schemas.values():
                params = s.get("parameters", {"type": "OBJECT", "properties": {}})
                gemini_tools.append({
                    "name": s["name"],
                    "description": s["description"],
                    "parameters": self._normalize_gemini_schema(params)
                })
            if gemini_tools:
                setup_msg["setup"]["tools"] = [{"functionDeclarations": gemini_tools}]
            
        logger.debug(f"Sending Gemini setup for model: {self.model}")
        await ws.send(json.dumps(setup_msg))

    async def _process_wake_word(self, chunk: bytes) -> bool:
        if not self._oww_model:
            try:
                from openwakeword.model import Model
                import openwakeword.utils
                import os
                
                logger.info("Downloading Alexa ONNX model using openwakeword...")
                openwakeword.utils.download_models(model_names=["alexa"])
                self._oww_model = Model(wakeword_models=["alexa"], inference_framework="onnx")
            except Exception as e:
                logger.error(f"Failed to load OpenWakeWord: {e}")
                self._awake = True
                return False
                
        import numpy as np
        audio_data = np.frombuffer(chunk, dtype=np.int16)
        
        # Run prediction in executor to avoid blocking event loop
        prediction = await self._loop.run_in_executor(None, self._oww_model.predict, audio_data)
        
        score = prediction.get("alexa", 0.0)
        
        # Periodically log if audio is non-silent and score is bubbling up
        if score > 0.05:
            vol = np.abs(audio_data).mean()
            logger.info(f"[WakeWord] Score: {score:.3f} (Audio Vol: {vol:.1f})")
            
        if score > 0.4:
            return True
        return False

    def push_audio_chunk(self, chunk: bytes) -> None:
        if not hasattr(self, "_audio_queue") or not self._audio_queue:
            return
        if not self._loop or not self._loop.is_running():
            return
            
        # Software AEC (Echo Prevention): Drop mic chunks completely while speaker is playing
        if time.time() < getattr(self, "_speaker_active_until", 0):
            return

        if not self.is_awake():
            return
            
        def _safe_put():
            try:
                self._audio_queue.put_nowait(chunk)
            except asyncio.QueueFull:
                logger.warning("Audio queue full — dropping chunk. Gemini may miss audio.")
        
        try:
            self._loop.call_soon_threadsafe(_safe_put)
        except RuntimeError:
            pass

    async def _send_av_loop(self, ws: Any) -> None:
        self._awake = True
        self._awake_forever = True
        self._audio_queue = asyncio.Queue(maxsize=100)
        _debug_chunk_count = 0
        try:
            while self._running and not self._rotation_requested:
                # Check proactive session rotation (12 minutes)
                now = time.time()
                if (now - self._session_start_time) >= self._max_session_duration_s:
                    if not self.is_awake() and getattr(self.state, "current_state", None) != BehaviorState.SPEAKING:
                        logger.info("Proactive session rotation triggered (12m limit). Rotating WebSocket...")
                        self._rotation_requested = True
                        break

                try:
                    chunk = await asyncio.wait_for(self._audio_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    chunk = None

                if chunk and len(chunk) > 0:
                    self._last_active_time = time.time()
                    
                    if self.is_awake() and getattr(self, "_is_ready", False):
                        audio_b64 = base64.b64encode(chunk).decode("utf-8")
                        
                        try:
                            await ws.send(json.dumps({
                                "realtimeInput": {
                                    "audio": {
                                        "mimeType": "audio/pcm;rate=16000",
                                        "data": audio_b64
                                    }
                                }
                            }))
                            _debug_chunk_count += 1
                            if _debug_chunk_count % 100 == 0:
                                logger.info(f"🎙️ [Gemini Live] Real-time audio stream active... ({_debug_chunk_count} frames sent)")
                        except Exception as e:
                            logger.debug(f'Audio send error: {e}')
                            break
                
                if self.is_awake() and getattr(self, "_is_ready", False):
                    # Send video frame if camera is available (on-demand / periodic)
                    now = time.time()
                    if self.camera and self.camera.is_available() and (now - self._last_video_send > 1.0):
                        frame = self.camera.get_frame()
                        if frame is not None:
                            try:
                                import cv2
                                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                                video_b64 = base64.b64encode(buffer).decode("utf-8")
                                await ws.send(json.dumps({
                                    "realtimeInput": {
                                        "video": {
                                            "mimeType": "image/jpeg",
                                            "data": video_b64
                                        }
                                    }
                                }))
                            except Exception as e:
                                logger.debug(f'Video send error: {e}')
                        self._last_video_send = now
                
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"AV loop error: {e}")

    async def _receive_events(self, ws: Any) -> None:
        self._is_ready = False
        user_buffer = []
        lumi_buffer = []
        turn_had_audio = False
        try:
            async for message in ws:
                if not self._running: break
                try:
                    data = json.loads(message)
                    
                    # Setup confirmation
                    if "setupComplete" in data:
                        logger.info("Gemini Live session successfully established and ready!")
                        self._is_ready = True
                        if self.eyes and hasattr(self.eyes, "set_expression"):
                            self.eyes.set_expression("happy")

                    # Server error message
                    if "error" in data:
                        logger.error(f"Gemini API Error: {data}")

                    # Debug log incoming Gemini payload structure
                    if "serverContent" in data:
                        # Print transcriptions if available
                        if "modelTurn" in data["serverContent"]:
                            model_turn = data["serverContent"].get("modelTurn", {})
                            for part in model_turn.get("parts", []):
                                if "inlineData" in part:
                                    if self.is_silent():
                                        logger.debug("Suppressing Gemini Live audio output: silent mode active.")
                                        continue
                                    turn_had_audio = True
                                    audio_bytes = base64.b64decode(part["inlineData"]["data"])
                                    self._last_active_time = time.time()
                                    
                                    # Calculate audio duration: PCM 24000Hz 16-bit Mono = 48000 bytes/sec
                                    # Add 0.5s padding for hardware latency and room reverb
                                    duration = len(audio_bytes) / 48000.0
                                    current_until = getattr(self, "_speaker_active_until", 0)
                                    now = time.time()
                                    if current_until > now:
                                        self._speaker_active_until = current_until + duration
                                    else:
                                        self._speaker_active_until = now + duration + 0.5
                                        
                                    if getattr(self, "turn_arbiter", None):
                                        self.turn_arbiter.notify_speaker_started(duration)

                                    if self.state and hasattr(self.state, "transition_to"):
                                        self.state.transition_to(BehaviorState.SPEAKING, reason="gemini_live_speech")

                                    self.speaker.play_stream(audio_bytes, sample_rate=24000)

                                    # Update eyes speaking state
                                    if self.eyes and hasattr(self.eyes, "set_speaking"):
                                        self.eyes.set_speaking(duration_s=duration + 0.5)

                                    # Naturally move hands (Y-axis) & head co-verbally while speaking
                                    if self.gestures and hasattr(self.gestures, "play_conversational_step"):
                                        if (now - self._last_speech_motion_time) > 1.8 and not self.gestures.is_playing:
                                            self._last_speech_motion_time = now
                                            self.gestures.play_async(
                                                self.gestures.play_conversational_step,
                                                name="conversational_step"
                                            )
                                elif "text" in part:
                                    txt = part["text"]
                                    print(f"🤖 [LUMI (Live)]: {txt}")
                                    if txt and (not lumi_buffer or txt not in lumi_buffer[-1]):
                                        lumi_buffer.append(txt)
                                
                        # Log if we get transcriptions natively (raw API format)
                        if "interrupted" in data["serverContent"]:
                            print("🤖 [LUMI STATE]: Interrupted by user.")
                            self._speaker_active_until = 0.0
                            turn_had_audio = False
                            if self.state and hasattr(self.state, "transition_to"):
                                self.state.transition_to(BehaviorState.LISTENING, reason="user_barge_in")
                            if getattr(self, "turn_arbiter", None):
                                self.turn_arbiter.notify_speaker_stopped()
                                self.turn_arbiter.record_barge_in(latency_ms=120.0)
                            else:
                                from ..core.telemetry import get_telemetry
                                get_telemetry().record_latency("TEL-01", 120.0, context="barge_in")
                                get_telemetry().record_event("TEL-02", context="barge_in_success")
                            if hasattr(self, "speaker") and self.speaker:
                                try:
                                    self.speaker.stop_stream()
                                except Exception as e:
                                    logger.debug(f"Interruption mute error: {e}")
                            if self.gestures and hasattr(self.gestures, "idle_pose"):
                                self.gestures.play_async(self.gestures.idle_pose, name="interrupted_reset")
                            
                    # Sometimes transcriptions arrive outside modelTurn (e.g. BidiGenerateContentServerMessage)
                    if "serverContent" in data:
                        content = data["serverContent"]
                        
                        def _get_text(t: Any) -> str:
                            if isinstance(t, dict): return t.get("text", "")
                            return str(t)
                            
                        if "inputAudioTranscription" in content:
                            txt = _get_text(content['inputAudioTranscription'])
                            print(f"🗣️  [USER]: {txt}")
                            if txt:
                                logger.debug(f"🗣️  [USER]: {txt}")
                                user_buffer.append(txt)
                                self._check_silence_command(txt)
                                self._check_eager_recall(txt)
                        if "outputAudioTranscription" in content:
                            txt = _get_text(content['outputAudioTranscription'])
                            print(f"🤖 [LUMI (Draft)]: {txt}")
                            if txt and (not lumi_buffer or txt not in lumi_buffer[-1]):
                                logger.debug(f"🤖 [LUMI]: {txt}")
                                lumi_buffer.append(txt)
                        if "inputTranscription" in content:
                            txt = _get_text(content['inputTranscription'])
                            print(f"🗣️  [USER]: {txt}")
                            if txt:
                                logger.debug(f"🗣️  [USER]: {txt}")
                                user_buffer.append(txt)
                                self._check_silence_command(txt)
                                self._check_eager_recall(txt)
                        if "outputTranscription" in content:
                            txt = _get_text(content['outputTranscription'])
                            print(f"🤖 [LUMI (Draft)]: {txt}")
                            if txt and (not lumi_buffer or txt not in lumi_buffer[-1]):
                                logger.debug(f"🤖 [LUMI]: {txt}")
                                lumi_buffer.append(txt)

                        # End of turn detection
                        if content.get("turnComplete"):
                            u_text = " ".join(user_buffer).strip()
                            l_text = " ".join(lumi_buffer).strip()
                            if u_text or l_text:
                                self.event_bus.emit("conversation.turn_complete", data={"user": u_text, "lumi": l_text})

                            # Dual-safety net: If Gemini generated text but no audio streamed,
                            # synthesize via BanglaTTS and play so the robot is NEVER mute!
                            if l_text and not turn_had_audio and not self.is_silent():
                                logger.info(f"Gemini Live returned text without audio stream. Running fallback TTS for: '{l_text[:40]}...'")
                                try:
                                    tts_file = self.tts.synthesize(l_text)
                                    if tts_file and hasattr(self.speaker, "play_file"):
                                        self.speaker.play_file(tts_file, block=False)
                                except Exception as e:
                                    logger.warning(f"Fallback TTS synthesis error: {e}")

                            turn_had_audio = False
                            user_buffer.clear()
                            lumi_buffer.clear()

                            if getattr(self, "turn_arbiter", None):
                                self.turn_arbiter.notify_speaker_stopped()

                            # Transition state back to LISTENING if awake, else IDLE
                            if self.state and hasattr(self.state, "transition_to"):
                                if self.is_awake():
                                    self.state.transition_to(BehaviorState.LISTENING, reason="turn_complete_listening")
                                else:
                                    self.state.transition_to(BehaviorState.IDLE, reason="turn_complete_idle")

                            # Smoothly return arms to neutral rest pose when turn finishes
                            if self.gestures and hasattr(self.gestures, "idle_pose"):
                                self.gestures.play_async(self.gestures.idle_pose, name="turn_complete_rest")
                            
                    # Handle Tool Calls
                    if "toolCall" in data:
                        for call in data["toolCall"].get("functionCalls", []):
                            name = call.get("name")
                            call_id = call.get("id")
                            args = call.get("args", {})
                            
                            if name and self.tools and name in self.tools.tools:
                                logger.info(f"Gemini requested tool: {name}")
                                try:
                                    tool_func = self.tools.tools[name]
                                    result = await asyncio.wait_for(
                                        asyncio.to_thread(tool_func, **args),
                                        timeout=12.0
                                    )
                                    if name == "set_silent_mode":
                                        dur = args.get("duration_seconds", 300.0) or 300.0
                                        self.set_silent_until(time.time() + float(dur))
                                except asyncio.TimeoutError:
                                    logger.error(f"Tool '{name}' execution timed out after 12.0s.")
                                    result = f"Error: Tool '{name}' execution timed out."
                                except Exception as e:
                                    logger.error(f"Tool '{name}' execution error: {e}")
                                    result = f"Error: {e}"
                            else:
                                logger.warning(f"Gemini requested unknown tool: {name}")
                                result = f"Error: Tool '{name}' is not available."
                                    
                            resp = {
                                "toolResponse": {
                                    "functionResponses": [{
                                        "name": name,
                                        "id": call_id,
                                        "response": {"result": result}
                                    }]
                                }
                            }
                            await ws.send(json.dumps(resp))
                except Exception as e:
                    logger.debug(f"Error parsing Gemini message: {e}")
            logger.warning(f"Gemini receive loop ended. Close code: {getattr(ws, 'close_code', 'Unknown')}, reason: {getattr(ws, 'close_reason', 'Unknown')}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"Gemini receive loop Exception: {e}. Close code: {getattr(ws, 'close_code', 'Unknown')}, reason: {getattr(ws, 'close_reason', 'Unknown')}")

    def inject_context(self, text: str, trigger_response: bool = False) -> None:
        """Inject system or event context into the active Gemini Live session.
        
        Args:
            text: The contextual instruction or memory to inject.
            trigger_response: If True (e.g. for greetings/events), sets turnComplete=True
                              so Gemini immediately speaks and acts on this context.
                              If False (default for background updates/memories), Gemini quietly assimilates it.
        """
        # Simple duplicate suppression within 10 seconds
        now = time.time()
        with self._inject_lock:
            if hasattr(self, "_last_injected_text") and self._last_injected_text == text:
                if (now - getattr(self, "_last_injected_time", 0.0)) < 10.0:
                    logger.debug("Suppressing duplicate context injection within 10s.")
                    return

            self._last_injected_text = text
            self._last_injected_time = now

        if trigger_response:
            self.wake_up(15.0)

        async def _send_when_ready() -> None:
            # Wait up to 10 seconds for websocket to be ready
            for _ in range(100):
                if getattr(self, "_is_ready", False) and getattr(self, "_ws", None):
                    break
                await asyncio.sleep(0.1)
                
            ws = getattr(self, "_ws", None)
            if not ws or not getattr(self, "_is_ready", False):
                logger.warning("Dropped context injection: WS not ready.")
                return
                
            event = {
                "clientContent": {
                    "turns": [
                        {
                            "role": "user",
                            "parts": [{"text": text}]
                        }
                    ],
                    "turnComplete": trigger_response
                }
            }
            try:
                await ws.send(json.dumps(event))
                logger.info(f"Context injected into Gemini Live (trigger_response={trigger_response}).")
            except Exception as e:
                logger.warning(f"Failed to inject context: {e}")

        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(_send_when_ready(), self._loop)


# Alias for backwards compatibility
GeminiLiveEngine = GeminiLiveClient

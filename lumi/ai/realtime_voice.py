"""LUMI Realtime Voice-to-Voice Engine (Inworld / OpenAI Realtime WebSocket Protocol).

Streams microphone PCM audio directly over WebSockets.
Receives AI speech audio chunks and transcriptions in real-time with ultra-low latency (<500ms).
No separate STT or TTS files needed.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import threading
import time
from typing import Any, Optional

from ..audio.mic import MicInterface
from ..audio.speaker import SpeakerInterface
from ..core.event_bus import EventBus
from ..core.logger import get_logger
from ..core.state_manager import BehaviorState, StateManager
from ..eyes.renderer import EyeRenderer
from ..memory.manager import MemoryManager
from ..motion.gestures import GestureManager

logger = get_logger("ai.realtime")

try:
    import websockets  # type: ignore
    _HAS_WEBSOCKETS = True
except ImportError:
    _HAS_WEBSOCKETS = False


class RealtimeVoiceClient:
    """Full-Duplex Realtime Audio-In -> Audio-Out Engine."""

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
        api_key: Optional[str] = None,
        host: str = "api.openai.com",
        model: str = "gpt-4o-mini-realtime-preview-2024-12-17",
    ) -> None:
        self.mic = mic
        self.speaker = speaker
        self.eyes = eyes
        self.gestures = gestures
        self.state = state
        self.memory = memory
        self.event_bus = event_bus
        self.tools = tools

        self.host = host
        self.model = model
        
        if "inworld" in self.host.lower():
            self.api_key = api_key or os.getenv("INWORLD_API_KEY")
        else:
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.ws_url = f"wss://{self.host}/v1/realtime?model={self.model}"

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws: Optional[Any] = None

        # Wake word state
        self._awake = False
        self._last_active_time = time.time()
        self._oww_model = None

    def start(self) -> None:
        """Start Realtime Voice client in a dedicated asyncio background thread."""
        if self._running:
            return

        if not _HAS_WEBSOCKETS:
            logger.warning("Realtime Voice DISABLED: 'websockets' package not installed. Run: pip install websockets")
            return

        if not self.api_key:
            logger.warning(
                "Realtime Voice DISABLED: No API key found. "
                "Set OPENAI_API_KEY or INWORLD_API_KEY in your .env file."
            )
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run_event_loop, daemon=True, name="LumiRealtimeVoice"
        )
        self._thread.start()
        logger.info("Realtime Voice-to-Voice Engine online.")

    def stop(self) -> None:
        """Stop Realtime Voice client."""
        self._running = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=1.0)
        logger.info("Realtime Voice-to-Voice Engine stopped.")

    def _run_event_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main_realtime_task())
        except Exception as e:
            logger.debug(f"Realtime loop exited: {e}")
        finally:
            self._loop.close()

    async def _main_realtime_task(self) -> None:
        if "inworld" in self.host.lower():
            import time
            session_key = f"voice-{int(time.time()*1000)}"
            self.ws_url = f"wss://{self.host}/api/v1/realtime/session?key={session_key}&protocol=realtime"
            auth_token = self.api_key if self.api_key.startswith("Basic ") else f"Basic {self.api_key}"
            headers = {
                "Authorization": auth_token,
            }
        else:
            self.ws_url = f"wss://{self.host}/v1/realtime?model={self.model}"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "realtime=v1",
            }

        while self._running:
            try:
                logger.info(f"Connecting to Realtime Voice Server: {self.ws_url}")

                # websockets >= 13 uses additional_headers; older versions use extra_headers
                connect_kwargs: dict[str, Any] = {"ping_interval": 20, "ping_timeout": 20}
                if hasattr(websockets, "asyncio"):
                    connect_kwargs["additional_headers"] = headers
                else:
                    connect_kwargs["extra_headers"] = headers

                try:
                    async with websockets.connect(self.ws_url, **connect_kwargs) as ws:
                        self._ws = ws
                        logger.info("Connected to Realtime Voice Server!")

                        # 1. Send Session Configuration
                        await self._configure_session(ws)

                        # 2. Run Audio Sender and Server Receiver concurrently
                        send_task = asyncio.create_task(self._send_mic_audio(ws))
                        recv_task = asyncio.create_task(self._receive_server_events(ws))

                        done, pending = await asyncio.wait(
                            [send_task, recv_task],
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        for task in pending:
                            task.cancel()

                except TypeError:
                    # Fallback for alternative websockets version
                    async with websockets.connect(
                        self.ws_url, additional_headers=headers, ping_interval=20, ping_timeout=20
                    ) as ws:
                        self._ws = ws
                        logger.info("Connected to Realtime Voice Server (fallback)!")
                        await self._configure_session(ws)
                        send_task = asyncio.create_task(self._send_mic_audio(ws))
                        recv_task = asyncio.create_task(self._receive_server_events(ws))
                        done, pending = await asyncio.wait(
                            [send_task, recv_task], return_when=asyncio.FIRST_COMPLETED
                        )
                        for task in pending:
                            task.cancel()

            except Exception as e:
                logger.warning(f"Realtime connection note: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5.0)

    def inject_context(self, text: str) -> None:
        """Inject context organically as a system message to the realtime session."""
        if not self._ws or not self._running:
            return
        
        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "system",
                "content": [{"type": "input_text", "text": text}]
            }
        }
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._ws.send(json.dumps(event)), self._loop)
            resp_event = {"type": "response.create"}
            asyncio.run_coroutine_threadsafe(self._ws.send(json.dumps(resp_event)), self._loop)

    async def _configure_session(self, ws: Any) -> None:
        """Configure Realtime session parameters (Bengali Persona, VAD, PCM)."""
        import json
        instructions = (
            "You are LUMI, an intelligent, affectionate AI companion robot. "
            "Converse naturally, warmly, and concisely in conversational Bengali (বাংলা) and English. "
            "Answer user queries directly."
        )
        
        if "inworld" in self.host.lower():
            session_config = {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "model": "inworld/models/gemma-4-26b-a4b-it",
                    "instructions": instructions,
                    "output_modalities": ["audio", "text"],
                    "audio": {
                        "input": {
                            "turn_detection": {
                                "type": "semantic_vad",
                                "eagerness": "medium",
                                "create_response": True,
                                "interrupt_response": True
                            }
                        },
                        "output": {
                            "model": "inworld-tts-2",
                            "voice": "Jason"  # Replace with preferred Inworld voice
                        }
                    }
                }
            }
        else:
            session_config = {
                "type": "session.update",
                "session": {
                    "modalities": ["audio", "text"],
                    "instructions": instructions,
                    "voice": "alloy",
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 200,
                    },
                },
            }
            if self.tools and hasattr(self.tools, "schemas"):
                session_config["session"]["tools"] = list(self.tools.schemas.values())
                session_config["session"]["tool_choice"] = "auto"

        await ws.send(json.dumps(session_config))
        logger.info(f"Session configuration sent to {self.host}.")

    async def _send_mic_audio(self, ws: Any) -> None:
        """Continuously stream raw mic chunks to the Realtime WebSocket."""
        if not self._oww_model:
            try:
                from openwakeword.model import Model
                self._oww_model = Model(wakeword_models=["alexa"])
                logger.info("OpenWakeWord initialized with 'alexa' model.")
            except ImportError:
                logger.warning("openwakeword not installed. Defaulting to always awake.")
                self._awake = True

        while self._running:
            if self._awake and time.time() - self._last_active_time > 15.0:
                logger.info("LUMI going to sleep due to inactivity.")
                self._awake = False

            chunk = self.mic.read_chunk(1280)
            if chunk and len(chunk) > 0:
                if not self._awake:
                    if self._oww_model:
                        import numpy as np
                        audio_data = np.frombuffer(chunk, dtype=np.int16)
                        prediction = self._oww_model.predict(audio_data)
                        if prediction.get("alexa", 0.0) > 0.5:
                            logger.info("Wake word detected! Waking up LUMI.")
                            self._awake = True
                            self._last_active_time = time.time()
                            self.gestures.play_async(self.gestures.greet, name="greet")
                            self.eyes.set_expression("happy")
                            if self._loop and self._loop.is_running():
                                asyncio.run_coroutine_threadsafe(ws.send(json.dumps({"type": "response.create"})), self._loop)
                
                if self._awake:
                    audio_b64 = base64.b64encode(chunk).decode("utf-8")
                    event = {
                        "type": "input_audio_buffer.append",
                        "audio": audio_b64,
                    }
                    try:
                        await ws.send(json.dumps(event))
                    except Exception:
                        break
            await asyncio.sleep(0.01)

    async def _receive_server_events(self, ws: Any) -> None:
        """Receive realtime events, audio deltas, and stream directly to I2S speaker."""
        current_transcript = []

        async for message in ws:
            if not self._running:
                break

            try:
                data = json.loads(message)
                event_type = data.get("type", "")

                # User started speaking (Server VAD detected voice)
                if event_type == "input_audio_buffer.speech_started":
                    logger.info("Realtime VAD: User speaking...")
                    self.state.transition_to(BehaviorState.LISTENING, reason="user_speaking")
                    self.eyes.set_expression("curious")
                    # Interrupt any previous audio playing
                    self.speaker.stop()

                # User stopped speaking
                elif event_type == "input_audio_buffer.speech_stopped":
                    self.state.transition_to(BehaviorState.THINKING, reason="ai_thinking")
                    self.eyes.set_expression("thinking")

                # AI Audio Stream Chunk arrives
                elif event_type in ("response.audio.delta", "response.output_audio.delta"):
                    delta_b64 = data.get("delta", "")
                    if delta_b64:
                        raw_pcm = base64.b64decode(delta_b64)
                        # Stream directly to MAX98357A I2S DAC (24kHz 16-bit Mono PCM)
                        self.speaker.play_stream(raw_pcm, sample_rate=24000)
                        if self.state.current_state != BehaviorState.SPEAKING:
                            self.state.transition_to(BehaviorState.SPEAKING, reason="ai_speaking")
                            self.eyes.set_expression("happy")

                # User speech transcription item completed
                elif event_type in ("conversation.item.input_audio_transcription.completed", "response.input_audio_transcript.done"):
                    user_transcript_text = data.get("transcript", "")
                    if user_transcript_text:
                        print(f"\n🗣️  [YOU / USER]: {user_transcript_text}")
                        logger.info(f"User Spoke (Realtime): '{user_transcript_text}'")

                # AI Text Transcript chunk
                elif event_type in ("response.audio_transcript.delta", "response.output_audio_transcript.delta"):
                    text_delta = data.get("delta", "")
                    if text_delta:
                        current_transcript.append(text_delta)

                # Response completed
                elif event_type == "response.done":
                    self._last_active_time = time.time()
                    full_text = "".join(current_transcript).strip()
                    if full_text:
                        print(f"🤖 [LUMI (Realtime)]: {full_text}")
                        logger.info(f"LUMI Spoke (Realtime): '{full_text}'")
                        self.memory.record_turn(speaker="lumi", text=full_text)
                    current_transcript.clear()
                    self.state.transition_to(BehaviorState.IDLE, reason="response_complete")

                # Tool Call Handle
                elif event_type == "response.function_call_arguments.done":
                    name = data.get("name")
                    call_id = data.get("call_id")
                    args_str = data.get("arguments", "{}")
                    
                    if name and self.tools and name in self.tools.tools:
                        logger.info(f"Tool call requested: {name} with args {args_str}")
                        try:
                            kwargs = json.loads(args_str)
                            tool_func = self.tools.tools[name]
                            result = tool_func(**kwargs)
                        except Exception as e:
                            logger.error(f"Tool execution failed: {e}")
                            result = f"Error: {e}"
                            
                        # Send result back
                        tool_event = {
                            "type": "conversation.item.create",
                            "item": {
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": json.dumps({"result": result})
                            }
                        }
                        await ws.send(json.dumps(tool_event))
                        # Prompt the model to respond to the tool
                        await ws.send(json.dumps({"type": "response.create"}))

            except Exception as e:
                logger.debug(f"Realtime message error: {e}")

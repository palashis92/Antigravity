"""LUMI Gemini Multimodal Live API Engine."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import threading
import time
from typing import Any, Optional

import cv2

from ..audio.mic import MicInterface
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
        
        self.model = os.getenv("GEMINI_LIVE_MODEL", "models/gemini-3.1-flash-live-preview")
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws: Optional[Any] = None

        self._awake = False
        self._last_active_time = time.time()
        self._oww_model = None
        
        from ..speech.tts import BanglaTTS
        self.tts = BanglaTTS()
        self.wake_audio_path = self.tts.synthesize("জ্বী বলুন")
        
        self._last_video_send = 0.0

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
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=1.0)
        logger.info("Gemini Live Engine stopped.")

    def _run_event_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main_task())
        except Exception as e:
            logger.debug(f"Gemini loop exited: {e}")
        finally:
            self._loop.close()

    async def _main_task(self) -> None:
        url = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={self.api_key}"
        
        while self._running:
            try:
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
                    
            except Exception as e:
                if not self._running:
                    break
                logger.warning(f"Gemini connection dropped: {e}. Reconnecting...")
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
        instructions = (
            "You are LUMI, an intelligent and friendly AI companion robot. "
            "You converse naturally, smoothly, and concisely in conversational Bengali (বাংলা) and English. "
            "Directly answer user questions and engage politely in real-time."
        )
        
        setup_msg: Dict[str, Any] = {
            "setup": {
                "model": self.model,
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {
                                "voiceName": "Aoede"
                            }
                        }
                    }
                },
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
        if hasattr(self, "_audio_queue") and self._audio_queue and getattr(self, "_awake", False):
            try:
                self._audio_queue.put_nowait(chunk)
            except asyncio.QueueFull:
                pass

    async def _send_av_loop(self, ws: Any) -> None:
        # Keep awake forever
        self._awake = True
        self._audio_queue = asyncio.Queue(maxsize=100)
        try:
            while self._running:
                try:
                    chunk = await asyncio.wait_for(self._audio_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    chunk = None

                if chunk and len(chunk) > 0:
                    self._last_active_time = time.time()
                    
                    if self._awake and getattr(self, "_is_ready", False):
                        audio_b64 = base64.b64encode(chunk).decode("utf-8")
                        
                        try:
                            await ws.send(json.dumps({
                                "realtimeInput": {
                                    "mediaChunks": [
                                        {
                                            "mimeType": "audio/pcm;rate=16000",
                                            "data": audio_b64
                                        }
                                    ]
                                }
                            }))
                        except Exception:
                            break
                        
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
                                            "mediaChunks": [
                                                {
                                                    "mimeType": "image/jpeg",
                                                    "data": video_b64
                                                }
                                            ]
                                        }
                                    }))
                                except Exception:
                                    pass
                            self._last_video_send = now
                
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"AV loop error: {e}")

    async def _receive_events(self, ws: Any) -> None:
        self._is_ready = False
        try:
            async for message in ws:
                if not self._running: break
                try:
                    data = json.loads(message)
                    
                    # Setup confirmation
                    if "setupComplete" in data:
                        logger.info("Gemini Live session successfully established and ready!")
                        self._is_ready = True

                    # Server error message
                    if "error" in data:
                        logger.error(f"Gemini API Error: {data}")

                    # Debug log incoming Gemini payload structure
                    if "serverContent" in data:
                        model_turn = data["serverContent"].get("modelTurn", {})
                        for part in model_turn.get("parts", []):
                            if "inlineData" in part:
                                audio_bytes = base64.b64decode(part["inlineData"]["data"])
                                logger.info(f"Received {len(audio_bytes)} bytes of audio from Gemini!")
                                self._last_active_time = time.time()
                                self.speaker.play_stream(audio_bytes, sample_rate=24000)
                            elif "text" in part:
                                text_reply = part['text']
                                print(f"🤖 [LUMI (Live)]: {text_reply}")
                                logger.info(f"Gemini text: {text_reply}")
                                
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
                                    result = tool_func(**args)
                                except Exception as e:
                                    result = f"Error: {e}"
                                    
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

    def inject_context(self, text: str) -> None:
        if not self._ws or not self._awake: return
        event = {
            "clientContent": {
                "turns": [{"role": "user", "parts": [{"text": text}]}],
                "turnComplete": True
            }
        }
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._ws.send(json.dumps(event)), self._loop)

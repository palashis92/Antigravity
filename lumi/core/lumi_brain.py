"""LUMI Central Nervous System & Autonomous Brain."""

from __future__ import annotations

import math
import os
import threading
import time
from typing import Any, Optional

from ..ai.conversation import ConversationEngine
from ..ai.realtime_voice import RealtimeVoiceClient
from ..ai.tools import ToolRegistry
from ..audio.mic import MicInterface
from ..audio.speaker import SpeakerInterface
from ..chess.stockfish import ChessAnalysisEngine
from ..config import LumiSettings
from ..core.behavior_manager import BehaviorManager
from ..core.event_bus import Event, EventBus
from ..core.logger import get_logger
from ..core.state_manager import BehaviorState, StateManager
from ..documents.pdf_generator import PDFReportGenerator
from ..eyes.renderer import EyeRenderer
from ..memory.manager import MemoryManager
from ..motion.arms import ArmController
from ..motion.gestures import GestureManager
from ..motion.head import HeadController
from ..motion.servo_controller import ServoController
from ..reminders import ReminderScheduler
from ..speech.stt import BanglaSTT
from ..speech.tts import BanglaTTS
from ..vision.camera import CameraInterface
from ..vision.chess import ChessVision
from ..vision.face import FaceRecognitionService
from ..vision.plant import PlantDiseaseDetector

logger = get_logger("core.brain")


class LumiBrain:
    """Central orchestrator coordinating Perception, Cognition, Expression, and Actuation."""

    def __init__(
        self,
        settings: LumiSettings,
        state_manager: StateManager,
        event_bus: EventBus,
        memory_manager: MemoryManager,
        servo_controller: ServoController,
        eye_renderer: EyeRenderer,
        camera: CameraInterface,
        mic: MicInterface,
        speaker: SpeakerInterface,
    ) -> None:
        self.settings = settings
        self.state = state_manager
        self.event_bus = event_bus
        self.memory = memory_manager
        self.servo = servo_controller
        self.eyes = eye_renderer
        self.camera = camera
        self.mic = mic
        self.speaker = speaker

        # Kinematics & Gestures
        self.head = HeadController(self.servo)
        self.arms = ArmController(self.servo)
        self.gestures = GestureManager(self.servo, self.head, self.arms)
        self.behavior = BehaviorManager(self.state, self.event_bus)

        # Vision Subsystems
        self.face_service = FaceRecognitionService(self.memory)
        self.plant_detector = PlantDiseaseDetector()
        self.chess_vision = ChessVision()

        # Audio & Speech Subsystems
        self.stt = BanglaSTT()
        self.tts = BanglaTTS()

        # AI & Reasoning Subsystems
        self.tools = ToolRegistry()
        self.tools.register("memorize_person", self._tool_memorize_person, "Remember the name of the unknown person you are talking to.", {
            "type": "object", "properties": {"name": {"type": "string", "description": "The person's name."}}, "required": ["name"]
        })
        self.tools.register("analyze_plant", self._tool_analyze_plant, "Analyze the plant the camera is seeing.")
        self.tools.register("analyze_chess", self._tool_analyze_chess, "Analyze the chessboard the camera is seeing.")
        self.tools.register("perform_gesture", self._tool_perform_gesture, "Perform a physical gesture.", {
            "type": "object", "properties": {"gesture_name": {"type": "string", "enum": ["greet", "thinking", "wave", "happy", "sad"]}}, "required": ["gesture_name"]
        })
        self.tools.register("move_servo", self._tool_move_servo, "Move a specific servo.", {
            "type": "object", "properties": {"servo_name": {"type": "string"}, "angle": {"type": "number"}}, "required": ["servo_name", "angle"]
        })
        self.tools.register("update_contact_info", self._tool_update_contact_info, "Save or update a person's name, phone, and address.", {
            "type": "object", "properties": {"name": {"type": "string"}, "phone": {"type": "string"}, "address": {"type": "string"}}, "required": ["name"]
        })
        self.tools.register("save_event_reminder", self._tool_save_event_reminder, "Save an event or reminder to memory.", {
            "type": "object", "properties": {"title": {"type": "string"}, "description": {"type": "string"}, "date_time": {"type": "string"}}, "required": ["title", "date_time"]
        })
        self.tools.register("send_email", self._tool_send_email, "Send an email.", {
            "type": "object", "properties": {"to_address": {"type": "string"}, "subject": {"type": "string"}, "message": {"type": "string"}}, "required": ["to_address", "subject", "message"]
        })
        self.tools.register("send_whatsapp", self._tool_send_whatsapp, "Send a WhatsApp message.", {
            "type": "object", "properties": {"phone_number": {"type": "string"}, "message": {"type": "string"}}, "required": ["phone_number", "message"]
        })
        
        from ..core.command_router import CommandRouter
        self.command_router = CommandRouter(self)

        from ..ai.gemini_live import GeminiLiveClient
        
        self.conversation = ConversationEngine(self.memory, self.tools)
        self.realtime_voice = GeminiLiveClient(
            mic=self.mic,
            speaker=self.speaker,
            eyes=self.eyes,
            gestures=self.gestures,
            state=self.state,
            memory=self.memory,
            event_bus=self.event_bus,
            tools=self.tools,
            camera=self.camera,
        )
        self.chess_engine = ChessAnalysisEngine()
        self.reminders = ReminderScheduler(self.memory, self.event_bus)
        self.documents = PDFReportGenerator()

        # Active interaction context
        self.active_person: Optional[Any] = None
        self._running = False
        self._perception_thread: Optional[threading.Thread] = None
        self._audio_thread: Optional[threading.Thread] = None

        self._subscribe_events()

    def _subscribe_events(self) -> None:
        self.event_bus.subscribe("reminder.due", self._on_reminder_due)
        self.event_bus.subscribe("vision.face_detected", self._on_face_detected)
        self.event_bus.subscribe("motion.idle_wander", self._on_idle_wander)

    def _on_idle_wander(self, event: Event) -> None:
        if self.state.current_state == BehaviorState.IDLE:
            self.gestures.play_async(self.gestures.idle_alive_motion, name="idle_wander")

    def _on_reminder_due(self, event: Event) -> None:
        reminder_data = event.data.get("reminder")
        if not reminder_data:
            return
        reminder_text = reminder_data.get("reminder_text", "")
        self.state.transition_to(BehaviorState.SPEAKING, reason="reminder_triggered")
        self.eyes.set_expression("thinking")
        speech_text = f"একটি রিমাইন্ডার রয়েছে: {reminder_text}"
        audio_path = self.tts.synthesize(speech_text)
        if audio_path:
            self.speaker.play_file(audio_path, block=True)
        self.state.transition_to(BehaviorState.IDLE, reason="reminder_delivered")

    def _on_face_detected(self, event: Event) -> None:
        person_data = event.data.get("person")
        if person_data:
            self.behavior.on_person_spotted(
                person_data.get("name", "Unknown"), person_data.get("is_known", False)
            )

    def _perception_loop(self) -> None:
        logger.info("Starting Perception Loop")
        last_frame_time = 0.0
        while self._running:
            if not self.camera.is_available():
                time.sleep(1.0)
                continue
            now = time.time()
            if now - last_frame_time >= 0.3:
                last_frame_time = now
                frame = self.camera.get_frame()
                if frame is not None:
                    self.process_person_interaction(frame)
            time.sleep(0.1)

    def _compute_rms(self, pcm_data: bytes) -> float:
        import struct

        count = len(pcm_data) // 2
        if count == 0:
            return 0.0
        shorts = struct.unpack(f"<{count}h", pcm_data)
        sum_sq = sum(s * s for s in shorts)
        return math.sqrt(sum_sq / count)

    def _audio_loop(self) -> None:
        logger.info("Starting Audio Loop")
        speech_started = False
        speech_start_time = 0.0
        speech_end_time = 0.0
        buffer = bytearray()
        ENERGY_THRESHOLD = 400

        while self._running:
            chunk = self.mic.read_chunk(1024)
            if not chunk:
                time.sleep(0.01)
                continue

            # Push audio to Gemini Live / Realtime engine if supported
            if hasattr(self, "realtime_voice") and hasattr(self.realtime_voice, "push_audio_chunk"):
                self.realtime_voice.push_audio_chunk(chunk)

            energy = self._compute_rms(chunk)
            now = time.time()

            if energy > ENERGY_THRESHOLD:
                if not speech_started:
                    if speech_start_time == 0:
                        speech_start_time = now
                    elif now - speech_start_time >= 0.25:
                        speech_started = True
                        speech_end_time = 0.0
                        buffer.extend(chunk)
                        logger.debug("VAD: Speech started")
                else:
                    speech_end_time = 0.0
                    buffer.extend(chunk)
            else:
                speech_start_time = 0.0
                if speech_started:
                    if speech_end_time == 0:
                        speech_end_time = now
                    elif now - speech_end_time >= 0.8:
                        logger.debug("VAD: Speech ended")
                        speech_started = False
                        if len(buffer) > 0:
                            pcm_audio = bytes(buffer)
                            buffer.clear()
                            try:
                                self.handle_user_speech_input(pcm_audio)
                            except Exception as e:
                                logger.error(f"Error handling speech: {e}")
                    else:
                        buffer.extend(chunk)
            time.sleep(0.01)

    def start_loops(self) -> None:
        """Starts the perception, audio listening, and Realtime Voice background threads."""
        if self._running:
            return
        self._running = True
        self._perception_thread = threading.Thread(
            target=self._perception_loop, daemon=True, name="LumiPerceptionLoop"
        )
        self._perception_thread.start()

        self._audio_thread = threading.Thread(
            target=self._audio_loop, daemon=True, name="LumiAudioLoop"
        )
        self._audio_thread.start()

        # Start Realtime Voice Engine (Inworld / OpenAI Realtime WebSocket)
        self.realtime_voice.start()

        logger.info("Lumi Brain background perception & Realtime voice loops started.")

    def run(self) -> None:
        """Starts all background threads and enters the main idle loop."""
        self.start_loops()
        logger.info("Lumi Brain running.")
        try:
            while self._running:
                self.behavior.tick_idle()
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        """Cleanly shuts down all threads and realtime sessions."""
        self._running = False
        self.realtime_voice.stop()
        if self._perception_thread:
            self._perception_thread.join(timeout=1.0)
        if self._audio_thread:
            self._audio_thread.join(timeout=1.0)
        logger.info("Lumi Brain stopped.")

    def process_person_interaction(self, face_frame: Any) -> None:
        """Autonomous visual pipeline: detect person -> track -> greet -> engage."""
        faces = self.face_service.detect_and_recognize(face_frame)
        if not faces:
            return

        face = faces[0]
        person_data = {
            "name": face.person.name if face.is_known and face.person else "Unknown",
            "is_known": face.is_known,
        }
        self.event_bus.emit("vision.face_detected", data={"person": person_data}, source="vision")

        # Smoothly track face with head servos
        self.head.track_bounding_box(face.center[0], face.center[1])
        
        # Track face with procedural eyes
        gaze_x = (face.center[0] / self.settings.vision.frame_width) * 2.0 - 1.0
        gaze_y = (face.center[1] / self.settings.vision.frame_height) * 2.0 - 1.0
        self.eyes.set_gaze(gaze_x, gaze_y)

        if face.is_known and face.person is not None:
            person = face.person
            self.active_person = person
            if self.face_service.should_interact(person.id, cooldown_s=60.0):
                self.state.transition_to(BehaviorState.GREETING, reason=f"spot_{person.name}")
                self.eyes.set_expression("happy")
                self.gestures.play_async(self.gestures.greet, name="greet")
                
                if hasattr(self.realtime_voice, "inject_context"):
                    prompt = f"You see {person.name} in front of you. Acknowledge them naturally in Bengali."
                    self.realtime_voice.inject_context(prompt)
                self.state.transition_to(BehaviorState.IDLE, reason="greeting_complete")
        else:
            # Unknown person learning hook
            if hasattr(self.face_service, "set_pending_face"):
                self.face_service.set_pending_face(face.embedding)
            
            if self.face_service.should_interact("unknown", cooldown_s=60.0):
                self.eyes.set_expression("curious")
                if hasattr(self.realtime_voice, "inject_context"):
                    prompt = "A new person is here. Respond naturally and politely in Bengali."
                    self.realtime_voice.inject_context(prompt)

    def handle_user_speech_input(self, pcm_bytes: bytes) -> str:
        """Process spoken audio buffer -> Command Router (Local Priority) -> LLM Fallback -> TTS."""
        self.state.transition_to(BehaviorState.LISTENING, reason="user_speaking")
        self.eyes.set_expression("curious")
        transcription = self.stt.transcribe_pcm_bytes(pcm_bytes)
        if not transcription:
            self.state.transition_to(BehaviorState.IDLE, reason="empty_speech")
            return ""

        print(f"\n🗣️  [YOU / USER]: {transcription}")
        logger.info(f"User Spoke: '{transcription}'")

        # ---------------------------------------------------------------------
        # PRIORITY 1: Check Local Skills & Commands First (Zero Cloud LLM Overhead)
        # ---------------------------------------------------------------------
        local_reply = self.command_router.route_speech(transcription)
        if local_reply is not None:
            print(f"🤖 [LUMI (Local)]: {local_reply}")
            logger.info(f"LUMI Executed Local Command Response: '{local_reply}'")
            if local_reply.strip():
                self.state.transition_to(BehaviorState.SPEAKING, reason="local_cmd_reply")
                audio_path = self.tts.synthesize(local_reply)
                if audio_path:
                    self.speaker.play_file(audio_path, block=True)
            self.state.transition_to(BehaviorState.IDLE, reason="local_cmd_finished")
            return local_reply

        # ---------------------------------------------------------------------
        # PRIORITY 2: Send to Conversational LLM if no local command matched
        # ---------------------------------------------------------------------
        self.state.transition_to(BehaviorState.THINKING, reason="ai_processing")
        self.eyes.set_expression("thinking")
        self.gestures.play_async(self.gestures.thinking, name="thinking")

        response = self.conversation.generate_response(
            transcription, current_person=self.active_person
        )
        print(f"🤖 [LUMI (AI)]: {response}")
        logger.info(f"LUMI Responded (LLM): '{response}'")

        self.state.transition_to(BehaviorState.SPEAKING, reason="robot_replying")
        self.eyes.set_expression("happy")

        audio_path = self.tts.synthesize(response)
        if audio_path:
            self.speaker.play_file(audio_path, block=True)

        self.state.transition_to(BehaviorState.IDLE, reason="reply_finished")
        return response

    def analyze_plant_leaf(self, leaf_frame: Any) -> str:
        """Run computer vision leaf pathology classifier."""
        self.state.transition_to(BehaviorState.VISION_ANALYSIS, reason="plant_analysis")
        self.eyes.set_expression("curious")
        result = self.plant_detector.analyze_leaf(leaf_frame)
        summary = self.plant_detector.generate_bangla_speech_summary(result)
        self.state.transition_to(BehaviorState.SPEAKING, reason="explain_plant")
        audio_path = self.tts.synthesize(summary)
        if audio_path:
            self.speaker.play_file(audio_path, block=True)
        self.state.transition_to(BehaviorState.IDLE, reason="plant_analysis_done")
        return summary

    # =========================================================================
    # Realtime Tools Implementation
    # =========================================================================
    def _tool_memorize_person(self, name: str) -> str:
        """Saves the pending unknown face with the given name."""
        encoding = self.face_service.get_pending_face()
        if not encoding:
            return "No unknown face is currently in view to memorize."
        
        from ..memory.models import ConsentStatus
        self.memory.remember_person(
            name=name,
            relationship="friend",
            consent_status=ConsentStatus.GRANTED,
            preferred_language="bn"
        )
        person = self.memory.find_person_by_name(name)
        if person:
            person.face_embedding = encoding
            self.memory.update_person(person)
            return f"Successfully memorized the face of {name}."
        return f"Failed to save {name} to database."

    def _tool_describe_vision(self) -> str:
        """Describes what the robot currently sees via GPT-4 Vision API."""
        frame = self.camera.get_frame()
        if frame is None:
            return "I cannot see anything right now. The camera is offline."
        
        import cv2
        import base64
        import urllib.request
        import json
        import os
        
        _, buffer = cv2.imencode('.jpg', frame)
        b64_img = base64.b64encode(buffer).decode('utf-8')
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "Vision API key is missing."
        
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image briefly in Bengali."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                    ]
                }
            ],
            "max_tokens": 150
        }
        
        req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(payload).encode('utf-8'), headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        })
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode())
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Vision API error: {e}")
            return "I'm having trouble understanding what I see right now."

    def _tool_analyze_plant(self) -> str:
        frame = self.camera.get_frame()
        if frame is None: return "Camera offline."
        return self.plant_detector.generate_bangla_speech_summary(self.plant_detector.analyze_leaf(frame))

    def _tool_analyze_chess(self) -> str:
        frame = self.camera.get_frame()
        if frame is None: return "Camera offline."
        chess_res = self.chess_vision.extract_fen_from_frame(frame)
        eval_res = self.chess_engine.analyze_position(chess_res.fen_string)
        return eval_res.explanation_bn

    def _tool_perform_gesture(self, gesture_name: str) -> str:
        func = getattr(self.gestures, gesture_name, None)
        if func:
            self.gestures.play_async(func, name=gesture_name)
            return f"Performed gesture {gesture_name}."
        return f"Gesture {gesture_name} not implemented."

    def _tool_move_servo(self, servo_name: str, angle: float) -> str:
        try:
            self.servos.set_angle(servo_name, angle)
            return f"Moved {servo_name} to {angle} degrees."
        except Exception as e:
            return f"Failed to move servo {servo_name}: {e}"

    def _tool_update_contact_info(self, name: str, phone: str = "", address: str = "") -> str:
        person = self.memory.find_person_by_name(name)
        if not person:
            encoding = self.face_service.get_pending_face()
            from ..memory.models import ConsentStatus
            self.memory.remember_person(
                name=name,
                relationship="friend",
                consent_status=ConsentStatus.GRANTED,
                preferred_language="bn"
            )
            person = self.memory.find_person_by_name(name)
            if person and encoding:
                person.face_embedding = encoding
                
        if person:
            if phone: person.metadata["phone"] = phone
            if address: person.metadata["address"] = address
            self.memory.update_person(person)
            return f"Updated contact info for {name}. Phone: {phone}, Address: {address}."
        return "Failed to find or create person."

    def _tool_save_event_reminder(self, title: str, description: str, date_time: str) -> str:
        from ..memory.models import Fact
        fact = Fact(
            category="event",
            fact_text=f"Event: {title} on {date_time}. Details: {description}"
        )
        self.memory.save_fact(fact)
        return "Event successfully saved to memory."

    def _tool_send_email(self, to_address: str, subject: str, message: str) -> str:
        logger.info(f"Mock sending Email to {to_address} with subject '{subject}': {message}")
        return f"Email successfully queued to {to_address}."

    def _tool_send_whatsapp(self, phone_number: str, message: str) -> str:
        logger.info(f"Mock sending WhatsApp to {phone_number}: {message}")
        return f"WhatsApp message successfully queued to {phone_number}."

    def analyze_chessboard(self, board_frame: Any) -> str:
        """Extract FEN from physical chessboard and query Stockfish UCI engine."""
        self.state.transition_to(BehaviorState.CHESS_ANALYSIS, reason="chess_analysis")
        self.eyes.set_expression("thinking")
        self.gestures.play_async(self.gestures.thinking, name="thinking")
        chess_res = self.chess_vision.extract_fen_from_frame(board_frame)
        eval_res = self.chess_engine.analyze_position(chess_res.fen_string)
        summary = eval_res.explanation_bn
        self.state.transition_to(BehaviorState.SPEAKING, reason="explain_chess")
        audio_path = self.tts.synthesize(summary)
        if audio_path:
            self.speaker.play_file(audio_path, block=True)
        self.state.transition_to(BehaviorState.IDLE, reason="chess_done")
        return summary

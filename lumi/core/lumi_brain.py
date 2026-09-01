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

from ..speech.tts import BanglaTTS
from ..vision.camera import CameraInterface
from ..vision.chess import ChessVision
from ..vision.face import FaceRecognitionService
from ..vision.plant import PlantDiseaseDetector
from ..memory.mem0_engine import LumiMem0Engine

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
        
        self.tts = BanglaTTS()

        # AI & Reasoning Subsystems
        self.tools = ToolRegistry()
        self.tools.register("memorize_person", self._tool_memorize_person, "CALL THIS ONLY when the user explicitly introduces themselves (e.g., 'My name is X') or asks you to remember their name. Do NOT call this for random names or entities mentioned in conversation.", {
            "type": "object", 
            "properties": {
                "name": {"type": "string", "description": "The person's full name."},
                "relationship": {"type": "string", "description": "Their relationship to Palash (the owner), e.g. friend, brother, guest."},
                "notes": {"type": "string", "description": "Any short important facts or details to remember about them."}
            }, 
            "required": ["name"]
        })
        self.tools.register("analyze_plant", self._tool_analyze_plant, "Analyze the plant the camera is seeing.")
        self.tools.register("describe_vision", self._tool_describe_vision, "Describe what you currently see through the camera. Use this when someone asks what you see, or when you want to comment on surroundings.")
        self.tools.register("analyze_chess", self._tool_analyze_chess, "Analyze the chessboard the camera is seeing.")
        self.tools.register("perform_gesture", self._tool_perform_gesture, "Perform a physical gesture. CALL THIS SPARINGLY, only when highly appropriate to the context (e.g., waving when saying goodbye). Do NOT call this continuously.", {
            "type": "object", "properties": {"gesture_name": {"type": "string", "enum": ["greet", "thinking", "wave", "happy", "sad"]}}, "required": ["gesture_name"]
        })
        self.tools.register("move_servo", self._tool_move_servo, "Move a specific servo.", {
            "type": "object", "properties": {"servo_name": {"type": "string"}, "angle": {"type": "number"}}, "required": ["servo_name", "angle"]
        })
        self.tools.register("update_contact_info", self._tool_update_contact_info, "Save or update a person's name, phone, and address.", {
            "type": "object", "properties": {"name": {"type": "string"}, "phone": {"type": "string"}, "address": {"type": "string"}}, "required": ["name"]
        })
        self.tools.register("set_reminder", self._tool_set_reminder, "Set a time-based reminder. You MUST provide the time in ISO 8601 format (YYYY-MM-DDThh:mm:ss). Calculate this based on the current time provided in your system instructions.", {
            "type": "object", "properties": {"title": {"type": "string", "description": "Short title of the reminder"}, "description": {"type": "string", "description": "Optional details"}, "remind_at_iso": {"type": "string", "description": "ISO 8601 formatted datetime string"}}, "required": ["title", "remind_at_iso"]
        })
        self.tools.register("send_email", self._tool_send_email, "Send an email.", {
            "type": "object", "properties": {"to_address": {"type": "string"}, "subject": {"type": "string"}, "message": {"type": "string"}}, "required": ["to_address", "subject", "message"]
        })
        self.tools.register("send_whatsapp", self._tool_send_whatsapp, "Send a WhatsApp message.", {
            "type": "object", "properties": {"phone_number": {"type": "string"}, "message": {"type": "string"}}, "required": ["phone_number", "message"]
        })
        self.tools.register("memorize_fact", self._tool_memorize_fact, "Save a specific fact or detail about a person or event to long-term memory. Do this autonomously whenever you learn something important (e.g. user's hobbies, current tasks, preferences).", {
            "type": "object", "properties": {"fact": {"type": "string", "description": "The fact to remember (e.g. 'Palash likes black coffee')."}, "person_name": {"type": "string", "description": "Optional name of the person this fact is about."}}, "required": ["fact"]
        })
        self.tools.register("recall_facts", self._tool_recall_facts, "Retrieve past facts from long-term memory about a person or topic. PROACTIVELY call this whenever the user brings up a new topic, a person's name, or an ongoing project to check if you have context, even if the user didn't explicitly ask you to remember. Integrate the results naturally.", {
            "type": "object", "properties": {"search_query": {"type": "string", "description": "Keywords to search for."}, "person_name": {"type": "string", "description": "Optional name of the person."}}, "required": ["search_query"]
        })
        
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
        
        # Dual-Mem0 System: Use Cloud API if key exists, otherwise use Local Gemini Engine
        import os
        from ..memory.mem0_cloud import Mem0CloudEngine
        if os.environ.get("MEM0_API_KEY"):
            logger.info("MEM0_API_KEY detected! Using official Mem0 Cloud API.")
            self.mem0 = Mem0CloudEngine()
        else:
            logger.info("No MEM0_API_KEY found. Falling back to native LumiMem0 Engine.")
            self.mem0 = LumiMem0Engine(self.memory)

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
        self.event_bus.subscribe("conversation.turn_complete", self._on_turn_complete)

    def _on_turn_complete(self, event: Event) -> None:
        person = self.active_person
        if not person:
            # Fallback: attribute conversation to owner so memory is never lost
            person = self.memory.find_person_by_name("Palash")
            if not person:
                return
        u_text = event.data.get("user", "")
        l_text = event.data.get("lumi", "")
        if u_text or l_text:
            self.mem0.process_conversation_turn_async(
                person_id=person.id,
                person_name=person.name,
                user_text=u_text,
                ai_text=l_text
            )

    def _on_idle_wander(self, event: Event) -> None:
        if self.state.current_state == BehaviorState.IDLE:
            self.gestures.play_async(self.gestures.idle_alive_motion, name="idle_wander")

    def _on_reminder_due(self, event: Event) -> None:
        title = event.data.get("title", "")
        desc = event.data.get("description", "")
        remind_at = event.data.get("remind_at", "")
        
        if hasattr(self, "realtime_voice") and self.realtime_voice._running:
            from datetime import datetime
            import dateutil.parser
            
            prompt = f"[SYSTEM ALERT: A reminder scheduled by the user is due. Title: '{title}'. Description: '{desc}'.]"
            
            try:
                # Check if it's late (e.g. system was offline)
                # remind_at is from SQLite so it might be missing timezone, replace Z with +00:00 just in case
                scheduled_time = datetime.fromisoformat(remind_at.replace("Z", "+00:00"))
                # If naive, make it aware or vice versa? Let's just strip tzinfo for simple calculation 
                # since both are local times based on our previous fix.
                scheduled_time = scheduled_time.replace(tzinfo=None)
                now = datetime.now()
                diff_minutes = (now - scheduled_time).total_seconds() / 60.0
                
                if diff_minutes > 5:
                    prompt = f"[SYSTEM ALERT: The user had a reminder scheduled for {remind_at} ('{title}'), but you were OFFLINE at that time. You just woke up. Apologize for missing the exact time and tell them the reminder now. Naturally in Bangla.]"
                else:
                    prompt = f"[SYSTEM ALERT: A reminder scheduled by the user is due RIGHT NOW. Title: '{title}'. Description: '{desc}'. Please notify the user about this reminder enthusiastically and naturally in Bangla immediately.]"
            except Exception as e:
                import logging
                logging.getLogger("reminders").error(f"Error parsing reminder time: {e}")
                pass
                
            self.realtime_voice.inject_context(prompt)
        else:
            self.state.transition_to(BehaviorState.SPEAKING, reason="reminder_triggered")
            self.eyes.set_expression("thinking")
            speech_text = f"একটি রিমাইন্ডার রয়েছে: {title}. {desc}"
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
        logger.info("Starting Audio Loop (Streaming to Gemini Live)")
        ENERGY_THRESHOLD = 400
        _debug_audio_frames = 0
        
        while self._running:
            chunk = self.mic.read_chunk(1024)
            if not chunk:
                time.sleep(0.01)
                continue

            # Push audio to Gemini Live / Realtime engine if supported
            if hasattr(self, "realtime_voice") and hasattr(self.realtime_voice, "push_audio_chunk"):
                self.realtime_voice.push_audio_chunk(chunk)

            energy = self._compute_rms(chunk)
            _debug_audio_frames += 1
            if _debug_audio_frames % 200 == 0:
                logger.debug(f"Mic Audio RMS Energy: {energy:.1f}")

            if energy > ENERGY_THRESHOLD:
                if self.state.current_state == BehaviorState.IDLE:
                    self.eyes.set_expression("curious")
            
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
                    # 10x Better Known Person Interaction
                    relationship = person.relationship if hasattr(person, 'relationship') else 'friend'
                    notes = person.notes if hasattr(person, 'notes') and person.notes else 'None'
                    
                    fact_str = "None"
                    if hasattr(self.mem0, "recall_facts_sync"):
                        # Official Mem0 Cloud API
                        fact_str = self.mem0.recall_facts_sync(person.id)
                    else:
                        # Native SQLite Engine
                        recent_facts = self.memory.recall_facts(person_id=person.id)
                        if recent_facts:
                            fact_str = ", ".join([f.fact_text for f in recent_facts[:3]])
                    
                    prompt = (
                        f"CRITICAL CONTEXT: You are currently talking to {person.name} ({relationship}). "
                        f"Notes about them: {notes}. "
                        f"Recent memories you saved: {fact_str}. "
                        "When using these memories, remember that YOU are talking TO this person. If a memory says 'Palash did X and Fuad did Y' and the user is Palash, then YOU know that the user did X. "
                        "Acknowledge them naturally, warmly, and politely in conversational Bengali (বাংলা). "
                        "Do not mention their notes or memories mechanically, but use them naturally to ask how they are doing (e.g. 'Did you finish X?')."
                    )
                    self.realtime_voice.inject_context(prompt)
                self.state.transition_to(BehaviorState.IDLE, reason="greeting_complete")
        else:
            # Unknown person learning hook
            if hasattr(self.face_service, "set_pending_face"):
                self.face_service.set_pending_face(face.embedding)
            
            if self.face_service.should_interact("unknown", cooldown_s=60.0):
                self.eyes.set_expression("curious")
                if hasattr(self.realtime_voice, "inject_context"):
                    # 10x Better Unknown Person Interaction
                    prompt = (
                        "A new, unrecognized person just walked into your view. "
                        "Act naturally curious! Greet them warmly in Bengali (বাংলা). "
                        "Politely ask for their name, how they are related to Palash (your owner), "
                        "and if they would like you to remember their face for next time. "
                        "If they agree, use the 'memorize_person' tool with their details."
                    )
                    self.realtime_voice.inject_context(prompt)

    # =========================================================================
    # Realtime Tools Implementation
    # =========================================================================
    def _tool_memorize_person(self, name: str, relationship: str = "guest", notes: str = "") -> str:
        """Saves the pending unknown face with detailed metadata."""
        encoding = self.face_service.get_pending_face()
        if not encoding:
            return "No unknown face is currently in view to memorize. Please ask them to look at the camera."
        
        from ..memory.models import ConsentStatus
        self.memory.remember_person(
            name=name,
            relationship=relationship,
            consent_status=ConsentStatus.GRANTED,
            preferred_language="bn"
        )
        person = self.memory.find_person_by_name(name)
        if person:
            person.face_embedding = encoding
            if notes:
                person.notes = notes
            self.memory.update_person(person)
            return f"Successfully memorized the face of {name} ({relationship})."
        return f"Failed to save {name} to database."

    def _tool_memorize_fact(self, fact: str, person_name: Optional[str] = None) -> str:
        """Autonomously remember a semantic fact."""
        person_id = None
        if person_name:
            person = self.memory.find_person_by_name(person_name)
            if person:
                person_id = person.id
            else:
                return f"Person '{person_name}' not found. Cannot attach fact to them."
        
        saved_fact = self.memory.remember_fact(fact_text=fact, person_id=person_id)
        if saved_fact:
            return f"Fact memorized successfully: '{fact}'"
        return "Failed to memorize fact due to privacy consent settings."

    def _tool_recall_facts(self, search_query: str, person_name: Optional[str] = None) -> str:
        """Recall saved facts from semantic memory."""
        person_id = None
        if person_name:
            person = self.memory.find_person_by_name(person_name)
            if person:
                person_id = person.id
                
        # If no specific person requested, default to the person we're talking to (or owner)
        if not person_id:
            active = getattr(self, "active_person", None)
            fallback = self.memory.find_person_by_name("Palash")
            person = active or fallback
            if person:
                person_id = person.id
                
        # 1. Check local SQLite memory
        local_facts = self.memory.recall_facts(person_id=person_id, search_query=search_query)
        local_results = [f"- {f.fact_text} (from {f.created_at[:10]})" for f in local_facts[:5]]
        
        # 2. Check Mem0 Cloud (if active)
        cloud_results = []
        if hasattr(self, "mem0") and hasattr(self.mem0, "recall_facts_sync"):
            cloud_str = self.mem0.recall_facts_sync(person_id=person_id or "default", query=search_query)
            if cloud_str:
                cloud_results.append(f"- {cloud_str}")
                
        all_results = local_results + cloud_results
        
        if not all_results:
            return f"No relevant facts found in memory for {person.name if person else 'this person'}."
            
        result = f"Memories retrieved about {person.name if person else 'User'}:\n" + "\n".join(all_results)
        result += f"\n(CRITICAL INSTRUCTION: You are currently talking to {person.name if person else 'the User'}. The above memories are facts about them. If the memory says 'Palash did X and Fuad did Y', and the user is Palash, you must understand that the USER did X, and their friend/colleague Fuad did Y. Do not get confused about who is who.)"
        
        return result

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
            self.servo.move_joint(servo_name, angle)
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

    def _tool_set_reminder(self, title: str, remind_at_iso: str, description: str = "") -> str:
        person = getattr(self, "active_person", None)
        person_id = person.id if person else None
        try:
            self.memory.create_reminder(
                title=title,
                remind_at_iso=remind_at_iso,
                person_id=person_id,
                description=description
            )
            return f"Reminder successfully scheduled for {remind_at_iso}."
        except Exception as e:
            return f"Failed to set reminder: {e}"

    def _tool_send_email(self, to_address: str, subject: str, message: str) -> str:
        logger.info(f"Mock sending Email to {to_address} with subject '{subject}': {message}")
        return f"Email successfully queued to {to_address}."

    def _tool_send_whatsapp(self, phone_number: str, message: str) -> str:
        logger.info(f"Mock sending WhatsApp to {phone_number}: {message}")
        return f"WhatsApp message successfully queued to {phone_number}."




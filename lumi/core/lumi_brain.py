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
        from ..audio.vad import VoiceActivityDetector
        from ..audio.speaker_id import SpeakerIdentifier
        self.vad = VoiceActivityDetector(aggressiveness=2)
        self.speaker_id = SpeakerIdentifier(self.memory, similarity_threshold=0.75)
        self._current_speaker: Optional[str] = None
        self._voice_buffer: bytearray = bytearray()  # Buffer for voice enrollment
        self._enrolling_voice_for: Optional[str] = None  # Person ID being enrolled
        
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
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Short title for the reminder (e.g. Take medicine)."},
                "remind_at_iso": {"type": "string", "description": "The exact date and time to trigger the reminder in ISO 8601 format."},
                "description": {"type": "string", "description": "Optional details about the reminder."}
            },
            "required": ["title", "remind_at_iso"]
        })
        self.tools.register("show_animal_animation", self._tool_show_animal_animation, "Show an animal animation/image on your screen. Call this when the user asks how an animal sounds or acts, while SIMULTANEOUSLY using your voice to mimic the animal sound.", {
            "type": "object",
            "properties": {
                "animal": {"type": "string", "description": "The name of the animal (e.g. cat, dog, bird)."}
            },
            "required": ["animal"]
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
        self.tools.register("leave_message", self._tool_leave_message, "Save a message for the owner or another person if they are not currently present. When someone asks for someone else, organically ask if they want to leave a message, and if yes, use this.", {
            "type": "object", "properties": {
                "recipient_name": {"type": "string"},
                "sender_name": {"type": "string"},
                "message_text": {"type": "string"}
            },
            "required": ["recipient_name", "sender_name", "message_text"]
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

        # --- Proactive Memory Systems (Phase 2 + Phase 4) ---
        # ProactiveRecallEngine: auto-injects relevant memories during conversation
        from ..memory.proactive_recall import ProactiveRecallEngine
        self.proactive_recall = ProactiveRecallEngine(
            memory=self.memory,
            mem0=self.mem0,
            realtime_voice=self.realtime_voice,
            event_bus=self.event_bus,
        )

        # MemoryConsolidator: background daemon for dedup, staleness decay, cleanup
        from ..memory.consolidation import MemoryConsolidator
        self.memory_consolidator = MemoryConsolidator(self.memory)
        self.memory_consolidator.start()

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

        # --- Person-mention detection (Phase 2, Change 5) ---
        # If the user mentions another known person by name, auto-inject their facts
        if u_text:
            try:
                for p in self.memory.list_people():
                    if p.id == person.id:
                        continue  # Skip the active person (already have their context)
                    name_lower = p.name.lower()
                    first_name = name_lower.split()[0] if name_lower else ""
                    if name_lower in u_text.lower() or (len(first_name) > 2 and first_name in u_text.lower()):
                        mentioned_facts = self.memory.recall_facts(person_id=p.id)
                        if mentioned_facts:
                            facts_str = ", ".join([f.fact_text for f in mentioned_facts[:3]])
                            context = (
                                f"[MEMORY CONTEXT: The user just mentioned {p.name} ({p.relationship}). "
                                f"What you remember about {p.name}: {facts_str}. "
                                f"Use this knowledge naturally.]"
                            )
                            if hasattr(self.realtime_voice, "inject_context"):
                                self.realtime_voice.inject_context(context)
                            break  # Only inject for first mentioned person per turn
            except Exception as e:
                logger.debug(f"Person-mention detection error: {e}")

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
        self._last_face_seen_time = time.time()
        person_data = event.data.get("person")
        if person_data:
            self.behavior.on_person_spotted(
                person_data.get("name", "Unknown"), person_data.get("is_known", False)
            )

    def _perception_loop(self) -> None:
        logger.info("Starting Perception Loop")
        last_frame_time = 0.0
        self._last_face_seen_time = time.time()
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

            # Clear active_person if no face detected for 30 seconds
            if self.active_person and (now - self._last_face_seen_time > 30.0):
                logger.info(f"Active person '{self.active_person.name}' timed out (no face for 30s).")
                self.active_person = None
                
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
        logger.info("Starting Audio Loop (Streaming to Gemini Live + VAD + Speaker ID)")
        ENERGY_THRESHOLD = 400
        _debug_audio_frames = 0

        # Wire up VAD callbacks
        self.vad.set_on_utterance_complete(self._on_speech_utterance)
        self.vad.set_on_overlap_detected(self._on_overlap_detected)
        
        while self._running:
            chunk = self.mic.read_chunk(1024)
            if not chunk:
                time.sleep(0.01)
                continue

            # 1. Always push audio to Gemini Live (uninterrupted stream)
            if hasattr(self, "realtime_voice") and hasattr(self.realtime_voice, "push_audio_chunk"):
                self.realtime_voice.push_audio_chunk(chunk)

            # 2. Feed chunk through VAD pipeline
            from ..audio.vad import SpeechEvent
            event = self.vad.process_chunk(chunk)

            # 3. Collect audio for voice enrollment if active
            if self._enrolling_voice_for:
                self._voice_buffer.extend(chunk)

            # 4. Eye animation on speech detection
            energy = self._compute_rms(chunk)
            _debug_audio_frames += 1
            if _debug_audio_frames % 200 == 0:
                logger.debug(f"Mic Audio RMS Energy: {energy:.1f}")

            if event == SpeechEvent.SPEECH_START or energy > ENERGY_THRESHOLD:
                if self.state.current_state == BehaviorState.IDLE:
                    self.eyes.set_expression("curious")

            # 5. Overlap detection (check periodically during speech)
            if event == SpeechEvent.SPEECH_CONTINUE and _debug_audio_frames % 50 == 0:
                # Count visible faces from last perception loop
                num_faces = len(getattr(self, '_last_detected_faces', []))
                self.vad.detect_overlap(num_faces)
            
            time.sleep(0.01)

    def _on_speech_utterance(self, audio_bytes: bytes, duration: float) -> None:
        """Called by VAD when a complete speech utterance is ready.
        
        Runs speaker identification in a background thread to avoid
        blocking the audio loop.
        """
        def _identify():
            # If we're enrolling a voice, skip identification
            if self._enrolling_voice_for:
                return

            if not self.speaker_id.is_available():
                return

            speaker_name, confidence = self.speaker_id.identify_speaker(audio_bytes)
            if speaker_name and confidence >= 0.75:
                if speaker_name != self._current_speaker:
                    self._current_speaker = speaker_name
                    logger.info(f"🎙️ Active speaker changed to: {speaker_name}")
                    # Tell Gemini who is speaking
                    if hasattr(self, "realtime_voice") and hasattr(self.realtime_voice, "inject_context"):
                        self.realtime_voice.inject_context(
                            f"SPEAKER UPDATE: The person currently speaking is {speaker_name}. "
                            f"Address them by name when responding."
                        )
            else:
                if self._current_speaker is not None:
                    self._current_speaker = None
                    logger.debug("🎙️ Speaker not recognized (unknown voice)")

        thread = threading.Thread(target=_identify, daemon=True, name="SpeakerID_Worker")
        thread.start()

    def _on_overlap_detected(self) -> None:
        """Called by VAD when overlapping speech from multiple people is detected."""
        logger.info("🔊 Overlapping speech detected! Asking people to take turns.")
        if hasattr(self, "realtime_voice") and hasattr(self.realtime_voice, "inject_context"):
            import random
            overlap_responses = [
                "Multiple people are talking at once! Say in Bengali: 'আরে আরে, একজন একজন করে বলো! আমি তো সবার কথা একসাথে ধরতে পারি না!' Then laugh friendly.",
                "You hear overlapping voices. Say in Bengali: 'একটু থামো থামো! কে আগে বলবে?' with a playful tone.",
                "Too many voices at once! Say in Bengali: 'ভাই একসাথে বললে তো আমি বুঝি না! একজন বলো আগে!' Keep it humorous.",
            ]
            self.realtime_voice.inject_context(random.choice(overlap_responses))

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
        if hasattr(self, "memory_consolidator"):
            self.memory_consolidator.stop()
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

        faces = self.face_service.confirm_identity(faces)
        
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

        # Steer microphone beamformer towards the active face
        if hasattr(self.mic, 'backend') and hasattr(self.mic.backend, 'spatial_processor'):
            spatial = self.mic.backend.spatial_processor
            if spatial:
                spatial.steer_towards_face(face.center[0], self.settings.vision.frame_width)

        if face.is_known and face.person is not None:
            person = face.person
            self.active_person = person
            self._last_face_seen_time = time.time()
            if self.face_service.should_interact(person.id, cooldown_s=60.0):
                self.state.transition_to(BehaviorState.GREETING, reason=f"spot_{person.name}")
                self.eyes.set_expression("happy")
                self.gestures.play_async(self.gestures.greet, name="greet")
                
                if hasattr(self.realtime_voice, "inject_context"):
                    relationship = person.relationship if hasattr(person, 'relationship') else 'friend'
                    notes = person.notes if hasattr(person, 'notes') and person.notes else 'None'
                    
                    fact_str = "None"
                    if hasattr(self.mem0, "recall_facts_sync"):
                        fact_str = self.mem0.recall_facts_sync(person.id)
                    else:
                        recent_facts = self.memory.recall_facts(person_id=person.id)
                        if recent_facts:
                            fact_str = ", ".join([f.fact_text for f in recent_facts[:3]])
                    
                    # Check for unread messages
                    unread_msgs = ""
                    if hasattr(self.memory, "get_unread_messages"):
                        msgs = self.memory.get_unread_messages(person.id)
                        if msgs:
                            msg_texts = [f"From {m['sender_name']}: {m['message_text']}" for m in msgs]
                            unread_msgs = f"\nURGENT: YOU HAVE UNREAD MESSAGES FOR {person.name}: {', '.join(msg_texts)}. YOU MUST TELL THEM THIS MESSAGE IMMEDIATELY AS SOON AS YOU GREET THEM!"
                            self.memory.mark_messages_read(person.id)
                    
                    prompt = (
                        f"CRITICAL CONTEXT: You are currently talking to {person.name} ({relationship}). "
                        f"Notes about them: {notes}. "
                        f"Recent memories you saved: {fact_str}. {unread_msgs}\n"
                        "When using these memories, remember that YOU are talking TO this person. "
                        "Acknowledge them naturally, warmly, and politely in conversational Bengali (বাংলা). "
                        "Do not mention their notes or memories mechanically, but use them naturally. "
                        
                        "Also casually ask what brings them here today, or if they have any message for Palash (assuming Palash is not here right now). "
                        "If they want to leave a message, use the 'leave_message' tool. "
                        "CRITICAL RULE FOR APPEARANCE: Only compliment their appearance (dress, hair, etc.) if you "
                        "CLEARLY and UNMISTAKABLY see something specific in the camera feed right now. "
                        "If the camera feed is unclear, or you just see a face/wall without distinct clothing, "
                        "DO NOT make up a compliment. A forced or fake compliment feels unnatural."
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
                    import random
                    # Varied unknown person greetings for natural feel
                    unknown_prompts = [
                        (
                            "A new person appeared! You don't know them. "
                            "In Bengali, greet warmly: 'তোমাকে তো চিনতে পারছি না! তুমি কে?' "
                            "Ask their name. If they are looking for Palash, say he's not here and ask if they want to leave a message. "
                            "If yes, use 'leave_message' tool. Also use 'memorize_person' to save their name."
                        ),
                        (
                            "Someone unfamiliar is standing in front of you. Be curious! "
                            "In Bengali, say 'আরে, নতুন কেউ এসেছে! পরিচয়টা দাও তো!' "
                            "Subtly ask if they came to see Palash or have a message for him. "
                            "When they tell their name, call 'memorize_person'."
                        ),
                        (
                            "A stranger appeared! Act naturally surprised and curious. "
                            "In Bengali, say 'ওহ, তোমাকে তো আগে দেখিনি! কী নাম তোমার?' "
                            "Ask their name and if they need to leave a message for your owner Palash (who is away). "
                            "Use 'memorize_person' tool once they introduce themselves."
                        ),
                        (
                            "You see an unknown face! Greet them in Bengali with friendly curiosity. "
                            "Say something like 'এই যে! তুমি নতুন মুখ! আমি তো তোমাকে চিনি না!' "
                            "Ask their name and if they have a message for Palash. If they share their name, save with 'memorize_person'. "
                            "CRITICAL RULE: Do not compliment their appearance unless you clearly see something very distinct."
                        ),
                    ]
                    prompt = random.choice(unknown_prompts)
                    self.realtime_voice.inject_context(prompt)

    # =========================================================================
    # Realtime Tools Implementation
    # =========================================================================
    def _tool_memorize_person(self, name: str, relationship: str = "guest", notes: str = "") -> str:
        """Saves the pending unknown face with detailed metadata and starts voice enrollment."""
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
            
            # Start voice enrollment in background
            # Collect audio for 5 seconds to build a voice profile
            self._voice_buffer = bytearray()
            self._enrolling_voice_for = person.id

            def _finish_enrollment():
                time.sleep(5.0)  # Collect 5 seconds of audio
                audio_data = bytes(self._voice_buffer)
                self._enrolling_voice_for = None
                self._voice_buffer = bytearray()
                if len(audio_data) > 16000 * 2 * 1.5:  # At least 1.5 sec
                    success = self.speaker_id.enroll_voice(person.id, audio_data)
                    if success:
                        logger.info(f"🎙️ Voice profile saved for {name}")
                    else:
                        logger.warning(f"Voice enrollment failed for {name}")

            enrollment_thread = threading.Thread(
                target=_finish_enrollment, daemon=True, name=f"VoiceEnroll_{name}"
            )
            enrollment_thread.start()

            return f"Successfully memorized the face of {name} ({relationship}). Also enrolling their voice profile..."
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
        """Recall saved facts from semantic memory with dedup and recency scoring."""
        from datetime import datetime as _dt

        person_id = None
        person = None
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

        # 1. Check local SQLite memory (prefer FTS5 if available, fallback to LIKE)
        if hasattr(self.memory, "recall_facts_fts"):
            local_facts = self.memory.recall_facts_fts(search_query, person_id=person_id, limit=10)
        else:
            local_facts = self.memory.recall_facts(person_id=person_id, search_query=search_query)
        
        # 2. Check Mem0 Cloud (if active)
        cloud_facts_text = ""
        if hasattr(self, "mem0") and hasattr(self.mem0, "recall_facts_sync"):
            cloud_facts_text = self.mem0.recall_facts_sync(person_id=person_id or "default", query=search_query)

        # 3. Apply recency-weighted scoring to local facts
        now = _dt.now()
        scored_facts = []
        for f in local_facts:
            try:
                created = _dt.fromisoformat(f.created_at)
                age_days = max((now - created).days, 0)
            except (ValueError, TypeError):
                age_days = 365
            recency_score = 0.5 ** (age_days / 30.0)
            score = f.confidence * recency_score
            scored_facts.append((f, score))
        
        scored_facts.sort(key=lambda x: x[1], reverse=True)
        top_facts = scored_facts[:7]

        # 4. Deduplicate local results
        seen_texts: set = set()
        unique_results: list = []
        for f, score in top_facts:
            normalized = " ".join(f.fact_text.lower().split())
            is_dup = False
            for seen in seen_texts:
                # Simple word-overlap check
                words_a, words_b = set(normalized.split()), set(seen.split())
                if words_a and words_b:
                    overlap = len(words_a & words_b) / len(words_a | words_b)
                    if overlap > 0.75:
                        is_dup = True
                        break
            if not is_dup:
                seen_texts.add(normalized)
                unique_results.append(f"- {f.fact_text} (from {f.created_at[:10]})")

        # 5. Add cloud results (deduped against local)
        if cloud_facts_text:
            cloud_normalized = " ".join(cloud_facts_text.lower().split())
            is_cloud_dup = False
            for seen in seen_texts:
                words_a, words_b = set(cloud_normalized.split()), set(seen.split())
                if words_a and words_b:
                    overlap = len(words_a & words_b) / len(words_a | words_b)
                    if overlap > 0.75:
                        is_cloud_dup = True
                        break
            if not is_cloud_dup:
                unique_results.append(f"- {cloud_facts_text}")
                
        if not unique_results:
            return f"No relevant facts found in memory for {person.name if person else 'this person'}."
            
        result = f"Memories retrieved about {person.name if person else 'User'}:\n" + "\n".join(unique_results)
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




    def _tool_show_animal_animation(self, animal: str) -> str:
        """Shows an animal animation on the screen and plays the ACTUAL animal sound from the speaker. DO NOT use TTS to mimic the sound yourself (e.g. do not say "Meow" or "Woof"). Just say something natural like "Here it is!" or "Look at this!"."""
        import os
        import urllib.request
        from PIL import Image
        
        animal_lower = animal.lower().strip()
        assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "animals")
        os.makedirs(assets_dir, exist_ok=True)
        
        # Audio playback (if user provided real MP3s/WAVs/OGGs, play them)
        audio_path = os.path.join(assets_dir, f"{animal_lower}.mp3")
        wav_path = os.path.join(assets_dir, f"{animal_lower}.wav")
        ogg_path = os.path.join(assets_dir, f"{animal_lower}.ogg")
        audio_found = False
        if os.path.exists(audio_path):
            self.speaker.play_audio_file(audio_path, block=False)
            audio_found = True
        elif os.path.exists(ogg_path):
            self.speaker.play_audio_file(ogg_path, block=False)
            audio_found = True
        elif os.path.exists(wav_path):
            self.speaker.play_audio_file(wav_path, block=False)
            audio_found = True
        
        # Synthesize audio if missing
        if not audio_found:
            try:
                import wave, math, struct, random
                try:
                    from .assets.animal_registry import ANIMALS
                except ImportError:
                    try:
                        from lumi.assets.animal_registry import ANIMALS
                    except ImportError:
                        ANIMALS = {}
                        
                animal_data = ANIMALS.get(animal_lower, {'audio': 'bark'})
                audio_type = animal_data.get('audio', 'bark')
                
                def save_wav(filename, samples, sample_rate=44100):
                    with wave.open(filename, 'w') as f:
                        f.setnchannels(1)
                        f.setsampwidth(2)
                        f.setframerate(sample_rate)
                        for s in samples:
                            f.writeframesraw(struct.pack('<h', int(max(-32767, min(32767, s * 32767)))))
                
                sr = 44100
                samples = []
                
                if audio_type in ['chirp', 'squeak', 'eagle_cry']:
                    duration = 0.2
                    for i in range(int(sr * duration)):
                        t = i / sr
                        freq = 2000 + 3000 * (t / duration)
                        amp = math.sin(t * math.pi / duration)
                        samples.append(math.sin(2 * math.pi * freq * t) * amp)
                        
                elif audio_type in ['meow', 'howl']:
                    duration = 0.8
                    for i in range(int(sr * duration)):
                        t = i / sr
                        freq = 600 + (400 * (t / 0.3)) if t < 0.3 else 1000 - (500 * ((t - 0.3) / 0.5))
                        env = math.sin(t * math.pi / duration)
                        samples.append((math.sin(2 * math.pi * freq * t) + 0.3 * math.sin(2 * math.pi * freq * 2 * t)) * env * 0.8)
                        
                elif audio_type in ['roar', 'grunt', 'trumpet', 'moo']:
                    duration = 1.0
                    for i in range(int(sr * duration)):
                        t = i / sr
                        env = math.sin(t * math.pi / duration)
                        noise = random.uniform(-1, 1)
                        tone = math.sin(2 * math.pi * 150 * t)
                        samples.append((noise * 0.7 + tone * 0.3) * env * 0.9)
                        
                elif audio_type in ['bubble', 'quack', 'croak', 'honk', 'gobble', 'caw']:
                    duration = 0.4
                    for i in range(int(sr * duration)):
                        t = i / sr
                        env = math.exp(-t * 8) * math.sin(t * 10 * math.pi) # Repeating bursts
                        noise = random.uniform(-1, 1)
                        freq = 400 + 200 * math.sin(t * 50)
                        tone = math.sin(2 * math.pi * freq * t)
                        samples.append((noise * 0.2 + tone * 0.8) * max(0, env))
                        
                elif audio_type in ['hiss']:
                    duration = 0.5
                    for i in range(int(sr * duration)):
                        t = i / sr
                        env = math.sin(t * math.pi / duration)
                        noise = random.uniform(-1, 1)
                        samples.append(noise * env * 0.6)
                        
                else: # default to bark/misc
                    duration = 0.3
                    for i in range(int(sr * duration)):
                        t = i / sr
                        env = math.exp(-t * 15)
                        noise = random.uniform(-1, 1)
                        freq = 300 - (100 * (t / duration))
                        tone = math.sin(2 * math.pi * freq * t)
                        samples.append((noise * 0.4 + tone * 0.6) * env)
                
                save_wav(wav_path, samples)
                if os.path.exists(wav_path) and hasattr(self.speaker, "play_file"):
                    self.speaker.play_file(wav_path, block=False)
                    audio_found = True
            except Exception as e:
                logger.error(f"Failed to synthesize audio: {e}")

        # Display Procedural Animation directly on the display
        if hasattr(self.eyes, "show_procedural_animal"):
            try:
                self.eyes.show_procedural_animal(animal_lower, duration=4.0)
            except Exception as e:
                logger.error(f"Failed to play procedural animation: {e}")
                
        if audio_found:
            return f"Successfully displayed procedural {animal} animation and played synthesized sound. Acknowledge this playfully."
        else:
            return f"Successfully displayed procedural {animal} animation. Make a cute {animal} sound with your voice now!"

    def _tool_leave_message(self, recipient_name: str, sender_name: str, message_text: str) -> str:
        if not hasattr(self.memory, "leave_message"):
            return "Error: Messaging system not initialized."
        return self.memory.leave_message(recipient_name, sender_name, message_text)

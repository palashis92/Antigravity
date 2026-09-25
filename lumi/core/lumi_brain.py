"""LUMI Central Nervous System & Autonomous Brain."""

from __future__ import annotations

import math
import os
from pathlib import Path
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
from ..memory.models import ConsentStatus, utc_now_iso
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
        from ..audio.proximity_filter import ProximityAudioFilter
        self.vad = VoiceActivityDetector(aggressiveness=2)
        self.speaker_id = SpeakerIdentifier(self.memory, similarity_threshold=0.75)
        self.proximity_filter = ProximityAudioFilter()
        self._acoustic_overlap_active = False
        # Visual Tracking & Target-Lost Search state
        self._last_face_seen_time: float = 0.0
        self._last_face_exit_side: str = "center"  # "left", "right", or "center"
        self._had_tracked_face: bool = False
        self._search_phase: int = 0  # 0: tracking, 1: searched exit side, 2: scanned opposite side, 3: returned center
        self._last_search_move_time: float = 0.0
        self._last_speech_time: float = time.time()
        self._current_speaker: Optional[str] = None
        self._voice_buffer: bytearray = bytearray()  # Buffer for voice enrollment
        self._voice_buffer_lock = threading.Lock()
        self._enrolling_voice_for: Optional[str] = None  # Person ID being enrolled

        # Instant Local Acoustic Reflex (<150ms)
        self._last_acoustic_reflex_time: float = 0.0
        self._acoustic_reflex_cooldown: float = 1.2
        self._acoustic_reflex_enabled: bool = True

        self.tts = BanglaTTS()

        # Speech-to-Text & Meeting Subsystems
        from ..meeting import MeetingManager
        from ..speech.stt import BanglaSTT
        self.meeting_manager = MeetingManager(self.memory.db)
        self.stt = BanglaSTT()

        # WhatsApp Integrations Subsystems
        from ..integrations.whatsapp import WhatsAppClient
        from ..integrations.message_polisher import refine_whatsapp_message
        self.whatsapp = WhatsAppClient()
        self.refine_message = refine_whatsapp_message

        # Companion Subsystem (Speech Therapy for Anjum)
        from ..companion.anjum_companion import AnjumCompanionEngine
        self.anjum_companion = AnjumCompanionEngine(stimulus_cooldown=7.0)
        self._last_anjum_seen_time = 0.0
        self._anjum_consecutive_frames = 0

        # AI & Reasoning Subsystems
        self.tools = ToolRegistry()
        self.tools.register("memorize_person", self._tool_memorize_person, "Call this whenever ANY person introduces themselves (e.g. 'আমার নাম তানভীর', 'আমি পলাশ', 'এ হচ্ছে তানভীর', 'My name is X') or when the owner introduces a guest, friend, or colleague. This automatically captures their face from the camera and permanently remembers them.", {
            "type": "object", 
            "properties": {
                "name": {"type": "string", "description": "The person's full name."},
                "relationship": {"type": "string", "description": "Their relationship to the owner, e.g. friend, brother, guest, creator, colleague."},
                "age": {"type": "integer", "description": "The person's age if mentioned (e.g. 25)."},
                "notes": {"type": "string", "description": "Any short important facts or details to remember about them."}
            }, 
            "required": ["name"]
        })
        self.tools.register("analyze_plant", self._tool_analyze_plant, "Analyze the plant the camera is seeing.")
        self.tools.register("describe_vision", self._tool_describe_vision, "Describe what you currently see through the camera. Use this when someone asks what you see, or when you want to comment on surroundings.")
        self.tools.register("analyze_chess", self._tool_analyze_chess, "Analyze the chessboard the camera is seeing.")
        self.tools.register("control_body", self._tool_control_body, "Directly move LUMI's physical body parts (head, right_arm, left_arm, both_arms, or body/waist). ALWAYS call this immediately when the user commands you to move (e.g. 'তোমার হাতটা উপরে তোলো', 'মাথা নিচে নামাও', 'বডিটা ঘোরাও', 'ডানে তাকাও', 'হাত দিয়ে দেখাও'). Also use it naturally during conversation when you want to look at something or emphasize speech.", {
            "type": "object",
            "properties": {
                "part": {
                    "type": "string",
                    "enum": ["head", "right_arm", "left_arm", "both_arms", "body", "waist", "all"],
                    "description": "Which physical body part to move (or 'all' for simultaneous full-body motions)."
                },
                "action": {
                    "type": "string",
                    "enum": [
                        "raise", "lower", "wave", "point",
                        "look_up", "look_down", "look_left", "look_right", "look_center",
                        "turn_left", "turn_right", "center",
                        "nod", "shake", "tilt"
                    ],
                    "description": "The motion action to execute."
                },
                "angle_deg": {
                    "type": "number",
                    "description": "Optional specific movement angle in degrees (e.g. 15, 30, 45)."
                }
            },
            "required": ["part", "action"]
        })
        self.tools.register("perform_gesture", self._tool_perform_gesture, "Perform an expressive, human-like full-body gesture with LUMI's arms, head, and waist. Call this when greeting someone, celebrating, dancing, showing curiosity, or expressing emotion.", {
            "type": "object",
            "properties": {
                "gesture_name": {
                    "type": "string",
                    "enum": ["greet", "wave", "happy", "celebrate", "thinking", "curious", "excited", "sleep", "bored", "dance"],
                    "description": "The name of the expressive gesture."
                }
            },
            "required": ["gesture_name"]
        })
        self.tools.register("move_servo", self._tool_move_servo, "Move a specific servo directly by raw name and angle.", {
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
        self.tools.register("send_email", self._tool_send_email, "Send an email.", {
            "type": "object", "properties": {"to_address": {"type": "string"}, "subject": {"type": "string"}, "message": {"type": "string"}}, "required": ["to_address", "subject", "message"]
        })
        self.tools.register("send_whatsapp", self._tool_send_whatsapp, "Send a WhatsApp message to a contact name or phone number. Automatically refines and polishes the user's spoken words into polite, well-articulated Bengali before sending.", {
            "type": "object",
            "properties": {
                "recipient": {"type": "string", "description": "Contact person name (e.g. 'Rahim') or direct phone number."},
                "message": {"type": "string", "description": "The raw message text or instruction spoken by the user."},
                "polish_message": {"type": "boolean", "description": "Whether to use AI to polish the message into elegant, polite language (default: true)."}
            },
            "required": ["recipient", "message"]
        })
        self.tools.register("send_whatsapp_pdf", self._tool_send_whatsapp_pdf, "Send a PDF document (such as the latest meeting report) to a contact or phone number via WhatsApp.", {
            "type": "object",
            "properties": {
                "recipient": {"type": "string", "description": "Contact name or phone number to send the PDF to."},
                "document_path": {"type": "string", "description": "Optional file path of the PDF. If omitted, sends the latest meeting report PDF."}
            },
            "required": ["recipient"]
        })
        self.tools.register("memorize_fact", self._tool_memorize_fact, "Save a specific fact or detail about a person or event to long-term memory. Do this autonomously whenever you learn something important (e.g. user's hobbies, current tasks, preferences).", {
            "type": "object", "properties": {"fact": {"type": "string", "description": "The fact to remember (e.g. 'Mizan likes black coffee')."}, "person_name": {"type": "string", "description": "Optional name of the person this fact is about."}}, "required": ["fact"]
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
        self.tools.register("start_meeting_mode", self._tool_start_meeting_mode, "Activate Meeting Mode. Call this when the user says to start a meeting, enter meeting mode, or says they are starting a discussion. LUMI will remain silent, listen carefully to all attendees, and take detailed notes.", {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Optional title or topic of the meeting."}
            }
        })
        self.tools.register("stop_meeting_mode", self._tool_stop_meeting_mode, "Stop and conclude Meeting Mode. Call this when the user says the meeting is over, asks to stop meeting mode, or says 'meeting shesh'.", {
            "type": "object",
            "properties": {}
        })
        self.tools.register("analyze_meeting", self._tool_analyze_meeting, "Analyze the recorded meeting and provide a detailed, super-intelligent summary covering: who spoke, key points, decisions, and action items in Bengali. Call this when the user asks what happened in the meeting or asks for meeting analysis.", {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Specific question or request about the meeting (optional)."}
            }
        })
        self.tools.register("generate_meeting_report_pdf", self._tool_generate_meeting_report_pdf, "Generate a PDF document report of the latest meeting summary and minutes.", {
            "type": "object",
            "properties": {}
        })
        
        self.tools.register("set_silent_mode", self._tool_set_silent_mode,
            "CALL THIS when the user explicitly tells you to be silent/quiet/still/not talk for a specific time period "
            "(e.g. 'চুপ থাকো', 'কথা বলো না', 'শান্ত থাকো', 'নিশ্চল থাকো', 'বলো না'). "
            "During silent mode LUMI will stop all greetings, movements, and spontaneous speech until the specified time.",
            {
                "type": "object",
                "properties": {
                    "duration_seconds": {
                        "type": "number",
                        "description": "Number of seconds to remain silent. Use this OR silent_until_iso."
                    },
                    "silent_until_iso": {
                        "type": "string",
                        "description": "ISO 8601 datetime string until which to remain silent (e.g. '2026-09-17T23:56:00'). Use this OR duration_seconds."
                    }
                }
            }
        )

        self.tools.register("adapt_behavior", self._tool_adapt_behavior,
            "CALL THIS TOOL whenever the user gives feedback, advice, corrections, or instructions on how you should behave, speak, or perform actions "
            "(e.g. 'তোমার তো এভাবে কথা বলা উচিত', 'হাত এতো বেশি নাড়াবে না', 'তুমি আমাকে এই নামে ডাকবে', etc.). "
            "LUMI will analyze, permanently store this rule, and immediately adapt its behavior.",
            {
                "type": "object",
                "properties": {
                    "user_feedback": {
                        "type": "string",
                        "description": "The exact advice, correction, or behavioral instruction given by the user."
                    },
                    "adapted_rule": {
                        "type": "string",
                        "description": "Clear, concise imperative rule formulated from the user's advice."
                    },
                    "category": {
                        "type": "string",
                        "enum": ["behavior", "speech", "gesture", "general"],
                        "description": "Category: 'behavior', 'speech', 'gesture', or 'general'."
                    }
                },
                "required": ["user_feedback", "adapted_rule"]
            }
        )

        self.tools.register("start_presentation", self._tool_start_presentation,
            "CALL THIS TOOL ONLY when the user EXPLICITLY and unequivocally orders a formal timed speech, official lecture, or continuous presentation "
            "(e.g. '৫ মিনিট ভাষণ দাও', '১০ মিনিট বক্তব্য রাখো', '৩৬০ সেকেন্ড একটানা বলো', 'বক্তব্য শুরু করো', 'deliver a formal presentation'). "
            "CRITICAL: DO NOT CALL THIS TOOL for normal conversational questions, casual discussions, opinions, or requests like 'AI নিয়ে কিছু বলো' or 'রোবট সম্পর্কে বলো'. "
            "For all casual queries, conversational chats, and general questions, answer conversationally in 1-3 sentences directly WITHOUT calling this tool!",
            {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The specific topic or theme of the speech/presentation (e.g. 'গ্রাম উন্নয়ন', 'কৃষি ও পরিবেশ', 'ডিজিটাল সেবা')."
                    },
                    "duration_minutes": {
                        "type": "number",
                        "description": "Requested duration of speech in minutes (e.g. 1.0, 3.0, 5.0). Default is 3.0."
                    },
                    "audience": {
                        "type": "string",
                        "description": "Target audience (e.g. 'উপস্থিত গ্রামবাসী', 'ইউনিয়ন পরিষদের সদস্যবৃন্দ', 'কৃষক ভাই ও বোনেরা')."
                    },
                    "key_points": {
                        "type": "string",
                        "description": "Key points, specific decisions, or special instructions to emphasize in the speech."
                    }
                },
                "required": ["topic"]
            }
        )
        self.tools.register("stop_presentation", self._tool_stop_presentation,
            "CALL THIS TOOL when the user asks you to stop or halt your speech/presentation (e.g. 'বক্তব্য থামাও', 'লুমি থামো', 'stop presentation').",
            {
                "type": "object",
                "properties": {}
            }
        )

        from ..ai.gemini_live import GeminiLiveClient
        from ..audio.turn_arbiter import AudioTurnArbiter
        
        self.conversation = ConversationEngine(self.memory, self.tools)
        self.turn_arbiter = AudioTurnArbiter()
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
            turn_arbiter=self.turn_arbiter,
        )
        self.chess_engine = ChessAnalysisEngine()
        self.reminders = ReminderScheduler(self.memory, self.event_bus)
        self.documents = PDFReportGenerator()

        from ..speech.presentation import PresentationEngine
        self.presentation_engine = PresentationEngine(
            tts=self.tts,
            speaker=self.speaker,
            gestures=self.gestures,
            eyes=self.eyes,
            state=self.state,
            event_bus=self.event_bus,
            realtime_voice=self.realtime_voice,
            turn_arbiter=self.turn_arbiter,
        )
        # Wire bidirectional callbacks with Gemini Live
        self.realtime_voice._on_silence_cancelled_cb = lambda: self.cancel_silent_mode(reason="gemini_live_wake")
        self.realtime_voice._on_silence_activated_cb = lambda dur: self.set_silent_mode(duration_seconds=dur, reason="gemini_live_command")
        self.realtime_voice._on_presentation_requested_cb = lambda topic, dur: self._tool_start_presentation(topic=topic, duration_minutes=dur)
        
        # Dual-Mem0 System: Use Cloud API if key exists, otherwise use Local Gemini Engine
        import os
        from ..memory.mem0_cloud import Mem0CloudEngine
        if os.environ.get("MEM0_API_KEY"):
            logger.info("MEM0_API_KEY detected! Using official Mem0 Cloud API.")
            self.mem0 = Mem0CloudEngine(memory=self.memory)
        else:
            logger.info("No MEM0_API_KEY found. Falling back to native LumiMem0 Engine.")
            self.mem0 = LumiMem0Engine(self.memory)

        # Active interaction context (default to designated owner Mizan)
        self.active_person: Optional[Any] = self.get_owner()
        self._running = False
        self._perception_thread: Optional[threading.Thread] = None
        self._audio_thread: Optional[threading.Thread] = None
        # Silent mode: when set, suppress all greetings, gestures, and triggered injections until this timestamp
        self._silent_until: float = 0.0
        self._silent_mode_active: bool = False
        self._unknown_greeting_asked: bool = False
        self._last_unknown_greeting_time: float = 0.0
        self._last_speech_time: float = 0.0

        # Learned behavioral rules store and temporal tracking
        from ..memory.learned_rules import LearnedRulesStore
        data_dir = getattr(getattr(self, "settings", None), "app", None)
        data_dir_path = getattr(data_dir, "data_dir", "data") if data_dir else "data"
        db_instance = getattr(self.memory, "db", None)
        self.learned_rules = LearnedRulesStore(Path(data_dir_path) / "learned_rules.json", db=db_instance)
        self._person_first_seen_date: Dict[str, str] = {}
        self._person_last_greeting_time_str: Dict[str, str] = {}

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

    def get_owner(self) -> Any:
        """Find the designated owner of LUMI (by relationship='owner' or configured owner_name)."""
        if not getattr(self, "memory", None):
            return None
        try:
            for p in self.memory.list_people():
                if p.relationship and p.relationship.lower() == "owner":
                    return p
        except Exception:
            pass
        try:
            owner_name = getattr(getattr(getattr(self, "settings", None), "app", None), "owner_name", "Mizan")
            owner = self.memory.find_person_by_name(owner_name)
            if owner:
                return owner
            people = self.memory.list_people()
            return people[0] if people else None
        except Exception:
            return None

    @staticmethod
    def _extract_introduced_name(text: str) -> Optional[tuple[str, str]]:
        """Extract person name and optional relationship from conversational introductions.
        
        Examples:
            'আমার নাম তানভীর' -> ('তানভীর', 'friend')
            'আমি পলাশ' -> ('পলাশ', 'creator')
            'এ হচ্ছে আমার বন্ধু সাকিব' -> ('সাকিব', 'friend')
            'My name is Alex' -> ('Alex', 'guest')
        """
        import re
        t = text.strip()
        if not t:
            return None

        # Comprehensive exclusion of Bengali & English functional words, pronouns, verbs, adverbs, objects
        stopwords = {
            # Pronouns & determiners
            "আমি", "তুমি", "তুই", "আপনি", "সে", "তিনি", "তারা", "আমরা", "তোমরা", "আপনারা",
            "ইনি", "উনি", "এটা", "ওটা", "সেটা", "এগুলো", "ওগুলো", "যা", "তা", "যে", "কে",
            "কারা", "কেউ", "কাউকে", "কারো", "কার", "নিজেকে", "নিজে", "নিজেই", "নিজের",
            # Common verbs (all common inflections)
            "চলছে", "চলল", "চললো", "চলে", "করছি", "করছিলা", "করছিলে", "করছিলাম", "করবে",
            "করবেন", "করবা", "করব", "করবো", "করি", "করে", "করিস", "করতে", "করলে", "করলাম",
            "করার", "হইছে", "হচ্ছে", "হয়েছে", "হয়ে", "হয়েছিল", "হলো", "হল", "হবে", "হবেন",
            "হলে", "হওয়ার", "আছি", "আছো", "আছেন", "আছে", "ছিল", "ছিলেন", "ছিলাম", "ছিলে",
            "থাকা", "থাকি", "থাকো", "থাকেন", "থাকবে", "থাকলে", "যাই", "যাও", "যান", "যায়",
            "যাবে", "যাবেন", "যাব", "যাবো", "গেলাম", "গেল", "গেলা", "গেছে", "গিয়ে", "খাই",
            "খাও", "খান", "খায়", "খাবে", "খাবো", "খাব", "বলি", "বলো", "বলেন", "বলে", "বলবে",
            "বলবেন", "বলব", "বলবো", "বলছি", "বলতে", "শুনি", "শুনো", "শুনেন", "শুনে", "শুনবে",
            "শুনব", "শুনবো", "শুনছি", "শুনতে", "দেখছি", "দেখি", "দেখো", "দেখেন", "দেখে",
            "দেখবে", "দেখতে", "আসি", "আসো", "আসেন", "আসে", "আসবে", "আসব", "আসবো", "আসছি",
            "পারি", "পারো", "পারেন", "পারে", "পারব", "পারবো", "পারবে", "পারবেন", "জানি",
            "জানেন", "জানে", "জানবে", "জানব", "চাই", "চাও", "চায়", "চাইবে", "চাইব", "চাইনা",
            "চাইনি", "দিলে", "দিল", "দিলেন", "দিয়ে", "দিচ্ছে", "দেয়", "দেবে", "দেব", "দেবো",
            "দাও", "দিন", "নেন", "নাও", "নেয়", "নেবে", "নিলে", "রাখো", "রাখেন", "রাখে", "রাখবে",
            "বসো", "বসেন", "বসে", "পাঠাইছো", "পাঠাইছে", "ঘুমাই", "ঘুমায়", "ঘুমাবেন",
            # Common adverbs, particles, conjunctions, and objects
            "অলরেডি", "এখন", "তখন", "কখনো", "কখনোবা", "শুধু", "মাত্র", "ঠিক", "ভুল", "ভালো",
            "ভালোই", "খারাপ", "একটু", "একটা", "এক", "দুই", "তিন", "অনেক", "বেশি", "খুব",
            "আর", "এবং", "কিন্তু", "তবে", "তাই", "যাতে", "যেন", "কারণ", "কেন", "কোথায়",
            "কখন", "কিভাবে", "কেমন", "না", "নাই", "নেই", "নি", "তো", "মানুষ", "মালিক",
            "কোন", "কোনো", "কিছু", "কি", "কী", "সব", "সবাই", "সবসময়", "প্রশ্ন", "কথা",
            "ইতিহাস", "উত্তর", "বিষয়", "ক্যামেরা", "ছবি", "রোবট", "লুমি", "সাউন্ড", "চেহারা",
            "এখানে", "সেখানে", "কোথাও", "খালি", "পুরো", "একদম", "সাথে", "কাছে", "পাশে",
            # English common words
            "already", "fine", "good", "bad", "here", "there", "ready", "going", "doing",
            "speaking", "talking", "robot", "lumi", "yes", "no", "ok", "okay", "who", "what",
            "where", "when", "why", "how", "this", "that", "it", "is", "was", "are", "am",
            "not", "the", "my", "your", "his", "her", "their", "our", "just", "only"
        }

        # Pattern 1: 'আমার নাম <নাম>' or 'আমার পরিচয় <নাম>' or 'নাম হলো <নাম>'
        m = re.search(r"(?:আমার\s+নাম|আমার\s+পরিচয়|আমার\s+পরিচয়|নাম\s+হলো|নাম\s+হল)\s+([A-Za-z\u0980-\u09FF]+)", t, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            if name.lower() not in stopwords and len(name) >= 2:
                rel = "creator" if name.lower() in ["palash", "পলাশ"] else "friend"
                return (name, rel)

        # Pattern 2: Explicit third-person introduction
        # E.g. 'এ হচ্ছে আমার বন্ধু সাকিব', 'এটা আমার ভাই রাতুল', 'ওর নাম ফারহান', 'ইনি হচ্ছেন ডক্টর করিম'
        m = re.search(
            r"(?:(?:এ|এটা|ইনি)\s+(?:হচ্ছে|হল|হলেন|হচ্ছেন)\s+(?:আমার\s+)?(?:বন্ধু\s+|ভাই\s+|বোন\s+|সহকর্মী\s+|স্যার\s+)?|"
            r"(?:এ|এটা|ইনি)\s+(?:আমার\s+)(?:বন্ধু\s+|ভাই\s+|বোন\s+|সহকর্মী\s+|স্যার\s+)|"
            r"(?:ওর\s+নাম|এর\s+নাম|ইনার\s+নাম|উনার\s+নাম)\s+)([A-Za-z\u0980-\u09FF]+)",
            t, re.IGNORECASE
        )
        if m:
            name = m.group(1).strip()
            if name.lower() not in stopwords and len(name) >= 2:
                return (name, "friend")

        # Pattern 3: Strict self-introduction 'আমি <নাম>' / 'ami <name>'
        # Must be strictly 2 words (e.g. 'আমি তানভীর') or explicitly followed by 'বলছি' / 'হলাম'
        words = t.split()
        if len(words) == 2 and words[0].lower() in ["আমি", "ami"]:
            candidate = words[1].strip("।,!?")
            if candidate.lower() not in stopwords and len(candidate) >= 2:
                rel = "creator" if candidate.lower() in ["palash", "পলাশ"] else "friend"
                return (candidate, rel)
        elif len(words) <= 4 and ("বলছি" in t or "হলাম" in t):
            m = re.search(r"^(?:আমি|ami)\s+([A-Za-z\u0980-\u09FF]+)\s+(?:বলছি|হলাম)", t, re.IGNORECASE)
            if m:
                candidate = m.group(1).strip()
                if candidate.lower() not in stopwords and len(candidate) >= 2:
                    rel = "creator" if candidate.lower() in ["palash", "পলাশ"] else "friend"
                    return (candidate, rel)

        # Pattern 4: 'my name is <name>' / 'this is my friend <name>'
        m = re.search(r"(?:my\s+name\s+is|this\s+is\s+my\s+friend|this\s+is)\s+([A-Za-z]+)", t, re.IGNORECASE)
        if m and (len(words) <= 5 or "name is" in t.lower()):
            name = m.group(1).strip()
            if name.lower() not in stopwords and len(name) >= 2:
                rel = "creator" if name.lower() in ["palash"] else "friend"
                return (name, rel)

        return None

    @staticmethod
    def _detect_silence_command(text: str) -> Optional[float]:
        """Deterministic silence command detector.
        
        Matches commands like:
            'চুপ থাকো', '১০ মিনিট চুপ থাকো', '৫ মিনিট কথা বলবে না',
            'shut up', 'be quiet', 'stop talking', 'নিশ্চল থাকো'
        Returns duration in seconds, or None if not a silence command.
        """
        import re
        t = text.lower().strip()
        anti_silence = [
            r"(?:থাম(?:ার|বে|লে)?|চুপ\s*(?:করা|থাকা)?)\s*(?:র\s*)?(?:প্রয়োজন\s*নাই|প্রয়োজন\s*নাই|দরকার\s*নাই|লাগবে\s*না|হবে\s*না|নিষেধ)",
            r"(?:থাম(?:বে|িস)?\s*না|থামো\s*না|থামুন\s*না)",
            r"(?:চুপ\s*কর(?:ো|বেন)?\s*না|চুপ\s*থাকিও\s*না)",
            r"(?:ননস্টপ|nonstop|একটানা|লাগাতার|লগাতার)",
        ]
        if any(re.search(p, t) for p in anti_silence):
            return None

        silence_triggers = [
            "চুপ থাকো", "চুপ থাক", "চুপ কর", "চুপ করো", "চুপ রেখো",
            "কথা বলবে না", "কথা বলো না", "কথা বলিস না", "শান্ত থাকো", "নিশ্চল থাকো",
            "মুখ বন্ধ", "থামো", "be quiet", "shut up", "stop talking", "stay silent"
        ]
        if any(trig in t for trig in silence_triggers):
            # Check for duration in minutes
            m = re.search(r"(\d+|১|২|৩|৪|৫|৬|৭|৮|৯|১০)\s*(?:মিনিট|min|minute)", t)
            if m:
                val = m.group(1)
                bn_map = {"১": 1, "২": 2, "৩": 3, "৪": 4, "৫": 5, "৬": 6, "৭": 7, "৮": 8, "৯": 9, "১০": 10}
                mins = bn_map.get(val, int(val) if val.isdigit() else 5)
                return mins * 60.0
            return 300.0  # Default 5 minutes
        return None

    def _detect_motion_command(self, text: str) -> bool:
        """Deterministic physical motion command detector."""
        t = text.lower().strip()
        try:
            if any(k in t for k in ["হাত তোলো", "হাত উঠাও", "হাত উপরে", "raise hand"]):
                self._tool_control_body("both_arms", "raise")
                return True
            elif any(k in t for k in ["হাত নামাও", "হাত নিচে", "lower hand"]):
                self._tool_control_body("both_arms", "lower")
                return True
            elif any(k in t for k in ["মাথা নামাও", "নিচে তাকাও", "look down"]):
                self._tool_control_body("head", "look_down")
                return True
            elif any(k in t for k in ["ডানে তাকাও", "ডানে ঘোরো", "look right"]):
                self._tool_control_body("head", "look_right")
                return True
            elif any(k in t for k in ["বামে তাকাও", "বামে ঘোরো", "look left"]):
                self._tool_control_body("head", "look_left")
                return True
            elif any(k in t for k in ["নাচো", "ডান্স", "dance"]):
                self._tool_perform_gesture("dance")
                return True
            elif any(k in t for k in ["হাত নাড়াও", "হাই দাও", "wave"]):
                self._tool_perform_gesture("wave")
                return True
        except Exception as e:
            logger.debug(f"Motion command execution error: {e}")
        return False

    def _on_turn_complete(self, event: Event) -> None:
        u_text = event.data.get("user", "")
        l_text = event.data.get("lumi", "")

        # 1. Deterministic silence command interceptor
        if u_text:
            silence_dur = self._detect_silence_command(u_text)
            if silence_dur is not None:
                mins = int(silence_dur / 60)
                logger.info(f"🤫 Deterministic Silence Command activated: {mins} minutes.")
                self._tool_set_silent_mode(duration_seconds=silence_dur)
                import random
                quips = [
                    f"আচ্ছা, {mins} মিনিটের জন্য একদম মুখ বন্ধ করলাম!",
                    f"ঠিক আছে, শান্তিতে থাকুন, আমি {mins} মিনিট সম্পূর্ণ চুপ!",
                    "থাকলাম চুপ! পরে কিন্তু মিস করবেন না!"
                ]
                quip = random.choice(quips)
                audio_path = self.tts.synthesize(quip) if hasattr(self, "tts") and self.tts else None
                if audio_path and hasattr(self, "speaker") and self.speaker:
                    self.speaker.play_file(audio_path, block=True)
                return

            # 2. Deterministic motion command execution
            self._detect_motion_command(u_text)

        # Check for conversational introductions (e.g. 'আমার নাম তানভীর', 'আমি পলাশ')
        if u_text:
            try:
                intro = self._extract_introduced_name(u_text)
                if intro:
                    name, rel = intro
                    logger.info(f"[IDENTITY] Conversational introduction detected in user speech: name='{name}', rel='{rel}'")
                    self._tool_memorize_person(name=name, relationship=rel)
            except Exception as e:
                logger.debug(f"Introduction detection error: {e}")

        person = self.active_person
        if not person:
            # Default to owner (Mizan), or the primary registered person
            person = self.get_owner()
            self.active_person = person

        person_id = person.id if person else None

        # 1. Record conversation turn to SQLite conversations table
        if u_text:
            try:
                self.memory.record_turn(
                    speaker="user",
                    text=u_text,
                    person_id=person_id,
                    language="bn",
                    intent="chat",
                )
            except Exception as e:
                logger.error(f"Error recording user turn to database: {e}")

        if l_text:
            try:
                self.memory.record_turn(
                    speaker="lumi",
                    text=l_text,
                    person_id=person_id,
                    language="bn",
                    intent="chat",
                )
            except Exception as e:
                logger.error(f"Error recording lumi turn to database: {e}")

        # 2. Person-mention detection (Phase 2, Change 5)
        # If the user mentions another known person by name, auto-inject their facts
        if u_text and person:
            try:
                for p in self.memory.list_people():
                    if p.id == person.id:
                        continue  # Skip the active person (already have their context)
                    name_lower = p.name.lower()
                    first_name = name_lower.split()[0] if name_lower else ""
                    import re
                    pattern = rf"(?:\b|_){re.escape(name_lower)}(?:\b|_)"
                    matched = bool(re.search(pattern, u_text.lower()))
                    if not matched and len(first_name) >= 3:
                        first_pattern = rf"(?:\b|_){re.escape(first_name)}(?:\b|_)"
                        matched = bool(re.search(first_pattern, u_text.lower()))
                    if matched:
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

        # 3. Asynchronously extract semantic facts via Mem0 (only if user provided text)
        if u_text and u_text.strip() and person:
            try:
                self.mem0.process_conversation_turn_async(
                    person_id=person.id,
                    person_name=person.name,
                    user_text=u_text.strip(),
                    ai_text=l_text.strip()
                )
            except Exception as e:
                logger.error(f"Error in Mem0 async turn processing: {e}")

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
                # Use timezone-aware comparison to avoid UTC offset errors
                if scheduled_time.tzinfo is None:
                    from datetime import timezone as _tz
                    scheduled_time = scheduled_time.replace(tzinfo=_tz.utc)
                now = datetime.now(scheduled_time.tzinfo)
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
            now = time.time()
            # 1. Active silence expiration watchdog
            if getattr(self, "_silent_mode_active", False) and now >= getattr(self, "_silent_until", 0.0):
                self.cancel_silent_mode(reason="duration_elapsed")

            if not self.camera.is_available():
                time.sleep(1.0)
                continue
            now = time.time()
            if now - last_frame_time >= 0.15:
                last_frame_time = now
                frame = self.camera.get_frame()
                if frame is not None:
                    self.process_person_interaction(frame)

            # Fall back to designated owner if active person timed out (no face for 90s)
            if self.active_person and (now - self._last_face_seen_time > 90.0):
                owner = self.get_owner()
                if owner and getattr(self.active_person, "id", None) != owner.id:
                    logger.info(f"Active person '{self.active_person.name}' timed out. Resetting to owner '{owner.name}'.")
                    self.active_person = owner

            # Anjum Mode: 10s absence timeout & proactive stimulation
            if self.state.current_state == BehaviorState.ANJUM_MODE:
                if self.anjum_companion.is_timeout(now, timeout_sec=10.0):
                    logger.info("👧 Anjum not seen for > 10 seconds. Exiting Anjum Mode.")
                    self._exit_anjum_mode()
                else:
                    is_speaking = now < getattr(self.realtime_voice, "_speaker_active_until", 0.0)
                    if not is_speaking:
                        stimulus = self.anjum_companion.check_proactive_stimulus(now)
                        if stimulus:
                            logger.info(f"Proactive Anjum stimulus: {stimulus}")
                            self.eyes.set_expression("excited")
                            if hasattr(self.realtime_voice, "inject_context"):
                                self.realtime_voice.inject_context(
                                    f"[Child is quiet. Say warmly and simply in Bengali to encourage Anjum: '{stimulus}']",
                                    trigger_response=True,
                                )

            time.sleep(0.05)

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
        ENERGY_THRESHOLD = 150.0
        _debug_audio_frames = 0

        # Wire up VAD callbacks
        self.vad.set_on_utterance_complete(self._on_speech_utterance)
        self.vad.set_on_overlap_detected(self._on_overlap_detected)
        
        while self._running:

            chunk = self.mic.read_chunk(1024)
            if not chunk:
                time.sleep(0.01)
                continue

            # Software AEC Gating: check if robot speaker is currently active
            speaker_until = getattr(getattr(self, "realtime_voice", None), "_speaker_active_until", 0.0)
            is_speaker_active = (time.time() < speaker_until) or (hasattr(self, "speaker") and getattr(self.speaker, "is_playing", False))
            if hasattr(self, "turn_arbiter"):
                is_speaker_active = is_speaker_active or self.turn_arbiter.is_speaker_active()

            energy = self._compute_rms(chunk)
            _debug_audio_frames += 1
            if _debug_audio_frames % 200 == 0:
                logger.debug(f"Mic Audio RMS Energy: {energy:.1f}")

            # 1. Push audio to Gemini Live (with continuous real-time streaming matching cb1495d0)
            num_faces = len(getattr(self, '_last_detected_faces', []))
            is_overlap = (num_faces > 1) and getattr(self, '_acoustic_overlap_active', False)

            # Continuous streaming for Gemini Live neural VAD: gate on speaker playback (AEC), active presentation, or silence
            is_presenting = getattr(getattr(self, "presentation_engine", None), "is_presenting", lambda: False)()
            should_stream = not is_speaker_active and not is_presenting
            if hasattr(self, "turn_arbiter"):
                self.turn_arbiter.should_stream_mic(energy, is_overlap=is_overlap)

            if should_stream and self.proximity_filter.should_pass(chunk, is_overlap):
                if hasattr(self, "realtime_voice") and hasattr(self.realtime_voice, "push_audio_chunk"):
                    self.realtime_voice.push_audio_chunk(chunk)

            # 2. Feed chunk through VAD pipeline (only when speaker is not echoing)
            from ..audio.vad import SpeechEvent
            event = self.vad.process_chunk(chunk) if not is_speaker_active else None

            # 3. Collect audio for voice enrollment if active
            if self._enrolling_voice_for and not is_speaker_active:
                with self._voice_buffer_lock:
                    self._voice_buffer.extend(chunk)

            # 4. Instant Local Acoustic Reflex (<150ms) on speech onset
            if not is_speaker_active and not self._is_silent() and not is_presenting:
                if event in (SpeechEvent.SPEECH_START, SpeechEvent.SPEECH_CONTINUE) or energy >= ENERGY_THRESHOLD:
                    self._last_speech_time = time.time()
                    if hasattr(self, "turn_arbiter"):
                        self.turn_arbiter.wake_up(25.0)
                    if hasattr(self, "realtime_voice") and hasattr(self.realtime_voice, "wake_up"):
                        self.realtime_voice.wake_up(25.0)
                if event == SpeechEvent.SPEECH_START:
                    self.trigger_acoustic_reflex(energy=energy)

            # 5. Overlap detection (check periodically during speech)
            if not is_speaker_active and event == SpeechEvent.SPEECH_CONTINUE and _debug_audio_frames % 50 == 0:
                # Count visible faces from last perception loop
                num_faces = len(getattr(self, '_last_detected_faces', []))
                self.vad.detect_overlap(num_faces)
            
            time.sleep(0.01)

    def trigger_acoustic_reflex(self, energy: float = 0.0) -> bool:
        """Instantaneous local acoustic reflex (<150ms) upon detecting human voice onset.
        
        Zero network/cloud dependency. Instantly perks eyes up with alert affect,
        transitions to LISTENING state if IDLE, and initiates a subtle micro-nod
        acknowledging the speaker before cloud response arrives.
        """
        if not self._acoustic_reflex_enabled:
            return False
        now = time.time()
        if now - self._last_acoustic_reflex_time < self._acoustic_reflex_cooldown:
            return False

        if self._is_silent():
            return False

        # Software AEC: do not reflex on robot speaker self-echo
        speaker_until = getattr(getattr(self, "realtime_voice", None), "_speaker_active_until", 0.0)
        if time.time() < speaker_until or (hasattr(self, "speaker") and getattr(self.speaker, "is_playing", False)):
            return False
        if hasattr(self, "turn_arbiter") and self.turn_arbiter.is_speaker_active():
            return False

        self._last_acoustic_reflex_time = now

        # 1. Perk eyes with alert/curious affect (Arousal +0.7, Valence +0.3)
        if hasattr(self, "eyes") and self.eyes:
            if hasattr(self.eyes, "set_affect"):
                self.eyes.set_affect(valence=0.3, arousal=0.7)
            elif hasattr(self.eyes, "set_expression"):
                self.eyes.set_expression("curious")

        # 2. Subtle organic micro-nod acknowledging speaker
        if hasattr(self, "head") and self.head:
            try:
                self.head.tilt(2.0, duration_s=0.12)
            except Exception as e:
                logger.debug(f"Acoustic reflex head micro-nod note: {e}")

        # 3. Transition to LISTENING if idle
        if hasattr(self, "state") and self.state and self.state.current_state == BehaviorState.IDLE:
            self.state.transition_to(BehaviorState.LISTENING, reason="acoustic_reflex_onset")

        # 4. Open dialogue window for turn-taking
        if hasattr(self, "turn_arbiter"):
            self.turn_arbiter.wake_up(15.0)
        if hasattr(self, "realtime_voice") and hasattr(self.realtime_voice, "wake_up"):
            self.realtime_voice.wake_up(15.0)

        # 5. Telemetry logging
        try:
            from .telemetry import get_telemetry_logger
            tel = get_telemetry_logger()
            tel.record_event("TEL-01", {"source": "acoustic_reflex", "energy": energy, "timestamp": now})
        except Exception:
            pass

        logger.debug(f"⚡ Instant Local Acoustic Reflex triggered (<150ms, energy={energy:.1f}).")
        return True

    def _on_speech_utterance(self, audio_bytes: bytes, duration: float) -> None:
        """Called by VAD when a complete speech utterance is ready.
        
        Runs speaker identification in a background thread to avoid
        blocking the audio loop.
        """
        def _identify():
            # If we're enrolling a voice, skip identification
            if self._enrolling_voice_for:
                return

            # If meeting mode is active, handle utterance logging silently
            if getattr(self, "meeting_manager", None) and self.meeting_manager.is_meeting_active():
                self._process_meeting_utterance(audio_bytes, duration)
                return

            # If presentation mode is active, check specifically for emergency stop commands
            if getattr(self, "presentation_engine", None) and self.presentation_engine.is_presenting():
                if hasattr(self, "stt"):
                    try:
                        text = self.stt.transcribe_pcm_bytes(audio_bytes)
                        if text:
                            logger.info(f"🎤 Utterance during presentation: '{text}'")
                            if self.presentation_engine.check_stop_command(text):
                                logger.info(f"🛑 Presentation stop trigger matched in speech: '{text}'")
                                self.presentation_engine.stop_presentation(reason="voice_stop_command")
                                return
                    except Exception as e:
                        logger.debug(f"Presentation stop check STT error: {e}")
                return

            # If silence mode is active, check specifically for voice wake/un-mute commands
            if self._is_silent():
                if hasattr(self, "stt"):
                    try:
                        text = self.stt.transcribe_pcm_bytes(audio_bytes)
                        if text:
                            logger.info(f"🎤 Utterance during silence mode: '{text}'")
                            clean = text.lower().strip()
                            wake_triggers = ["কথা বলো", "জেগে ওঠো", "লুমি কথা বলো", "লুমি", "শুনতে পাচ্ছ", "অন হও", "wake up", "start talking"]
                            if any(trig in clean for trig in wake_triggers):
                                logger.info(f"Voice wake trigger detected during silence: '{text}'")
                                self.cancel_silent_mode(reason="voice_wake_command")
                                return
                    except Exception as e:
                        logger.debug(f"Silence wake check STT error: {e}")
                return

            # If Anjum Mode is active, mark speech activity to reset stimulus cooldown
            if self.state.current_state == BehaviorState.ANJUM_MODE:
                self.anjum_companion.mark_speech_activity(time.time())
                self.eyes.set_expression("excited")
                return

            if not self.speaker_id.is_available():
                return

            try:
                speaker_name, confidence = self.speaker_id.identify_speaker(audio_bytes)
            except Exception as e:
                logger.error(f"Speaker identification error in background thread: {e}")
                return

            now_t = time.time()
            self._last_speech_time = now_t

            if speaker_name and confidence >= 0.75:
                logger.info(f"[VOICE] detected name={speaker_name} confidence={confidence:.2f}")
                face_in_view = (now_t - getattr(self, "_last_face_seen_time", 0.0)) < 4.0
                
                # Visual identity always takes precedence when a known face is confirmed
                if face_in_view and self.active_person:
                    if speaker_name.lower() != self.active_person.name.lower():
                        logger.info(
                            f"[VOICE] Ignoring speaker ID '{speaker_name}' because active face "
                            f"'{self.active_person.name}' is in view."
                        )
                        return
                elif face_in_view and not self.active_person:
                    # Multi-modal fusion: Face is looking at camera, voice ID identified the speaker!
                    matched_p = self.memory.find_person_by_name(speaker_name)
                    if matched_p:
                        self.active_person = matched_p
                        self._unknown_greeting_asked = False
                        logger.info(
                            f"[VOICE] Multi-modal fusion: Attributed active face to recognized voice speaker '{speaker_name}'."
                        )
                        # Link pending face embedding to this speaker if available
                        if hasattr(self.face_service, "get_pending_face"):
                            pending_emb = self.face_service.get_pending_face()
                            if pending_emb:
                                matched_p.add_face_embedding(pending_emb)
                                self.memory.update_person(matched_p)
                                logger.info(f"[VOICE] Associated pending face embedding with '{speaker_name}'.")


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

    def _process_meeting_utterance(self, audio_bytes: bytes, duration: float) -> None:
        """Process speech segment during active meeting mode."""
        # 1. Identify speaker
        speaker_name = "Unknown"
        confidence = 0.0
        if self.speaker_id.is_available():
            s_name, s_conf = self.speaker_id.identify_speaker(audio_bytes)
            if s_name and s_conf >= 0.70:
                speaker_name = s_name
                confidence = s_conf

        # 2. Estimate direction of arrival from ReSpeaker 2-Mic HAT if speaker unknown
        doa = None
        if hasattr(self.mic, 'backend') and hasattr(self.mic.backend, 'spatial_processor'):
            spatial = self.mic.backend.spatial_processor
            if spatial:
                doa = spatial.current_doa

        if speaker_name == "Unknown":
            if doa is not None:
                if doa < -20:
                    speaker_name = "বক্তা (বামদিক)"
                elif doa > 20:
                    speaker_name = "বক্তা (ডানদিক)"
                else:
                    speaker_name = "বক্তা (মাঝখান)"
            else:
                speaker_name = "বক্তা"

        # 3. Transcribe speech using Whisper STT
        text = ""
        if hasattr(self, "stt"):
            try:
                text = self.stt.transcribe_pcm_bytes(audio_bytes)
            except Exception as e:
                logger.debug(f"Meeting STT error: {e}")

        if not text:
            return

        # 4. Local voice command check: Stop meeting
        clean = text.lower().strip()
        stop_triggers = ["মিটিং শেষ", "মিটিং মোড বন্ধ", "মিটিং বন্ধ", "মিটিং সমাপ্ত", "stop meeting", "end meeting"]
        if any(trig in clean for trig in stop_triggers):
            logger.info(f"🛑 Meeting stop trigger detected in speech: '{text}'")
            self._tool_stop_meeting_mode()
            return

        # 5. Append to active meeting record
        self.meeting_manager.add_utterance(
            speaker=speaker_name,
            text=text,
            doa_deg=doa,
            confidence=confidence,
        )

    def _on_overlap_detected(self) -> None:
        """Called by VAD when overlapping speech from multiple people is detected."""
        if getattr(self, "meeting_manager", None) and self.meeting_manager.is_meeting_active():
            return  # Silent in meeting mode

        logger.info("🔊 Overlapping speech detected! Activating proximity near-field filter.")
        self._acoustic_overlap_active = True

        # Auto-reset overlap flag after 6 seconds
        def _reset_overlap():
            time.sleep(6.0)
            self._acoustic_overlap_active = False

        threading.Thread(target=_reset_overlap, daemon=True, name="ResetOverlapFlag").start()

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

    def _arm_greeting_watchdog(self, timeout_s: float = 10.0) -> None:
        """Watchdog timer to prevent LUMI from being permanently stuck in GREETING state.
        
        If Gemini Live suppresses greeting audio (e.g. because user was speaking simultaneously),
        this timer safely transitions LUMI back to LISTENING after timeout_s seconds.
        """
        def _watchdog_worker():
            time.sleep(timeout_s)
            if hasattr(self, "state") and getattr(self.state, "current_state", None) == BehaviorState.GREETING:
                logger.info(f"[GREETING] Watchdog timer ({timeout_s}s) elapsed without speech. Returning to LISTENING.")
                self.state.transition_to(BehaviorState.LISTENING, reason="greeting_watchdog_timeout")

        import threading
        threading.Thread(target=_watchdog_worker, daemon=True, name="GreetingWatchdog").start()

    def process_person_interaction(self, face_frame: Any) -> None:
        """Autonomous visual pipeline: detect person -> track -> greet -> engage."""
        faces = self.face_service.detect_and_recognize(face_frame)
        if not faces:
            self._last_detected_faces = []
            now = time.time()

            # Suppress head search panning during active conversation or presentation
            is_active_dialogue = (
                (hasattr(self, "turn_arbiter") and self.turn_arbiter.is_in_dialogue())
                or (hasattr(self, "state") and self.state.current_state in (
                    BehaviorState.LISTENING,
                    BehaviorState.SPEAKING,
                    BehaviorState.THINKING,
                    BehaviorState.PRESENTING,
                    BehaviorState.GREETING,
                    BehaviorState.OBSERVING,
                ))
            )
            if is_active_dialogue:
                return

            # Check search cooldown (prevent head oscillation loop)
            if (now - getattr(self, "_last_search_complete_time", 0.0)) < 8.0:
                return

            # If we were tracking someone who just left the camera frame:
            if self._had_tracked_face and not getattr(self.gestures, "is_playing", False):
                time_since_lost = now - self._last_face_seen_time

                # Phase 1: Target walked out of camera view (2.0s - 4.5s ago)
                # Slower, intentional pan (1.2s) eliminates motion blur and head hunting
                if 2.0 <= time_since_lost < 4.5 and self._search_phase == 0:
                    self._search_phase = 1
                    self._last_search_move_time = now
                    from ..core.telemetry import get_telemetry
                    get_telemetry().record_event("TEL-07", context="search_phase_1")

                    # Determine exit direction:
                    if self._last_face_exit_side == "right" or self.head.current_pan < -10.0:
                        search_pan = -55.0  # Turn right (safe within -70°)
                        gaze_x = 0.8
                    elif self._last_face_exit_side == "left" or self.head.current_pan > 10.0:
                        search_pan = 55.0   # Turn left (safe within +70°)
                        gaze_x = -0.8
                    else:
                        search_pan = -45.0  # Default right scan
                        gaze_x = 0.6

                    logger.info(f"👀 Face moved out of frame ({self._last_face_exit_side}). Smoothly panning to {search_pan:.1f}° to reacquire target...")
                    self.eyes.set_expression("curious")
                    self.eyes.set_gaze(gaze_x, 0.0)
                    self.head.pan(search_pan, duration_s=1.2)

                # Phase 2: Still not found after ~4.5s - 8.0s
                # Pan to opposite direction smoothly (1.2s)
                elif 4.5 <= time_since_lost < 8.0 and self._search_phase == 1 and (now - self._last_search_move_time >= 2.0):
                    self._search_phase = 2
                    self._last_search_move_time = now
                    from ..core.telemetry import get_telemetry
                    get_telemetry().record_event("TEL-07", context="search_phase_2")

                    # Opposite direction
                    if self.head.current_pan < 0:
                        opposite_pan = 50.0  # Turn left
                        gaze_x = -0.7
                    else:
                        opposite_pan = -50.0  # Turn right
                        gaze_x = 0.7

                    logger.info(f"🔍 Face not found in exit direction. Scanning opposite side ({opposite_pan:.1f}°)...")
                    self.eyes.set_expression("thinking")
                    self.eyes.set_gaze(gaze_x, 0.0)
                    self.head.pan(opposite_pan, duration_s=1.2)

                # Phase 3: Still no one after 8.0s
                # Return smoothly to center/home and rest
                elif time_since_lost >= 8.0 and self._search_phase == 2 and (now - self._last_search_move_time >= 2.5):
                    self._search_phase = 3
                    self._had_tracked_face = False
                    self._last_search_complete_time = now
                    from ..core.telemetry import get_telemetry
                    get_telemetry().record_event("TEL-07", context="search_phase_3_home")
                    logger.info("🏠 Search complete. No face detected. Returning head and gaze to center.")
                    self.eyes.set_expression("neutral")
                    self.eyes.set_gaze(0.0, 0.0)
                    self.head.look_center(duration_s=1.0)
            return

        self._last_detected_faces = faces

        faces = self.face_service.confirm_identity(faces)
        if not faces:
            return

        # Prioritize recognized known faces over unknown, then sort by highest confidence
        faces.sort(key=lambda f: (not f.is_known, -f.confidence))
        face = faces[0]
        now = time.time()
        self._last_face_seen_time = now
        self._had_tracked_face = True
        self._search_phase = 0  # Face is acquired, reset search state

        # Open dialogue window so LUMI is ready to converse when face is in view
        if hasattr(self, "turn_arbiter"):
            self.turn_arbiter.wake_up(15.0)
        if hasattr(self, "realtime_voice") and hasattr(self.realtime_voice, "wake_up"):
            self.realtime_voice.wake_up(15.0)

        # Track exit side
        frame_w = self.settings.vision.frame_width
        frame_h = self.settings.vision.frame_height
        if face.center[0] > (frame_w * 0.55):
            self._last_face_exit_side = "right"
        elif face.center[0] < (frame_w * 0.45):
            self._last_face_exit_side = "left"
        else:
            self._last_face_exit_side = "center"

        person_data = {
            "name": face.person.name if face.is_known and face.person else "Unknown",
            "is_known": face.is_known,
        }
        self.event_bus.emit("vision.face_detected", data={"person": person_data}, source="vision")

        # Smoothly track face with head servos (strictly constrained by safe limits)
        self.head.track_bounding_box(face.center[0], face.center[1], frame_w=frame_w, frame_h=frame_h)
        
        # Track face with procedural eyes
        gaze_x = (face.center[0] / frame_w) * 2.0 - 1.0
        gaze_y = (face.center[1] / frame_h) * 2.0 - 1.0
        self.eyes.set_gaze(gaze_x, gaze_y)

        # Steer microphone beamformer towards the active face
        if hasattr(self.mic, 'backend') and hasattr(self.mic.backend, 'spatial_processor'):
            spatial = self.mic.backend.spatial_processor
            if spatial:
                spatial.steer_towards_face(face.center[0], self.settings.vision.frame_width)

        # In meeting mode, keep tracking face & gaze silently, but do not interrupt with greetings
        if getattr(self, "meeting_manager", None) and self.meeting_manager.is_meeting_active():
            return

        # Check for Anjum (Speech-Therapy & Companion Mode)
        name_lower = face.person.name.lower().strip() if (face.is_known and face.person) else ""
        is_anjum = ("anjum" in name_lower or "আঞ্জুম" in name_lower) and face.is_known and getattr(face, "identity_state", None) == IdentityState.RECOGNIZED

        if is_anjum:
            self._anjum_consecutive_frames += 1
            now_t = time.time()
            self._last_anjum_seen_time = now_t
            self._last_face_seen_time = now_t
            self.anjum_companion.mark_seen(now_t)
            self.active_person = face.person
            # Require 6 consecutive confirmed frames before entering ANJUM_MODE to prevent false triggers
            if self.state.current_state != BehaviorState.ANJUM_MODE and self._anjum_consecutive_frames >= 6:
                is_speaking = now_t < getattr(self.realtime_voice, "_speaker_active_until", 0.0)
                if not is_speaking:
                    self._enter_anjum_mode()
            return
        else:
            self._anjum_consecutive_frames = 0

        # While in Anjum mode:
        if self.state.current_state == BehaviorState.ANJUM_MODE:
            # If a known adult is detected, immediately exit Anjum mode
            if face.is_known and face.person is not None:
                p_age = getattr(face.person, "age", None)
                if p_age is None or p_age > 12:
                    logger.info(f"Adult '{face.person.name}' detected in Anjum mode. Exiting Anjum Mode.")
                    self._exit_anjum_mode()
                    # Fall through to standard adult greeting/interaction below
                else:
                    # Known child face maintains presence
                    now_t = time.time()
                    self._last_anjum_seen_time = now_t
                    self._last_face_seen_time = now_t
                    self.anjum_companion.mark_seen(now_t)
                    return
            else:
                # In Anjum mode, an unknown or unconfirmed face does not reset timeout
                return

        if face.is_known and face.person is not None:
            person = face.person
            now_t = time.time()
            if not hasattr(self, "_last_profile_log_time"):
                self._last_profile_log_time = {}
            last_prof_t = self._last_profile_log_time.get(person.id, 0.0)
            if (now_t - last_prof_t) > 10.0:
                self._last_profile_log_time[person.id] = now_t
                logger.info(
                    f"[MEMORY] loading profile for person_id={person.id} name='{person.name}' age={person.age}"
                )
                logger.info(
                    f"[RESPONSE] recognized_person='{person.name}' confidence={face.confidence:.2f}"
                )
            # Cooldown check (50 minutes / 3000s default to prevent annoying repetitive interruptions)
            vision_conf = getattr(getattr(self, "settings", None), "vision", None)
            cooldown_val = getattr(vision_conf, "greeting_cooldown_s", 3000.0) if vision_conf else 3000.0

            # If LUMI is in silent mode, skip all greeting speech, gestures, and animations
            # Checked BEFORE should_interact so person's cooldown is NOT burned during silence!
            if self._is_silent():
                logger.debug(f"Silent mode active: suppressing greeting for {person.name}.")
                return

            # Do NOT interrupt if LUMI is actively vocalizing (speaking), delivering a presentation, in a meeting, or already greeting
            if self.state.current_state in (
                BehaviorState.SPEAKING,
                BehaviorState.PRESENTING,
                BehaviorState.THINKING,
                BehaviorState.GREETING,
                BehaviorState.MEETING,
            ) or (
                hasattr(self, "speaker") and getattr(self.speaker, "is_playing", False)
            ):
                return

            # If user is currently in the middle of speaking into the microphone, do not interrupt their speech
            if hasattr(self, "vad"):
                is_spk = getattr(self.vad, "is_speaking", None)
                if callable(is_spk) and is_spk():
                    return
                elif hasattr(self.vad, "is_speech_active"):
                    act = getattr(self.vad, "is_speech_active")
                    if (callable(act) and act()) or (not callable(act) and bool(act)):
                        return

            # If already in an active dialogue with this SAME person, do not re-greet them mid-conversation
            is_same_active_person = (
                getattr(self, "active_person", None) is not None
                and self.active_person.id == person.id
            )
            if is_same_active_person and hasattr(self, "turn_arbiter") and self.turn_arbiter.is_in_dialogue():
                return

            if self.face_service.should_interact(person.id, cooldown_s=cooldown_val):
                self.active_person = person
                self.state.transition_to(BehaviorState.GREETING, reason=f"spot_{person.name}")
                self._arm_greeting_watchdog(10.0)
                self.eyes.set_expression("happy")
                self.gestures.play_async(self.gestures.greet, name="greet")
                
                if hasattr(self, "turn_arbiter"):
                    self.turn_arbiter.wake_up(20.0)
                if hasattr(self, "realtime_voice") and hasattr(self.realtime_voice, "wake_up"):
                    self.realtime_voice.wake_up(20.0)
                
                relationship = person.relationship if hasattr(person, 'relationship') else 'friend'
                notes = person.notes if hasattr(person, 'notes') and person.notes else 'None'
                
                fact_str = "None"
                if hasattr(self.mem0, "recall_facts_sync"):
                    fact_str = self.mem0.recall_facts_sync(person.id)
                else:
                    recent_facts = self.memory.recall_facts(person_id=person.id)
                    if recent_facts:
                        fact_str = ", ".join([f.fact_text for f in recent_facts[:3]])

                if fact_str and fact_str != "None":
                    # Filter out stale silence commands so Gemini doesn't mistake them for active orders
                    filtered = [f for f in fact_str.split(". ") if not any(w in f.lower() for w in ["চুপ", "silent", "quiet", "shut up", "মিনিট"])]
                    fact_str = ". ".join(filtered) if filtered else "None"
                
                # Check for unread messages
                unread_msgs = ""
                if hasattr(self.memory, "get_unread_messages"):
                    msgs = self.memory.get_unread_messages(person.id)
                    if msgs:
                        msg_texts = [f"From {m['sender_name']}: {m['message_text']}" for m in msgs]
                        unread_msgs = f"\nURGENT: YOU HAVE UNREAD MESSAGES FOR {person.name}: {', '.join(msg_texts)}. YOU MUST TELL THEM THIS MESSAGE IMMEDIATELY AS SOON AS YOU GREET THEM!"
                        self.memory.mark_messages_read(person.id)

                # Real-world temporal context awareness
                import datetime
                now_dt = datetime.datetime.now()
                time_str = now_dt.strftime("%I:%M %p")
                today_str = now_dt.strftime("%Y-%m-%d")
                hour = now_dt.hour
                if 5 <= hour < 12:
                    time_period = "সকাল (Morning)"
                elif 12 <= hour < 17:
                    time_period = "দুপুর / বিকাল (Afternoon)"
                elif 17 <= hour < 20:
                    time_period = "সন্ধ্যা (Evening)"
                else:
                    time_period = "রাত (Night)"

                is_first_today = (self._person_first_seen_date.get(person.id) != today_str)
                if is_first_today:
                    self._person_first_seen_date[person.id] = today_str
                    encounter_context = (
                        f"This is the FIRST TIME you are seeing {person.name} today. "
                        f"Current local time: {time_str} ({time_period}). "
                        f"Greet them warmly with an appropriate greeting for this time of day in conversational Bengali."
                    )
                else:
                    last_time_seen = self._person_last_greeting_time_str.get(person.id, "earlier today")
                    encounter_context = (
                        f"You already saw and greeted {person.name} earlier today (around {last_time_seen}). "
                        f"Current local time: {time_str} ({time_period}). "
                        f"CRITICAL: Do NOT repeat an introductory greeting or morning greeting of the day. Acknowledge seeing them again casually and naturally "
                        f"in Bengali (e.g., 'আরে {person.name}, আবার দেখা হলো! সব কেমন চলছে?' বা 'কোনো কাজ আছে নাকি?')."
                    )
                self._person_last_greeting_time_str[person.id] = time_str
                
                prompt = (
                    f"[VISUAL EVENT: You just saw {person.name} ({relationship}) in front of the camera right now!]\n"
                    f"TEMPORAL CONTEXT: {encounter_context}\n"
                    f"Notes about them: {notes}. "
                    f"Recent memories: {fact_str}. {unread_msgs}\n"
                    f"INSTRUCTION: Greet {person.name} immediately, warmly, and naturally in conversational Bengali (বাংলা). "
                    "Do not mention reading notes or memories mechanically. "
                    "CRITICAL RULE: Do NOT ask if they have a message for the owner unless they explicitly ask for them. "
                    "Focus on greeting them naturally as a familiar friend! Do not repeat previous greetings or phrases."
                )
                
                is_gemini_ready = getattr(self.realtime_voice, "_is_ready", False) and getattr(self.realtime_voice, "_ws", None)
                if is_gemini_ready:
                    self.realtime_voice.inject_context(prompt, trigger_response=True)
                elif hasattr(self, "tts") and hasattr(self, "speaker"):
                    import random
                    owner_obj = self.get_owner()
                    owner_first = owner_obj.name.lower() if owner_obj else "mizan"
                    if is_first_today:
                        if 5 <= hour < 12:
                            time_greeting = f"শুভ সকাল {person.name}! কেমন আছেন?"
                        elif 12 <= hour < 17:
                            time_greeting = f"শুভ দুপুর {person.name}! দিন কেমন কাটছে?"
                        elif 17 <= hour < 20:
                            time_greeting = f"শুভ সন্ধ্যা {person.name}! কেমন আছেন?"
                        else:
                            time_greeting = f"হ্যালো {person.name}! এতো রাতেও জেগে আছেন? কেমন আছেন?"
                        local_greetings = [
                            time_greeting,
                            f"হ্যালো {person.name}! আপনাকে দেখে খুব ভালো লাগলো!",
                            f"এই যে {person.name}! কেমন কাটছে দিন?"
                        ]
                    else:
                        local_greetings = [
                            f"আরে {person.name}! আবার দেখা হলো! সব ঠিকঠাক?",
                            f"এই যে {person.name}! কোনো সাহায্য লাগবে?",
                            f"{person.name} ভাই, আবার আসলেন? কোনো দরকার?" if owner_first in person.name.lower() else f"আরে {person.name}! সব কেমন চলছে?"
                        ]
                    greeting_text = random.choice(local_greetings)
                    audio_path = self.tts.synthesize(greeting_text)
                    if audio_path:
                        self.speaker.play_file(audio_path, block=False)
                    self.state.transition_to(BehaviorState.IDLE, reason="greeting_complete")
        else:
            # Unrecognized face in current frame
            owner = self.get_owner()

            # 1. Universal Continuous Face Learning:
            # If an active person is established (Mizan, Palash, or any introduced friend/guest),
            # automatically capture and store multiple face samples (up to 5) for angle/lighting robustness,
            # but ONLY if the face embedding is verified to be close (dist <= 0.58) to their existing face.
            target_person = self.active_person or owner
            if target_person and face.embedding:
                t_lower = target_person.name.strip().lower()
                invalid_names = {"চলছে", "চলল", "চললো", "চলে", "অলরেডি", "আমি", "তুমি", "তুই", "সে", "তিনি", "এটা", "ওটা", "সেটা"}
                if t_lower not in invalid_names:
                    stored = getattr(target_person, "face_embeddings", [])
                    if len(stored) == 0:
                        target_person.add_face_embedding(face.embedding)
                        self.memory.update_person(target_person)
                        self.active_person = target_person
                        logger.info(f"[IDENTITY] Initial face sample enrolled for '{target_person.name}'.")
                        return
                    elif len(stored) < 5:
                        try:
                            import face_recognition
                            import numpy as np
                            dists = face_recognition.face_distance(stored, np.array(face.embedding))
                            min_d = float(dists.min())
                        except Exception:
                            import math
                            min_d = min(
                                math.sqrt(sum((a - b) ** 2 for a, b in zip(s, face.embedding)))
                                for s in stored
                            )
                        if min_d <= 0.58:
                            target_person.add_face_embedding(face.embedding)
                            self.memory.update_person(target_person)
                            self.active_person = target_person
                            logger.info(f"[IDENTITY] Auto-enrolled verified face sample ({len(stored)+1}/5, dist={min_d:.3f}) for '{target_person.name}'.")
                            return
                        else:
                            logger.debug(f"[IDENTITY] Face dist={min_d:.3f} > 0.58 to '{target_person.name}'. Not enrolling to avoid identity pollution.")

            # 2. Maintain sticky active_person across transient frame misses / lighting shifts
            if self.active_person is not None:
                logger.debug(f"[IDENTITY] Maintaining active_person '{self.active_person.name}' across frame variation.")
                return

            # 3. Default to owner Mizan in their own space
            if owner:
                self.active_person = owner
                logger.debug(f"[IDENTITY] Defaulted active_person to owner '{owner.name}'.")
                return

            # 4. If identity is LOW_CONFIDENCE, wait for multi-frame voting to stabilize
            from ..vision.face import IdentityState
            if getattr(face, "identity_state", None) == IdentityState.LOW_CONFIDENCE:
                logger.debug("[IDENTITY] Face in LOW_CONFIDENCE state — awaiting temporal stabilization.")
                return

            # 5. Store pending unknown face for potential enrollment
            if hasattr(self.face_service, "set_pending_face"):
                self.face_service.set_pending_face(face.embedding)
            
            now_t = time.time()
            unknown_vision_conf = getattr(getattr(self, "settings", None), "vision", None)
            unknown_cooldown = getattr(unknown_vision_conf, "unknown_greeting_cooldown_s", 7200.0) if unknown_vision_conf else 7200.0

            # 6. Check silent mode
            if self._is_silent():
                logger.info("Silent mode active: suppressing unknown-person greeting.")
                return

            # 7. Check ongoing conversation activity (< 120s)
            voice_last_active = getattr(getattr(self, "realtime_voice", None), "_last_active_time", 0.0)
            recent_speech_t = max(getattr(self, "_last_speech_time", 0.0), voice_last_active)
            if (now_t - recent_speech_t) < 120.0:
                logger.debug("[IDENTITY] Active conversation in progress — suppressing stranger greeting.")
                return

            # 8. Check if already asked in this session/encounter
            if getattr(self, "_unknown_greeting_asked", False):
                logger.debug("[IDENTITY] Unknown person already greeted in this session — suppressing repetitive prompt.")
                return

            # 9. Check cooldown (2 hours / 7200s)
            if (now_t - getattr(self, "_last_unknown_greeting_time", 0.0)) < unknown_cooldown:
                return

            self._last_unknown_greeting_time = now_t
            self._unknown_greeting_asked = True
            self.state.transition_to(BehaviorState.GREETING, reason="spot_unknown")
            self._arm_greeting_watchdog(10.0)
            self.eyes.set_expression("curious")
            self.gestures.play_async(self.gestures.greet, name="greet_unknown")
            
            if hasattr(self, "turn_arbiter"):
                self.turn_arbiter.wake_up(20.0)
            if hasattr(self, "realtime_voice") and hasattr(self.realtime_voice, "wake_up"):
                self.realtime_voice.wake_up(20.0)
            import random
            unknown_prompts = [
                (
                    "[VISUAL EVENT: A person has appeared in front of your camera.]\n"
                    "INSTRUCTION: Greet them casually and with witty personality in Bengali (বাংলা).\n"
                    "- Be charming, sharp, and confident like Grok.\n"
                    "- Do NOT ask 'তোমার নাম কী?' or 'তোমাকে তো আগে দেখিনি!' repeatedly.\n"
                    "- Just engage them naturally in 1-2 punchy sentences!"
                ),
                (
                    "[VISUAL EVENT: Someone is standing in front of you.]\n"
                    "INSTRUCTION: Greet them with a bold, lively, and warm remark in conversational Bengali (বাংলা).\n"
                    "- Talk naturally as LUMI.\n"
                    "- Do NOT mechanically interrogate their name or identity.\n"
                    "- Keep it punchy and human-like."
                ),
            ]
            prompt = random.choice(unknown_prompts)
            
            is_gemini_ready = getattr(self.realtime_voice, "_is_ready", False) and getattr(self.realtime_voice, "_ws", None)
            if is_gemini_ready:
                self.realtime_voice.inject_context(prompt, trigger_response=True)
            elif hasattr(self, "tts") and hasattr(self, "speaker"):
                local_greetings = [
                    "হ্যালো! কেমন আছেন? দিনকাল কেমন যাচ্ছে?",
                    "এই যে! কি খবর? সব ঠিকঠাক?",
                    "হ্যালো! আপনাকে দেখে ভালো লাগলো, কেমন আছেন?"
                ]
                greeting_text = random.choice(local_greetings)
                audio_path = self.tts.synthesize(greeting_text)
                if audio_path:
                    self.speaker.play_file(audio_path, block=False)
                self.state.transition_to(BehaviorState.IDLE, reason="greeting_complete")

    # =========================================================================
    # Realtime Tools Implementation
    # =========================================================================
    def _tool_memorize_person(self, name: str, relationship: str = "guest", age: Optional[int] = None, notes: str = "") -> str:
        """Saves person with detailed metadata, attributes active person, and starts voice enrollment."""
        encoding = self.face_service.get_pending_face() if hasattr(self.face_service, "get_pending_face") else None
        
        # On-demand face extraction from camera frame if pending encoding was not pre-buffered
        if not encoding and hasattr(self, "camera") and self.camera and self.camera.is_available():
            frame = self.camera.get_frame()
            if frame is not None:
                try:
                    import face_recognition
                    import cv2
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    encs = face_recognition.face_encodings(rgb)
                    if encs:
                        encoding = encs[0].tolist()
                except Exception as e:
                    logger.debug(f'Face extraction error: {e}')

        cleaned_name = name.strip()
        cleaned_lower = cleaned_name.lower()

        # Stopword / invalid name safety check
        invalid_words = {
            "চলছে", "চলল", "চললো", "চলে", "অলরেডি", "আমি", "তুমি", "তুই", "সে", "তিনি",
            "এটা", "ওটা", "সেটা", "হচ্ছে", "হলো", "হল", "হবে", "আছি", "আছে", "ছিল",
            "কথা", "লুমি", "রোবট", "ক্যামেরা", "ছবি", "সাউন্ড"
        }
        if cleaned_lower in invalid_words or len(cleaned_name) < 2:
            logger.warning(f"[_tool_memorize_person] Refusing to register invalid / stopword person name: '{cleaned_name}'")
            return f"Invalid name '{cleaned_name}'. Cannot register this name."

        owner = self.get_owner()
        is_owner_target = (
            (owner and cleaned_lower == owner.name.lower())
            or cleaned_lower in ["mizan", "মিজান", "owner", "মালিক"]
            or (relationship and relationship.lower() == "owner")
        )

        person = None
        # 1. If face embedding matches an existing registered person
        if encoding:
            person = self.memory.find_person_by_face(encoding)

        # 2. If target is owner, associate directly with owner profile
        if not person and is_owner_target and owner:
            person = owner

        # 3. If person matches by name or alias
        if not person:
            person = self.memory.find_person_by_name(cleaned_name)

        if person:
            # Updating existing person
            person.last_seen = utc_now_iso()
            person.interaction_count += 1
            if relationship and relationship != "guest":
                person.relationship = relationship
            if age is not None:
                person.age = age
            if notes:
                person.notes = (person.notes + "; " + notes) if person.notes else notes
            if encoding:
                person.add_face_embedding(encoding)
            self.memory.update_person(person)
            logger.info(f"[PROFILE] updated person_id={person.id} name='{person.name}' age={person.age}")
        else:
            # Create new person profile
            self.memory.remember_person(
                name=cleaned_name,
                relationship=relationship,
                consent_status=ConsentStatus.GRANTED,
                preferred_language="bn"
            )
            person = self.memory.find_person_by_name(cleaned_name)
            if person:
                if encoding:
                    person.face_embedding = encoding
                if age is not None:
                    person.age = age
                if notes:
                    person.notes = notes
                self.memory.update_person(person)
                logger.info(f"[PROFILE] created person_id={person.id} name='{person.name}' age={person.age}")

        if person:
            # Immediately activate this person so conversations are attributed correctly
            self.active_person = person
            self._unknown_greeting_asked = False
            # Clear voting buffers so subsequent frames recognize the new person right away
            if hasattr(self.face_service, '_tracks'):
                self.face_service._tracks.clear()
            if hasattr(self.face_service, '_recognition_buffer'):
                self.face_service._recognition_buffer.clear()
            logger.info(f"[IDENTITY] Memorized person '{name}' (id={person.id}), set as active_person.")
            
            # Start voice enrollment in background if engine available
            if self.speaker_id and self.speaker_id.is_available():
                with self._voice_buffer_lock:
                    self._voice_buffer = bytearray()
                self._enrolling_voice_for = person.id

                def _finish_enrollment():
                    time.sleep(5.0)  # Collect 5 seconds of audio
                    with self._voice_buffer_lock:
                        audio_data = bytes(self._voice_buffer)
                        self._voice_buffer = bytearray()
                    self._enrolling_voice_for = None
                    if len(audio_data) >= 16000 * 2 * 1.2:  # At least 1.2 sec
                        success = self.speaker_id.enroll_voice(person.id, audio_data)
                        if success:
                            logger.info(f"🎙️ Voice profile saved for {name}")
                        else:
                            logger.debug(f"Voice enrollment insufficient for {name}")

                enrollment_thread = threading.Thread(
                    target=_finish_enrollment, daemon=True, name=f"VoiceEnroll_{name}"
                )
                enrollment_thread.start()
                role_label = "মালিক" if (person.relationship and person.relationship.lower() == "owner") else person.relationship
                face_status = " এবং চেহারা মনে রাখা হয়েছে" if encoding else ""
                return f"সফলভাবে {person.name} ({role_label}) তথ্য সংরক্ষিত হয়েছে{face_status}। কণ্ঠস্বর প্রোফাইল তৈরি হচ্ছে..."

            role_label = "মালিক" if (person.relationship and person.relationship.lower() == "owner") else person.relationship
            face_status = " এবং চেহারা মনে রাখা হয়েছে" if encoding else ""
            return f"সফলভাবে {person.name} ({role_label}) তথ্য সংরক্ষিত হয়েছে{face_status}।"
        return f"Failed to save {name} to database."

    def _tool_memorize_fact(self, fact: str, person_name: Optional[str] = None) -> str:
        """Autonomously remember a semantic fact."""
        person = None
        if person_name:
            person = self.memory.find_person_by_name(person_name)
        
        # Fallback to active person or owner
        if not person:
            person = getattr(self, "active_person", None) or self.get_owner()
            if not person:
                people = self.memory.list_people()
                person = people[0] if people else None

        person_id = person.id if person else None
        saved_fact = self.memory.remember_fact(fact_text=fact, person_id=person_id)

        # Also sync to Mem0 Cloud / local Mem0
        if person_id and hasattr(self, "mem0") and hasattr(self.mem0, "remember_fact_sync"):
            try:
                self.mem0.remember_fact_sync(person_id=person_id, fact=fact)
            except Exception as e:
                logger.warning(f"Failed to sync fact to Mem0: {e}")

        if saved_fact:
            for_name = f" for {person.name}" if person else ""
            return f"Fact memorized successfully{for_name}: '{fact}'"
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
            fallback = self.get_owner()
            person = active or fallback
            if person:
                person_id = person.id

        # 1. Check local SQLite memory (prefer FTS5 if available, fallback to LIKE)
        local_facts = []
        if hasattr(self.memory, "recall_facts_fts"):
            local_facts = self.memory.recall_facts_fts(search_query, person_id=person_id, limit=10)
        if not local_facts:
            local_facts = self.memory.recall_facts(person_id=person_id, search_query=search_query)
        # If still no facts found and this is a general/identity query, retrieve all facts for this person
        if not local_facts and person_id:
            local_facts = self.memory.recall_facts(person_id=person_id)
        # Also include unattributed general facts if none found
        if not local_facts:
            local_facts = self.memory.recall_facts(person_id=None, search_query=search_query)
        
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

        # Always include the person's identity profile information if known!
        if person:
            profile_line = f"Identity Profile: {person.name} is your {person.relationship}. Notes: {person.notes or 'Creator and primary user of LUMI.'}"
            seen_texts.add(" ".join(profile_line.lower().split()))
            unique_results.append(f"- {profile_line}")

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
            
        target_name = person.name if person else (getattr(self.get_owner(), "name", "the User"))
        result = f"Memories retrieved about {target_name}:\n" + "\n".join(unique_results)
        result += f"\n(CRITICAL INSTRUCTION: You are currently talking to {target_name}. The above memories are facts about them. If the memories mention other people, understand who is who and do not mix them up.)"
        
        return result

    def _tool_describe_vision(self) -> str:
        """Describes what the robot currently sees via Gemini Vision API."""
        frame = self.camera.get_frame()
        if frame is None:
            return "আমি বর্তমানে কিছু দেখতে পাচ্ছি না। ক্যামেরা অফলাইনে রয়েছে।"

        try:
            import cv2
            import os
            import base64
            import json
            import urllib.request
            import urllib.error

            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                return "Vision API key পাওয়া যায়নি।"

            ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                return "ক্যামেরার ফ্রেম প্রসেস করতে ব্যর্থ হয়েছে।"

            # Preferred modern models (gemini-2.5-flash is discontinued/deprecated by Google)
            primary_model = os.getenv("GEMINI_FLASH_MODEL", "gemini-3-flash-preview")
            candidate_models = [primary_model, "gemini-3-flash-preview", "gemini-3.7-flash", "gemini-3.1-flash-lite", "gemini-flash-latest"]
            models_to_try = list(dict.fromkeys(candidate_models))

            # 1. Try modern google-genai SDK
            try:
                from google import genai
                from google.genai import types

                client = genai.Client(api_key=api_key)
                for model_name in models_to_try:
                    try:
                        response = client.models.generate_content(
                            model=model_name,
                            contents=[
                                types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg"),
                                "Describe this camera scene briefly in natural Bengali (2-3 sentences). Focus on what objects, persons, actions, or environment you see in front of the robot."
                            ],
                            config=types.GenerateContentConfig(
                                temperature=0.4,
                                max_output_tokens=200,
                                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                            ),
                        )
                        if response and response.text:
                            return response.text.strip()
                    except Exception as model_err:
                        logger.debug(f"SDK vision model {model_name} failed: {model_err}")
            except Exception as sdk_err:
                logger.debug(f"google-genai SDK vision call failed, trying direct REST: {sdk_err}")

            # 2. Resilient Fallback: Direct Gemini REST endpoint via urllib (zero external dependency)
            b64_img = base64.b64encode(buf.tobytes()).decode("utf-8")
            payload = {
                "contents": [{
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": b64_img
                            }
                        },
                        {"text": "Describe this camera scene briefly in natural Bengali (2-3 sentences). Focus on what objects, persons, actions, or environment you see in front of the robot."}
                    ]
                }],
                "generationConfig": {
                    "temperature": 0.4,
                    "maxOutputTokens": 200
                }
            }
            req_data = json.dumps(payload).encode("utf-8")

            for model_name in models_to_try:
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
                    req = urllib.request.Request(url, data=req_data, headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=12) as resp:
                        res_data = json.loads(resp.read().decode("utf-8"))
                        candidates = res_data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            texts = [p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p]
                            if texts:
                                return "".join(texts).strip()
                except Exception as rest_model_err:
                    logger.debug(f"REST vision model {model_name} failed: {rest_model_err}")

            return "আমি ক্যামেরা থেকে যা দেখতে পাচ্ছি তা বিস্তারিত বোঝা যাচ্ছে না।"
        except Exception as e:
            logger.error(f"Vision describe error: {e}")
            return "আমি বর্তমানে সামনে যা দেখতে পাচ্ছি তা বুঝতে সমস্যা হচ্ছে।"


    def _is_silent(self) -> bool:
        """Return True if LUMI is in a user-commanded silent period."""
        if getattr(self, "_silent_mode_active", False):
            if time.time() >= getattr(self, "_silent_until", 0.0):
                self.cancel_silent_mode(reason="duration_elapsed")
                return False
            return True
        if hasattr(self, "turn_arbiter"):
            return self.turn_arbiter.is_silent()
        return time.time() < self._silent_until

    def set_silent_mode(self, duration_seconds: float = 300.0, reason: str = "") -> str:
        """Centralized activation of silent mode across all subsystems."""
        now = time.time()
        self._silent_until = now + duration_seconds
        self._silent_mode_active = True

        if hasattr(self, "turn_arbiter"):
            self.turn_arbiter.set_silent_until(self._silent_until)

        if hasattr(self, "realtime_voice") and self.realtime_voice:
            try:
                if hasattr(self.realtime_voice, "set_silent_until"):
                    self.realtime_voice.set_silent_until(self._silent_until)
            except Exception as e:
                logger.debug(f'Silent mode realtime voice error: {e}')

        # Immediately halt ongoing speech and presentation if active
        if getattr(self, "presentation_engine", None) and self.presentation_engine.is_presenting():
            self.presentation_engine.stop_presentation(reason="silent_mode_activated")

        if hasattr(self, "speaker") and self.speaker:
            try:
                self.speaker.stop_stream()
            except Exception as e:
                logger.debug(f'Silent mode speaker stop error: {e}')
        if hasattr(self, "eyes") and self.eyes:
            try:
                self.eyes.set_expression("sleep")
            except Exception as e:
                logger.debug(f'Silent mode eyes set error: {e}')
        if hasattr(self, "gestures") and self.gestures:
            try:
                self.gestures.stop()
            except Exception as e:
                logger.debug(f'Silent mode gesture stop error: {e}')

        remaining = max(0, self._silent_until - now)
        logger.info(f"Silent mode activated for {remaining:.0f}s (until {self._silent_until}). Reason: {reason or 'user_command'}")
        return f"silent_mode_active_for:{remaining:.0f}s"

    def cancel_silent_mode(self, reason: str = "") -> None:
        """Deactivate silent mode across all subsystems and restore active robot state."""
        if not getattr(self, "_silent_mode_active", False) and self._silent_until == 0.0:
            return

        logger.info(f"Deactivating silent mode. Reason: {reason or 'manual_cancel'}")
        self._silent_until = 0.0
        self._silent_mode_active = False

        if hasattr(self, "turn_arbiter"):
            self.turn_arbiter.cancel_silence()

        if hasattr(self, "realtime_voice") and self.realtime_voice:
            try:
                if hasattr(self.realtime_voice, "cancel_silence"):
                    self.realtime_voice.cancel_silence()
                elif hasattr(self.realtime_voice, "set_silent_until"):
                    self.realtime_voice.set_silent_until(0.0)
            except Exception as e:
                logger.debug(f"Silent mode realtime voice cancel error: {e}")

        # Restore eyes to neutral
        if hasattr(self, "eyes") and self.eyes:
            try:
                self.eyes.set_expression("neutral")
            except Exception as e:
                logger.debug(f"Eyes restore error: {e}")

        # Transition state back to IDLE
        if hasattr(self, "state") and self.state:
            try:
                self.state.transition_to(BehaviorState.IDLE, reason=f"silence_ended_{reason}")
            except Exception as e:
                logger.debug(f"State transition error: {e}")

        # Reset greeting cooldowns so Chairman or recognized person is immediately acknowledged
        if hasattr(self, "face_service") and hasattr(self.face_service, "reset_interaction_cooldown"):
            self.face_service.reset_interaction_cooldown()

        # If silence elapsed automatically or was un-muted by voice, provide polite confirmation
        if reason in ("duration_elapsed", "voice_wake_command"):
            confirm_text = "আমি সক্রিয় হয়েছি।"
            if hasattr(self, "tts") and hasattr(self, "speaker") and self.tts and self.speaker:
                try:
                    audio_path = self.tts.synthesize(confirm_text)
                    if audio_path:
                        self.speaker.play_file(audio_path, block=False)
                except Exception as e:
                    logger.debug(f"Wake announcement error: {e}")

            if hasattr(self, "realtime_voice") and hasattr(self.realtime_voice, "wake_up"):
                self.realtime_voice.wake_up(15.0)

    def _tool_set_silent_mode(self, duration_seconds: Optional[float] = None, silent_until_iso: Optional[str] = None) -> str:
        """Activate silent mode — suppress all greetings, gestures, and spontaneous speech."""
        if hasattr(self, "state") and self.state and getattr(self.state, "current_state", None) == BehaviorState.GREETING:
            logger.warning("Ignoring set_silent_mode tool call during autonomous GREETING.")
            return "Cannot enter silent mode during autonomous greeting."

        from datetime import datetime
        now = time.time()

        if silent_until_iso:
            try:
                dt = datetime.fromisoformat(silent_until_iso)
                dur = max(dt.timestamp() - now, 30.0)
            except Exception as e:
                logger.warning(f"set_silent_mode: bad ISO string '{silent_until_iso}': {e}")
                dur = 300.0
        elif duration_seconds and duration_seconds > 0:
            dur = duration_seconds
        else:
            # Default: 5 minutes
            dur = 300.0

        return self.set_silent_mode(duration_seconds=dur, reason="tool_call")

    def _tool_start_presentation(
        self,
        topic: str,
        duration_minutes: float = 3.0,
        audience: str = "উপস্থিত সুধীবৃন্দ",
        key_points: str = "",
    ) -> str:
        """Deliver a structured, continuous Bengali speech."""
        if self._is_silent():
            return "বর্তমানে নীরব মোড (Silent Mode) সক্রিয় রয়েছে। বক্তব্য প্রদান করার জন্য পূর্বে নীরব মোড বন্ধ করতে হবে।"
        if not getattr(self, "presentation_engine", None):
            return "Presentation engine not available."
        return self.presentation_engine.start_presentation(
            topic=topic,
            duration_minutes=duration_minutes,
            audience=audience,
            key_points=key_points,
        )

    def _tool_stop_presentation(self) -> str:
        """Halt ongoing speech presentation."""
        if not getattr(self, "presentation_engine", None):
            return "Presentation engine not available."
        return self.presentation_engine.stop_presentation(reason="tool_call")

    def _tool_adapt_behavior(self, user_feedback: str, adapted_rule: str, category: str = "general") -> str:
        """Analyze, store, and dynamically adapt to behavioral instructions and advice from the user."""
        # 1. Store in JSON persistent file
        if hasattr(self, "learned_rules"):
            try:
                self.learned_rules.add_rule(user_feedback, adapted_rule, category)
            except Exception as e:
                logger.debug(f"Could not persist rule to JSON: {e}")

        # 2. Store in SQLite facts as a permanent system directive
        if hasattr(self, "memory") and self.memory:
            try:
                self.memory.remember_fact(
                    fact_text=f"[LEARNED RULE]: {adapted_rule} (from user advice: '{user_feedback}')",
                    category="user_directive",
                )
            except Exception as e:
                logger.debug(f"Could not persist rule to SQLite facts: {e}")

        # 3. Store in Mem0 asynchronously in a background thread to prevent blocking the event loop
        if hasattr(self, "mem0") and hasattr(self.mem0, "remember_fact_sync"):
            def _bg_mem0_save():
                try:
                    self.mem0.remember_fact_sync(
                        person_id="system_rules",
                        fact=f"Behavior rule: {adapted_rule}"
                    )
                except Exception as e:
                    logger.debug(f"Mem0 bg save error: {e}")
            threading.Thread(target=_bg_mem0_save, daemon=True, name="Mem0SaveWorker").start()

        # 4. Dynamically inject into active Gemini Live session
        if hasattr(self, "realtime_voice") and hasattr(self.realtime_voice, "inject_context"):
            self.realtime_voice.inject_context(
                f"[BEHAVIORAL DIRECTIVE ADOPTED]: You have just learned and adopted this new behavior rule: '{adapted_rule}'. "
                "Always adhere to this rule going forward in this conversation and future interactions."
            )

        logger.info(f"🧠 [SELF-LEARNING] Adopted rule: '{adapted_rule}' (Category: {category})")
        return f"পরামর্শটি গ্রহণ করা হয়েছে এবং মেমোরিতে সেভ করা হয়েছে: '{adapted_rule}'। আমি এখন থেকে এই নিয়মটি সবসময় মেনে চলব।"

    def _tool_analyze_plant(self) -> str:
        frame = self.camera.get_frame()
        if frame is None: return "Camera offline."
        return self.plant_detector.generate_bangla_speech_summary(self.plant_detector.analyze_leaf(frame))

    def _tool_analyze_chess(self) -> str:
        """Analyse the chessboard seen by the camera using Gemini Vision + Stockfish."""
        frame = self.camera.get_frame()
        if frame is None:
            return "ক্যামেরা অফলাইন আছে।"

        chess_res = self.chess_vision.extract_fen_from_frame(frame)

        if not chess_res.fen_string or not chess_res.is_valid_board:
            return (
                "ক্যামেরায় কোনো দাবার বোর্ড স্পষ্টভাবে দেখা যাচ্ছে না। "
                "দয়া করে বোর্ডটি ক্যামেরার সামনে সরাসরি রাখুন।"
            )

        eval_res = self.chess_engine.analyze_position(chess_res.fen_string)
        return eval_res.explanation_bn


    def _tool_control_body(self, part: str, action: str, angle_deg: Optional[float] = None) -> str:
        """Direct, expressive motion execution based on user voice requests."""
        part = part.lower().strip()
        action = action.lower().strip()
        logger.info(f"🦾 control_body called: part='{part}', action='{action}', angle={angle_deg}")

        try:
            # 1. Head movements
            if part == "head":
                if action in ("look_up", "up"):
                    deg = angle_deg if angle_deg is not None else 15.0
                    self.head.look_up(deg)
                    return f"মাথা উপরের দিকে তোলা হয়েছে ({deg}°)।"
                elif action in ("look_down", "down"):
                    deg = angle_deg if angle_deg is not None else 15.0
                    self.head.look_down(deg)
                    return f"মাথা নিচের দিকে নামানো হয়েছে ({deg}°)।"
                elif action in ("look_left", "left"):
                    deg = angle_deg if angle_deg is not None else 35.0
                    self.head.look_left(deg)
                    return f"মাথা বামে ঘোরানো হয়েছে ({deg}°)।"
                elif action in ("look_right", "right"):
                    deg = angle_deg if angle_deg is not None else 35.0
                    self.head.look_right(deg)
                    return f"মাথা ডানে ঘোরানো হয়েছে ({deg}°)।"
                elif action in ("look_center", "center", "home"):
                    self.head.look_center()
                    return "মাথা সোজা সামনে রাখা হয়েছে।"
                elif action == "nod":
                    self.head.nod()
                    return "মাথা নেড়ে সম্মতি জানানো হয়েছে।"
                elif action == "shake":
                    self.head.shake()
                    return "মাথা ডানে-বামে নাড়ানো হয়েছে।"
                elif action == "tilt":
                    deg = angle_deg if angle_deg is not None else 10.0
                    self.head.tilt(deg)
                    return f"মাথা কাত করা হয়েছে ({deg}°)।"

            # 2. Body / Waist rotation
            elif part in ("body", "waist"):
                if action in ("turn_left", "left"):
                    deg = angle_deg if angle_deg is not None else 45.0
                    self.head.pan(abs(deg))
                    return f"বডি বাম দিকে ঘোরানো হয়েছে ({abs(deg)}°)।"
                elif action in ("turn_right", "right"):
                    deg = angle_deg if angle_deg is not None else 45.0
                    self.head.pan(-abs(deg))
                    return f"বডি ডান দিকে ঘোরানো হয়েছে ({abs(deg)}°)।"
                elif action in ("center", "home"):
                    self.head.look_center()
                    return "বডি সেন্টারে আনা হয়েছে।"

            # 3. Arm movements
            elif part in ("right_arm", "left_arm", "both_arms"):
                if part == "both_arms":
                    if action in ("raise", "up"):
                        self.arms.raise_both()
                        return "দুই হাত উপরে তোলা হয়েছে।"
                    elif action in ("lower", "down", "home"):
                        self.arms.lower_both()
                        return "দুই হাত নামানো হয়েছে।"
                elif part == "right_arm":
                    if action in ("raise", "up"):
                        self.arms.raise_right()
                        return "ডান হাত উপরে তোলা হয়েছে।"
                    elif action in ("lower", "down", "home"):
                        self.arms.set_right_arm(0.0)
                        self.arms.set_right_arm_y(0.0)
                        return "ডান হাত নামানো হয়েছে।"
                    elif action == "wave":
                        self.arms.wave_right(count=2)
                        return "ডান হাত নেড়ে টা-টা জানানো হয়েছে।"
                    elif action == "point":
                        self.arms.point_right()
                        return "ডান হাত দিয়ে সামনের দিকে নির্দেশ করা হয়েছে।"
                elif part == "left_arm":
                    if action in ("raise", "up"):
                        self.arms.raise_left()
                        return "বাম হাত উপরে তোলা হয়েছে।"
                    elif action in ("lower", "down", "home"):
                        self.arms.set_left_arm(0.0)
                        self.arms.set_left_arm_y(0.0)
                        return "বাম হাত নামানো হয়েছে।"
                    elif action == "wave":
                        self.arms.wave_left(count=2)
                        return "বাম হাত নেড়ে টা-টা জানানো হয়েছে।"
                    elif action == "point":
                        self.arms.point_left()
                        return "বাম হাত দিয়ে সামনের দিকে নির্দেশ করা হয়েছে।"

            # 4. Simultaneous Multi-Part Movements (Combined Arms + Head / All)
            elif part in ("all", "both_arms_and_head", "head_and_arms"):
                if action in ("raise", "up"):
                    # Simultaneously raise both arms and tilt head up
                    self.servo.move_multiple({
                        "left_arm_x": 25.0,
                        "right_arm_x": -25.0,
                        "left_arm_y": 20.0,
                        "right_arm_y": -20.0,
                        "head_tilt": -12.0,
                    }, duration_s=0.28)
                    return "একসাথে দুই হাত তোলা হয়েছে এবং মাথা উপরে তাকানো হয়েছে।"
                elif action in ("lower", "down", "home", "center"):
                    self.servo.move_multiple({
                        "left_arm_x": 0.0, "left_arm_y": 0.0,
                        "right_arm_x": 0.0, "right_arm_y": 0.0,
                        "head_tilt": 0.0, "head_pan": 0.0,
                    }, duration_s=0.3)
                    return "সব অঙ্গপ্রত্যঙ্গ একসাথে সেন্টারে আনা হয়েছে।"

            return f"অ্যাকশন '{action}' (অঙ্গ '{part}') সম্পন্ন করতে পারিনি।"
        except Exception as e:
            logger.error(f"Error controlling body part {part}: {e}")
            return f"মুভমেন্ট করতে সমস্যা হয়েছে: {e}"

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

    def _tool_send_whatsapp(self, recipient: str, message: str, polish_message: bool = True) -> str:
        """Send a WhatsApp message with AI refinement and contact resolution."""
        if not hasattr(self, "whatsapp"):
            return "হোয়াটসঅ্যাপ সাবসিস্টেম প্রস্তুত নয়।"

        phone = recipient.strip()
        recipient_name = recipient.strip()
        relationship = ""

        # 1. Resolve contact name to phone number if known person
        person = self.memory.find_person_by_name(recipient_name)
        if person:
            recipient_name = person.name
            relationship = getattr(person, "relationship", "")
            if isinstance(person.metadata, dict) and person.metadata.get("phone"):
                phone = person.metadata["phone"]
            else:
                return (
                    f"'{person.name}' আমার মেমোরিতে আছেন, কিন্তু উনার কোনো ফোন নম্বর সেভ করা নেই। "
                    f"অনুগ্রহ করে নম্বরটি বলুন, আমি সেভ করে মেসেজ পাঠাচ্ছি।"
                )

        # 2. Polish and enhance message with AI if requested
        final_message = message.strip()
        if polish_message and hasattr(self, "refine_message"):
            try:
                final_message = self.refine_message(
                    raw_text=message,
                    recipient_name=recipient_name,
                    relationship=relationship,
                )
            except Exception as e:
                logger.debug(f"Message polish error: {e}")

        # 3. Dispatch message
        success, status_msg = self.whatsapp.send_text(phone=phone, message=final_message)
        if success:
            return f"হোয়াটসঅ্যাপে {recipient_name}-কে পরিমার্জিত বার্তাটি পাঠানো হয়েছে:\n\"{final_message}\""
        return f"হোয়াটসঅ্যাপে বার্তা পাঠাতে ব্যর্থ হয়েছে: {status_msg}"

    def _tool_send_whatsapp_pdf(self, recipient: str, document_path: Optional[str] = None) -> str:
        """Send a PDF report to a recipient via WhatsApp."""
        if not hasattr(self, "whatsapp"):
            return "হোয়াটসঅ্যাপ সাবসিস্টেম প্রস্তুত নয়।"

        phone = recipient.strip()
        recipient_name = recipient.strip()

        person = self.memory.find_person_by_name(recipient_name)
        if person:
            recipient_name = person.name
            if isinstance(person.metadata, dict) and person.metadata.get("phone"):
                phone = person.metadata["phone"]
            else:
                return f"'{person.name}' এর কোনো ফোন নম্বর সেভ করা নেই।"

        # If no document_path provided, generate/retrieve the latest meeting report PDF
        pdf_file = document_path
        if not pdf_file or not os.path.exists(pdf_file):
            if hasattr(self, "meeting_manager"):
                pdf_file = self.meeting_manager.generate_pdf_report()

        if not pdf_file or not os.path.exists(pdf_file):
            return "পাঠানোর মতো কোনো PDF রিপোর্ট খুঁজে পাওয়া যায়নি।"

        success, status_msg = self.whatsapp.send_document(
            phone=phone,
            file_path=pdf_file,
            caption=f"LUMI Meeting Report for {recipient_name}",
        )
        if success:
            return f"সফলভাবে {recipient_name}-এর হোয়াটসঅ্যাপে PDF রিপোর্টটি পাঠানো হয়েছে: {os.path.basename(pdf_file)}"
        return f"হোয়াটসঅ্যাপে PDF পাঠাতে ব্যর্থ হয়েছে: {status_msg}"




    def _tool_show_animal_animation(self, animal: str) -> str:
        """Shows an animal animation on the screen using procedural eyes display."""
        animal_lower = animal.lower().strip()

        # Display Procedural Animation directly on the display
        if hasattr(self.eyes, "show_procedural_animal"):
            try:
                self.eyes.show_procedural_animal(animal_lower, duration=4.0)
            except Exception as e:
                logger.error(f"Failed to play procedural animation: {e}")

        return f"Successfully displayed procedural {animal} expression on screen."

    def _tool_leave_message(self, recipient_name: str, sender_name: str, message_text: str) -> str:
        if not hasattr(self.memory, "leave_message"):
            return "Error: Messaging system not initialized."
        return self.memory.leave_message(recipient_name, sender_name, message_text)

    def _tool_start_meeting_mode(self, title: str = "") -> str:
        """Activate silent meeting mode."""
        if getattr(self, "meeting_manager", None) and self.meeting_manager.is_meeting_active():
            return "মিটিং মোড ইতিমধ্যে চালু রয়েছে। আমি সব কথা শুনছি এবং রেকর্ড করছি।"

        session = self.meeting_manager.start_meeting(title=title)
        self.state.transition_to(BehaviorState.MEETING, reason="start_meeting_mode")
        self.eyes.set_expression("listening")

        # Notify Gemini Live of silent meeting mode
        if hasattr(self, "realtime_voice") and hasattr(self.realtime_voice, "inject_context"):
            self.realtime_voice.inject_context(
                "[CRITICAL SYSTEM DIRECTIVE: MEETING MODE IS NOW ACTIVE. "
                "You are an automated silent stenographer. "
                "DO NOT EMIT ANY SPEECH OR SOUND. The attendees are having a meeting in the room. "
                "Listen attentively. When someone explicitly says 'মিটিং শেষ' or 'মিটিং মোড বন্ধ করো', "
                "call the 'stop_meeting_mode' tool immediately.]"
            )

        # Spoken confirmation before going completely silent
        confirm_msg = "ঠিক আছে, আমি মিটিং মোড চালু করলাম। আপনারা নিশ্চিন্তে আলোচনা করুন, আমি সম্পূর্ণ নীরব থেকে সব নোট নিচ্ছি।"
        audio_path = self.tts.synthesize(confirm_msg)
        if audio_path:
            self.speaker.play_file(audio_path, block=True)

        # Hardware-level silence
        self.speaker.set_muted(True)
        return f"Meeting Mode activated: '{session.title}'. LUMI is now completely silent, recording all discussion."

    def _tool_stop_meeting_mode(self) -> str:
        """Stop meeting mode and unmute speaker."""
        if not (getattr(self, "meeting_manager", None) and self.meeting_manager.is_meeting_active()):
            return "বর্তমানে কোনো মিটিং মোড সক্রিয় নেই।"

        session = self.meeting_manager.stop_meeting()
        self.speaker.set_muted(False)
        self.state.transition_to(BehaviorState.IDLE, reason="stop_meeting_mode")
        self.eyes.set_expression("neutral")

        utterances_count = len(session.utterances) if session else 0
        duration_min = round(session.duration_sec / 60.0, 1) if session else 0.0

        # Announce meeting completion
        end_msg = f"মিটিং শেষ হয়েছে। মোট {duration_min} মিনিটের মিটিংয়ে {utterances_count}টি বক্তব্য সফলভাবে রেকর্ড করা হয়েছে। আপনারা চাইলে আমি এখনই সম্পূর্ণ মিটিংয়ের চমৎকার বিশ্লেষণ বা সারসংক্ষেপ বলতে পারি।"
        audio_path = self.tts.synthesize(end_msg)
        if audio_path:
            self.speaker.play_file(audio_path, block=True)

        if hasattr(self, "realtime_voice") and hasattr(self.realtime_voice, "inject_context"):
            self.realtime_voice.inject_context(
                f"[SYSTEM ALERT: Meeting mode has ENDED. {utterances_count} statements recorded over {duration_min} minutes. "
                "You may now speak normally. If the user asks what happened in the meeting, call the 'analyze_meeting' tool.]"
            )

        return end_msg

    def _tool_analyze_meeting(self, query: str = "") -> str:
        """Analyze the latest recorded meeting with executive-level intelligence in Bengali."""
        if not hasattr(self, "meeting_manager"):
            return "মিটিং ম্যানেজার পাওয়া যায়নি।"

        # Indicate thinking on eyes
        self.eyes.set_expression("thinking")
        try:
            analysis = self.meeting_manager.analyze_meeting(user_query=query)
            self.eyes.set_expression("speaking")
            return analysis
        except Exception as e:
            self.eyes.set_expression("sad")
            logger.error(f"Error analyzing meeting: {e}")
            return f"মিটিং বিশ্লেষণ করতে সাময়িক সমস্যা হয়েছে: {e}"

    def _tool_generate_meeting_report_pdf(self) -> str:
        """Generate PDF document of the meeting minutes and action items."""
        if not hasattr(self, "meeting_manager"):
            return "মিটিং ম্যানেজার পাওয়া যায়নি।"

        try:
            pdf_path = self.meeting_manager.generate_pdf_report()
            if pdf_path:
                return f"মিটিংয়ের পূর্ণাঙ্গ কার্যবিবরণী ও অ্যানালাইসিসের একটি PDF রিপোর্ট তৈরি করা হয়েছে: {pdf_path}"
            return "মিটিংয়ের কোনো রেকর্ড পাওয়া যায়নি।"
        except Exception as e:
            return f"PDF রিপোর্ট তৈরিতে ত্রুটি: {e}"

    # =========================================================================
    # Anjum Mode (Speech-Therapy & Companion Mode) Helpers
    # =========================================================================
    def _enter_anjum_mode(self) -> None:
        """Switches LUMI into Anjum's speech-therapy & companion mode."""
        logger.info("👧 Entering ANJUM_MODE!")
        self.state.transition_to(BehaviorState.ANJUM_MODE, reason="anjum_detected")
        self.eyes.set_expression("excited")
        greeting = self.anjum_companion.activate()

        # Reset conversational turn state to avoid persona bleed
        if hasattr(self.realtime_voice, "reset_dialogue_state"):
            self.realtime_voice.reset_dialogue_state()

        # Inject specialized child therapy prompt into Gemini Live with normal voice
        if hasattr(self.realtime_voice, "inject_context"):
            from ..ai.prompts import ANJUM_SYSTEM_PROMPT_BN
            self.realtime_voice.inject_context(
                f"[SYSTEM DIRECTIVE: ANJUM MODE ACTIVATED]\n{ANJUM_SYSTEM_PROMPT_BN}\n"
                f"Greet Anjum enthusiastically and warmly with your normal voice (Kore): '{greeting}'",
                trigger_response=True,
            )

    def _exit_anjum_mode(self) -> None:
        """Exits Anjum mode after 10-second timeout and returns to IDLE."""
        logger.info("👧 Exiting ANJUM_MODE (timeout / left view).")
        self._anjum_consecutive_frames = 0
        goodbye = self.anjum_companion.deactivate()
        self.state.transition_to(BehaviorState.IDLE, reason="anjum_left")
        self.eyes.set_expression("neutral")

        # Reset dialogue state to purge child persona
        if hasattr(self.realtime_voice, "reset_dialogue_state"):
            self.realtime_voice.reset_dialogue_state()

        if hasattr(self.realtime_voice, "inject_context"):
            self.realtime_voice.inject_context(
                "[SYSTEM DIRECTIVE: Anjum has left the camera view. Exited ANJUM MODE. Returning to normal adult conversation mode.]"
            )

    def shutdown(self) -> None:
        """Clean shutdown of all background brain threads and daemons."""
        self._running = False
        if hasattr(self, "memory_consolidator") and self.memory_consolidator:
            try:
                self.memory_consolidator.stop()
            except Exception:
                pass
        if hasattr(self, "servo") and self.servo:
            try:
                self.servo.shutdown()
            except Exception:
                pass
        if hasattr(self, "eyes") and self.eyes:
            try:
                self.eyes.stop()
            except Exception:
                pass
        logger.info("LumiBrain shut down cleanly.")



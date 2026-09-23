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
        tts: Optional[Any] = None,
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
        
        # Default to Gemini Live production model (models/gemini-3.1-flash-live-preview) with override support
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
        
        if tts is not None:
            self.tts = tts
        else:
            from ..speech.tts import BanglaTTS
            self.tts = BanglaTTS()
        try:
            self.wake_audio_path = self.tts.synthesize("জ্বী বলুন") if hasattr(self.tts, "synthesize") else None
        except Exception:
            self.wake_audio_path = None
        
        self._last_video_send = 0.0
        self._last_speech_motion_time = 0.0
        self._silent_until = 0.0
        self._active_speech_target_end = 0.0
        self._active_speech_topic = ""
        self._speech_continuation_count = 0

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

    def cancel_silence(self) -> None:
        """Cancel silence mode immediately."""
        self._silent_until = 0.0
        if getattr(self, "turn_arbiter", None):
            self.turn_arbiter.cancel_silence()

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
        if not text:
            return False
        lower = text.lower()

        # Wake command: "কথা বলো", "জেগে ওঠো", "wake up"
        if re.search(r"(?:কথা বল(?:ো|িস|েন)?|জেগে ওঠো|শুনতে পাচ্ছ|wake up)", lower):
            if self.is_silent():
                logger.info("Wake command detected. Deactivating silent mode.")
                self.cancel_silence()
                if getattr(self, "eyes", None) and hasattr(self.eyes, "set_expression"):
                    self.eyes.set_expression("happy")
                if getattr(self, "_on_silence_cancelled_cb", None) and callable(self._on_silence_cancelled_cb):
                    try:
                        self._on_silence_cancelled_cb()
                    except Exception as e:
                        logger.debug(f"Silence cancel cb error: {e}")
                return True

        # Check negation / anti-silence (e.g. "থামার প্রয়োজন নাই", "থামবে না", "ননস্টপ")
        anti_silence = [
            r"(?:থাম(?:ার|বে|লে)?|চুপ\s*(?:করা|থাকা)?)\s*(?:র\s*)?(?:প্রয়োজন\s*নাই|প্রয়োজন\s*নাই|দরকার\s*নাই|লাগবে\s*না|হবে\s*না|নিষেধ)",
            r"(?:থাম(?:বে|িস)?\s*না|থামো\s*না|থামুন\s*না)",
            r"(?:চুপ\s*কর(?:ো|বেন)?\s*না|চুপ\s*থাকিও\s*না)",
            r"(?:ননস্টপ|nonstop|একটানা|লাগাতার|লগাতার)",
        ]
        if any(re.search(p, lower) for p in anti_silence):
            return False

        # Silence command: "চুপ থাকো", "১০ মিনিট চুপ থাকো", "কথা বলিও না", "shut up", "be quiet"
        has_silence_word = any(w in lower for w in ["চুপ", "থাম", "shut up", "be quiet", "silence", "quiet"])
        if has_silence_word and re.search(r"(?:^|\s+)(?:চুপ\s*(?:থাকো?|থাকুন|করো?|করুন)|থামো|থামুন|থেমে\s*(?:যাও|যান)|shut\s*up|be\s*quiet|stop)(?:\s+|$|[।!?])", lower):
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
            self._active_speech_target_end = 0.0
            self._active_speech_topic = ""
            if getattr(self, "turn_arbiter", None):
                self.turn_arbiter.set_presenting(False)
            logger.info(f"Silence command matched from user speech! Silencing LUMI for {duration_s:.0f}s.")
            if getattr(self, "speaker", None):
                try:
                    self.speaker.stop_stream()
                except Exception as e:
                    logger.debug(f'Silent mode speaker stop error: {e}')
            if getattr(self, "eyes", None) and hasattr(self.eyes, "set_expression"):
                self.eyes.set_expression("sleep")
            if getattr(self, "_on_silence_activated_cb", None) and callable(self._on_silence_activated_cb):
                try:
                    self._on_silence_activated_cb(duration_s)
                except Exception as e:
                    logger.debug(f"Silence activated cb error: {e}")
            return True
        return False

    def _check_presentation_command(self, text: str) -> bool:
        """Detect speech/presentation requests (e.g. '৫ মিনিট কথা বলো', '৩৬০ সেকেন্ড বলো', 'ভাষণ দাও', 'give a speech')."""
        import re
        if not text:
            return False

        lower = text.lower()
        # Exclude actual silences, cancellations, reminders
        if any(w in lower for w in ["মনে করিয়ে", "রিমাইন্ডার", "alarm", "remind"]):
            return False
        if self._check_silence_command(text):
            return False

        norm = text
        bn_digits = "০১২৩৪৫৬৭৮৯"
        en_digits = "0123456789"
        dev_digits = "०१२३४५६७८९"
        for b, e in zip(bn_digits, en_digits):
            norm = norm.replace(b, e)
        for d, e in zip(dev_digits, en_digits):
            norm = norm.replace(d, e)
        norm_lower = norm.lower()

        # Match seconds and minutes
        sec_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:সেকেন্ড(?:ের|ে)?|सेकंड|sec(?:ond)?s?)", norm_lower)
        min_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:মিনিট(?:ের|ে)?|मिनट|min(?:ute)?s?)", norm_lower)

        duration = None
        if sec_match:
            duration = max(0.5, float(sec_match.group(1)) / 60.0)
        elif min_match:
            duration = max(0.5, float(min_match.group(1)))
        elif any(w in norm_lower for w in ["পাঁচ মিনিট", "পাচ মিনিট", "pach minute", "panch minute", "five minute", "5 min", "पांच मिनट"]):
            duration = 5.0
        elif any(w in norm_lower for w in ["দশ মিনিট", "dosh minute", "ten minute", "10 min", "दस मिनट"]):
            duration = 10.0
        elif any(w in norm_lower for w in ["তিন মিনিট", "tin minute", "three minute", "3 min", "तीन मिनट"]):
            duration = 3.0
        elif any(w in norm_lower for w in ["দুই মিনিট", "dui minute", "two minute", "2 min", "दो मिनट"]):
            duration = 2.0
        elif any(w in norm_lower for w in ["এক মিনিট", "ek minute", "one minute", "1 min", "एक मिनट"]):
            duration = 1.0

        speech_verbs = [
            "বক্তব্য", "ভাষণ", "বক্তৃতা", "উপস্থাপন", "লেকচার", "আলোচনা",
            "কথা বল", "কথা বলো", "কথা বলুন", "কথা বলবি", "কথা বলতে", "কথা বলবো", "কথা বলব", "কথা বলবা",
            "বলো", "বলুন", "বলব", "বলবো", "কিছু বল", "কিছু বলো", "একটানা বল", "একটানা বলো",
            "ননস্টপ", "nonstop", "লাগাতার", "লগাতার",
            "बात करें", "बात करो", "लगातार",
            "katha bolo", "kotha bolo", "katha bol", "kotha bol", "katha bolun", "kotha bolun",
            "katha bolte", "kotha bolte", "kotha", "katha", "bolo", "bolun", "kichu bolo",
            "boktobbo", "bhashon", "vashon",
            "speech", "presentation", "lecture", "talk", "speak"
        ]
        has_speech_intent = any(w in norm_lower for w in speech_verbs)

        speech_nouns = ["বক্তব্য", "ভাষণ", "বক্তৃতা", "উপস্থাপন", "speech", "presentation", "lecture", "boktobbo", "bhashon"]
        action_words = ["দাও", "দিন", "শুরু", "বল", "কর", "give", "deliver", "start"]
        has_speech_word = any(w in norm_lower for w in speech_nouns)
        has_action_word = any(w in norm_lower for w in action_words)

        is_presentation = False
        if duration is not None and has_speech_intent:
            is_presentation = True
        elif has_speech_word and has_action_word:
            is_presentation = True
            duration = 3.0

        if not is_presentation or duration is None:
            return False

        # Smart Topic Extraction
        # 1. Look for targeted clauses preceding postpositions ('নিয়ে', 'সম্পর্কে', 'বিষয়ে', 'niye', 'somporke')
        clause_pattern = r"([^,।\.\?\!\:\;]+?)\s*(?:নিয়ে|নিয়ে|সম্পর্কে|সম্বন্ধে|বিষয়ে|বিষয়ক|এর\s*ওপর|niye|somporke)(?:\s+|$)"
        matches = re.findall(clause_pattern, norm, flags=re.IGNORECASE)
        cleaned_clauses = []
        for m in matches:
            cand = m.strip()
            for _ in range(2):
                cand = re.sub(r"(\d+|পাঁচ|দশ|তিন|দুই|এক)\s*(?:মিনিট(?:ের|ে)?|সেকেন্ড(?:ের|ে)?|min(?:ute)?s?|sec(?:ond)?s?|सेकंड|मिनट)", "", cand, flags=re.IGNORECASE)
                cand = re.sub(r"(?:^|\s+)(?:হামে|আমি|তুমি|লুমি|রোবট|বলতেছি|বলছি|বলবো|বলব|জানতে\s*চাই\s*না|জানতে\s*চাই|যান্তে\s*চাই\s*না|যান্তে\s*চাই|তোমাকে|দয়া\s*করে|প্লিজ|একটি|একটা|শুধু|কোন|কোনো|প্রশ্ন|করবো|করব|না|বাত|लगातार|नॉनस्टॉप)(?:\s+|$)", " ", cand, flags=re.IGNORECASE)
                cand = re.sub(r"\s+", " ", cand).strip()
            if len(cand) > 2 and cand not in cleaned_clauses:
                cleaned_clauses.append(cand)

        if cleaned_clauses:
            topic = " ও ".join(cleaned_clauses)
        else:
            cleaned = norm
            cleaned = re.sub(r"\(\s*\d+(?:\.\d+)?\s*(?:মিনিট(?:ের|ে)?|সেকেন্ড(?:ের|ে)?|min(?:ute)?s?|sec(?:ond)?s?|सेकंड|मिनट)\s*\)", " ", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\d+(?:\.\d+)?\s*(?:মিনিট(?:ের|ে)?|সেকেন্ড(?:ের|ে)?|min(?:ute)?s?|sec(?:ond)?s?|सेकंड|मिनट)", " ", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"(?:পাঁচ|দশ|তিন|দুই|এক)\s*(?:মিনিট(?:ের|ে)?|সেকেন্ড(?:ের|ে)?|सेकंड|मिनट)", " ", cleaned, flags=re.IGNORECASE)

            noise_patterns = [
                r"কথা\s*বল(?:তে\s*বলা\s*হয়|তে|বে|ব|বেন|ছেন|বো|বা)?",
                r"কথা\s*বলো", r"কথা\s*বলুন", r"কথা\s*বল",
                r"katha\s*bolo", r"kotha\s*bolo", r"katha\s*bol", r"kotha\s*bol",
                r"কিছু\s*বলো", r"কিছু\s*বলুন", r"কিছু\s*বল",
                r"একটানা\s*বলো", r"একটানা\s*বলুন", r"একটানা\s*বল", r"একটানা",
                r"ননস্টপ", r"nonstop", r"লাগাতার", r"লগাতার",
                r"बात\s*करें", r"बात\s*करो", r"लगातार", r"तक", r"তাক", r"পর্যন্ত",
                r"বক্তব্য\s*(?:দাও|দিন|শুরু\s*করো|শুরু\s*করুন|দিতে\s*হবে|রাখো)?",
                r"ভাষণ\s*(?:দাও|দিন|শুরু\s*করো|শুরু\s*করুন|দিতে\s*হবে)?",
                r"বক্তৃতা\s*(?:দাও|দিন|শুরু\s*করো|শুরু\s*করুন|দিতে\s*হবে)?",
                r"উপস্থাপন\s*(?:করো|করুন|করা\s*হোক)?",
                r"লেকচার\s*(?:দাও|দিন)?",
                r"রোবট(?:টাকে|কে|টি)?",
                r"দয়া\s*করে", r"একটু", r"প্লিজ", r"দিতে\s*হবে", r"শুরু\s*করো", r"শুরু\s*করুন",
                r"তুমি", r"লুমি", r"একটি", r"একটা", r"কোনো", r"কোন", r"বিষয়ে", r"বিষয়ক",
                r"সম্পর্কে", r"সম্বন্ধে", r"নিয়ে", r"নিয়ে", r"ওপর",
                r"হামে\s*যান্তে\s*চাই\s*না", r"হামে\s*যান্তে\s*চাই", r"হামে\s*বলতেছি",
                r"যান্তে\s*চাই\s*না", r"জানতে\s*চাই\s*না", r"যান্তে\s*চাই", r"জানতে\s*চাই",
                r"পারো\s*কিনা\s*দেখি", r"পারি\s*কিনা", r"দেখি",
                r"বলো", r"বলুন", r"বল", r"বলবো", r"বলব", r"দাও", r"দিন", r"bolo", r"kotha", r"katha",
                r"give\s*a\s*speech(?:\s*on)?", r"deliver\s*a\s*presentation(?:\s*on)?",
                r"give\s*a\s*presentation(?:\s*on)?", r"give\s*a\s*lecture(?:\s*on)?",
                r"talk\s*about", r"speak\s*about", r"presentation\s*on",
                r"speech\s*on", r"talk\s*for", r"speak\s*for", r"presentation", r"speech",
                r"about", r"on", r"for"
            ]
            for np in noise_patterns:
                cleaned = re.sub(np, " ", cleaned, flags=re.IGNORECASE)

            cleaned = re.sub(r"[\(\)\[\]\{\}\-\_\,\.\?\!\'\"\:।]", " ", cleaned)
            cleaned = re.sub(r"^\s*[\u09be-\u09cd]+\s*", "", cleaned)
            cleaned = re.sub(r"\s+[\u09be-\u09cd]+(?:\s+|$)", " ", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()

            if len(cleaned) > 2:
                topic = cleaned
            elif any(w in lower for w in ["স্বাধীনতা", "মুক্তিযুদ্ধ", "স্বাধীনতার"]):
                topic = "বাংলাদেশের স্বাধীনতা সংগ্রাম ও মুক্তিযুদ্ধের ইতিহাস"
            elif any(w in lower for w in ["ইতিহাস"]):
                topic = "ইতিহাস ও ঐতিহ্য"
            else:
                topic = "বিজ্ঞান ও আধুনিক প্রযুক্তি"

        duration_s = duration * 60.0
        self._active_speech_target_end = time.time() + duration_s
        self._active_speech_topic = topic
        self._speech_continuation_count = 0
        logger.info(f"Speech/Presentation session initiated: '{topic}' for {duration:.1f} mins ({duration_s:.0f}s).")

        # Engage presenting guard on turn arbiter so mic audio doesn't self-interrupt
        if getattr(self, "turn_arbiter", None):
            self.turn_arbiter.set_presenting(True)
            self.turn_arbiter.wake_up(duration_s + 30.0)

        # Transition state
        if getattr(self, "state", None) and hasattr(self.state, "transition_to"):
            self.state.transition_to(BehaviorState.SPEAKING, reason=f"speech:{topic[:15]}")

        # Inject commanding prompt into active Gemini Live session
        speech_cmd = (
            f"[MANDATORY CONTINUOUS SPEECH DIRECTIVE: {duration:.1f}-MINUTE SPEECH ON '{topic}']:\n"
            f"Deliver an uninterrupted, comprehensive monologue directly in your voice for the full {duration:.1f} minutes (~{int(duration*120)} words). "
            f"Do NOT stop after 1 minute! NEVER ask 'আমি কি বলতেই থাকবো?', 'আমি কি আরো বলব?', or any check-in questions! "
            f"Speak continuously with deep analysis, historical background, real-world examples, and inspiring vision."
        )
        if hasattr(self, "inject_context") and callable(self.inject_context):
            try:
                self.inject_context(speech_cmd, trigger_response=True)
            except Exception as e:
                logger.debug(f"Speech context injection notice: {e}")

        if hasattr(self, "_on_presentation_requested_cb") and callable(self._on_presentation_requested_cb):
            try:
                self._on_presentation_requested_cb(topic, duration)
            except Exception as e:
                logger.debug(f"Presentation request cb error: {e}")
        return True

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

        instructions += (
            "\n\n[MANDATORY DIRECTIVE: CONTINUOUS MONOLOGUES & 5-MINUTE SPEECHES]:\n"
            "When the user asks you to talk, speak, or lecture for a duration (e.g. '৫ মিনিট কথা বলো', '৩৬০ সেকেন্ড বলো', 'talk for 5 minutes', 'ভাষণ দাও', or any time-bounded topic request):\n"
            "1. You MUST deliver a continuous, deep, articulate, multi-paragraph monologue directly in your voice!\n"
            "2. DO NOT stop after 1 minute! Keep speaking continuously.\n"
            "3. NEVER ask 'আমি কি বলতেই থাকবো?', 'আমি কি আরো বলবো?', 'Do you want me to keep going?', or any check-in/counter questions. Such questions are STRICTLY FORBIDDEN!\n"
            "4. The 1-3 sentence brevity rule is COMPLETELY SUSPENDED for speeches. Deliver a comprehensive address covering:\n"
            "   (a) Subject introduction, context, and philosophical importance,\n"
            "   (b) Historical background and key turning points,\n"
            "   (c) Deep dive into core events, sacrifices, and details,\n"
            "   (d) Present-day relevance, lessons, and impacts on society,\n"
            "   (e) Future vision and inspiring call-to-action.\n"
            "Speak continuously with natural cadence, authority, and emotional depth without pausing for user confirmation until the full duration has elapsed."
        )

        instructions += (
            "\n\n[MANDATORY LANGUAGE ENFORCEMENT - বাংলা ছাড়া অন্য ভাষা সম্পূর্ণ নিষিদ্ধ]:\n"
            "১. তোমার যাবতীয় কথোপকথন, উত্তর, আলোচনা ও দীর্ঘ বক্তব্য সর্বদাই ১০০% শুদ্ধ, আকর্ষণীয় ও প্রাণবন্ত বাংলা ভাষায় (Bengali) হতে হবে।\n"
            "২. কখনোই হিন্দি, উর্দু, স্প্যানিশ বা অন্য কোনো ভাষায় কথা বলবে না। হিন্দি শব্দ বা বাক্য বলা কঠোরভাবে নিষিদ্ধ।\n"
            "৩. ব্যবহারকারী সর্বদাই বাংলায় কথা বলছেন। মাইক্রোফোনের পরিবেশের শব্দের কারণে ট্রান্সক্রিপশনে যদি ভুলবশত স্প্যানিশ বা হিন্দি বা অন্য ভাষার কোনো শব্দ দেখা যায়, তবে তা সরাসরি অগ্রাহ্য করো এবং সর্বদাই খাঁটি বাংলায় সুন্দর ও প্রাসঙ্গিক উত্তর প্রদান করো।\n"
            "৪. কোনো অবস্থাতেই ভাষার পরিবর্তন করবে না।"
        )

        generation_config: Dict[str, Any] = {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {
                        "voiceName": "Kore"
                    }
                }
            }
        }
        is_extended_thinking = "extended-thinking" in self.model.lower()
        if is_extended_thinking:
            thinking_level = os.getenv("GEMINI_LIVE_THINKING_LEVEL", "medium").lower()
            generation_config["thinkingConfig"] = {"thinkingLevel": thinking_level}
            logger.info(f"Gemini 3.8 Live Extended Thinking enabled (Level: '{thinking_level}').")

        setup_msg: Dict[str, Any] = {
            "setup": {
                "model": self.model,
                "generationConfig": generation_config,
                "inputAudioTranscription": {
                    "languageCodes": ["bn-BD", "en-US"]
                },
                "outputAudioTranscription": {
                    "languageCodes": ["bn-BD"]
                },
                "realtimeInputConfig": {
                    "automaticActivityDetection": {
                        "disabled": False,
                        "startOfSpeechSensitivity": "START_SENSITIVITY_HIGH",
                        "endOfSpeechSensitivity": "END_SENSITIVITY_HIGH",
                        "silenceDurationMs": 800
                    }
                },
                "contextWindowCompression": {
                    "slidingWindow": {}
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
                tool_decl: Dict[str, Any] = {
                    "name": s["name"],
                    "description": s["description"],
                    "parameters": self._normalize_gemini_schema(params)
                }
                if is_extended_thinking:
                    tool_decl["behavior"] = "NON_BLOCKING"
                gemini_tools.append(tool_decl)
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

        # Drop mic audio if LUMI is currently delivering a presentation / speech
        if getattr(self.state, "current_state", None) == BehaviorState.PRESENTING or (getattr(self, "turn_arbiter", None) and self.turn_arbiter.is_presenting()):
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

                if getattr(self.state, "current_state", None) == BehaviorState.PRESENTING or (getattr(self, "turn_arbiter", None) and self.turn_arbiter.is_presenting()):
                    try:
                        while not self._audio_queue.empty():
                            self._audio_queue.get_nowait()
                    except Exception:
                        pass
                    await asyncio.sleep(0.05)
                    continue

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
                    # Send video frame if camera is available (throttled to 4.0s to avoid turn starvation)
                    now = time.time()
                    if self.camera and self.camera.is_available() and (now - self._last_video_send > 4.0):
                        # Don't send video frame if head is in mid-pan movement (avoids motion blur)
                        is_head_moving = getattr(getattr(self, "gestures", None), "is_playing", False) or getattr(self, "_is_panning", False)
                        if not is_head_moving:
                            frame = self.camera.get_frame()
                            if frame is not None:
                                try:
                                    import cv2
                                    h, w = frame.shape[:2]
                                    if w > 320:
                                        frame = cv2.resize(frame, (320, int(h * 320 / w)))
                                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 35])
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
        was_interrupted_this_turn = False
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
                        server_content = data["serverContent"]
                        interact_status = server_content.get("interactionStatus") or server_content.get("interaction_status")
                        if interact_status:
                            logger.debug(f"[Gemini 3.8 Live] Interaction Status: {interact_status}")

                        # Print transcriptions if available
                        if "modelTurn" in server_content:
                            model_turn = server_content.get("modelTurn", {})
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
                                        if getattr(self.state, "current_state", None) != BehaviorState.PRESENTING:
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
                            was_interrupted_this_turn = True
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
                                if any('\u0900' <= char <= '\u097f' and char not in ('\u0964', '\u0965') for char in txt):
                                    logger.info(f"User transcription contains Devanagari phonemes: '{txt}'. Reminding Gemini user is speaking Bengali.")
                                    self.inject_context(
                                        "[ASR CLARIFICATION: The user is speaking Bengali (বাংলা). Ambient room acoustics produced non-Bengali phonemes. Always answer in pure, natural Bengali.]",
                                        trigger_response=False
                                    )
                                self._check_silence_command(txt)
                                self._check_presentation_command(txt)
                                self._check_eager_recall(txt)
                        if "outputAudioTranscription" in content:
                            txt = _get_text(content['outputAudioTranscription'])
                            print(f"🤖 [LUMI (Draft)]: {txt}")
                            if txt and (not lumi_buffer or txt not in lumi_buffer[-1]):
                                logger.debug(f"🤖 [LUMI]: {txt}")
                                lumi_buffer.append(txt)
                                if any('\u0900' <= char <= '\u097f' and char not in ('\u0964', '\u0965') for char in txt):
                                    logger.warning(f"Detected Hindi/Devanagari in LUMI draft: '{txt}'. Nudging Bengali lock.")
                                    self.inject_context(
                                        "[CRITICAL LANGUAGE LOCK: Hindi is STRICTLY FORBIDDEN! Immediately switch and speak ONLY in pure, natural Bengali (বাংলা)!]",
                                        trigger_response=False
                                    )
                                elif getattr(self, "_active_speech_target_end", 0.0) > time.time():
                                    import re
                                    if re.search(r"(?:আমি\s*কি\s*(?:বলতেই|বলতে|আরো|আরও)\s*থাকব|আমি\s*কি\s*(?:আরো|আরও)\s*বলব)", txt):
                                        logger.warning(f"Detected check-in phrase in speech: '{txt}'. Nudging continuous monologue.")
                                        self.inject_context(
                                            f"[DO NOT ASK PERMISSION]: Continue speaking on '{self._active_speech_topic}' continuously. Do not ask check-in questions!",
                                            trigger_response=False
                                        )
                        if "inputTranscription" in content:
                            txt = _get_text(content['inputTranscription'])
                            print(f"🗣️  [USER]: {txt}")
                            if txt:
                                logger.debug(f"🗣️  [USER]: {txt}")
                                user_buffer.append(txt)
                                if any('\u0900' <= char <= '\u097f' and char not in ('\u0964', '\u0965') for char in txt):
                                    logger.info(f"User transcription contains Devanagari phonemes: '{txt}'. Reminding Gemini user is speaking Bengali.")
                                    self.inject_context(
                                        "[ASR CLARIFICATION: The user is speaking Bengali (বাংলা). Ambient room acoustics produced non-Bengali phonemes. Always answer in pure, natural Bengali.]",
                                        trigger_response=False
                                    )
                                self._check_silence_command(txt)
                                self._check_presentation_command(txt)
                                self._check_eager_recall(txt)
                        if "outputTranscription" in content:
                            txt = _get_text(content['outputTranscription'])
                            print(f"🤖 [LUMI (Draft)]: {txt}")
                            if txt and (not lumi_buffer or txt not in lumi_buffer[-1]):
                                logger.debug(f"🤖 [LUMI]: {txt}")
                                lumi_buffer.append(txt)
                                if any('\u0900' <= char <= '\u097f' and char not in ('\u0964', '\u0965') for char in txt):
                                    logger.warning(f"Detected Hindi/Devanagari in LUMI draft: '{txt}'. Nudging Bengali lock.")
                                    self.inject_context(
                                        "[CRITICAL LANGUAGE LOCK: Hindi is STRICTLY FORBIDDEN! Immediately switch and speak ONLY in pure, natural Bengali (বাংলা)!]",
                                        trigger_response=False
                                    )
                                elif getattr(self, "_active_speech_target_end", 0.0) > time.time():
                                    import re
                                    if re.search(r"(?:আমি\s*কি\s*(?:বলতেই|বলতে|আরো|আরও)\s*থাকব|আমি\s*কি\s*(?:আরো|আরও)\s*বলব)", txt):
                                        logger.warning(f"Detected check-in phrase in speech: '{txt}'. Nudging continuous monologue.")
                                        self.inject_context(
                                            f"[DO NOT ASK PERMISSION]: Continue speaking on '{self._active_speech_topic}' continuously. Do not ask check-in questions!",
                                            trigger_response=False
                                        )

                        if "interimInputTranscription" in content:
                            txt = _get_text(content['interimInputTranscription'])
                            if txt:
                                logger.debug(f"🗣️  [USER (interim)]: {txt}")
                                self.wake_up(15.0)

                        if content.get("waitingForInput"):
                            logger.debug("Gemini Live waiting for user input...")

                        # End of turn detection
                        if content.get("turnComplete"):
                            u_text = " ".join(user_buffer).strip()
                            l_text = " ".join(lumi_buffer).strip()
                            if u_text or l_text:
                                self.event_bus.emit("conversation.turn_complete", data={"user": u_text, "lumi": l_text})

                            # Check presentation command on the full turn transcription
                            if u_text and getattr(self.state, "current_state", None) != BehaviorState.PRESENTING:
                                self._check_presentation_command(u_text)

                            # Active Continuous Speech Session Tracker
                            now = time.time()
                            is_active_speech = getattr(self, "_active_speech_target_end", 0.0) > (now + 15.0)
                            if is_active_speech and not self.is_silent():
                                rem_sec = int(self._active_speech_target_end - now)
                                self._speech_continuation_count = getattr(self, "_speech_continuation_count", 0) + 1
                                logger.info(
                                    f"[SPEECH SESSION] Continuing speech on '{self._active_speech_topic}': "
                                    f"{rem_sec}s remaining (Pass {self._speech_continuation_count})."
                                )
                                cont_prompt = (
                                    f"[CONTINUOUS SPEECH SESSION: {rem_sec} SECONDS REMAINING]:\n"
                                    f"Continue your spoken monologue on '{self._active_speech_topic}' immediately! "
                                    f"Do not stop and NEVER ask 'আমি কি বলতেই থাকবো?' or any check-in questions. "
                                    f"Seamlessly transition into the next chapter: present-day impact, real-world examples, future vision, and inspiring concluding thoughts. "
                                    f"Speak continuously with rich, eloquent Bengali sentences without waiting for the user!"
                                )
                                self.inject_context(cont_prompt, trigger_response=True)
                            elif getattr(self, "_active_speech_target_end", 0.0) > 0.0 and getattr(self, "_active_speech_target_end", 0.0) <= (now + 15.0):
                                logger.info(f"[SPEECH SESSION] Speech on '{self._active_speech_topic}' completed full duration.")
                                self._active_speech_target_end = 0.0
                                self._active_speech_topic = ""
                                if getattr(self, "turn_arbiter", None):
                                    self.turn_arbiter.set_presenting(False)

                            # Dual-safety net: If Gemini generated text but no audio streamed,
                            # synthesize via BanglaTTS and play so the robot is NEVER mute!
                            # Guard: Do NOT synthesize if user interrupted/barged-in (they deliberately stopped LUMI)
                            if l_text and not turn_had_audio and not was_interrupted_this_turn and not self.is_silent():
                                logger.info(f"Gemini Live returned text without audio stream. Running fallback TTS for: '{l_text[:40]}...'")
                                try:
                                    tts_file = self.tts.synthesize(l_text)
                                    if tts_file and hasattr(self.speaker, "play_file"):
                                        self.speaker.play_file(tts_file, block=False)
                                except Exception as e:
                                    logger.warning(f"Fallback TTS synthesis error: {e}")

                            turn_had_audio = False
                            was_interrupted_this_turn = False
                            user_buffer.clear()
                            lumi_buffer.clear()

                            # Clear software AEC speaker ducking so user's subsequent reply is not dropped
                            self._speaker_active_until = min(getattr(self, "_speaker_active_until", 0.0), time.time())

                            if getattr(self, "turn_arbiter", None):
                                self.turn_arbiter.notify_speaker_stopped()
                                self.turn_arbiter.wake_up(25.0)

                            # Transition state back to LISTENING if awake (only if not delivering presentation/speech), else IDLE
                            if self.state and hasattr(self.state, "transition_to"):
                                if getattr(self.state, "current_state", None) != BehaviorState.PRESENTING and not (getattr(self, "turn_arbiter", None) and self.turn_arbiter.is_presenting()) and not is_active_speech:
                                    if self.is_awake():
                                        self.state.transition_to(BehaviorState.LISTENING, reason="turn_complete_listening")
                                        self.wake_up(25.0)
                                    else:
                                        self.state.transition_to(BehaviorState.IDLE, reason="turn_complete_idle")

                            # Smoothly return arms to neutral rest pose when turn finishes (unless presenting or in speech session)
                            if getattr(self.state, "current_state", None) != BehaviorState.PRESENTING and not (getattr(self, "turn_arbiter", None) and self.turn_arbiter.is_presenting()) and not is_active_speech:
                                if self.gestures and hasattr(self.gestures, "idle_pose"):
                                    self.gestures.play_async(self.gestures.idle_pose, name="turn_complete_rest")
                            
                    # Handle Tool Calls (support top-level toolCall and serverContent.toolCall)
                    tool_call_data = data.get("toolCall")
                    if not tool_call_data and "serverContent" in data and isinstance(data["serverContent"], dict):
                        tool_call_data = data["serverContent"].get("toolCall")

                    if tool_call_data and "functionCalls" in tool_call_data:
                        function_responses = []
                        for call in tool_call_data.get("functionCalls", []):
                            name = call.get("name")
                            call_id = call.get("id")
                            args = call.get("args", {})
                            
                            if name and self.tools and name in self.tools.tools:
                                logger.info(f"Gemini requested tool: {name} (id: {call_id})")
                                try:
                                    tool_func = self.tools.tools[name]
                                    result = await asyncio.wait_for(
                                        asyncio.to_thread(tool_func, **args),
                                        timeout=12.0
                                    )
                                    if name == "set_silent_mode":
                                        dur = args.get("duration_seconds", 300.0) or 300.0
                                        self.set_silent_until(time.time() + float(dur))
                                    elif name == "start_presentation":
                                        dur = float(args.get("duration_minutes", 3.0) or 3.0)
                                        topic = args.get("topic", "বক্তব্য")
                                        self._active_speech_target_end = time.time() + (dur * 60.0)
                                        self._active_speech_topic = topic
                                        self._speech_continuation_count = 0
                                        if getattr(self, "turn_arbiter", None):
                                            self.turn_arbiter.set_presenting(True)
                                            if hasattr(self.turn_arbiter, "wake_up"):
                                                self.turn_arbiter.wake_up(dur * 60.0 + 30.0)
                                    elif name == "stop_presentation":
                                        self._active_speech_target_end = 0.0
                                        self._active_speech_topic = ""
                                        if getattr(self, "turn_arbiter", None):
                                            self.turn_arbiter.set_presenting(False)
                                        if self.speaker:
                                            try:
                                                self.speaker.stop_stream()
                                            except Exception:
                                                pass
                                except asyncio.TimeoutError:
                                    logger.error(f"Tool '{name}' execution timed out after 12.0s.")
                                    result = f"Error: Tool '{name}' execution timed out."
                                except Exception as e:
                                    logger.error(f"Tool '{name}' execution error: {e}")
                                    result = f"Error: {e}"
                            else:
                                logger.warning(f"Gemini requested unknown tool: {name}")
                                result = f"Error: Tool '{name}' is not available."
                                    
                            func_resp: Dict[str, Any] = {
                                "name": name,
                                "response": {"result": result if isinstance(result, (dict, list, str, int, float, bool)) else str(result)}
                            }
                            if call_id:
                                func_resp["id"] = call_id
                            function_responses.append(func_resp)

                        # CRITICAL: Send ALL function responses in a SINGLE toolResponse message.
                        # Splitting multiple tool calls into separate messages breaks the Gemini Live WebSocket protocol!
                        if function_responses:
                            resp = {
                                "toolResponse": {
                                    "functionResponses": function_responses
                                }
                            }
                            await ws.send(json.dumps(resp))
                            logger.info(f"✅ Sent consolidated toolResponse with {len(function_responses)} function(s) to Gemini Live.")
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
        lock = getattr(self, "_inject_lock", None)
        if lock is None:
            self._inject_lock = threading.Lock()
            lock = self._inject_lock

        with lock:
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
                
            # Gemini Live API: To trigger an immediate response turn from Gemini, send clientContent with turnComplete=True.
            # For background assimilation without interruption, send via realtimeInput.
            if trigger_response:
                event = {
                    "clientContent": {
                        "turns": [
                            {
                                "role": "user",
                                "parts": [{"text": text}]
                            }
                        ],
                        "turnComplete": True
                    }
                }
            else:
                event = {
                    "realtimeInput": {
                        "text": text
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

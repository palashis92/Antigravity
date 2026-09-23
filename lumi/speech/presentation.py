"""Presentation Mode Engine for LUMI.

Enables structured, continuous, multi-minute Bengali speech delivery
for civic, public, and formal community gatherings without premature
conversational turn interruptions or false room-reverberation barge-in.
"""

from __future__ import annotations

import os
import re
import threading
import time
from typing import Any, Dict, List, Optional

from ..audio.speaker import SpeakerInterface
from ..core.event_bus import EventBus
from ..core.logger import get_logger
from ..core.state_manager import BehaviorState, StateManager
from ..eyes.renderer import EyeRenderer
from ..motion.gestures import GestureManager
from .tts import BanglaTTS

logger = get_logger("speech.presentation")


class PresentationEngine:
    """Orchestrates continuous monologue presentation delivery for LUMI."""

    def __init__(
        self,
        tts: BanglaTTS,
        speaker: SpeakerInterface,
        gestures: Optional[GestureManager],
        eyes: Optional[EyeRenderer],
        state: StateManager,
        event_bus: EventBus,
        realtime_voice: Optional[Any] = None,
        turn_arbiter: Optional[Any] = None,
    ) -> None:
        self.tts = tts
        self.speaker = speaker
        self.gestures = gestures
        self.eyes = eyes
        self.state = state
        self.event_bus = event_bus
        self.realtime_voice = realtime_voice
        self.turn_arbiter = turn_arbiter

        self._lock = threading.Lock()
        self._is_presenting = False
        self._stop_requested = False
        self._presentation_thread: Optional[threading.Thread] = None
        self._current_topic: str = ""

    def is_presenting(self) -> bool:
        """Return True if a presentation is currently being delivered."""
        with self._lock:
            return self._is_presenting

    def check_stop_command(self, text: str) -> bool:
        """Check if incoming utterance is an explicit command to halt the presentation."""
        clean = text.lower().strip()
        stop_patterns = [
            r"থাম(?:ো|েন|িস)?",
            r"বক্তব্য\s*বন্ধ",
            r"বক্তব্য\s*থামা(?:ও|ন)",
            r"কথা\s*বন্ধ",
            r"চুপ\s*কর(?:ো|েন)?",
            r"বক্তব্য\s*সমাপ্ত",
            r"stop\s*(?:presentation|speech|talking)",
            r"cancel\s*(?:presentation|speech)",
            r"halt\s*(?:speech|presentation)",
        ]
        return any(re.search(pat, clean) for pat in stop_patterns)

    def generate_speech_script(
        self,
        topic: str,
        duration_minutes: float = 3.0,
        audience: str = "উপস্থিত সুধীবৃন্দ",
        key_points: str = "",
    ) -> List[str]:
        """Generate structured Bengali speech segments tailored for public address.
        
        Scales paragraph count and content depth strictly based on duration_minutes
        (~110-120 Bengali words per minute, so 5 minutes produces ~550-650 words).
        Attempts dynamic LLM generation (via google-genai, direct REST, or OpenAI),
        with a deterministic high-grade topic-adaptive multi-paragraph fallback.
        """
        # Target ~110-120 Bengali words per minute
        word_target = int(max(duration_minutes, 0.5) * 120)
        target_paragraphs = max(4, min(12, int(duration_minutes * 2)))

        # 1. Try LLM generation if API key is present
        api_key = os.getenv("GEMINI_API_KEY")
        prompt = (
            f"You are LUMI, an intelligent and eloquent physical AI companion robot delivering a formal, inspiring Bengali speech.\n"
            f"Topic: {topic}\n"
            f"Target Audience: {audience}\n"
            f"Special Key Points: {key_points or 'None'}\n"
            f"Requested Duration: {duration_minutes:.1f} minutes (Strict Target: ~{word_target} spoken Bengali words).\n\n"
            f"INSTRUCTIONS FOR CONTINUOUS SPEECH DELIVERY:\n"
            f"1. You must write a complete, in-depth monologue of exactly {target_paragraphs} substantial paragraphs separated by a blank line (double newline).\n"
            f"2. Each paragraph must be rich, articulate, and 50 to 70 words long, so the total speech genuinely lasts {duration_minutes:.1f} minutes when spoken aloud (~{word_target} words).\n"
            f"3. Cover: (a) Respectful formal greetings & importance of {topic}, (b) Foundational background & historical context, "
            f"(c) Present-day real-world situation & developments, (d) Direct positive impacts on {audience}, "
            f"(e) Specific key points ({key_points or 'practical solutions'}), (f) Practical challenges & how to overcome them, "
            f"(g) Modern technological and social advancements, (h) Future vision & long-term goals, (i) Inspiring call to action, (j) Formal gratitude & closing blessings.\n"
            f"4. CRITICAL RULES: This is a ONE-WAY MONOLOGUE. Do NOT ask any questions back to the audience. Do NOT write short summaries. "
            f"Do NOT include stage directions, bullet points, asterisks, or markdown formatting. Output ONLY the pure spoken Bengali text."
        )

        if api_key:
            # Try Tier 1: modern google.genai SDK
            try:
                from google import genai
                from google.genai import types

                client = genai.Client(api_key=api_key)
                models_to_try = [os.getenv("GEMINI_FLASH_MODEL", "gemini-3-flash-preview"), "gemini-3-flash-preview", "gemini-3.7-flash", "gemini-flash-latest"]
                models_to_try = list(dict.fromkeys(models_to_try))

                for m in models_to_try:
                    try:
                        resp = client.models.generate_content(
                            model=m,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                temperature=0.7,
                                max_output_tokens=3000,
                                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                            ),
                        )
                        if resp and resp.text:
                            raw_text = resp.text.strip()
                            paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
                            if len(paragraphs) >= 3:
                                logger.info(f"Generated {len(paragraphs)} presentation segments via google.genai SDK ({sum(len(p.split()) for p in paragraphs)} words).")
                                return paragraphs
                    except Exception as m_err:
                        logger.debug(f"SDK presentation model {m} failed: {m_err}")
            except Exception as e:
                logger.debug(f"google.genai SDK speech generation error: {e}")

            # Try Tier 2: Direct REST call via urllib.request (zero third-party dependency)
            try:
                import json
                import urllib.request

                for m in [os.getenv("GEMINI_FLASH_MODEL", "gemini-3-flash-preview"), "gemini-3-flash-preview", "gemini-3.7-flash"]:
                    try:
                        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key}"
                        payload = {
                            "contents": [{"parts": [{"text": prompt}]}],
                            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 3000},
                        }
                        req = urllib.request.Request(
                            url,
                            data=json.dumps(payload).encode("utf-8"),
                            headers={"Content-Type": "application/json"},
                            method="POST",
                        )
                        with urllib.request.urlopen(req, timeout=12) as response:
                            data = json.loads(response.read().decode("utf-8"))
                        cand = data.get("candidates", [])[0]
                        raw_text = cand.get("content", {}).get("parts", [])[0].get("text", "")
                        paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
                        if len(paragraphs) >= 3:
                            logger.info(f"Generated {len(paragraphs)} presentation segments via Gemini REST API ({sum(len(p.split()) for p in paragraphs)} words).")
                            return paragraphs
                    except Exception as e_m:
                        logger.debug(f"Gemini REST attempt ({m}) failed: {e_m}")
                        continue
            except Exception as e:
                logger.debug(f"Direct Gemini REST API speech generation error: {e}")

        # Try Tier 3: OpenAI API if configured
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                from openai import OpenAI

                client = OpenAI(api_key=openai_key)
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=3000,
                )
                if resp and resp.choices:
                    raw_text = resp.choices[0].message.content.strip()
                    paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
                    if len(paragraphs) >= 3:
                        logger.info(f"Generated {len(paragraphs)} presentation segments via OpenAI ({sum(len(p.split()) for p in paragraphs)} words).")
                        return paragraphs
            except Exception as e:
                logger.debug(f"OpenAI speech generation error: {e}")

        # 2. Deterministic high-quality Bengali Civic & Topic-Adaptive Speech Generator
        return self._generate_fallback_script(topic, duration_minutes, audience, key_points, target_paragraphs)

    def _generate_fallback_script(
        self,
        topic: str,
        duration_minutes: float,
        audience: str,
        key_points: str,
        target_paragraphs: int,
    ) -> List[str]:
        """Deterministic topic-adaptive multi-paragraph Bengali speech generator."""
        p_greeting = (
            f"বিসমিল্লাহির রাহমানির রাহিম। পরম করুণাময় মহান সৃষ্টিকর্তার নামে শুরু করছি। "
            f"আজকের এই গুরুত্বপূর্ণ সমাবেশে উপস্থিত সম্মানিত {audience}, আসসালামু আলাইকুম এবং আন্তরিক শুভেচ্ছা। "
            f"একটি সুন্দর, সুশৃঙ্খল ও সমৃদ্ধ সমাজ গঠনে আপনাদের সকলের সক্রিয় ও স্বতঃস্ফূর্ত উপস্থিতি আমাদের গভীর অনুপ্রেরণা যোগায়।"
        )

        p_intro = (
            f"আজকের এই আয়োজনে আমাদের মূল আলোচ্য বিষয় হচ্ছে: {topic}। "
            f"আমাদের সমাজ, দৈনন্দিন জীবনযাত্রা এবং ভবিষ্যতের সার্বিক কল্যাণে এই বিষয়ের গুরুত্ব ও তাৎপর্য অপরিসীম। "
            f"যেকোনো জাতিকে উন্নতির পথে দৃঢ়ভাবে এগিয়ে নিতে হলে সচেতনতা এবং বিষয়ভিত্তিক স্পষ্ট ধারণাই সবচেয়ে বড় নিয়ামক হিসেবে কাজ করে।"
        )

        p_foundation = (
            f"যদি আমরা এই বিষয়ের পেছনের পটভূমি ও প্রাসঙ্গিকতার দিকে তাকাই, তবে দেখতে পাব যে সময়ের পরিবর্তনের সাথে সাথে "
            f"{topic} আমাদের প্রতিটি স্তরকে প্রভাবিত করছে। অতীতে আমরা যেভাবে চিন্তা করতাম, বর্তমান আধুনিক যুগে এসে "
            f"আমাদের সেই ভাবনায় অনেক ইতিবাচক রূপান্তর এসেছে এবং এর মাধ্যমে নতুন নতুন সম্ভাবনার দ্বার উন্মোচিত হয়েছে।"
        )

        p_context = (
            f"বাস্তব অভিজ্ঞতার আলোকে বিচার করলে স্পষ্ট বোঝা যায়, {topic} শুধুমাত্র কোনো তত্ত্বীয় বিষয় নয়। "
            f"বরং এর প্রতিটি দিক আমাদের প্রতিটি নাগরিকের জীবনমান উন্নয়ন, কর্মদক্ষতা বৃদ্ধি এবং সামাজিক ভারসাম্যের সাথে ওতপ্রোতভাবে জড়িত। "
            f"সঠিক কৌশল, সুদূরপ্রসারী চিন্তা ও বাস্তবমুখী দৃষ্টিভঙ্গি নিয়ে সম্মিলিতভাবে এগিয়ে গেলে আমরা এর সর্বোত্তম সুফল সহজেই অর্জন করতে সক্ষম হব।"
        )

        if key_points:
            p_keypoints = (
                f"বিশেষ করে এই বিষয়ে আমাদের যে মূল বিষয়গুলোর ওপর সর্বাধিক গুরুত্ব দেওয়া প্রয়োজন তা হলো: {key_points}। "
                f"উন্নয়ন ও অগ্রগতির প্রতিটি পদক্ষেপে স্বচ্ছতা, জবাবদিহিতা এবং আমাদের সকলের সক্রিয় অংশগ্রহণ নিশ্চিত করতে হবে। "
                f"সুনির্দিষ্ট এই লক্ষ্যগুলো অর্জনে পরিকল্পিত ও নিয়মতান্ত্রিক কর্মপরিকল্পনা বাস্তবায়নই আমাদের প্রত্যাশিত সাফল্য এনে দেবে।"
            )
        else:
            p_keypoints = (
                f"আমাদের উন্নয়ন ও অগ্রগতির প্রতিটি পদক্ষেপে সঠিক তথ্য, পারস্পরিক সহযোগিতা এবং আন্তরিক দায়বদ্ধতা বজায় রাখা আবশ্যক। "
                f"শিক্ষা, স্বাস্থ্য, সামাজিক নিরাপত্তা এবং আধুনিক প্রযুক্তির সমন্বয় ঘটিয়ে প্রতিটি মানুষের কাছে এর সুফল "
                f"পৌঁছে দিতে আমরা সবাই ঐক্যবদ্ধভাবে কাজ করতে আজ অঙ্গীকারবদ্ধ।"
            )

        p_impact = (
            f"যখন আমরা সফলভাবে এই উদ্যোগগুলো বাস্তবায়ন করব, তখন সমাজে একটি সুদূরপ্রসারী ইতিবাচক পরিবর্তন সূচিত হবে। "
            f"এর ফলে কেবল বর্তমান প্রজন্মই উপকৃত হবে না, বরং আমাদের ভবিষ্যৎ প্রজন্মের জন্যও একটি নিরাপদ, টেকসই এবং "
            f"সমৃদ্ধ ভিত্তি রচিত হবে, যা আমাদের স্বনির্ভরতার পথে বহু ধাপ এগিয়ে নিয়ে যাবে এবং একটি উজ্জ্বল ভবিষ্যতের নিশ্চয়তা দেবে।"
        )

        p_challenges = (
            f"তবে যেকোনো বড় লক্ষ্য অর্জনের পথে কিছু প্রতিবন্ধকতা ও চ্যালেঞ্জ থাকা অত্যন্ত স্বাভাবিক। "
            f"রিসোর্সের সঠিক ব্যবহার, প্রাথমিক অনীহা কিংবা অসচেতনতা অনেক সময় কাজের গতি কিছুটা মন্থর করতে পারে। "
            f"কিন্তু ধৈর্য, শৃঙ্খলা এবং সম্মিলিত সদিচ্ছা থাকলে কোনো বাধাই স্থায়ী হতে পারে না; সঠিক দিকনির্দেশনা ও পরিকল্পনাই প্রতিটি সংকট সমাধানের চাবিকাঠি।"
        )

        p_action = (
            f"এখানে উপস্থিত সম্মানিত {audience}, আপনারা প্রত্যেকেই এই পরিবর্তনের অন্যতম প্রধান অংশীদার। "
            f"আমরা যদি প্রত্যেকে নিজ নিজ অবস্থান থেকে দায়িত্বশীল ভূমিকা পালন করি এবং একে অপরের পাশে দাঁড়াই, তবে "
            f"{topic} নিয়ে আমাদের সকল স্বপ্ন ও পরিকল্পনা বাস্তবে রূপ নেবে। আত্মবিশ্বাস ও ইতিবাচক মানসিকতাই আমাদের অগ্রযাত্রার প্রধান চালিকাশক্তি।"
        )

        p_future = (
            f"ভবিষ্যতের রূপরেখা যদি আমরা পর্যবেক্ষণ করি, তবে আধুনিক বিশ্বের সাথে তাল মিলিয়ে আমাদেরও নিত্যনতুন প্রযুক্তি, "
            f"জ্ঞান এবং সৃজনশীলতাকে সানন্দে গ্রহণ করতে হবে। পরিবর্তনের এই ধারায় নিজেদেরকে সবসময় যুগোপযোগী করে গড়ে তুলতে হবে, "
            f"যাতে আমরা যে কোনো বৈশ্বিক ও স্থানীয় চ্যালেঞ্জ মোকাবিলায় সবসময় সম্পূর্ণ সক্ষম, আত্মপ্রত্যয়ী ও প্রস্তুত থাকি।"
        )

        p_conclusion = (
            f"পরিশেষে আমি উদাত্ত আহ্বান জানাই, আসুন আমরা সকল ভেদাভেদ ভুলে একটি সুন্দর, ন্যায়নিষ্ঠ, সুসংগঠিত ও আধুনিক সমাজ বিনির্মাণে "
            f"একসাথে কাঁধে কাঁধ মিলিয়ে কাজ করি। আমাদের পারস্পরিক সৌহার্দ্য, সম্মিলিত নিষ্ঠা ও অটল সততাই আমাদের কাঙ্ক্ষিত জাতীয় অগ্রযাত্রাকে দ্রুত ত্বরান্বিত করবে।"
        )

        p_close = (
            f"আপনারা অত্যন্ত ধৈর্য, আন্তরিকতা ও গভীর মনোযোগ সহকারে আমার এই বক্তব্য শুনেছেন, এজন্য আমি আপনাদের সবাইকে "
            f"অন্তরের অন্তস্তল থেকে আন্তরিক ধন্যবাদ ও মোবারকবাদ জানাচ্ছি। মহান আল্লাহ আমাদের সবাইকে সুস্থ, সুন্দর ও শান্তিতে রাখুন। "
            f"ধন্যবাদ সবাইকে। খোদা হাফেজ।"
        )

        all_pool = [
            p_greeting,
            p_intro,
            p_foundation,
            p_context,
            p_keypoints,
            p_impact,
            p_challenges,
            p_action,
            p_future,
            p_conclusion,
            p_close,
        ]

        if target_paragraphs <= 5:
            # 5 core paragraphs
            chosen = [p_greeting, p_intro, p_keypoints, p_action, p_close]
        elif target_paragraphs <= 8:
            chosen = [p_greeting, p_intro, p_foundation, p_context, p_keypoints, p_challenges, p_action, p_close]
        else:
            # Full 11 paragraphs for 5+ minutes
            chosen = all_pool

        return chosen

    def start_presentation(
        self,
        topic: str,
        duration_minutes: float = 3.0,
        audience: str = "উপস্থিত সুধীবৃন্দ",
        key_points: str = "",
    ) -> str:
        """Initiate presentation delivery on a background worker thread."""
        with self._lock:
            if self._is_presenting:
                return "লুমি ইতিমধ্যে একটি বক্তব্য প্রদান করছে। বক্তব্য থামাতে বললে 'লুমি থামো' বলুন।"

            self._is_presenting = True
            self._stop_requested = False
            self._current_topic = topic

        # Transition behavior state to PRESENTING
        self.state.transition_to(BehaviorState.PRESENTING, reason=f"start_presentation:{topic[:20]}")
        if self.turn_arbiter and hasattr(self.turn_arbiter, "set_presenting"):
            self.turn_arbiter.set_presenting(True)
            if hasattr(self.turn_arbiter, "wake_up"):
                self.turn_arbiter.wake_up(duration_minutes * 60.0 + 30.0)
        if self.eyes and hasattr(self.eyes, "set_expression"):
            self.eyes.set_expression("speaking")

        duration_s = duration_minutes * 60.0
        is_gemini_running = (
            self.realtime_voice is not None
            and getattr(self.realtime_voice, "_running", False)
        )
        if is_gemini_running:
            self.realtime_voice._active_speech_target_end = time.time() + duration_s
            self.realtime_voice._active_speech_topic = topic
            self.realtime_voice._speech_continuation_count = 0

            # Gemini Live delivers speech directly over WebSocket.
            # Start background gesture accompaniment worker immediately without blocking.
            self._presentation_thread = threading.Thread(
                target=self._delivery_worker,
                args=([], topic),
                name="LumiPresentationWorker",
                daemon=True,
            )
            self._presentation_thread.start()

            msg = f"বক্তব্য শুরু হচ্ছে: '{topic}'।"
            logger.info(msg)
            return msg

        # Offline / Fallback mode: Generate script and deliver via TTS on background worker
        def _bg_generate_and_deliver():
            segments = self.generate_speech_script(
                topic=topic,
                duration_minutes=duration_minutes,
                audience=audience,
                key_points=key_points,
            )
            self._delivery_worker(segments, topic)

        self._presentation_thread = threading.Thread(
            target=_bg_generate_and_deliver,
            name="LumiPresentationWorker",
            daemon=True,
        )
        self._presentation_thread.start()

        msg = f"বক্তব্য শুরু হচ্ছে: '{topic}'।"
        logger.info(msg)
        return msg

    def _delivery_worker(self, segments: List[str], topic: str) -> None:
        """Sequential non-blocking speech delivery with gestural pacing."""
        logger.info(f"Presentation worker started for topic: '{topic}' ({len(segments)} segments).")
        try:
            voice_end = getattr(self.realtime_voice, "_active_speech_target_end", 0.0)
            is_gemini_active = (
                self.realtime_voice is not None
                and getattr(self.realtime_voice, "_running", False)
                and isinstance(voice_end, (int, float))
                and voice_end > time.time()
            )

            if is_gemini_active:
                logger.info(f"Gemini Live is actively streaming speech on '{topic}'. Worker providing co-verbal gesture accompaniment.")
                step_idx = 0
                while not self._stop_requested:
                    current_end = getattr(self.realtime_voice, "_active_speech_target_end", 0.0)
                    if not isinstance(current_end, (int, float)) or time.time() >= current_end:
                        break

                    # Dynamic expressive co-verbal gestures while Gemini streams speech
                    if self.gestures and not getattr(self.gestures, "is_playing", False):
                        if step_idx == 0 and hasattr(self.gestures, "greet"):
                            self.gestures.play_async(self.gestures.greet, name="pres_greet")
                        elif hasattr(self.gestures, "play_conversational_step"):
                            self.gestures.play_async(self.gestures.play_conversational_step, name=f"pres_step_{step_idx}")
                    step_idx += 1

                    # Sleep in responsive slices to detect stop requests immediately
                    for _ in range(30):
                        if self._stop_requested:
                            break
                        cur_end = getattr(self.realtime_voice, "_active_speech_target_end", 0.0)
                        if isinstance(cur_end, (int, float)) and time.time() >= cur_end:
                            break
                        time.sleep(0.1)
            else:
                # Offline / Fallback Monologue Delivery via TTS
                for idx, text in enumerate(segments):
                    if self._stop_requested:
                        logger.info("Presentation worker detected stop request. Halting delivery.")
                        break

                    logger.info(f"[Presentation Segment {idx + 1}/{len(segments)}]: {text[:50]}...")

                    # Dynamic expressive co-verbal gestures
                    if self.gestures and not getattr(self.gestures, "is_playing", False):
                        if idx == 0 and hasattr(self.gestures, "greet"):
                            self.gestures.play_async(self.gestures.greet, name="pres_greet")
                        elif hasattr(self.gestures, "play_conversational_step"):
                            self.gestures.play_async(self.gestures.play_conversational_step, name=f"pres_step_{idx}")

                    audio_path = self.tts.synthesize(text)
                    if self._stop_requested:
                        break

                    if audio_path and hasattr(self.speaker, "play_file"):
                        self.speaker.play_file(audio_path, block=True)

                    if self._stop_requested:
                        break

                    # Rhetorical pause between paragraphs (0.6 - 0.9s)
                    time.sleep(0.7)

        except Exception as e:
            logger.error(f"Error during presentation speech delivery: {e}")

        finally:
            with self._lock:
                was_stopped = self._stop_requested
                self._is_presenting = False
                self._stop_requested = False

            if self.turn_arbiter and hasattr(self.turn_arbiter, "set_presenting"):
                self.turn_arbiter.set_presenting(False)

            # Reset eyes and arms
            if self.eyes and hasattr(self.eyes, "set_expression"):
                self.eyes.set_expression("neutral")
            if self.gestures and hasattr(self.gestures, "idle_pose"):
                self.gestures.play_async(self.gestures.idle_pose, name="pres_idle_reset")

            # Transition cleanly back to LISTENING so audience can interact/ask questions
            if self.state.current_state == BehaviorState.PRESENTING:
                self.state.transition_to(BehaviorState.LISTENING, reason="presentation_complete")

            # Notify Gemini Live that presentation has finished and normal interaction resumes
            if self.realtime_voice:
                if was_stopped:
                    self.realtime_voice._active_speech_target_end = 0.0
                    self.realtime_voice._active_speech_topic = ""
                if hasattr(self.realtime_voice, "inject_context"):
                    try:
                        status_desc = "halted by user command" if was_stopped else "concluded successfully"
                        self.realtime_voice.inject_context(
                            f"[SYSTEM ALERT: Presentation on '{topic}' has {status_desc}. "
                            f"LUMI is now listening and ready to engage in normal conversation.]",
                            trigger_response=False
                        )
                    except Exception as e:
                        logger.debug(f"Presentation completion inject notice: {e}")

            self.event_bus.emit(
                "presentation.complete",
                data={"topic": topic, "stopped": was_stopped},
            )
            logger.info(f"Presentation finished (stopped={was_stopped}).")

    def stop_presentation(self, reason: str = "") -> str:
        """Halt active speech presentation immediately."""
        with self._lock:
            if not self._is_presenting:
                return "বর্তমানে কোনো বক্তব্য চালু নেই।"
            self._stop_requested = True
            self._is_presenting = False

        if self.turn_arbiter and hasattr(self.turn_arbiter, "set_presenting"):
            self.turn_arbiter.set_presenting(False)

        if self.realtime_voice:
            self.realtime_voice._active_speech_target_end = 0.0
            self.realtime_voice._active_speech_topic = ""

        logger.info(f"Halting presentation delivery. Reason: {reason or 'user_stop'}")

        # Stop speaker output immediately
        if self.speaker and hasattr(self.speaker, "stop_stream"):
            try:
                self.speaker.stop_stream()
            except Exception as e:
                logger.debug(f"Presentation speaker stop error: {e}")

        # Stop physical gestures
        if self.gestures and hasattr(self.gestures, "stop"):
            try:
                self.gestures.stop()
            except Exception as e:
                logger.debug(f"Presentation gestures stop error: {e}")

        # Restore eyes
        if self.eyes and hasattr(self.eyes, "set_expression"):
            try:
                self.eyes.set_expression("neutral")
            except Exception as e:
                logger.debug(f"Eyes reset error: {e}")

        # Transition back to LISTENING
        if self.state.current_state == BehaviorState.PRESENTING:
            self.state.transition_to(BehaviorState.LISTENING, reason=f"stop_presentation_{reason}")

        # Spoken confirmation of halt
        stop_notice = "বক্তব্য স্থগিত করা হলো।"
        if self.tts and self.speaker:
            try:
                audio_path = self.tts.synthesize(stop_notice)
                if audio_path:
                    self.speaker.play_file(audio_path, block=False)
            except Exception as e:
                logger.debug(f"Stop confirmation speech error: {e}")

        return "বক্তব্য সফলভাবে বন্ধ করা হয়েছে।"

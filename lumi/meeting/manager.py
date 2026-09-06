"""Meeting Manager for LUMI.

Coordinates silent meeting mode:
- Real-time utterance logging with speaker attribution and spatial DOA
- Persistent meeting session recording in SQLite
- Super-intelligent, executive-level Bengali meeting analysis using LLMs (Gemini / GPT-4)
- Automated PDF meeting minutes and action item generation
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from ..core.logger import get_logger
from ..memory.database import Database

logger = get_logger("meeting.manager")


@dataclass
class MeetingUtterance:
    """A single speech segment recorded during a meeting."""
    speaker: str
    text: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    time_str: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))
    doa_deg: Optional[float] = None
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MeetingUtterance:
        return cls(
            speaker=data.get("speaker", "Unknown"),
            text=data.get("text", ""),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            time_str=data.get("time_str", datetime.now().strftime("%H:%M:%S")),
            doa_deg=data.get("doa_deg"),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class MeetingSession:
    """A complete recorded meeting session."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    ended_at: Optional[str] = None
    duration_sec: float = 0.0
    utterances: List[MeetingUtterance] = field(default_factory=list)
    participants: Set[str] = field(default_factory=set)
    summary: str = ""
    key_points: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    analysis: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_transcript_text(self) -> str:
        """Format utterances into a clean chronological dialogue transcript."""
        if not self.utterances:
            return "কোনো কথোপকথন রেকর্ড করা হয়নি।"
        
        lines = []
        for u in self.utterances:
            dir_tag = ""
            if u.doa_deg is not None and "(" not in u.speaker:
                if u.doa_deg < -20:
                    dir_tag = " (বামদিক)"
                elif u.doa_deg > 20:
                    dir_tag = " (ডানদিক)"
                else:
                    dir_tag = " (মাঝখান)"
            lines.append(f"[{u.time_str}] {u.speaker}{dir_tag}: {u.text}")
        return "\n".join(lines)


class MeetingManager:
    """Manages active meeting lifecycle, transcript collection, and AI analysis."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self._active_session: Optional[MeetingSession] = None
        self._last_session: Optional[MeetingSession] = None
        self._lock = threading.RLock()
        self._load_latest_meeting_from_db()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def is_meeting_active(self) -> bool:
        """Check if meeting mode is currently ongoing."""
        with self._lock:
            return self._active_session is not None

    def start_meeting(self, title: str = "") -> MeetingSession:
        """Start a new silent meeting session."""
        with self._lock:
            if self._active_session:
                logger.warning(f"Meeting '{self._active_session.id}' is already active. Stopping it first.")
                self.stop_meeting()

            now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            session_title = title.strip() or f"মিটিং ({now_str})"
            session = MeetingSession(title=session_title)
            self._active_session = session
            logger.info(f"🎙️ Meeting Mode STARTED: '{session.title}' (ID: {session.id})")
            return session

    def stop_meeting(self) -> Optional[MeetingSession]:
        """End the current meeting session and save it to SQLite."""
        with self._lock:
            if not self._active_session:
                logger.warning("No active meeting to stop.")
                return self._last_session

            session = self._active_session
            session.ended_at = datetime.now().isoformat()
            try:
                start_dt = datetime.fromisoformat(session.started_at)
                end_dt = datetime.fromisoformat(session.ended_at)
                session.duration_sec = max(0.0, (end_dt - start_dt).total_seconds())
            except Exception:
                session.duration_sec = 0.0

            # Collect unique participants
            for u in session.utterances:
                if u.speaker:
                    session.participants.add(u.speaker)

            # Save to persistent database
            self.save_meeting_to_db(session)
            self._last_session = session
            self._active_session = None

            logger.info(
                f"🛑 Meeting Mode STOPPED: '{session.title}' "
                f"({len(session.utterances)} utterances, {session.duration_sec:.1f}s)"
            )
            return session

    def add_utterance(
        self,
        speaker: str,
        text: str,
        doa_deg: Optional[float] = None,
        confidence: float = 1.0,
    ) -> None:
        """Append an utterance to the active meeting transcript."""
        with self._lock:
            if not self._active_session:
                return

            text = text.strip()
            if not text:
                return

            utterance = MeetingUtterance(
                speaker=speaker or "বক্তা",
                text=text,
                doa_deg=doa_deg,
                confidence=confidence,
            )
            self._active_session.utterances.append(utterance)
            self._active_session.participants.add(utterance.speaker)

            dir_str = f" [DOA: {doa_deg:.0f}°]" if doa_deg is not None else ""
            logger.info(f"📝 [MEETING NOTE] {utterance.speaker}{dir_str}: {utterance.text}")

    def get_active_meeting(self) -> Optional[MeetingSession]:
        """Retrieve current active meeting session."""
        with self._lock:
            return self._active_session

    def get_last_meeting(self) -> Optional[MeetingSession]:
        """Retrieve the most recently completed meeting session."""
        with self._lock:
            return self._last_session

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_meeting_to_db(self, session: MeetingSession) -> None:
        """Save or update meeting record in SQLite."""
        transcript_json = json.dumps([u.to_dict() for u in session.utterances], ensure_ascii=False)
        participants_json = json.dumps(list(session.participants), ensure_ascii=False)
        key_points_json = json.dumps(session.key_points, ensure_ascii=False)
        action_items_json = json.dumps(session.action_items, ensure_ascii=False)
        metadata_json = json.dumps(session.metadata, ensure_ascii=False)

        query = """
        INSERT OR REPLACE INTO meetings (
            id, title, started_at, ended_at, duration_sec,
            transcript, summary, key_points, action_items,
            participants, metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            session.id,
            session.title,
            session.started_at,
            session.ended_at or "",
            session.duration_sec,
            transcript_json,
            session.summary,
            key_points_json,
            action_items_json,
            participants_json,
            metadata_json,
        )

        try:
            with self.db.get_connection() as conn:
                conn.execute(query, params)
            logger.info(f"Meeting '{session.id}' successfully saved to database.")
        except Exception as e:
            logger.error(f"Failed to save meeting '{session.id}' to database: {e}", exc_info=True)

    def _load_latest_meeting_from_db(self) -> None:
        """Load the most recent meeting from database into cache."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute("SELECT * FROM meetings ORDER BY started_at DESC LIMIT 1")
                row = cursor.fetchone()
                if row:
                    self._last_session = self._row_to_session(row)
                    logger.info(f"Loaded previous meeting from database: '{self._last_session.title}'")
        except Exception as e:
            logger.debug(f"Could not load previous meeting from DB: {e}")

    @staticmethod
    def _row_to_session(row: Any) -> MeetingSession:
        """Convert SQLite row to MeetingSession object."""
        try:
            raw_transcript = json.loads(row["transcript"] or "[]")
            utterances = [MeetingUtterance.from_dict(u) for u in raw_transcript]
        except Exception:
            utterances = []

        try:
            participants = set(json.loads(row["participants"] or "[]"))
        except Exception:
            participants = set()

        try:
            key_points = json.loads(row["key_points"] or "[]")
        except Exception:
            key_points = []

        try:
            action_items = json.loads(row["action_items"] or "[]")
        except Exception:
            action_items = []

        try:
            metadata = json.loads(row["metadata"] or "{}")
        except Exception:
            metadata = {}

        return MeetingSession(
            id=row["id"],
            title=row["title"],
            started_at=row["started_at"],
            ended_at=row["ended_at"] if row["ended_at"] else None,
            duration_sec=float(row["duration_sec"] or 0.0),
            utterances=utterances,
            participants=participants,
            summary=row["summary"] or "",
            key_points=key_points,
            action_items=action_items,
            analysis=metadata.get("analysis", ""),
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Super-Intelligent Meeting Analysis
    # ------------------------------------------------------------------

    def analyze_meeting(
        self,
        session: Optional[MeetingSession] = None,
        user_query: str = "",
    ) -> str:
        """Produce deep, executive-level meeting analysis in Bengali using LLM.

        Evaluates:
        1. মূল সারসংক্ষেপ ও প্রতিপাদ্য বিষয় (Executive Summary)
        2. অংশগ্রহণকারী ও তাদের বক্তব্য (Speaker Contributions)
        3. প্রধান সিদ্ধান্ত ও গুরুত্বপূর্ণ পয়েন্ট (Key Decisions & Highlights)
        4. অ্যাকশন আইটেমস ও পরবর্তী করণীয় (Action Items & Responsibilities)
        5. সামগ্রিক কৌশলগত পর্যালোচনা (Strategic Intelligence Assessment)
        """
        target = session or self._active_session or self._last_session
        if not target:
            return "বিশ্লেষণ করার মতো কোনো মিটিংয়ের তথ্য বা রেকর্ড খুঁজে পাওয়া যায়নি।"

        if not target.utterances:
            return f"'{target.title}' মিটিংটিতে কোনো কথা বা বক্তব্য রেকর্ড করা হয়নি।"

        # If analysis already generated and no custom query asked, return cached
        if target.analysis and not user_query:
            return target.analysis

        transcript_text = target.to_transcript_text()
        participants_str = ", ".join(target.participants) if target.participants else "অনির্দিষ্ট"
        duration_min = round(target.duration_sec / 60.0, 1) if target.duration_sec else "চলমান"

        system_instruction = (
            "You are LUMI's Executive Chief of Staff & Intelligence Analyst. "
            "You were present in the room silently recording a meeting. "
            "Now you must provide a super-intelligent, articulate, and deeply insightful analysis "
            "of the meeting in fluent, natural Bengali (বাংলা).\n\n"
            "Analyze the meeting transcript and structure your response with clarity:\n"
            "1. 📌 **মিটিংয়ের মূল সারসংক্ষেপ (Executive Summary)**: মিটিংটি মূলত কী উদ্দেশ্যে হয়েছিল এবং সামগ্রিক নির্যাস কী?\n"
            "2. 🗣️ **বক্তাদের বক্তব্য ও মতামত (Speaker-wise Breakdown)**: কে কী বিষয়ে কথা বলেছে, কার কী ভূমিকা ছিল?\n"
            "3. 🎯 **গুরুত্বপূর্ণ সিদ্ধান্ত ও হাইলাইটেড পয়েন্ট (Key Decisions & Highlights)**: মিটিংয়ের সবচেয়ে তাৎপর্যপূর্ণ সিদ্ধান্তগুলো কী কী?\n"
            "4. 📋 **অ্যাকশন আইটেম ও পরবর্তী করণীয় (Action Items & Next Steps)**: কার কী দায়িত্ব, কাকে কী করতে বলা হয়েছে?\n"
            "5. 💡 **কৌশলগত অন্তর্দৃষ্টি (Strategic Assessment)**: মিটিংয়ের সাফল্য বা গুরুত্বপূর্ণ কোনো লক্ষ্যণীয় দিক।\n\n"
            "Rules:\n"
            "- Speak directly, intelligently, and professionally in Bengali (বাংলা).\n"
            "- Maintain 100% factual accuracy strictly based on what was said.\n"
            "- If someone had a disagreement or reached an agreement, highlight it clearly.\n"
            "- Keep the tone respectful, sharp, and helpful."
        )

        user_prompt = (
            f"মিটিংয়ের শিরোনাম: {target.title}\n"
            f"অংশগ্রহণকারী: {participants_str}\n"
            f"সময়কাল: {duration_min} মিনিট\n\n"
            f"--- সম্পূর্ণ মিটিং ট্রানস্ক্রিপ্ট ---\n"
            f"{transcript_text}\n"
            f"-----------------------------------\n"
        )
        if user_query:
            user_prompt += f"\nব্যবহারকারীর সুনির্দিষ্ট প্রশ্ন/অনুরোধ: {user_query}\nউপরে উল্লেখিত প্রশ্নের ওপর ভিত্তি করে প্রয়োজনীয় বিশ্লেষণ উপস্থাপন করো।"

        analysis_text = self._call_llm(system_instruction, user_prompt)
        if analysis_text:
            target.analysis = analysis_text
            # Extract high-level summary (first paragraph or up to 300 chars)
            target.summary = analysis_text[:400]
            target.metadata["analysis"] = analysis_text
            self.save_meeting_to_db(target)
            return analysis_text

        # Fallback if LLM call fails
        fallback = (
            f"📊 **{target.title} এর সংক্ষিপ্ত রিপোর্ট:**\n"
            f"- অংশগ্রহণকারী: {participants_str}\n"
            f"- মোট বক্তব্য রেকর্ড: {len(target.utterances)} টি\n\n"
            f"ট্রানস্ক্রিপ্ট সারাংশ:\n{transcript_text[:500]}..."
        )
        return fallback

    def _call_llm(self, system_instruction: str, user_prompt: str) -> str:
        """Call available LLM (Gemini GenAI or OpenAI) for analysis."""
        # 1. Try Gemini via google-genai
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            try:
                from google import genai
                from google.genai import types

                client = genai.Client(api_key=gemini_key)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.3,
                    ),
                )
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                logger.warning(f"Gemini API analysis failed: {e}. Trying fallback...")

        # 2. Try OpenAI API
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=openai_key)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                )
                if response and response.choices:
                    return response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"OpenAI analysis failed: {e}")

        return ""

    # ------------------------------------------------------------------
    # PDF Report Export
    # ------------------------------------------------------------------

    def generate_pdf_report(
        self,
        session: Optional[MeetingSession] = None,
        output_dir: str = "data/documents",
    ) -> str:
        """Generate a PDF document of the meeting summary and action items."""
        from ..documents.pdf_generator import PDFReportGenerator

        target = session or self._active_session or self._last_session
        if not target:
            return ""

        # Ensure analysis is run
        if not target.analysis:
            self.analyze_meeting(target)

        pdf_gen = PDFReportGenerator(output_dir=output_dir)
        sections = {
            "Meeting Information": (
                f"Title: {target.title}\n"
                f"Date: {target.started_at[:10]}\n"
                f"Duration: {round(target.duration_sec / 60, 1)} min\n"
                f"Participants: {', '.join(target.participants)}"
            ),
            "Executive Analysis & Minutes": target.analysis or target.summary,
            "Chronological Transcript Excerpt": target.to_transcript_text()[:1500],
        }

        pdf_path = pdf_gen.generate_summary_pdf(
            title=f"LUMI Meeting Report: {target.title}",
            content_sections=sections,
            author="LUMI AI Companion Robot",
        )
        logger.info(f"Meeting report PDF generated: {pdf_path}")
        return pdf_path

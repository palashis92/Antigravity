"""LUMI Meeting Subsystem.

Provides silent meeting mode, real-time multi-speaker transcription,
conversation capture, and super-intelligent Bengali meeting analysis.
"""

from .manager import MeetingManager, MeetingSession, MeetingUtterance

__all__ = ["MeetingManager", "MeetingSession", "MeetingUtterance"]

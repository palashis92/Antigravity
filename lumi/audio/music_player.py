"""On-Demand Music Player Subsystem for LUMI.

Enables LUMI to search, download, cache, and play songs on-demand
via YouTube (yt-dlp) and local audio playback through MAX98357A I2S DAC.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Optional, Tuple

from ..core.logger import get_logger
from .speaker import SpeakerInterface

logger = get_logger("audio.music_player")


class MusicPlayer:
    """Handles searching, caching, and playing music tracks."""

    def __init__(
        self,
        speaker: SpeakerInterface,
        music_dir: str = "data/music",
    ) -> None:
        self.speaker = speaker
        self.music_dir = Path(music_dir)
        self.music_dir.mkdir(parents=True, exist_ok=True)
        self._current_track: Optional[str] = None
        self._is_playing = False
        self._lock = threading.Lock()

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @property
    def current_track(self) -> Optional[str]:
        return self._current_track

    def play(self, query: str) -> Tuple[bool, str]:
        """Search and play a song by title or query.

        Args:
            query: The song name or search query (e.g. "Stereo Love").

        Returns:
            Tuple of (success: bool, status_message: str)
        """
        clean_query = query.strip()
        if not clean_query:
            return False, "কোনো গানের নাম উল্লেখ করা হয়নি।"

        # Check local cache first
        cached_file = self._find_cached_song(clean_query)
        if cached_file:
            track_title = Path(cached_file).stem
            logger.info(f"Playing cached music track: {cached_file}")
            return self._start_playback(cached_file, track_title)

        # Download from YouTube in background or synchronously if fast
        audio_file, title = self._download_track(clean_query)
        if audio_file and os.path.exists(audio_file):
            return self._start_playback(audio_file, title or clean_query)

        return False, f"দুঃখিত, '{clean_query}' গানটি খুঁজে বা ডাউনলোড করতে সমস্যা হয়েছে।"

    def stop(self) -> bool:
        """Stop currently playing music."""
        with self._lock:
            self.speaker.stop()
            self._is_playing = False
            track = self._current_track
            self._current_track = None
            logger.info(f"Music playback stopped (last track: {track}).")
            return True

    def _start_playback(self, file_path: str, title: str) -> Tuple[bool, str]:
        """Initiate non-blocking audio playback through speaker."""
        with self._lock:
            # Stop any ongoing playback first
            self.speaker.stop()
            self._current_track = title
            self._is_playing = True

        success = self.speaker.play_file(file_path, block=False)
        if success:
            logger.info(f"🎶 Music playing now: '{title}' ({file_path})")
            return True, f"'{title}' গানটি বাজানো হচ্ছে..."
        else:
            self._is_playing = False
            return False, f"'{title}' গানটি প্লে করতে ব্যর্থ হয়েছে।"

    def _find_cached_song(self, query: str) -> Optional[str]:
        """Check if matching audio file exists in cache."""
        sanitized = self._sanitize_filename(query).lower()
        if not sanitized:
            return None

        for ext in (".mp3", ".wav", ".m4a", ".ogg"):
            for f in self.music_dir.glob(f"*{ext}"):
                f_name = f.stem.lower()
                if sanitized in f_name or f_name in sanitized:
                    return str(f)
        return None

    def _download_track(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """Search and download best audio using yt-dlp."""
        sanitized = self._sanitize_filename(query)
        target_template = str(self.music_dir / f"{sanitized}.%(ext)s")

        # 1. Try python yt_dlp library
        try:
            import yt_dlp  # type: ignore

            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": target_template,
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "192",
                    }
                ],
                "noplaylist": True,
                "quiet": True,
                "no_warnings": True,
                "default_search": "ytsearch1",
            }
            logger.info(f"Searching YouTube for '{query}' via yt_dlp library...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"ytsearch1:{query}", download=True)
                if "entries" in info and info["entries"]:
                    entry = info["entries"][0]
                    title = entry.get("title", query)
                else:
                    title = info.get("title", query)

                expected_mp3 = str(self.music_dir / f"{sanitized}.mp3")
                if os.path.exists(expected_mp3):
                    return expected_mp3, title

                # Look for whatever was downloaded
                for ext in (".mp3", ".m4a", ".opus", ".webm", ".wav"):
                    cand = str(self.music_dir / f"{sanitized}{ext}")
                    if os.path.exists(cand):
                        return cand, title
        except ImportError:
            logger.debug("yt_dlp Python package not installed. Trying CLI...")
        except Exception as e:
            logger.warning(f"yt_dlp library error: {e}. Trying CLI fallback...")

        # 2. Try yt-dlp CLI tool
        if shutil.which("yt-dlp"):
            try:
                cmd = [
                    "yt-dlp",
                    "-x",
                    "--audio-format", "mp3",
                    "--audio-quality", "192K",
                    "-o", target_template,
                    "--no-playlist",
                    "--quiet",
                    f"ytsearch1:{query}",
                ]
                subprocess.run(cmd, check=True, timeout=60)
                expected_mp3 = str(self.music_dir / f"{sanitized}.mp3")
                if os.path.exists(expected_mp3):
                    return expected_mp3, query
            except Exception as e:
                logger.error(f"yt-dlp CLI failed: {e}")

        return None, None

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Strip illegal filename characters."""
        return re.sub(r'[\\/*?:"<>|]', "_", name).strip()
